"""evaluation of model_orig (undefended baseline) vs model_cb (RR-patched). Everything is
written to a JSON file (--out-json) as well as printed.

  - clean zero-shot top-1 accuracy, on the eval dataset and on a second held-out dataset
    (--capability-dataset, CIFAR-10 by default) as a check that clean capability is retained
  - clean representation drift: cos_sim(model_cb, model_orig) on clean test images, at the
    RR target layer and at the final image embedding
  - for every eps in --eps-list, under targeted PGD:
      targeted ASR       prediction == attacker's target label
      robust accuracy    prediction == true label
      untargeted error   prediction != true label
    for three settings:
      model_orig         attacked directly
      model_cb adaptive  PGD regenerated directly against model_cb, gradients flowing through
                         the LoRA adapters -- the real robustness test
      model_cb transfer  model_orig's adversarial images replayed against model_cb

Targeted ASR alone can be gamed: a defense that turns adversarial images into arbitrary
wrong predictions lowers it without restoring the true label, so robust accuracy and
untargeted error are reported next to it. Transfer is the gradient-masking check: an
attack crafted on a *different* model shouldn't beat one crafted on model_cb itself, so if
transfer ASR > adaptive ASR (by more than --masking-tolerance), model_cb's gradients are
likely obfuscated and the adaptive number overstates robustness."""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from attacks import pgd_targeted
from common import NormalizingCLIPWrapper, RepExtractor, build_zero_shot_classifier, load_clip, predict, \
    sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset
from lora import inject_lora, load_lora_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)
    p.add_argument("--capability-dataset", default="cifar10", choices=["cifar10", "none"])
    p.add_argument("--capability-data-root", default="./data")
    p.add_argument("--num-clean-images", type=int, default=0,
                    help="clean accuracy on the first N test images of each dataset (0 = full test set); "
                         "both models see the same images, so the paired delta stays meaningful")
    p.add_argument("--num-eval-images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for the clean-accuracy passes (CPU resize is the bottleneck)")
    p.add_argument("--eps-list", default="1,2,4,8", help="comma-separated L-inf budgets, in units of 1/255")
    p.add_argument("--pgd-alpha-ratio", type=float, default=0.125,
                    help="PGD step size as a fraction of eps (0.125 -> 1/255 at eps=8/255)")
    p.add_argument("--pgd-steps", type=int, default=100, help="stronger than train-time PGD")
    p.add_argument("--masking-tolerance", type=float, default=0.05,
                    help="flag gradient masking only if transfer ASR exceeds adaptive ASR by more than this; "
                         "PGD's random start alone makes the two differ slightly")
    p.add_argument("--orig-cache", default=None,
                    help="optional .pt cache for model_orig's adversarial images, shared across "
                         "checkpoints (they don't depend on model_cb); used by run_layer_sweep.py")
    p.add_argument("--seed", type=int, default=1234, help="different from training seed on purpose")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-json", default=None, help="defaults to <checkpoint>_eval.json")
    return p.parse_args()


def attack_metrics(preds, labels, target_labels):
    return {
        "targeted_asr": (preds == target_labels).float().mean().item(),
        "robust_accuracy": (preds == labels).float().mean().item(),
        "untargeted_error": (preds != labels).float().mean().item(),
    }


def generate_adv_set(wrapped_model, images, text_embeds, target_labels, eps, alpha, steps, batch_size):
    device = text_embeds.device
    logit_scale = wrapped_model.logit_scale
    adv_chunks = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        tgt = target_labels[i:i + batch_size]
        adv_chunks.append(pgd_targeted(wrapped_model, batch, text_embeds, tgt, logit_scale,
                                        eps, alpha, steps).cpu())
    return torch.cat(adv_chunks)


@torch.no_grad()
def representation_drift(wrapped_orig, wrapped_cb, model_orig, model_cb, images, layer, rep_tokens, batch_size):
    """Mean cos_sim(model_cb, model_orig) on clean images, at the RR layer and at the output."""
    device = next(model_orig.parameters()).device
    ext_orig = RepExtractor(model_orig, layer, rep_tokens)
    ext_cb = RepExtractor(model_cb, layer, rep_tokens)
    layer_cos, embed_cos = [], []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        emb_orig = wrapped_orig.encode_image(batch)
        emb_cb = wrapped_cb.encode_image(batch)
        layer_cos.append(F.cosine_similarity(ext_cb.get(), ext_orig.get(), dim=-1).flatten().cpu())
        embed_cos.append(F.cosine_similarity(emb_cb, emb_orig, dim=-1).cpu())
    ext_orig.remove()
    ext_cb.remove()
    return {"target_layer_cos": torch.cat(layer_cos).mean().item(),
            "final_embedding_cos": torch.cat(embed_cos).mean().item()}


def load_orig_cache(path, key):
    if path and os.path.exists(path):
        cache = torch.load(path, map_location="cpu")
        if cache["key"] == key:
            return cache["adv"]
        print(f"ignoring {path}: built with a different config")
    return {}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    eps_list = [float(e) for e in args.eps_list.split(",")]
    out_json = args.out_json or os.path.splitext(args.checkpoint)[0] + "_eval.json"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rep_tokens = ckpt.get("rep_tokens", "cls")  # checkpoints from before --rep-tokens existed are CLS-only
    print(f"loaded checkpoint: LoRA blocks [{ckpt['start_layer']}, {ckpt['end_layer']}], "
          f"RR loss at block {ckpt['target_layer']} on {rep_tokens} token(s), trained with seed={ckpt['seed']}")

    model_orig, resize_crop, tokenizer = load_clip(ckpt["model_name"], ckpt["pretrained"], device)
    wrapped_orig = NormalizingCLIPWrapper(model_orig, device).to(device)

    model_cb, _, _ = load_clip(ckpt["model_name"], ckpt["pretrained"], device)
    inject_lora(model_cb, ckpt["start_layer"], ckpt["end_layer"], r=ckpt["lora_r"], alpha=ckpt["lora_alpha"])
    load_lora_state_dict(model_cb, ckpt["start_layer"], ckpt["end_layer"], ckpt["lora_state"], device)
    model_cb.eval()
    wrapped_cb = NormalizingCLIPWrapper(model_cb, device).to(device)

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "target_layer": ckpt["target_layer"],
        "lora_blocks": [ckpt["start_layer"], ckpt["end_layer"]],
        "rep_tokens": rep_tokens,
        "eval_args": vars(args),
    }

    dataset = build_dataset(args.dataset, args.data_root, False, resize_crop, args.classnames_json)
    classnames = dataset.classnames
    text_embeds = build_zero_shot_classifier(model_orig, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)

    print("\n=== Clean zero-shot top-1 accuracy ===")
    clean = {}
    eval_sets = [(args.dataset, dataset, text_embeds)]
    if args.capability_dataset != "none":
        cap_dataset = build_dataset(args.capability_dataset, args.capability_data_root, False, resize_crop)
        cap_text = build_zero_shot_classifier(model_orig, tokenizer, cap_dataset.classnames, ZERO_SHOT_TEMPLATE, device)
        eval_sets.append((args.capability_dataset, cap_dataset, cap_text))
    for name, ds, txt in eval_sets:
        if args.num_clean_images:
            ds = torch.utils.data.Subset(ds, range(min(args.num_clean_images, len(ds))))
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        correct = {"model_orig": 0, "model_cb": 0}
        total = 0
        for images, labels in loader:
            correct["model_orig"] += (predict(wrapped_orig, images, txt, args.batch_size) == labels).sum().item()
            correct["model_cb"] += (predict(wrapped_cb, images, txt, args.batch_size) == labels).sum().item()
            total += labels.shape[0]
        clean[name] = {k: v / total for k, v in correct.items()}
        print(f"{name:10s} model_orig: {clean[name]['model_orig']:.4f}   model_cb: {clean[name]['model_cb']:.4f}  "
              f"(delta: {clean[name]['model_cb'] - clean[name]['model_orig']:+.4f})")
    results["clean_accuracy"] = clean

    eval_subset = torch.utils.data.Subset(dataset, range(min(args.num_eval_images, len(dataset))))
    images, labels = next(iter(torch.utils.data.DataLoader(eval_subset, batch_size=len(eval_subset), shuffle=False)))
    target_labels = sample_wrong_targets(labels, len(classnames), args.seed)

    drift = representation_drift(wrapped_orig, wrapped_cb, model_orig, model_cb, images,
                                 ckpt["target_layer"], rep_tokens, args.batch_size)
    results["clean_representation_drift"] = drift
    print(f"\nclean-image cos_sim(model_cb, model_orig): block {ckpt['target_layer']} "
          f"{drift['target_layer_cos']:.4f}, final embedding {drift['final_embedding_cos']:.4f}")

    cache_key = {"dataset": args.dataset, "n": len(eval_subset), "eps_list": eps_list, "steps": args.pgd_steps,
                 "alpha_ratio": args.pgd_alpha_ratio, "seed": args.seed,
                 "model": [ckpt["model_name"], ckpt["pretrained"]]}
    adv_orig_cache = load_orig_cache(args.orig_cache, cache_key)

    results["attacks"] = []
    for eps_255 in eps_list:
        eps, alpha = eps_255 / 255, eps_255 / 255 * args.pgd_alpha_ratio
        print(f"\n=== Targeted PGD, eps={eps_255:g}/255, step={alpha * 255:.3f}/255, steps={args.pgd_steps} ===")
        if eps_255 not in adv_orig_cache:
            adv_orig_cache[eps_255] = generate_adv_set(wrapped_orig, images, text_embeds, target_labels,
                                                       eps, alpha, args.pgd_steps, args.batch_size)
            if args.orig_cache:
                torch.save({"key": cache_key, "adv": adv_orig_cache}, args.orig_cache)
        adv_orig = adv_orig_cache[eps_255]
        adv_cb = generate_adv_set(wrapped_cb, images, text_embeds, target_labels,
                                  eps, alpha, args.pgd_steps, args.batch_size)

        row = {
            "eps_255": eps_255,
            "model_orig": attack_metrics(predict(wrapped_orig, adv_orig, text_embeds, args.batch_size),
                                         labels, target_labels),
            "model_cb_adaptive": attack_metrics(predict(wrapped_cb, adv_cb, text_embeds, args.batch_size),
                                                labels, target_labels),
            "model_cb_transfer": attack_metrics(predict(wrapped_cb, adv_orig, text_embeds, args.batch_size),
                                                labels, target_labels),
        }
        row["gradient_masking_suspected"] = row["model_cb_transfer"]["targeted_asr"] > \
            row["model_cb_adaptive"]["targeted_asr"] + args.masking_tolerance
        results["attacks"].append(row)

        print(f"{'':20s}{'targeted ASR':>14s}{'robust acc':>12s}{'untargeted err':>16s}")
        for setting in ("model_orig", "model_cb_adaptive", "model_cb_transfer"):
            m = row[setting]
            print(f"{setting:20s}{m['targeted_asr']:14.4f}{m['robust_accuracy']:12.4f}{m['untargeted_error']:16.4f}")
        if row["gradient_masking_suspected"]:
            print(f"WARNING: transfer ASR > adaptive ASR + {args.masking_tolerance} -- likely gradient masking; "
                  "the adaptive number overstates robustness")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results to {os.path.abspath(out_json)}")


if __name__ == "__main__":
    main()
