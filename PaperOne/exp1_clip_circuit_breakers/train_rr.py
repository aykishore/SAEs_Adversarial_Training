"""minimal representation-rerouting (circuit-breakers) adaptation for
CLIP's vision encoder.

Loss:

    L_s = ReLU( cos_sim( rep_cb(x_c), rep_orig(x_c) ) )      circuit-breaker loss
    L_r = || rep_orig(x_r) - rep_cb(x_r) ||_2                retain loss
    c_s(t) = alpha * (1 - t / (2T))                          decays over training
    c_r(t) = alpha * (t / (2T))                              grows over training
    L = c_s(t) * L_s + c_r(t) * L_r

rep(.) is the residual stream at one chosen vision-transformer block (pick it with
probe_layer_separability.py first). With --rep-tokens cls it is the CLS token only; with
--rep-tokens all, cos_sim / L2 are taken per token over CLS + all 49 patch tokens and
averaged, which is what the paper does (its losses run over every token position, not a
single summary token). model_orig is frozen and used only as a reference; model_cb has
LoRA adapters on the MLP linears of a small window of blocks around the target layer,
e.g. [target_layer-2, target_layer], not from block 0 onward — that "from block 0"
placement is specific to the paper's text-only LLM experiment; their multimodal (LLaVA)
setup instead uses a narrow window (LLM layers 14-16) around the RR layer (16), which is
what this mirrors.

Circuit-breaker set: only (clean, adversarial) pairs where model_orig classifies the clean
image correctly AND the targeted attack succeeds. A failed attack still represents the true
class, so rerouting it would directly fight L_r; the paper's circuit-breaker set likewise
contains only actually-harmful completions.

Retain set: the clean twin of each circuit-breaker image (paired, a cleaner signal for
CLIP) plus an unpaired pool of other clean training images, sampled independently as in
the paper's Algorithm 1 (x_r ~ D_r), so L_r covers more of the input distribution than
just the 2000 attacked images."""

import argparse
import hashlib
import json
import os
import tempfile

import torch
import torch.nn.functional as F
from tqdm import tqdm

from attacks import pgd_targeted
from common import NormalizingCLIPWrapper, RepExtractor, build_zero_shot_classifier, load_clip, sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset
from lora import inject_lora, lora_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--target-layer", type=int, required=True, help="from probe_layer_separability.py")
    p.add_argument("--rep-tokens", default="cls", choices=["cls", "all"],
                    help="cls: RR loss on the CLS token only; all: per-token over CLS + patch tokens")
    p.add_argument("--lora-window", type=int, default=3, help="LoRA covers [target_layer-window+1, target_layer]")
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-b-init-std", type=float, default=1e-3,
                    help="nonzero init breaks the zero-gradient saddle point L_s sits at when "
                         "model_cb starts bit-identical to model_orig; see lora.py docstring")

    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)

    p.add_argument("--num-circuit-breaker-images", type=int, default=2000,
                    help="number of *successful* attack pairs to collect")
    p.add_argument("--num-retain-images", type=int, default=2000,
                    help="size of the extra unpaired clean retain pool (0 = paired retain only)")
    p.add_argument("--retain-batch-size", type=int, default=32,
                    help="unpaired retain images per step, on top of the paired clean twins")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=1 / 255)
    p.add_argument("--pgd-steps", type=int, default=20, help="cheap PGD for building x_c; keep small")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rr-alpha", type=float, default=5.0, help="the paper's alpha scaling constant for c_s/c_r")
    p.add_argument("--ls-warmup-frac", type=float, default=0.0,
                    help="fraction of steps at the start with c_r forced to 0 (our addition, not in the paper). "
                         "Near model_cb == model_orig, L_s = cos has a ~zero gradient while L_r = ||.||_2 has an "
                         "O(1) one, so even a small c_r pulls the LoRA back to a no-op before L_s can escape; "
                         "a short L_s-only phase lets it escape first, then the paper's schedule resumes")
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="rr_checkpoint.pt")
    p.add_argument("--cache-dir", default=os.path.join(tempfile.gettempdir(), "clip_rr_cache"),
                    help="where the circuit-breaker set / retain pool are cached; they don't depend on the "
                         "target layer or token mode, so a layer sweep builds them once. Defaults to the "
                         "system temp dir to keep multi-GB files out of synced folders. '' disables caching")
    return p.parse_args()


def training_sets_cache_path(args):
    """Cache file keyed on every argument the training sets depend on."""
    key = (args.model_name, args.pretrained, args.dataset, os.path.abspath(args.data_root), args.classnames_json,
           args.num_circuit_breaker_images, args.num_retain_images, args.eps, args.pgd_alpha, args.pgd_steps,
           args.batch_size, args.seed)
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    return os.path.join(args.cache_dir, f"training_sets_{digest}.pt")


def build_training_sets(wrapped_orig, tokenizer, model_orig, dataset, classnames, args, device):
    """Precomputes, once up front, the filtered paired (clean, adversarial) circuit-breaker
    set and the unpaired retain pool. The two are drawn from disjoint images."""
    text_embeds = build_zero_shot_classifier(model_orig, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)
    logit_scale = wrapped_orig.logit_scale
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()

    clean_chunks, adv_chunks = [], []
    stats = {"images_seen": 0, "clean_correct": 0, "attack_success": 0}
    seed = args.seed
    pos = 0
    pbar = tqdm(total=args.num_circuit_breaker_images, desc="building circuit-breaker set")
    while stats["attack_success"] < args.num_circuit_breaker_images:
        if pos >= len(order):
            raise RuntimeError(f"dataset exhausted after {stats}; lower --num-circuit-breaker-images")
        batch_idx = order[pos:pos + args.batch_size]
        pos += len(batch_idx)
        images = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        labels = torch.tensor([dataset[i][1] for i in batch_idx], device=device)
        stats["images_seen"] += len(batch_idx)

        with torch.no_grad():
            correct = (wrapped_orig.encode_image(images) @ text_embeds.T).argmax(dim=-1) == labels
        images, labels = images[correct], labels[correct]
        stats["clean_correct"] += int(correct.sum())
        if images.shape[0] == 0:
            continue

        target_labels = sample_wrong_targets(labels, len(classnames), seed).to(device)
        seed += 1
        adv_images = pgd_targeted(wrapped_orig, images, text_embeds, target_labels, logit_scale,
                                   args.eps, args.pgd_alpha, args.pgd_steps)
        with torch.no_grad():
            success = (wrapped_orig.encode_image(adv_images) @ text_embeds.T).argmax(dim=-1) == target_labels

        clean_chunks.append(images[success].cpu())
        adv_chunks.append(adv_images[success].cpu())
        stats["attack_success"] += int(success.sum())
        pbar.update(int(success.sum()))
    pbar.close()

    # stored as float16 to halve RAM / cache size: the rounding (<0.1/255 per pixel) is far below
    # the perturbation scale and only affects training data, never evaluation
    n = args.num_circuit_breaker_images
    x_clean, x_adv = torch.cat(clean_chunks)[:n].half(), torch.cat(adv_chunks)[:n].half()

    retain_idx = order[pos:pos + args.num_retain_images]
    x_retain = torch.stack([dataset[i][0] for i in tqdm(retain_idx, desc="building retain pool")]).half() \
        if retain_idx else None
    stats["retain_pool"] = len(retain_idx)
    return x_clean, x_adv, x_retain, stats


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    start_layer = max(0, args.target_layer - args.lora_window + 1)
    end_layer = args.target_layer

    model_orig, resize_crop, tokenizer = load_clip(args.model_name, args.pretrained, device)
    wrapped_orig = NormalizingCLIPWrapper(model_orig, device).to(device)

    model_cb, _, _ = load_clip(args.model_name, args.pretrained, device)
    lora_params = inject_lora(model_cb, start_layer, end_layer, r=args.lora_r, alpha=args.lora_alpha,
                               b_init_std=args.lora_b_init_std)
    model_cb.train()
    wrapped_cb = NormalizingCLIPWrapper(model_cb, device).to(device)
    print(f"LoRA in blocks [{start_layer}, {end_layer}], RR loss applied at block {args.target_layer} "
          f"on {args.rep_tokens} token(s)")

    dataset = build_dataset(args.dataset, args.data_root, True, resize_crop, args.classnames_json)
    classnames = dataset.classnames

    cache_path = training_sets_cache_path(args) if args.cache_dir else None
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path)
        x_clean, x_adv, x_retain, set_stats = cached["x_clean"], cached["x_adv"], cached["x_retain"], cached["stats"]
        print(f"loaded cached training sets from {cache_path}")
    else:
        x_clean, x_adv, x_retain, set_stats = build_training_sets(
            wrapped_orig, tokenizer, model_orig, dataset, classnames, args, device)
        if cache_path:
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save({"x_clean": x_clean, "x_adv": x_adv, "x_retain": x_retain, "stats": set_stats}, cache_path)
            print(f"cached training sets to {cache_path}")
    print(f"circuit-breaker set: {x_clean.shape[0]} successful (clean, adversarial) pairs "
          f"from {set_stats['images_seen']} images seen ({set_stats['clean_correct']} clean-correct); "
          f"unpaired retain pool: {set_stats['retain_pool']} images")

    # stop_forward: only blocks [0, target_layer] are computed, see RepExtractor docstring
    extractor_orig = RepExtractor(model_orig, args.target_layer, args.rep_tokens, stop_forward=True)
    extractor_cb = RepExtractor(model_cb, args.target_layer, args.rep_tokens, stop_forward=True)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    n = x_clean.shape[0]
    log = []

    for step in range(args.total_steps):
        idx = torch.randint(0, n, (args.batch_size,))
        batch_adv = x_adv[idx].to(device).float()
        batch_retain = x_clean[idx]
        if x_retain is not None and args.retain_batch_size > 0:
            retain_idx = torch.randint(0, x_retain.shape[0], (args.retain_batch_size,))
            batch_retain = torch.cat([batch_retain, x_retain[retain_idx]])
        batch_retain = batch_retain.to(device).float()

        c_s = args.rr_alpha * (1 - step / (2 * args.total_steps))
        c_r = args.rr_alpha * (step / (2 * args.total_steps))
        if step < args.ls_warmup_frac * args.total_steps:
            c_r = 0.0

        with torch.no_grad():
            rep_orig_retain = extractor_orig.run(wrapped_orig, batch_retain).detach()
            rep_orig_adv = extractor_orig.run(wrapped_orig, batch_adv).detach()

        rep_cb_retain = extractor_cb.run(wrapped_cb, batch_retain)
        rep_cb_adv = extractor_cb.run(wrapped_cb, batch_adv)

        # cos / L2 over the hidden dim; mean over batch (and tokens, for --rep-tokens all)
        adv_cos_raw = F.cosine_similarity(rep_cb_adv, rep_orig_adv, dim=-1)
        retain_cos_raw = F.cosine_similarity(rep_cb_retain, rep_orig_retain, dim=-1)
        l_s = F.relu(adv_cos_raw).mean()
        l_r = (rep_orig_retain - rep_cb_retain).norm(dim=-1).mean()
        loss = c_s * l_s + c_r * l_r

        is_log_step = step % args.log_every == 0 or step == args.total_steps - 1
        grad_norm_s = None
        if is_log_step:
            # gradient of L_s alone, isolated from L_r and from the c_s/c_r schedule weights,
            # to check whether the rerouting loss itself is producing a meaningful gradient --
            # retain_graph=True so this extra backward doesn't free the graph loss.backward() needs
            grads = torch.autograd.grad(l_s, lora_params, retain_graph=True, allow_unused=True)
            grad_norm_s = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_log_step:
            adv_cos = adv_cos_raw.mean().item()
            retain_cos = retain_cos_raw.mean().item()
            # L_s/adv_cos sit within ~1e-5 of their ceiling (1.0); :.4f rounds away exactly the
            # signal we care about, so report both the raw value and 1-x for readability
            print(f"step {step:4d}  c_s={c_s:.3f} c_r={c_r:.3f}  "
                  f"adv_cos={adv_cos:.8f} (1-adv_cos={1 - adv_cos:.3e})  "
                  f"retain_cos={retain_cos:.8f} (1-retain_cos={1 - retain_cos:.3e})  "
                  f"L_s={l_s.item():.8f} L_r={l_r.item():.6f} loss={loss.item():.6f}  "
                  f"|grad L_s|={grad_norm_s:.6e}")
            log.append({"step": step, "c_s": c_s, "c_r": c_r, "adv_cos": adv_cos, "retain_cos": retain_cos,
                        "l_s": l_s.item(), "l_r": l_r.item(), "loss": loss.item(), "grad_norm_l_s": grad_norm_s})

    extractor_orig.remove()
    extractor_cb.remove()

    checkpoint = {
        "lora_state": lora_state_dict(model_cb, start_layer, end_layer),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "target_layer": args.target_layer,
        "rep_tokens": args.rep_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_b_init_std": args.lora_b_init_std,
        "ls_warmup_frac": args.ls_warmup_frac,
        "train_eps": args.eps,
        "training_set_stats": set_stats,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.out)
    log_path = os.path.splitext(args.out)[0] + "_train_log.json"
    with open(log_path, "w") as f:
        json.dump({"args": vars(args), "training_set_stats": set_stats, "log": log}, f, indent=2)
    print(f"saved checkpoint to {os.path.abspath(args.out)} and training log to {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
