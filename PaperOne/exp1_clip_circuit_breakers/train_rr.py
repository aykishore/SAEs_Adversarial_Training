"""minimal representation-rerouting (circuit-breakers) adaptation for
CLIP's vision encoder.

Loss:

    L_s = ReLU( cos_sim( rep_cb(x_c), rep_orig(x_c) ) )      circuit-breaker loss
    L_r = || rep_orig(x_r) - rep_cb(x_r) ||_2                retain loss
    c_s(t) = alpha * (1 - t / (2T))                          decays over training
    c_r(t) = alpha * (t / (2T))                              grows over training
    L = c_s(t) * L_s + c_r(t) * L_r

rep(.) is the CLS-token residual stream at one chosen vision-transformer block
(pick it with probe_layer_divergence.py first). model_orig is frozen and used only
as a reference; model_cb has LoRA adapters on the MLP linears of a small window of
blocks around the target layer, e.g. [target_layer-2, target_layer], not from block 0
onward — that "from block 0" placement is specific to the paper's text-only LLM
experiment; their multimodal (LLaVA) setup instead uses a narrow window (LLM layers
14-16) around the RR layer (16), which is what this mirrors."""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from attacks import pgd_targeted
from common import ClsTokenExtractor, NormalizingCLIPWrapper, build_zero_shot_classifier, load_clip, sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset
from lora import inject_lora, lora_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--target-layer", type=int, required=True, help="from probe_layer_divergence.py")
    p.add_argument("--lora-window", type=int, default=3, help="LoRA covers [target_layer-window+1, target_layer]")
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-b-init-std", type=float, default=1e-3,
                    help="nonzero init breaks the zero-gradient saddle point L_s sits at when "
                         "model_cb starts bit-identical to model_orig; see lora.py docstring")

    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)

    p.add_argument("--num-circuit-breaker-images", type=int, default=2000)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=1 / 255)
    p.add_argument("--pgd-steps", type=int, default=20, help="cheap PGD for building x_c; keep small")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--total-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rr-alpha", type=float, default=5.0, help="the paper's alpha scaling constant for c_s/c_r")
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="rr_checkpoint.pt")
    return p.parse_args()


def build_circuit_breaker_set(wrapped_orig, tokenizer, model_orig, dataset, classnames, args, device):
    """Precomputes the paired (clean, adversarial) circuit-breaker set once, up front."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    text_embeds = build_zero_shot_classifier(model_orig, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)
    logit_scale = wrapped_orig.logit_scale

    clean_chunks, adv_chunks = [], []
    n_collected = 0
    seed = args.seed
    for images, labels in tqdm(loader, desc="building circuit-breaker set"):
        if n_collected >= args.num_circuit_breaker_images:
            break
        images, labels = images.to(device), labels.to(device)
        target_labels = sample_wrong_targets(labels, len(classnames), seed)
        seed += 1
        adv_images = pgd_targeted(wrapped_orig, images, text_embeds, target_labels, logit_scale,
                                   args.eps, args.pgd_alpha, args.pgd_steps)

        clean_chunks.append(images.cpu())
        adv_chunks.append(adv_images.cpu())
        n_collected += images.shape[0]

    return torch.cat(clean_chunks)[:args.num_circuit_breaker_images], \
        torch.cat(adv_chunks)[:args.num_circuit_breaker_images]


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
    print(f"LoRA in blocks [{start_layer}, {end_layer}], RR loss applied at block {args.target_layer}")

    dataset = build_dataset(args.dataset, args.data_root, True, resize_crop, args.classnames_json)
    classnames = dataset.classnames

    x_clean, x_adv = build_circuit_breaker_set(wrapped_orig, tokenizer, model_orig, dataset, classnames, args, device)
    print(f"circuit-breaker set: {x_clean.shape[0]} paired (clean, adversarial) images")

    extractor_orig = ClsTokenExtractor(model_orig, args.target_layer)
    extractor_cb = ClsTokenExtractor(model_cb, args.target_layer)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    n = x_clean.shape[0]

    for step in range(args.total_steps):
        idx = torch.randint(0, n, (args.batch_size,))
        batch_clean = x_clean[idx].to(device)
        batch_adv = x_adv[idx].to(device)

        c_s = args.rr_alpha * (1 - step / (2 * args.total_steps))
        c_r = args.rr_alpha * (step / (2 * args.total_steps))

        with torch.no_grad():
            wrapped_orig.encode_image(batch_clean)
            rep_orig_clean = extractor_orig.get_cls().detach()
            wrapped_orig.encode_image(batch_adv)
            rep_orig_adv = extractor_orig.get_cls().detach()

        wrapped_cb.encode_image(batch_clean)
        rep_cb_clean = extractor_cb.get_cls()
        wrapped_cb.encode_image(batch_adv)
        rep_cb_adv = extractor_cb.get_cls()

        adv_cos_raw = F.cosine_similarity(rep_cb_adv, rep_orig_adv, dim=-1)
        clean_cos_raw = F.cosine_similarity(rep_cb_clean, rep_orig_clean, dim=-1)
        l_s = F.relu(adv_cos_raw).mean()
        l_r = (rep_orig_clean - rep_cb_clean).norm(dim=-1).mean()
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
            clean_cos = clean_cos_raw.mean().item()
            # L_s/adv_cos sit within ~1e-5 of their ceiling (1.0); :.4f rounds away exactly the
            # signal we care about, so report both the raw value and 1-x for readability
            print(f"step {step:4d}  c_s={c_s:.3f} c_r={c_r:.3f}  "
                  f"adv_cos={adv_cos:.8f} (1-adv_cos={1 - adv_cos:.3e})  "
                  f"clean_cos={clean_cos:.8f} (1-clean_cos={1 - clean_cos:.3e})  "
                  f"L_s={l_s.item():.8f} L_r={l_r.item():.6f} loss={loss.item():.6f}  "
                  f"|grad L_s|={grad_norm_s:.6e}")

    extractor_orig.remove()
    extractor_cb.remove()

    checkpoint = {
        "lora_state": lora_state_dict(model_cb, start_layer, end_layer),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "target_layer": args.target_layer,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_b_init_std": args.lora_b_init_std,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.out)
    print(f"saved checkpoint to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
