"""Preliminary layer-selection analysis for choosing where to apply the RR loss.

This is NOT a reproduction of anything in the circuit-breakers paper's layer choice.
Their text-only LLM experiment targets layers 10/20 of a 32-layer model; their
multimodal (LLaVA) experiment applies the RR loss at layer 16 of the Mistral backbone.
Neither number transfers to a 12-layer CLIP ViT-B/32 vision encoder, which has a
different architecture and depth. Because of that mismatch, we run our own small
probe here: generate adversarial images against the frozen model, and check where
clean vs. adversarial CLS-token representations start to diverge across the 12
vision-transformer blocks, as a starting point for picking a target layer."""
import argparse

import torch
import torch.nn.functional as F

from attacks import pgd_targeted
from common import ClsTokenExtractor, NormalizingCLIPWrapper, build_zero_shot_classifier, load_clip, sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--attack-batch-size", type=int, default=32,
                    help="PGD is run in chunks of this size, not on all --num-images at once, "
                         "to avoid exhausting GPU memory on small cards during the backward pass")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=1 / 255)
    p.add_argument("--pgd-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--plot-out", default="layer_divergence.png")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    model, resize_crop, tokenizer = load_clip(args.model_name, args.pretrained, device)
    wrapped = NormalizingCLIPWrapper(model, device).to(device)

    dataset = build_dataset(args.dataset, args.data_root, True, resize_crop, args.classnames_json)
    classnames = dataset.classnames

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.num_images, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    text_embeds = build_zero_shot_classifier(model, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)
    target_labels = sample_wrong_targets(labels, len(classnames), args.seed)

    print(f"Generating {args.num_images} adversarial images (eps={args.eps:.4f}, steps={args.pgd_steps}, "
          f"attack batch size={args.attack_batch_size})...")
    logit_scale = wrapped.logit_scale
    adv_chunks = []
    for i in range(0, images.shape[0], args.attack_batch_size):
        batch = images[i:i + args.attack_batch_size]
        tgt = target_labels[i:i + args.attack_batch_size]
        adv_chunks.append(pgd_targeted(wrapped, batch, text_embeds, tgt, logit_scale,
                                        args.eps, args.pgd_alpha, args.pgd_steps))
    adv_images = torch.cat(adv_chunks)

    num_layers = len(model.visual.transformer.resblocks)
    sims = []
    with torch.no_grad():
        for layer in range(num_layers):
            extractor = ClsTokenExtractor(model, layer)
            wrapped.encode_image(images)
            cls_clean = extractor.get_cls()
            wrapped.encode_image(adv_images)
            cls_adv = extractor.get_cls()
            extractor.remove()
            sim = F.cosine_similarity(cls_clean, cls_adv, dim=-1).mean().item()
            sims.append(sim)
            print(f"layer {layer:2d}: mean cos_sim(clean, adv) = {sim:.4f}")

    drops = [sims[i - 1] - sims[i] for i in range(1, num_layers)]
    recommended = max(range(1, num_layers), key=lambda i: drops[i - 1])
    print(f"\nCandidate target layer (largest single-step similarity drop): {recommended}")
    print("This is a starting point, not a definitive choice — sanity-check it against "
          "training stability before committing.")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(range(num_layers), sims, marker="o")
        plt.axvline(recommended, color="red", linestyle="--", label=f"candidate layer {recommended}")
        plt.xlabel("vision-transformer block")
        plt.ylabel("mean cos_sim(clean, adversarial) CLS token")
        plt.title("Layer divergence probe (our own analysis, not from the paper)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_out)
        print(f"Saved plot to {args.plot_out}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
