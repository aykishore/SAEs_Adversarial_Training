"""evaluation: exactly two numbers, for both model_orig (undefended
baseline) and model_cb (RR-patched):

  - clean zero-shot top-1 accuracy
  - adaptive targeted attack success rate (PGD regenerated directly against each
    model, gradients flowing through the LoRA adapters for model_cb — this is the
    real robustness test; replaying a static attack from model_orig against model_cb
    would overstate robustness)"""
import argparse

import torch

from attacks import pgd_targeted
from common import ClsTokenExtractor, NormalizingCLIPWrapper, build_zero_shot_classifier, load_clip, sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset
from lora import inject_lora, load_lora_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)
    p.add_argument("--num-eval-images", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=1 / 255)
    p.add_argument("--pgd-steps", type=int, default=200, help="stronger than train-time PGD")
    p.add_argument("--seed", type=int, default=1234, help="different from training seed on purpose")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def zero_shot_accuracy(wrapped_model, loader, text_embeds, device):
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        img_embeds = wrapped_model.encode_image(images)
        preds = (img_embeds @ text_embeds.T).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / total


@torch.no_grad()
def targeted_asr(wrapped_model, adv_images, target_labels, text_embeds, device, batch_size):
    correct, total = 0, 0
    for i in range(0, adv_images.shape[0], batch_size):
        batch = adv_images[i:i + batch_size].to(device)
        tgt = target_labels[i:i + batch_size].to(device)
        img_embeds = wrapped_model.encode_image(batch)
        preds = (img_embeds @ text_embeds.T).argmax(dim=-1)
        correct += (preds == tgt).sum().item()
        total += batch.shape[0]
    return correct / total


def generate_adaptive_adv_set(wrapped_model, images, text_embeds, target_labels, eps, alpha, steps, batch_size):
    logit_scale = wrapped_model.logit_scale
    adv_chunks = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size]
        tgt = target_labels[i:i + batch_size]
        adv_chunks.append(pgd_targeted(wrapped_model, batch, text_embeds, tgt, logit_scale,
                                        eps, alpha, steps).cpu())
    return torch.cat(adv_chunks)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"loaded checkpoint: LoRA blocks [{ckpt['start_layer']}, {ckpt['end_layer']}], "
          f"RR loss at block {ckpt['target_layer']}, trained with seed={ckpt['seed']}")

    model_orig, resize_crop, tokenizer = load_clip(ckpt["model_name"], ckpt["pretrained"], device)
    wrapped_orig = NormalizingCLIPWrapper(model_orig, device).to(device)

    model_cb, _, _ = load_clip(ckpt["model_name"], ckpt["pretrained"], device)
    inject_lora(model_cb, ckpt["start_layer"], ckpt["end_layer"], r=ckpt["lora_r"], alpha=ckpt["lora_alpha"])
    load_lora_state_dict(model_cb, ckpt["start_layer"], ckpt["end_layer"], ckpt["lora_state"], device)
    model_cb.eval()
    wrapped_cb = NormalizingCLIPWrapper(model_cb, device).to(device)

    dataset = build_dataset(args.dataset, args.data_root, False, resize_crop, args.classnames_json)
    classnames = dataset.classnames
    text_embeds = build_zero_shot_classifier(model_orig, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)

    full_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n=== Clean zero-shot top-1 accuracy ===")
    acc_orig = zero_shot_accuracy(wrapped_orig, full_loader, text_embeds, device)
    acc_cb = zero_shot_accuracy(wrapped_cb, full_loader, text_embeds, device)
    print(f"model_orig: {acc_orig:.4f}")
    print(f"model_cb  : {acc_cb:.4f}  (delta: {acc_cb - acc_orig:+.4f})")

    eval_subset = torch.utils.data.Subset(dataset, range(min(args.num_eval_images, len(dataset))))
    images, labels = next(iter(torch.utils.data.DataLoader(eval_subset, batch_size=len(eval_subset), shuffle=False)))
    images, labels = images.to(device), labels.to(device)
    target_labels = sample_wrong_targets(labels, len(classnames), args.seed)

    print(f"\n=== Adaptive targeted PGD ASR (eps={args.eps:.4f}, steps={args.pgd_steps}) ===")
    adv_orig = generate_adaptive_adv_set(wrapped_orig, images, text_embeds, target_labels,
                                          args.eps, args.pgd_alpha, args.pgd_steps, args.batch_size)
    adv_cb = generate_adaptive_adv_set(wrapped_cb, images, text_embeds, target_labels,
                                        args.eps, args.pgd_alpha, args.pgd_steps, args.batch_size)
    asr_orig = targeted_asr(wrapped_orig, adv_orig, target_labels, text_embeds, device, args.batch_size)
    asr_cb = targeted_asr(wrapped_cb, adv_cb, target_labels, text_embeds, device, args.batch_size)
    print(f"model_orig: {asr_orig:.4f}")
    print(f"model_cb  : {asr_cb:.4f}  (delta: {asr_cb - asr_orig:+.4f})")


if __name__ == "__main__":
    main()
