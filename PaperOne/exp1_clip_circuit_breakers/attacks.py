"""Targeted L-infinity PGD against CLIP's zero-shot classification decision."""
import torch
import torch.nn.functional as F


def pgd_targeted(wrapped_model, images_01: torch.Tensor, text_embeds: torch.Tensor,
                  target_labels: torch.Tensor, logit_scale: float,
                  eps: float, alpha: float, steps: int) -> torch.Tensor:
    """Optimizes pixels to make CLIP's zero-shot prediction flip to target_labels.

    Uses cross-entropy over the full zero-shot logits (logit_scale * image_embed @
    text_embeds.T), i.e. the same decision rule build_zero_shot_classifier is scored
    with, rather than only maximizing similarity to the target class in isolation —
    this makes the attack explicitly compete against the true class and every other
    class, not just push toward one target direction.

    images_01: [N, 3, H, W] in [0, 1], not yet CLIP-normalized.
    text_embeds: [num_classes, D], L2-normalized.
    target_labels: [N] int64 class indices.
    Returns adversarial images in [0, 1], same shape as input, detached.
    """
    images_01 = images_01.detach()
    target_labels = target_labels.to(images_01.device)
    delta = torch.empty_like(images_01).uniform_(-eps, eps)
    delta = torch.clamp(images_01 + delta, 0, 1) - images_01
    delta.requires_grad_(True)

    for _ in range(steps):
        adv = torch.clamp(images_01 + delta, 0, 1)
        img_embeds = wrapped_model.encode_image(adv)
        logits = logit_scale * img_embeds @ text_embeds.T
        loss = F.cross_entropy(logits, target_labels)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()  # minimize CE -> push prediction toward target_labels
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(images_01 + delta, 0, 1) - images_01)

    return torch.clamp(images_01 + delta, 0, 1).detach()
