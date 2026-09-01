"""Shared model/data utilities."""
import random

import open_clip
import torch
import torch.nn as nn
import torchvision.transforms as T

# values from openCLIP's preprocessing 
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(model_name: str, pretrained: str, device: str):
    """Loads a frozen open_clip model plus a resize/crop transform with normalization
    stripped out (normalization is re-applied inside NormalizingCLIPWrapper)."""
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    resize_crop = T.Compose([t for t in preprocess_val.transforms if not isinstance(t, T.Normalize)])
    return model, resize_crop, tokenizer


class NormalizingCLIPWrapper(nn.Module):
    """normalization is moved inside the forward pass here."""

    def __init__(self, model: nn.Module, device):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1))

    @property
    def logit_scale(self) -> float:
        return self.model.logit_scale.exp().item()

    def encode_image(self, pixel_images_01: torch.Tensor) -> torch.Tensor:
        x = (pixel_images_01 - self.mean) / self.std
        feats = self.model.encode_image(x)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)


class ClsTokenExtractor:
    """Captures the CLS-token residual stream at one vision-transformer block via a
    forward hook. Assumes batch_first=True"""

    def __init__(self, clip_model: nn.Module, layer: int):
        resblocks = clip_model.visual.transformer.resblocks
        assert 0 <= layer < len(resblocks), f"layer {layer} out of range for {len(resblocks)} blocks"
        self.layer = layer
        self._captured = None
        self._handle = resblocks[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        assert output.dim() == 3, f"expected [N, L, D] residual stream, got shape {tuple(output.shape)}"
        self._captured = output

    def get_cls(self) -> torch.Tensor:
        assert self._captured is not None, "no forward pass captured yet"
        return self._captured[:, 0, :]

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def build_zero_shot_classifier(model: nn.Module, tokenizer, classnames, template: str, device: str) -> torch.Tensor:
    """Returns an [num_classes, dim] L2-normalized text-embedding matrix."""
    prompts = [template.format(c) for c in classnames]
    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def sample_wrong_targets(true_labels: torch.Tensor, num_classes: int, seed: int) -> torch.Tensor:
    """Samples one random wrong target label per example, uniformly over all classes
    except the true one."""
    rng = random.Random(seed)
    targets = []
    for y in true_labels.tolist():
        choices = [c for c in range(num_classes) if c != y]
        targets.append(rng.choice(choices))
    return torch.tensor(targets, dtype=torch.long)
