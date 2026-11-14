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


class _StopForward(Exception):
    pass


class RepExtractor:
    """Captures the residual stream at one vision-transformer block via a forward hook.
    Assumes batch_first=True.

    tokens="cls" -> get() returns the CLS token, [N, D]
    tokens="all" -> get() returns CLS + all patch tokens, [N, L, D]

    With stop_forward=True the hook aborts the forward pass right after the block, so
    run() only computes blocks [0, layer]: the later blocks have no LoRA and nothing reads
    their output during training, so running them (and keeping their autograd graph) is
    pure waste. Never call encode_image directly on a model with a stopping extractor
    attached -- use run().
    """

    def __init__(self, clip_model: nn.Module, layer: int, tokens: str = "cls", stop_forward: bool = False):
        resblocks = clip_model.visual.transformer.resblocks
        assert 0 <= layer < len(resblocks), f"layer {layer} out of range for {len(resblocks)} blocks"
        assert tokens in ("cls", "all"), f"unknown tokens mode {tokens}"
        self.layer = layer
        self.tokens = tokens
        self.stop_forward = stop_forward
        self._captured = None
        self._handle = resblocks[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        assert output.dim() == 3, f"expected [N, L, D] residual stream, got shape {tuple(output.shape)}"
        self._captured = output
        if self.stop_forward:
            raise _StopForward

    def get(self) -> torch.Tensor:
        assert self._captured is not None, "no forward pass captured yet"
        return self._captured[:, 0, :] if self.tokens == "cls" else self._captured

    def run(self, wrapped_model, pixel_images_01: torch.Tensor) -> torch.Tensor:
        try:
            wrapped_model.encode_image(pixel_images_01)
        except _StopForward:
            pass
        return self.get()

    def remove(self):
        self._handle.remove()


class AllLayerCapture:
    """Captures the residual stream of every vision-transformer block in one forward pass."""

    def __init__(self, clip_model: nn.Module):
        self._captured = {}
        self._handles = [block.register_forward_hook(self._make_hook(i))
                         for i, block in enumerate(clip_model.visual.transformer.resblocks)]

    def _make_hook(self, layer):
        def hook(module, inputs, output):
            self._captured[layer] = output.detach()
        return hook

    def get(self, layer: int) -> torch.Tensor:
        return self._captured[layer]

    def remove(self):
        for handle in self._handles:
            handle.remove()


@torch.no_grad()
def build_zero_shot_classifier(model: nn.Module, tokenizer, classnames, template: str, device: str) -> torch.Tensor:
    """Returns an [num_classes, dim] L2-normalized text-embedding matrix."""
    prompts = [template.format(c) for c in classnames]
    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def predict(wrapped_model, images_01: torch.Tensor, text_embeds: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Zero-shot predictions for a CPU or GPU image tensor, computed in batches. Returns CPU int64."""
    device = text_embeds.device
    preds = []
    for i in range(0, images_01.shape[0], batch_size):
        img_embeds = wrapped_model.encode_image(images_01[i:i + batch_size].to(device))
        preds.append((img_embeds @ text_embeds.T).argmax(dim=-1).cpu())
    return torch.cat(preds)


def sample_wrong_targets(true_labels: torch.Tensor, num_classes: int, seed: int) -> torch.Tensor:
    """Samples one random wrong target label per example, uniformly over all classes
    except the true one."""
    rng = random.Random(seed)
    targets = []
    for y in true_labels.tolist():
        choices = [c for c in range(num_classes) if c != y]
        targets.append(rng.choice(choices))
    return torch.tensor(targets, dtype=torch.long)
