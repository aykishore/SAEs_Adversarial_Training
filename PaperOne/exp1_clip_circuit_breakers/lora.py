"""Minimal LoRA injection for open_clip's VisionTransformer blocks.

lora_B init note: standard LoRA practice zero-initializes B so the adapter starts as
a no-op, which is exactly wrong for this loss. At B=0, model_cb is bit-identical to
model_orig, so rep_cb == rep_orig for every input, and cos_sim(rep_cb, rep_orig) sits
at its own maximum -- a point where its gradient is the exact zero vector (not just
small: cos_sim doesn't depend on magnitude, and equal-direction is the maximizer over
the remaining degrees of freedom, so both angular and radial gradient components are
exactly zero there). L_s = ReLU(cos_sim(...)) therefore contributes exactly zero
gradient to every LoRA parameter at initialization; only L_r's (technically also
degenerate, but not exactly-zero in practice) gradient does anything, and it does so
by accident, not because it's meant to be what drives L_s off its own saddle point.
Measured directly after 100 training steps: cos_sim(rep_cb_adv, rep_orig_adv) was
still >0.9999998 (six nines) -- L_s had made essentially no progress. Initializing
lora_B with a small nonzero random perturbation instead breaks this saddle point
directly, giving L_s a real gradient from step 0, at a scale small enough not to
meaningfully disturb clean-image behavior before L_r has a chance to act.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, b_init_std: float = 1e-3):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if b_init_std > 0:
            nn.init.normal_(self.lora_B, std=b_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())


def inject_lora(clip_model: nn.Module, start_layer: int, end_layer: int, r: int = 4, alpha: float = 8.0,
                 b_init_std: float = 1e-3) -> list:
    """Adds LoRA adapters to the MLP linears of vision-transformer blocks
    [start_layer, end_layer] (inclusive). Returns the list of trainable LoRA parameters.
    """
    resblocks = clip_model.visual.transformer.resblocks
    assert 0 <= start_layer <= end_layer < len(resblocks), \
        f"invalid window [{start_layer}, {end_layer}] for {len(resblocks)} blocks"
    lora_params = []
    for i in range(start_layer, end_layer + 1):
        block = resblocks[i]
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, r=r, alpha=alpha, b_init_std=b_init_std)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, r=r, alpha=alpha, b_init_std=b_init_std)
        lora_params += [
            block.mlp.c_fc.lora_A, block.mlp.c_fc.lora_B,
            block.mlp.c_proj.lora_A, block.mlp.c_proj.lora_B,
        ]
    return lora_params


def lora_state_dict(clip_model: nn.Module, start_layer: int, end_layer: int) -> dict:
    resblocks = clip_model.visual.transformer.resblocks
    state = {}
    for i in range(start_layer, end_layer + 1):
        block = resblocks[i]
        state[f"{i}.c_fc.lora_A"] = block.mlp.c_fc.lora_A.detach().cpu()
        state[f"{i}.c_fc.lora_B"] = block.mlp.c_fc.lora_B.detach().cpu()
        state[f"{i}.c_proj.lora_A"] = block.mlp.c_proj.lora_A.detach().cpu()
        state[f"{i}.c_proj.lora_B"] = block.mlp.c_proj.lora_B.detach().cpu()
    return state


def load_lora_state_dict(clip_model: nn.Module, start_layer: int, end_layer: int, state: dict, device: str):
    resblocks = clip_model.visual.transformer.resblocks
    for i in range(start_layer, end_layer + 1):
        block = resblocks[i]
        block.mlp.c_fc.lora_A.data.copy_(state[f"{i}.c_fc.lora_A"].to(device))
        block.mlp.c_fc.lora_B.data.copy_(state[f"{i}.c_fc.lora_B"].to(device))
        block.mlp.c_proj.lora_A.data.copy_(state[f"{i}.c_proj.lora_A"].to(device))
        block.mlp.c_proj.lora_B.data.copy_(state[f"{i}.c_proj.lora_B"].to(device))
