# CLIP + Representation Rerouting (minimal)

Adapts the loss from "Improving Alignment and Robustness with Circuit Breakers"
(Zou et al., arXiv:2406.04313) to a CLIP vision encoder. Undesirable behavior =
targeted misclassification under PGD. Backbone: OpenCLIP ViT-B/32 (DataComp).

**Question this experiment answers:** can representation rerouting be adapted to a
CLIP vision encoder to reduce targeted adversarial misclassification without
substantially degrading clean zero-shot accuracy?


## Pipeline

1. **Pick a target layer** (a preliminary probe, not a reproduction of the paper's
   layer choice — see the module docstring in `probe_layer_divergence.py` for why
   their layer numbers don't transfer to CLIP's architecture):
   ```
   python probe_layer_divergence.py --dataset cifar100 --data-root ./data
   ```

2. **Train the circuit breaker:**
   ```
   python train_rr.py --target-layer <layer-from-step-1> --dataset cifar100 --data-root ./data
   ```
   Saves LoRA weights to `rr_checkpoint.pt`. CIFAR-100 here is a quick sanity check
   of the pipeline; swap in `--dataset imagefolder --data-root <path-to-imagenet-100>`
   for the real run.

3. **Evaluate:**
   ```
   python evaluate.py --checkpoint rr_checkpoint.pt --dataset cifar100 --data-root ./data
   ```
   Reports clean zero-shot accuracy and adaptive (white-box, regenerated per model)
   targeted PGD attack success rate, for model_orig and model_cb.

## What's faithful to the paper, and what's an adaptation

**Faithful:** the loss itself —
`L_s = ReLU(cos_sim(rep_cb, rep_orig))` on the circuit-breaker set,
`L_r = ||rep_orig - rep_cb||_2` on the retain set, combined with the paper's exact
coefficient schedule `c_s(t) = alpha(1 - t/2T)`, `c_r(t) = alpha*t/2T`.

**Adapted, not reproduced — see code comments for detail on each:**
- **LoRA placement.** A narrow window of blocks around the target layer
  (`[target_layer - lora_window + 1, target_layer]`, default window 3), matching
  the paper's multimodal (LLaVA) setup — LoRA in Mistral layers 14-16, RR loss at
  layer 16 — not its text-only setup, which puts LoRA everywhere from block 0 to
  the target layer.
- **Layer selection.** Our own preliminary probe (`probe_layer_divergence.py`),
  since CLIP's 12-layer ViT-B/32 has no equivalent to the paper's 32-layer LLM or
  Mistral-7B depth to inherit a layer number from.
- **Retain-set pairing.** `x_r` is the exact clean image `x_c` was perturbed from,
  not an independently-sampled retain example as in the paper's Algorithm 1
  (`x_s ~ D_s`, `x_r ~ D_r`). This gives a cleaner signal for CLIP but is our
  choice.
- **Training on PGD itself.** The paper trains RR against harmful *behaviors* and
  only encounters PGD at evaluation time — its central claim is that RR targets the
  undesirable process rather than a specific attack. Here, PGD generates both the
  circuit-breaker training set and the eval-time attack, which is closer to
  representation-space adversarial training than to that behavior-level framing.
  The paper itself notes standalone image classification is a harder setting for
  this reason; we keep this design because CLIP has no generation step to define
  an attack-independent "undesirable behavior" against, but this should be read
  as an adaptation, not a reproduction, on this specific point.

**Attack objective:** targeted PGD minimizes cross-entropy over CLIP's actual
zero-shot logits (`logit_scale * image_embed @ text_embeds.T`) against the target
label, so the attack explicitly competes against the true class and every other
class — not just a bare cosine-similarity push toward one target embedding.

- **LoRA-B initialization.** Standard LoRA practice zero-initializes `B` so the
  adapter starts as a no-op. For this loss that's actively harmful: at `B=0`,
  `model_cb` is bit-identical to `model_orig`, so `cos_sim(rep_cb, rep_orig)` sits
  exactly at its own maximum — a point where its gradient is the exact zero vector.
  `L_s` therefore contributes zero gradient at initialization; measured directly
  after 100 training steps on the zero-init version, `cos_sim` was still >0.9999998
  and `L_s` had made essentially no progress, with only `L_r`'s incidental (and
  technically also degenerate, but not exactly-zero-in-practice) gradient doing any
  work. `lora.py`'s `LoRALinear` now initializes `B` with small nonzero noise
  (`--lora-b-init-std`, default `1e-3`) to break this saddle point directly. See
  `lora.py`'s module docstring for the full derivation.

**Kept from the original scaffold, unchanged:** LoRA targets only the MLP linears
(`mlp.c_fc`, `mlp.c_proj`) in the vision transformer, not `attn.out_proj` —
PyTorch's fused attention kernel reads `out_proj.weight`/`.bias` directly, bypassing
its `forward()`, so a LoRA wrapper there would train an adapter with zero effect on
the model's output. Only the vision tower is patched; the text tower is untouched,
since the threat model is image-space perturbation only.

## Files

| file | purpose |
|---|---|
| `common.py` | model loading, pixel-space normalization wrapper, CLS-token hook, zero-shot classifier builder |
| `lora.py` | LoRA injection into a window of vision-transformer MLP linears |
| `attacks.py` | targeted PGD over CLIP's zero-shot logits |
| `data.py` | CIFAR-100 (quick sanity check) and ImageFolder (ImageNet-100) zero-shot datasets |
| `probe_layer_divergence.py` | preliminary layer-selection probe |
| `train_rr.py` | representation-rerouting training loop |
| `evaluate.py` | clean accuracy + adaptive ASR |
