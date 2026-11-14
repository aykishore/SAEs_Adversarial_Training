# CLIP + Sparse Feature Perturbation

Adapts the sparse-feature perturbation idea from "Layer-Wise Perturbations via Sparse Autoencoders
for Adversarial Text Generation" (SFPF, [arXiv:2508.10404](https://arxiv.org/abs/2508.10404)) to
CLIP's image encoder (OpenCLIP ViT-B/32, DataComp weights).

**Question:** do the sparse autoencoder (SAE) features that are active in adversarial images
actually *cause* adversarial behavior? If we add them to a clean image's internal activations, does
CLIP's prediction move toward the class the real attack produced?

**Short answer:** partly. Adding the attack features raises the score of the attack's class much
more than adding the same number of random features (up to +1.7 vs +0.1 at block 2), and the
effect grows with the amount added. But the push is rarely strong enough to change the prediction
(at most 12% of images, vs 5% for random features), and the effect is weaker in deeper blocks.

## Key terms

- **SAE (sparse autoencoder):** a model trained to rewrite each activation vector as a sparse
  combination of learned **features** (here 6,144 features for 768-dim activations). We reuse the
  SAEs trained in PaperThree.
- **Block:** one of the 12 transformer blocks of CLIP's image encoder (numbered 0-11). We use
  blocks 2, 5 and 8.
- **Patch tokens / CLS token:** the image is cut into 49 patches (tokens) plus one summary token
  (CLS). CLIP's prediction is read from the CLS token after the last block.
- **PGD attack:** an untargeted L2 attack (eps 3.0, 20 steps) that changes CLIP's prediction.
  The new prediction is the **PGD class**.
- **Attack features (SFPF mask):** the SAE features most active, on average, in successfully
  attacked images.
- **Injection strength alpha:** how much is added to each selected feature (0.1, 0.3, 0.5).
- **Random control:** the same number of randomly chosen features, injected the same way.

## How it works

The original paper finds SAE features tied to successful text attacks, perturbs them, and decodes
the result into new adversarial text. CLIP has no decoder from internal activations back to pixels,
so instead we inject the change directly into the running model:

1. **Find the attack features** (`discover_sfp_features.py`). For 100 successfully attacked images,
   encode every patch token with the SAE, average over patches and images (one cluster centroid, as
   in the paper), normalize it to unit length, and keep the features above 0.03 (the paper's
   threshold).
2. **Inject them into clean images** (`run_sfp_injection.py`). During CLIP's forward pass on 100
   other clean test images, at the chosen block: encode each patch token with the SAE (`z`), add
   `alpha` to the selected features (`z' = z + alpha * mask`), decode both, and add only the
   difference `decode(z') - decode(z)` to the activation. Using the difference avoids adding the
   SAE's reconstruction error.
3. **Measure the effect** against the clean prediction: how often the prediction changes (flip
   rate), the change in the true class's score, the change in the PGD class's score, and how often
   the new prediction equals the PGD class.

## Running it

Needs the SAEs from PaperThree and the clean/attacked image splits from PaperFour.

```
python discover_sfp_features.py    # -> sfp_feature_masks.pt
python run_sfp_injection.py        # -> sfp_injection_results.pt
```

## Results

Attack features found per block:

| block | features selected | fraction of 6,144 |
|---|---|---|
| 2 | 190 | 3.1% |
| 5 | 134 | 2.2% |
| 8 | 70 | 1.1% |
| 11 | 31 | 0.5% |

Injection into 100 clean test images (score changes are in logit units):

| block | alpha | flip rate: attack / random | PGD-class score change: attack / random | true-class score change: attack / random | matches PGD class: attack / random |
|---|---|---|---|---|---|
| 2 | 0.1 | 0.02 / 0.02 | +0.08 / +0.00 | -0.03 / -0.01 | 0.00 / 0.01 |
| 2 | 0.3 | 0.07 / 0.04 | +0.54 / +0.03 | -0.06 / -0.02 | 0.03 / 0.02 |
| 2 | 0.5 | 0.12 / 0.05 | **+1.70** / +0.10 | +0.50 / -0.04 | 0.06 / 0.02 |
| 5 | 0.1 | 0.04 / 0.01 | +0.05 / -0.04 | +0.01 / -0.04 | 0.01 / 0.00 |
| 5 | 0.3 | 0.05 / 0.01 | +0.20 / -0.10 | +0.00 / -0.13 | 0.01 / 0.00 |
| 5 | 0.5 | 0.09 / 0.03 | **+0.37** / -0.12 | -0.07 / -0.24 | 0.02 / 0.01 |
| 8 | 0.1 | 0.01 / 0.01 | +0.04 / -0.00 | +0.03 / +0.01 | 0.01 / 0.00 |
| 8 | 0.3 | 0.01 / 0.02 | +0.13 / -0.01 | +0.06 / +0.01 | 0.01 / 0.01 |
| 8 | 0.5 | 0.02 / 0.02 | **+0.21** / -0.01 | +0.08 / +0.01 | 0.01 / 0.01 |

### What this means

1. **The attack features push toward the attack's class.** At every block and strength, they raise
   the PGD class's score more than random features do, and random features barely move it.
2. **The effect grows with strength** (dose-response): at block 2 the PGD-class score rises from
   +0.08 to +0.54 to +1.70 as alpha goes from 0.1 to 0.5.
3. **The effect is strongest early and weaker deeper** (+1.70 at block 2, +0.37 at block 5, +0.21
   at block 8 at alpha 0.5). Fewer features are selected in deeper blocks (190, 134, 70), so part of
   this drop may come from the smaller mask rather than the depth itself.
4. **The push rarely changes the prediction.** At most 12% of images flip (vs 5% for random), and
   only up to 6% land on the exact class the real attack chose. The features carry part of the
   attack's effect, not all of it.
5. **Small sample.** These are 100 images per condition, so differences of a few percentage points
   in flip rate are within noise; the score changes are the more reliable signal.

### Why block 11 is not in the injection results

An earlier run included block 11 and reported exactly 0 for every metric. That was a property of
the setup, not of the features: the injection edits patch tokens at the block's output, and CLIP's
prediction is read from the CLS token only. After the last block nothing mixes patch tokens back
into CLS, so no injection there can change the output. Block 11 is therefore left out of
`run_sfp_injection.py`. Its attack features are still found (table above); testing them would need
an injection into the CLS token or into block 11's input, which we did not do.

## Files

| file | purpose |
|---|---|
| `discover_sfp_features.py` | finds the attack features per block -> `sfp_feature_masks.pt` |
| `run_sfp_injection.py` | injects attack and random features into clean images, measures the effect -> `sfp_injection_results.pt` |
