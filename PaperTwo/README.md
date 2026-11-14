# CLIP + Representation Engineering

Reproduces the visual representation-reading experiment from "Representation Engineering: A
Top-Down Approach to AI Transparency" (Zou et al., 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405),
Table 12) on CLIP's image encoder (`openai/clip-vit-base-patch32`).

**Question:** does CLIP's image encoder contain a direction for each facial emotion, one that
can be found without training a classifier and that tells that emotion apart from a neutral face?

**Short answer:** yes. Directions learned on one cartoon character perfectly separate emotion
from neutral faces on four other characters (AUC 1.0) at the last blocks of the encoder. Our first
version reported ~100% accuracy on the paper's own setup, but that number was inflated by
near-duplicate images; the cross-character test is the reliable one.

## Key terms

- **FERG-DB:** a dataset of cartoon faces: 6 characters (Aia, Bonnie, Jules, Malcolm, Mery, Ray),
  each drawn with 7 expressions (neutral, joy, sadness, anger, fear, surprise, disgust). The images
  are animation frames, so many are nearly identical.
- **Block:** one of the 12 transformer blocks of CLIP's image encoder (numbered 0-11). We use
  each block's CLS-token output, a 768-dim vector per image (the **representation**).
- **LAT (Linear Artificial Tomography):** RepE's method for finding a concept direction. Take
  differences between pairs of representations and run PCA; the top principal component is the
  direction. An image's **score** is its representation projected onto that direction.
- **Random-pair LAT:** the paper's version. Images (emotion and neutral mixed) are paired at random,
  so the direction is found without labels; labels are only used to decide which end is "emotion".
- **Paired LAT:** our variant. Each emotion image is paired with a neutral image, which uses labels.
- **AUC:** how well the scores rank emotion images above neutral ones, independent of any
  threshold (0.5 = chance, 1.0 = perfect).

## Setup

- Model: `openai/clip-vit-base-patch32` (Hugging Face), CLS token at every block.
- Training: 256 neutral and 256 emotion images of Mery per emotion, as in the paper (512 per task).
- Test sets:
  - **Same character (the paper's setup):** 64 + 64 other Mery images.
  - **Cross-character (added):** 64 + 64 images of each other character. The block is chosen on
    Aia, and accuracy is reported on Bonnie, Jules, Malcolm and Ray (512 images per emotion).

## Running it

```
python extract_activations.py          # Mery train/test CLS representations, all blocks
python lat.py                          # paired LAT directions
python lat_random_pairs.py             # random-pair LAT directions (paper's method)
python evaluate.py                     # same-character evaluation, paired LAT
python evaluate_random_pairs.py        # same-character evaluation, random-pair LAT
python extract_other_characters.py     # representations for the other 5 characters
python evaluate_cross_character.py     # cross-character evaluation -> results_cross_character.json
```

FERG-DB goes in `data/FERG_DB/FERG_DB_256/` (not included; see the citation below).

## Results

### Same character (the paper's setup): inflated

Our first evaluation scored 99-100% for every emotion with both methods, against the paper's
61-74%. Two problems make those numbers unreliable:

1. **Near-duplicate images.** A random split of Mery's animation frames puts near-copies in train
   and test. For 97% of neutral test images, the closest training image differs by less than
   0.01 RMS in pixel value, while two typical training images differ by 0.035.
2. **The best block was chosen on the test set.**

The clearest sign: paired LAT scores 100% on Mery at blocks 0-1, but only ~50% (chance) on other
characters at those blocks. Early blocks are close to the pixels, so they match frames rather than
recognize emotions.

### Cross-character: the reliable result

Directions learned on Mery, block chosen on Aia, tested on Bonnie, Jules, Malcolm and Ray
(`results_cross_character.json`):

| emotion | random-pair LAT: block | accuracy [95% CI] | AUC | paired LAT: block | accuracy [95% CI] | AUC | paper (Mery) |
|---|---|---|---|---|---|---|---|
| joy | 11 | 0.986 [0.976, 0.996] | 1.000 | 8 | 0.922 [0.899, 0.945] | 1.000 | 0.742 |
| sadness | 11 | 0.979 [0.966, 0.991] | 1.000 | 11 | 0.984 [0.974, 0.995] | 1.000 | 0.617 |
| anger | 11 | 0.875 [0.846, 0.904] | 1.000 | 11 | 1.000 [1.000, 1.000] | 1.000 | 0.727 |
| fear | 11 | 0.971 [0.956, 0.985] | 1.000 | 11 | 1.000 [1.000, 1.000] | 1.000 | 0.734 |
| surprise | 11 | 0.865 [0.836, 0.895] | 1.000 | 8 | 0.932 [0.910, 0.954] | 1.000 | 0.688 |
| disgust | 10 | 0.809 [0.775, 0.843] | 1.000 | 10 | 0.783 [0.748, 0.819] | 1.000 | 0.609 |

Accuracy averaged over the 6 emotions, by block (random-pair LAT):
blocks 0-9 stay between 0.50 and 0.63, block 10 reaches 0.64, and block 11 0.93.

- **The emotion directions are real and generalize across characters.** At the last blocks, every
  direction ranks every test character's emotion images above its neutral ones (AUC 1.0 for all 6
  emotions on all 4 characters).
- **Accuracy below 1.0 comes from the threshold, not the direction.** The cut-off learned on Mery
  sits in a slightly different place for other characters; the ranking itself is perfect.
- **Emotion is only readable late in the encoder.** Blocks 0-9 are near chance on new characters.
- **Our numbers are higher than the paper's,** even without the leakage. We can't explain the gap
  from the paper's description of its protocol; it may evaluate differently (for example, a
  different threshold or pairing at test time).

### Cross-emotion directions

`evaluate_cross_emotion.py` applies each emotion's direction (random-pair LAT, block 11) to every
other emotion. Many directions also separate other emotions from neutral: for example, the joy
direction scores 100% on fear and disgust. Because every test is against neutral faces, this
shows that the directions largely detect "a non-neutral expression", not one specific emotion.
Separating emotions from each other would need an emotion-vs-emotion test, which we did not run.

An earlier version of this README concluded that the cosine similarity between directions doesn't
match how they behave. The numbers don't support that (joy and disgust have cosine 0.78 and behave
alike; joy and sadness have 0.19 and behave differently), so we have dropped the claim.

### Patch tokens

`extract_patch_activations.py`, `lat_patch.py` and `evaluate_patch.py` repeat the analysis with the
mean of the patch tokens instead of CLS. They were only evaluated on Mery, so they have the same
near-duplicate problem and are not part of the results above.

## Files

| file | purpose |
|---|---|
| `extract_activations.py` | Mery train/test CLS representations at all 12 blocks |
| `lat.py`, `lat_random_pairs.py` | paired and random-pair LAT directions |
| `evaluate.py`, `evaluate_random_pairs.py` | same-character evaluation (inflated, see above) |
| `extract_other_characters.py` | representations for the other 5 characters |
| `evaluate_cross_character.py` | cross-character evaluation, block chosen on Aia |
| `evaluate_cross_emotion.py`, `direction_similarity.py` | each direction on other emotions; cosine between directions |
| `extract_patch_activations.py`, `lat_patch.py`, `evaluate_patch.py` | patch-token version (Mery only) |
| `dataset.py`, `inspect_model.py` | dataset and model sanity checks |

## Data

FERG-DB: Deepali Aneja, Alex Colburn, Gary Faigin, Linda G. Shapiro, Barbara Mones, "Modeling
Stylized Character Expressions via Deep Learning." Asian Conference on Computer Vision, Springer, 2016.
