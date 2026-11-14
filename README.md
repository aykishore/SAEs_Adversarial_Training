# SAEs_Adversarial_Training
Research on SAE for Adversarial Training
with Dr. Nageen Himayat and Dr. Xin Liu (UC Davis) started in August 2026. In progress

## What this is
Working toward using sparse autoencoders to understand and improve
adversarial robustness. Starting by reproducing experiments from
prior work before extending them.

## Current Status
Five groundwork experiments, each adapting one paper to CLIP's image encoder (ViT-B/32). Each folder's README has the setup, results and limitations.

| folder | paper | what we found |
|---|---|---|
| [PaperOne/exp1_clip_circuit_breakers](PaperOne/exp1_clip_circuit_breakers) | Circuit Breakers (Zou et al., 2024) | Representation rerouting at early blocks blocks transferred attacks with no loss in clean accuracy, but not adaptive attacks. A layer probe predicts which blocks work. |
| [PaperTwo](PaperTwo) | Representation Engineering (Zou et al., 2023) | Emotion directions learned on one cartoon character generalize to four others (AUC 1.0), but only at the last blocks. |
| [PaperThree](PaperThree) | Adversarial Examples Are Not Bugs, They Are Superposition (Gorton & Lewis, 2025) | Adversarial images activate more SAE features than clean ones at every block tested; the increase shrinks with depth. |
| [PaperFour](PaperFour) | SAEgis (2026) | SAE features at block 11 detect PGD images with ROC-AUC 0.92, but recall is low at a 1% false-positive rate. |
| [PaperFive](PaperFive) | Sparse Feature Perturbation (SFPF, 2025) | Injecting attack-associated SAE features into clean images pushes predictions toward the attack's class, but rarely flips them. |

Next: the main project, combining SAEs with adversarial training.
