# CLIP ADAPTATION OF https://arxiv.org/pdf/2508.17456 

Paper: Adversarial Examples Are Not Bugs, They Are Superposition

This project adapts one experiment from “Adversarial Examples Are Not Bugs, They Are Superposition” (Gorton & Lewis, 2025) from ResNet18 to a CLIP Vision Transformer.

The original paper investigates whether adversarial examples activate more sparse features than clean inputs. This implementation asks the same question inside OpenCLIP ViT-B/32 using Sparse Autoencoders (SAEs).

Experiment to answer: Do adversarial images activate more SAE features than clean images inside CLIP?

We measure:

[L_0 = {number of active SAE features}]
for clean and adversarial representations at several depths of the CLIP vision encoder.

# model 
OpenCLIP:
Architecture: ViT-B-32
Pretrained weights: datacomp_xl_s13b_b90k
Vision transformer blocks: 12
Hidden dimension: 768
Patch size: 32 × 32
Input resolution: 224 × 224
Patch tokens per image: 49

The experiment analyzes transformer blocks:
Block 2
Block 5
Block 8
Block 11

# dataset
CIFAR-100 is used for both zero-shot classification and activation collection.

CLIP achieved approximately 80.2% zero-shot accuracy on the first 500 CIFAR-100 test images using class-name text prompts.

SAE training activations were collected from 2,000 separate CIFAR-100 training images.

# adversarial attack
Adversarial Attack

We construct an untargeted L2 PGD attack against the CLIP zero-shot classifier.

Attack configuration:

L2 epsilon: 3.0
Step size: 0.5
PGD steps: 20
Random initialization inside the L2 ball

The attack is evaluated on 100 examples that CLIP originally classified correctly.

Results:

Clean accuracy: 100%
Adversarial accuracy: 0%
Attack success rate: 100%
Maximum observed L2 perturbation: approximately 3.0

# activation extraction 
Each 224 × 224 image is divided into 49 ViT patch tokens.

For every selected transformer block, each image produces:

[49, 768]

patch-token activations.

Each patch activation is treated as one SAE training example.

For 2,000 training images, each transformer block therefore produces:

98,000 × 768

activation vectors.

Activations are standardized independently at each transformer block using statistics computed from clean training activations.

# sparse autoencoder 
A separate ReLU + L1 Sparse Autoencoder is trained for each selected CLIP block.

Architecture:

768 → 6144 → 768

This corresponds to an 8× expansion factor.

Training configuration:

Learning rate: 5e-4
Batch size: 512
Epochs: 10
L1 coefficient: 1.0
Optimizer: Adam
Gradient clipping: 1.0

The SAE encoder and decoder are initialized with tied initial directions, and decoder columns begin with small normalized weights.

The SAEs are trained only on clean CIFAR-100 training activations.

After training, each SAE is frozen and applied to both clean and PGD representations.

# results 
Average SAE feature activity:

CLIP Block	Clean L0	PGD L0	Difference	PGD / Clean
2	561.29	1081.10	+519.80	1.926×
5	655.43	937.58	+282.15	1.430×
8	589.08	704.89	+115.81	1.197×
11	687.04	778.36	+91.33	1.133×
PGD increased SAE feature activity at every analyzed layer.

At blocks 2, 5, and 8, all 100 attacked examples showed increased L0.

At block 11, 94 of 100 examples showed increased L0.

# main observation 
The central qualitative result from the original paper transfers to CLIP:

[L_0({PGD}) > L_0({clean})]
Adversarial examples activate substantially more SAE features than their clean counterparts.

However, the depth-dependent trend differs.

The original paper reports increasingly large feature proliferation in deeper ResNet layers. In this CLIP experiment, the relative increase instead decreases with transformer depth:

Block 2   1.926×
Block 5   1.430×
Block 8   1.197×
Block 11  1.133×

This suggests that adversarial feature proliferation may depend on architecture or on the type of internal representation being analyzed.

One possible explanation is that this experiment analyzes ViT patch-token activations, while later CLIP layers increasingly aggregate global information through the class token.

This interpretation is only a hypothesis and is not established by the current experiment.

