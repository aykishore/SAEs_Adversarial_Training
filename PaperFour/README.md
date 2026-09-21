# CLIP Experiment arxiv.org/pdf/2605.07447 SAEgis for CLIP

This project adapts the adversarial detection method proposed in “Sparse Autoencoders as Plug-and-Play Firewalls for Adversarial Attack Detection in VLMs” (SAEgis) to an OpenCLIP vision transformer. 

The original paper studies whether sparse internal features can be used to detect adversarial inputs in vision-language models. Rather than reproducing the paper's Qwen2.5-VL setup directly, this implementation applies the same core idea to:

OpenCLIP ViT-B/32
pretrained weights: datacomp_xl_s13b_b90k
CIFAR-100
L2-PGD adversarial examples
Sparse Autoencoders trained on CLIP patch-token activations

Experiment Question: Can SAE features inside CLIP be used to detect PGD adversarial images?

# overview

clean CLIP activations -> train top-k sparse autoencoder -> clean + PGD discovery examples -> identify attack-associated SAE features -> clean development set -> calibrate detection threshold -> untouched clean + PGD test set -> evaluate adversarial detector 

the clip model is never adversarially trained or fine tuned

# model 
ncoder
        ↓
Clean + PGD discovery examples
        ↓
Identify attack-associated SAE features
        ↓
Clean development set
        ↓
Calibrate detection threshold
        ↓
Untouched clean + PGD test set
        ↓
Evaluate adversarial detector

The CLIP model itself is never adversarially trained or fine-tuned.

Model

The backbone is:

OpenCLIP ViT-B/32
Pretrained: datacomp_xl_s13b_b90k

Relevant architecture details:

12 transformer blocks
hidden dimension: 768
image resolution: 224 × 224
patch size: 32 × 32
49 image patch tokens + 1 CLS token

This experiment analyzes block 11 patch-token activations.

The CLS token is excluded.
Each image therefore produces:

49 × 768

internal activation vectors.

# Sparse AutoEncoder
Sparse Autoencoder

A separate Top-K Sparse Autoencoder is trained on clean block-11 CLIP activations collected previously from CIFAR-100 training images.

Architecture:

768 → 6144 → 768

Configuration:

Expansion factor: 8×
Top-K: 64
Learning rate: 5e-4
Batch size: 512
Epochs: 10

For each patch token, the SAE encoder first produces non-negative features using ReLU and then retains only the 64 largest activations.

The SAE is trained only to reconstruct clean CLIP activations.

No adversarial examples are used for SAE training.

Activations are standardized using the mean and standard deviation computed from the clean CLIP activation dataset.


# detector data

The detector uses CIFAR-100 test images that the CLIP zero-shot classifier originally classifies correctly.

Using seed 42, these examples are divided into three disjoint groups:
feature discovery, clean developement, final test

# adversarial attack 
An untargeted L2-PGD attack is applied to the discovery and final-test images.

Configuration:

L2 epsilon: 3.0
Step size: 0.5
Steps: 20
Random initialization: Yes

The attack operates in raw image space [0,1], while CLIP normalization is applied internally before model inference.

# discovering attack-associated SAE features
The core idea of SAEgis is not simply to count all active SAE features.

Instead, the detector identifies features whose activation patterns differ systematically between clean and adversarial inputs.

For each SAE feature (i) and image (x), an image-level score is computed

that combines the maximum activation magnitude of the feature, and how broadly the feature appears across the image.

for each feature, an attack-association score is calculated. 

# detection score
For a new image, each of its 49 patch-token representations is passed through the frozen Top-K SAE.

The detector checks how many of the active SAE features belong to the previously selected set of 256 attack-associated features. This gives the average number of attack-associated SAE features active per patch token. Higher scores indicate stronger similarity to the sparse activation patterns observed under PGD.

# threshold calibration 
The detection threshold is calibrated using only the 100 clean development examples.

The threshold is set to the 98th percentile of their detector scores, corresponding to a target false-positive rate of approximately 2%.

Observed clean development statistics:\

Mean score:       3.0706\
Standard deviation: 0.8939\
Minimum:          1.5918\
Maximum:          6.8776\
Threshold:        5.4547\

Two of the 100 clean development examples fell above the threshold.

# final test results 
The calibrated detector was evaluated on:

100 clean test images
100 PGD adversarial test images
Detector Score Distributions
Condition	Mean	Std	Min	Max\
Clean	3.0053	0.7691	1.7143	5.6122\
PGD	4.7329	0.9323	2.6327	7.0408\

PGD examples therefore produced substantially higher attack-feature scores on average.

# classification results
Using the clean-calibrated threshold:

Threshold: 5.4547\

True negatives:   99\
False positives:   1\
False negatives:  77\
True positives:   23\

Final metrics:\

Metric	Result\
Accuracy	0.610\
Precision	0.958\
Recall	0.230\
F1	0.371\
False Positive Rate	0.010\
ROC-AUC	0.923\

# interpretation
The most important result is the strong separation between the underlying detector scores.

The mean detector score increased from:

[
3.01
]

for clean inputs to:

[
4.73
]

for PGD inputs.

The detector achieved:

[
\boxed{\text{ROC-AUC}=0.923}
]

showing that the selected SAE features contain substantial information about whether an internal CLIP representation originated from a clean or adversarial image.

However, the strict clean-calibrated threshold produced a different tradeoff:

Precision: 95.8%\
Recall:    23.0%\
FPR:        1.0%\

The detector is therefore highly conservative.

the result shows that a threshold selected specifically to maintain a very low clean false-positive rate leads to poor recall in this single-layer CLIP adaptation.

# main finding
PGD attacks produce identifiable sparse feature signatures inside CLIP
A subset of SAE features discovered using adversarial examples generalized to previously unseen test images and produced strong clean-vs-PGD ranking performance.\
good score separation does not automatically imply high-recall detection
under a strict false-positive constraint.

The block-11 detector achieved high precision and low false-positive rate, but relatively low recall.

# limitations
single layer only block 11 is used for detection
single attack only L2-PGD
small evaluation set 
thresholf tradeoff
no detector-aware adaptive attack 


