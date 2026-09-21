import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from pathlib import Path
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"

BLOCK = 11

INPUT_DIM = 768
LATENT_DIM = 6144
TOP_K = 64

BATCH_SIZE = 10

FALSE_POSITIVE_TARGET = 0.02



model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

model = model.to(device)
model.eval()


normalizer = None

for transform in preprocess.transforms:

    if isinstance(transform, transforms.Normalize):
        normalizer = transform
        break


if normalizer is None:
    raise RuntimeError(
        "Could not find CLIP normalization."
    )


clip_mean = torch.tensor(
    normalizer.mean,
    device=device
).view(1, 3, 1, 1)

clip_std = torch.tensor(
    normalizer.std,
    device=device
).view(1, 3, 1, 1)


def normalize(images):

    return (
        images - clip_mean
    ) / clip_std



class TopKSAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        k
    ):

        super().__init__()

        self.k = k

        self.encoder = nn.Linear(
            input_dim,
            latent_dim
        )

        self.decoder = nn.Linear(
            latent_dim,
            input_dim
        )


    def get_topk(self, x):

        dense = F.relu(
            self.encoder(x)
        )

        values, indices = torch.topk(
            dense,
            k=self.k,
            dim=1
        )

        return values, indices



checkpoint = torch.load(
    "topk_sae_block11.pt",
    map_location=device
)


sae = TopKSAE(
    INPUT_DIM,
    LATENT_DIM,
    TOP_K
).to(device)


sae.load_state_dict(
    checkpoint["state_dict"]
)

sae.eval()



attack_feature_data = torch.load(
    "attack_features_block11.pt"
)


top_features = (
    attack_feature_data[
        "top_features"
    ]
    .to(device)
)


print(
    "\nAttack features loaded:",
    len(top_features)
)


# Make boolean lookup table:
#
# attack_feature_mask[i] = True
# if SAE feature i is one of top 256

attack_feature_mask = torch.zeros(
    LATENT_DIM,
    dtype=torch.bool,
    device=device
)

attack_feature_mask[
    top_features
] = True



paper3_dir = Path("../PaperThree")


stats = torch.load(
    paper3_dir / "sae_stats_block11.pt"
)


activation_mean = (
    stats["mean"]
    .to(device)
)

activation_std = (
    stats["std"]
    .to(device)
)



data = torch.load(
    "saegis_data.pt"
)


dev_clean = data[
    "dev_clean"
]

test_clean = data[
    "test_clean"
]

test_adv = data[
    "test_adv"
]


print(
    "\nClean dev:",
    dev_clean.shape
)

print(
    "Clean test:",
    test_clean.shape
)

print(
    "PGD test:",
    test_adv.shape
)



captured = {}


def hook_fn(
    module,
    inputs,
    output
):

    captured["activation"] = (
        output.detach()
    )


hook_handle = (
    model.visual
    .transformer
    .resblocks[BLOCK]
    .register_forward_hook(
        hook_fn
    )
)



@torch.no_grad()
def detector_scores(images):

    all_scores = []


    for start in range(
        0,
        len(images),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(images)
        )


        batch = (
            images[start:end]
            .to(device)
        )


        # ------------------------------------------
        # CLIP
        # ------------------------------------------

        model.encode_image(
            normalize(batch)
        )


        activation = captured[
            "activation"
        ]


        # Remove CLS:
        #
        # [B,50,768] -> [B,49,768]

        patches = activation[
            :, 1:, :
        ]


        current_batch_size = (
            patches.shape[0]
        )


        # Flatten:
        #
        # [B,49,768] -> [B*49,768]

        flat = patches.reshape(
            -1,
            INPUT_DIM
        )


        # Same clean-training normalization
        # used to train the SAE.

        flat = (
            flat - activation_mean
        ) / activation_std


        # ------------------------------------------
        # TopK SAE
        # ------------------------------------------

        values, indices = sae.get_topk(
            flat
        )


        # ------------------------------------------
        # Is each TopK feature one of our
        # 256 attack-associated features?
        # ------------------------------------------

        selected = attack_feature_mask[
            indices
        ]


        # TopK can technically contain zero values
        # if fewer than K positive ReLU activations exist.
        # Count only genuinely active features.

        active = (
            values > 0
        )


        selected_and_active = (
            selected & active
        )


        # Number of attack features per patch
        #
        # [B*49,64] -> [B*49]

        per_patch_count = (
            selected_and_active
            .sum(dim=1)
            .float()
        )


        # Restore image/token layout
        #
        # [B*49] -> [B,49]

        per_patch_count = (
            per_patch_count.view(
                current_batch_size,
                49
            )
        )


        # N(x):
        # mean attack-feature count over tokens

        image_score = (
            per_patch_count.mean(
                dim=1
            )
        )


        all_scores.append(
            image_score.cpu()
        )


    return torch.cat(
        all_scores
    )


print(
    "\nCalculating clean DEV scores..."
)

dev_scores = detector_scores(
    dev_clean
)



threshold = torch.quantile(
    dev_scores,
    1.0 - FALSE_POSITIVE_TARGET
).item()


print("\n" + "=" * 60)
print("THRESHOLD CALIBRATION")
print("=" * 60)

print(
    "Mean clean dev score:",
    dev_scores.mean().item()
)

print(
    "Std clean dev score:",
    dev_scores.std().item()
)

print(
    "Minimum clean dev score:",
    dev_scores.min().item()
)

print(
    "Maximum clean dev score:",
    dev_scores.max().item()
)

print(
    "98th percentile threshold:",
    threshold
)


dev_false_positives = (
    dev_scores > threshold
).sum().item()


print(
    "Clean dev examples above threshold:",
    dev_false_positives,
    "/",
    len(dev_scores)
)



print(
    "\nCalculating FINAL clean scores..."
)

clean_scores = detector_scores(
    test_clean
)


print(
    "Calculating FINAL PGD scores..."
)

adv_scores = detector_scores(
    test_adv
)


hook_handle.remove()



print("\n" + "=" * 60)
print("DETECTOR SCORE DISTRIBUTIONS")
print("=" * 60)


print("\nCLEAN test:")

print(
    "Mean:",
    clean_scores.mean().item()
)

print(
    "Std:",
    clean_scores.std().item()
)

print(
    "Min:",
    clean_scores.min().item()
)

print(
    "Max:",
    clean_scores.max().item()
)


print("\nPGD test:")

print(
    "Mean:",
    adv_scores.mean().item()
)

print(
    "Std:",
    adv_scores.std().item()
)

print(
    "Min:",
    adv_scores.min().item()
)

print(
    "Max:",
    adv_scores.max().item()
)


clean_predictions = (
    clean_scores > threshold
).long()


adv_predictions = (
    adv_scores > threshold
).long()


# Clean label = 0
# Adversarial label = 1

y_true = torch.cat([
    torch.zeros(
        len(clean_scores),
        dtype=torch.long
    ),

    torch.ones(
        len(adv_scores),
        dtype=torch.long
    )
])


y_pred = torch.cat([
    clean_predictions,
    adv_predictions
])


y_scores = torch.cat([
    clean_scores,
    adv_scores
])


y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()
y_scores_np = y_scores.numpy()


accuracy = accuracy_score(
    y_true_np,
    y_pred_np
)


precision = precision_score(
    y_true_np,
    y_pred_np,
    zero_division=0
)


recall = recall_score(
    y_true_np,
    y_pred_np,
    zero_division=0
)


f1 = f1_score(
    y_true_np,
    y_pred_np,
    zero_division=0
)


auc = roc_auc_score(
    y_true_np,
    y_scores_np
)


tn, fp, fn, tp = confusion_matrix(
    y_true_np,
    y_pred_np
).ravel()


fpr = fp / (
    fp + tn
)


print("\n")
print("=" * 60)
print("FINAL SAEGIS DETECTION RESULTS")
print("=" * 60)


print(
    "Threshold:",
    threshold
)

print(
    "\nTrue negatives:",
    tn
)

print(
    "False positives:",
    fp
)

print(
    "False negatives:",
    fn
)

print(
    "True positives:",
    tp
)


print(
    "\nAccuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)

print(
    "False positive rate:",
    fpr
)

print(
    "ROC AUC:",
    auc
)



torch.save(
    {
        "threshold": threshold,

        "dev_scores":
            dev_scores,

        "clean_test_scores":
            clean_scores,

        "adv_test_scores":
            adv_scores,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positive_rate":
            fpr,

        "roc_auc":
            auc,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    },

    "saegis_detection_results.pt"
)


print(
    "\nSaved saegis_detection_results.pt"
)