import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import open_clip



device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"

BLOCK = 11

INPUT_DIM = 768
LATENT_DIM = 6144
TOP_K = 64

NUM_ATTACK_FEATURES = 256

BATCH_SIZE = 10



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
        "Could not locate CLIP normalization."
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

        self.input_dim = input_dim
        self.latent_dim = latent_dim
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

        # Dense SAE activations
        dense = F.relu(
            self.encoder(x)
        )


        # Keep the 64 largest features
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


print("\nLoaded TopK SAE")

print(
    "Latent dimension:",
    LATENT_DIM
)

print(
    "TopK:",
    TOP_K
)



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


discovery_clean = data[
    "discovery_clean"
]

discovery_adv = data[
    "discovery_adv"
]


print("\nDiscovery clean:")
print(discovery_clean.shape)

print("Discovery PGD:")
print(discovery_adv.shape)


captured = {}


def activation_hook(
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
        activation_hook
    )
)


@torch.no_grad()
def get_sparse_activations(images):

    all_values = []
    all_indices = []


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
        # CLIP forward pass
        # ------------------------------------------

        model.encode_image(
            normalize(batch)
        )


        activation = captured[
            "activation"
        ]


        # ------------------------------------------
        # Remove CLS token
        #
        # [B,50,768]
        # ->
        # [B,49,768]
        # ------------------------------------------

        patches = activation[
            :, 1:, :
        ]


        current_batch_size = (
            patches.shape[0]
        )


        # ------------------------------------------
        # Flatten patches
        #
        # [B,49,768]
        # ->
        # [B*49,768]
        # ------------------------------------------

        flat_patches = patches.reshape(
            -1,
            INPUT_DIM
        )


        # ------------------------------------------
        # Standardize with CLEAN training stats
        # ------------------------------------------

        flat_patches = (
            flat_patches
            - activation_mean
        ) / activation_std


        # ------------------------------------------
        # TopK SAE
        # ------------------------------------------

        values, indices = sae.get_topk(
            flat_patches
        )


        # ------------------------------------------
        # Restore token structure
        #
        # [B*49,64]
        # ->
        # [B,49,64]
        # ------------------------------------------

        values = values.view(
            current_batch_size,
            49,
            TOP_K
        )


        indices = indices.view(
            current_batch_size,
            49,
            TOP_K
        )


        all_values.append(
            values.cpu()
        )

        all_indices.append(
            indices.cpu()
        )


    return (
        torch.cat(
            all_values,
            dim=0
        ),
        torch.cat(
            all_indices,
            dim=0
        )
    )


print(
    "\nExtracting CLEAN sparse activations..."
)

clean_values, clean_indices = (
    get_sparse_activations(
        discovery_clean
    )
)


print(
    "Clean values:",
    clean_values.shape
)

print(
    "Clean indices:",
    clean_indices.shape
)


print(
    "\nExtracting PGD sparse activations..."
)

adv_values, adv_indices = (
    get_sparse_activations(
        discovery_adv
    )
)


print(
    "PGD values:",
    adv_values.shape
)

print(
    "PGD indices:",
    adv_indices.shape
)


# Remove CLIP hook now that extraction is finished
hook_handle.remove()



def score_one_image(
    values,
    indices
):

    # values:
    # [49,64]
    #
    # indices:
    # [49,64]


    flat_values = values.reshape(
        -1
    ).to(device)


    flat_indices = indices.reshape(
        -1
    ).to(device)


    # Ignore zero-valued TopK entries
    positive = (
        flat_values > 0
    )


    flat_values = flat_values[
        positive
    ]

    flat_indices = flat_indices[
        positive
    ]


    # ------------------------------------------
    # Piece A:
    #
    # Maximum activation magnitude per feature
    # ------------------------------------------

    max_activation = torch.zeros(
        LATENT_DIM,
        device=device
    )


    max_activation.scatter_reduce_(
        0,
        flat_indices,
        flat_values,
        reduce="amax",
        include_self=True
    )


    # ------------------------------------------
    # Piece B:
    #
    # Number of image tokens where each feature
    # was selected.
    #
    # Because torch.topk cannot select the same
    # feature twice within one token, counting
    # occurrences is equivalent to token count.
    # ------------------------------------------

    token_count = torch.bincount(
        flat_indices,
        minlength=LATENT_DIM
    ).float()


    # ------------------------------------------
    # Final SAEgis feature score
    # ------------------------------------------

    score = (
        max_activation
        *
        torch.log1p(
            token_count
        )
    )


    return score



def average_feature_scores(
    values,
    indices,
    name
):

    total_score = torch.zeros(
        LATENT_DIM,
        device=device
    )


    print(
        f"\nScoring {name} images..."
    )


    for image_index in range(
        len(values)
    ):

        score = score_one_image(
            values[image_index],
            indices[image_index]
        )


        total_score += score


        if (
            (image_index + 1) % 20 == 0
            or image_index + 1 == len(values)
        ):

            print(
                f"{image_index + 1}/{len(values)}"
            )


    mean_score = (
        total_score
        / len(values)
    )


    return mean_score.cpu()



clean_feature_scores = (
    average_feature_scores(
        clean_values,
        clean_indices,
        "CLEAN"
    )
)


adv_feature_scores = (
    average_feature_scores(
        adv_values,
        adv_indices,
        "PGD"
    )
)



attack_scores = (
    adv_feature_scores
    - clean_feature_scores
)



top_values, top_indices = torch.topk(
    attack_scores,
    k=NUM_ATTACK_FEATURES
)


print("\n")
print("=" * 78)
print("TOP ATTACK-ASSOCIATED SAE FEATURES")
print("=" * 78)


print(
    f"{'Rank':<8}"
    f"{'Feature':<12}"
    f"{'Clean score':<16}"
    f"{'PGD score':<16}"
    f"{'Attack score':<16}"
)

print("-" * 70)


# Print top 20 so output stays readable
for rank in range(20):

    feature = (
        top_indices[rank]
        .item()
    )


    print(
        f"{rank + 1:<8}"
        f"{feature:<12}"
        f"{clean_feature_scores[feature].item():<16.4f}"
        f"{adv_feature_scores[feature].item():<16.4f}"
        f"{attack_scores[feature].item():<16.4f}"
    )



num_positive = (
    attack_scores > 0
).sum().item()


num_negative = (
    attack_scores < 0
).sum().item()


print("\nFeature-score summary:")

print(
    "Features more active under PGD:",
    num_positive,
    "/",
    LATENT_DIM
)

print(
    "Features more active under clean:",
    num_negative,
    "/",
    LATENT_DIM
)


print(
    "Mean attack score of top 256:",
    top_values.mean().item()
)

print(
    "Minimum attack score in top 256:",
    top_values.min().item()
)


torch.save(
    {
        "block": BLOCK,

        "top_k_sae": TOP_K,

        "num_attack_features":
            NUM_ATTACK_FEATURES,

        "top_features":
            top_indices,

        "top_attack_scores":
            top_values,

        "all_attack_scores":
            attack_scores,

        "clean_feature_scores":
            clean_feature_scores,

        "adv_feature_scores":
            adv_feature_scores
    },

    "attack_features_block11.pt"
)


# Also save human-readable feature IDs
with open(
    "attack_features_block11.json",
    "w"
) as f:

    json.dump(
        {
            "top_256_features":
                top_indices.tolist(),

            "attack_scores":
                top_values.tolist()
        },

        f,
        indent=2
    )


print(
    "\nSaved attack_features_block11.pt"
)

print(
    "Saved attack_features_block11.json"
)