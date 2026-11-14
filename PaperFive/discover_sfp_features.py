import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from pathlib import Path
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "datacomp_xl_s13b_b90k"

BLOCKS = [2, 5, 8, 11]

INPUT_DIM = 768
LATENT_DIM = 6144

# SFPF paper example threshold
TAU = 0.03

BATCH_SIZE = 10

paper3_dir = Path("../PaperThree")
paper4_dir = Path("../PaperFour")



model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer(
    MODEL_NAME
)


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



# Rebuild CIFAR-100 zero-shot classifier
#
# want SUCCESSFUL PGD attacks for feature
# discovery, matching the SFPF setup.


dataset = datasets.CIFAR100(
    root=paper3_dir / "data",
    train=False,
    download=True
)

class_names = [
    name.replace("_", " ")
    for name in dataset.classes
]

prompts = [
    f"a photo of a {name}"
    for name in class_names
]

text_tokens = tokenizer(
    prompts
).to(device)


with torch.no_grad():

    text_features = model.encode_text(
        text_tokens
    )

    text_features = F.normalize(
        text_features,
        dim=-1
    )


@torch.no_grad()
def get_predictions(images):

    predictions = []

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

        image_features = model.encode_image(
            normalize(batch)
        )

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        logits = (
            model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        predictions.append(
            logits.argmax(dim=1).cpu()
        )

    return torch.cat(
        predictions
    )




# Must match PaperThree.


class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim
    ):

        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            latent_dim
        )

        self.decoder = nn.Linear(
            latent_dim,
            input_dim
        )


    def encode(self, x):

        return F.relu(
            self.encoder(x)
        )


    def decode(self, features):

        return self.decoder(
            features
        )


    def forward(self, x):

        features = self.encode(x)

        reconstruction = self.decode(
            features
        )

        return reconstruction, features



saes = {}
stats = {}


for block in BLOCKS:

    checkpoint = torch.load(
        paper3_dir
        / f"sae_block{block}_corrected.pt",
        map_location=device
    )

    sae = SparseAutoencoder(
        INPUT_DIM,
        LATENT_DIM
    ).to(device)

    sae.load_state_dict(
        checkpoint["state_dict"]
    )

    sae.eval()

    saes[block] = sae


    block_stats = torch.load(
        paper3_dir
        / f"sae_stats_block{block}.pt"
    )

    stats[block] = {
        "mean":
            block_stats["mean"].to(device),

        "std":
            block_stats["std"].to(device)
    }


print(
    "\nLoaded SAEs for blocks:",
    BLOCKS
)



data = torch.load(
    paper4_dir / "saegis_data.pt"
)

discovery_clean = data[
    "discovery_clean"
]

discovery_adv = data[
    "discovery_adv"
]

discovery_labels = data[
    "discovery_labels"
]


print("\nDiscovery clean:")
print(discovery_clean.shape)

print("Discovery PGD:")
print(discovery_adv.shape)



print(
    "\nChecking which PGD attacks succeeded..."
)

adv_predictions = get_predictions(
    discovery_adv
)

successful_mask = (
    adv_predictions
    != discovery_labels
)

num_successful = (
    successful_mask
    .sum()
    .item()
)


print(
    "Successful attacks:",
    num_successful,
    "/",
    len(discovery_adv)
)


if num_successful == 0:

    raise RuntimeError(
        "No successful PGD attacks found."
    )


successful_adv = discovery_adv[
    successful_mask
]



captured = {}
hooks = []


def make_hook(block):

    def hook(
        module,
        inputs,
        output
    ):

        captured[block] = (
            output.detach()
        )

    return hook


for block in BLOCKS:

    handle = (
        model.visual
        .transformer
        .resblocks[block]
        .register_forward_hook(
            make_hook(block)
        )
    )

    hooks.append(handle)



#
# For each image:
#
#   CLIP patches
#       ↓
#   SAE encode each patch
#       ↓
#   average 49 SAE vectors
#
# Result per image:
#
#   [6144]

@torch.no_grad()
def get_image_sae_features(images):

    results = {
        block: []
        for block in BLOCKS
    }


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
        # One CLIP pass captures all four blocks
        # ------------------------------------------

        model.encode_image(
            normalize(batch)
        )


        for block in BLOCKS:

            activation = captured[
                block
            ]


            # --------------------------------------
            # Remove CLS token
            #
            # [B,50,768]
            # ->
            # [B,49,768]
            # --------------------------------------

            patches = activation[
                :, 1:, :
            ]

            current_batch_size = (
                patches.shape[0]
            )


            # --------------------------------------
            # Flatten patches
            #
            # [B,49,768]
            # ->
            # [B*49,768]
            # --------------------------------------

            flat = patches.reshape(
                -1,
                INPUT_DIM
            )


            # --------------------------------------
            # Same clean normalization used to
            # train this SAE
            # --------------------------------------

            flat = (
                flat
                - stats[block]["mean"]
            ) / stats[block]["std"]


            # --------------------------------------
            # SAE sparse representation
            # --------------------------------------

            features = saes[
                block
            ].encode(
                flat
            )


            # --------------------------------------
            # Restore image structure
            #
            # [B*49,6144]
            # ->
            # [B,49,6144]
            # --------------------------------------

            features = features.view(
                current_batch_size,
                49,
                LATENT_DIM
            )


            # --------------------------------------
            # Pool over patch tokens
            #
            # [B,49,6144]
            # ->
            # [B,6144]
            # --------------------------------------

            image_features = (
                features.mean(
                    dim=1
                )
            )


            results[
                block
            ].append(
                image_features.cpu()
            )


    for block in BLOCKS:

        results[block] = torch.cat(
            results[block],
            dim=0
        )


    return results


print(
    "\nExtracting SAE representations "
    "from successful PGD images..."
)

attack_features = get_image_sae_features(
    successful_adv
)


# Remove hooks
for handle in hooks:
    handle.remove()



# Paper:
#
# c = mean(z_attack)
#
# c_normalized = c / ||c||_2
#
# mask_i = 1 if |c_i| > tau


saved_results = {
    "tau": TAU,
    "blocks": BLOCKS,
    "num_successful_attacks":
        num_successful,
    "masks": {}
}


print("\n")
print("=" * 72)
print("SFPF ATTACK-FEATURE DISCOVERY")
print("=" * 72)


for block in BLOCKS:

    features = attack_features[
        block
    ]


    # ------------------------------------------
    # k=1 centroid
    # ------------------------------------------

    centroid = features.mean(
        dim=0
    )


    # ------------------------------------------
    # L2 normalize
    # ------------------------------------------

    centroid_norm = centroid.norm(
        p=2
    )

    normalized_centroid = (
        centroid
        / centroid_norm.clamp(
            min=1e-12
        )
    )


    # ------------------------------------------
    # Binary SFPF danger mask
    # ------------------------------------------

    mask = (
        normalized_centroid.abs()
        > TAU
    )


    selected_features = (
        mask
        .nonzero(
            as_tuple=False
        )
        .squeeze(1)
    )


    num_selected = (
        len(selected_features)
    )


    # ------------------------------------------
    # For inspection: top 10 centroid features
    # ------------------------------------------

    top_values, top_indices = torch.topk(
        normalized_centroid.abs(),
        k=10
    )


    print(
        f"\nBlock {block}"
    )

    print(
        "Centroid L2 norm:",
        centroid_norm.item()
    )

    print(
        "Maximum normalized feature:",
        normalized_centroid.abs()
        .max()
        .item()
    )

    print(
        f"Features selected at tau={TAU}:",
        num_selected,
        "/",
        LATENT_DIM
    )

    print(
        "Top 10 feature IDs:",
        top_indices.tolist()
    )

    print(
        "Top 10 normalized values:",
        [
            round(x, 6)
            for x in top_values.tolist()
        ]
    )


    saved_results["masks"][
        block
    ] = {
        "centroid":
            centroid.cpu(),

        "normalized_centroid":
            normalized_centroid.cpu(),

        "mask":
            mask.cpu(),

        "selected_features":
            selected_features.cpu(),

        "num_selected":
            num_selected
    }




torch.save(
    saved_results,
    "sfp_feature_masks.pt"
)


print(
    "\nSaved sfp_feature_masks.pt"
)