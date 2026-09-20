import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from torchvision import transforms




device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model_name = "ViT-B-32"
pretrained = "datacomp_xl_s13b_b90k"

selected_blocks = [2, 5, 8, 11]

INPUT_DIM = 768
LATENT_DIM = 6144

BATCH_SIZE = 10





model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

model = model.to(device)
model.eval()




normalize_transform = None

for transform in preprocess.transforms:

    if isinstance(transform, transforms.Normalize):

        normalize_transform = transform
        break


if normalize_transform is None:

    raise RuntimeError(
        "Could not find CLIP normalization."
    )


mean = torch.tensor(
    normalize_transform.mean,
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    normalize_transform.std,
    device=device
).view(1, 3, 1, 1)


def normalize(images):

    return (images - mean) / std




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


for block in selected_blocks:

    checkpoint = torch.load(
        f"sae_block{block}_corrected.pt",
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
        f"sae_stats_block{block}.pt"
    )

    stats[block] = {
        "mean": block_stats["mean"].to(device),
        "std": block_stats["std"].to(device)
    }


print("\nLoaded SAEs for blocks:")
print(selected_blocks)



pairs = torch.load(
    "pgd_test_pairs.pt"
)

clean_images = pairs["clean_images"]
adv_images = pairs["adv_images"]
labels = pairs["labels"]


print("\nClean images:")
print(clean_images.shape)

print("PGD images:")
print(adv_images.shape)

print("Labels:")
print(labels.shape)




captured = {}

hooks = []


def make_hook(block_index):

    def hook(module, inputs, output):

        # [batch, 50, 768]

        captured[block_index] = (
            output.detach()
        )

    return hook




for block in selected_blocks:

    handle = (
        model.visual
        .transformer
        .resblocks[block]
        .register_forward_hook(
            make_hook(block)
        )
    )

    hooks.append(handle)




@torch.no_grad()
def get_l0_for_images(images):

    # Store one per-image value for each block
    #
    # We average the 49 patch L0 values
    # to get one L0 measurement per image.

    results = {
        block: []
        for block in selected_blocks
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

        batch = images[
            start:end
        ].to(device)


        # ------------------------------------------
        # Run through CLIP
        # ------------------------------------------

        normalized_batch = normalize(
            batch
        )

        model.encode_image(
            normalized_batch
        )


        # ------------------------------------------
        # Each selected layer
        # ------------------------------------------

        for block in selected_blocks:

            activation = captured[block]

            # Remove CLS token:
            #
            # [B, 50, 768]
            # ->
            # [B, 49, 768]

            patches = activation[
                :, 1:, :
            ]


            batch_size = patches.shape[0]


            # Flatten patch tokens:
            #
            # [B, 49, 768]
            # ->
            # [B*49, 768]

            flat_patches = patches.reshape(
                -1,
                INPUT_DIM
            )


            # --------------------------------------
            # Standardize using CLEAN training stats
            # --------------------------------------

            flat_patches = (
                flat_patches
                - stats[block]["mean"]
            ) / stats[block]["std"]


            # --------------------------------------
            # SAE encoding
            # --------------------------------------

            features = saes[
                block
            ].encode(
                flat_patches
            )


            # --------------------------------------
            # L0 = number of active SAE features
            # --------------------------------------

            patch_l0 = (
                features > 0
            ).sum(
                dim=1
            ).float()


            # Restore:
            #
            # [B*49]
            # ->
            # [B,49]

            patch_l0 = patch_l0.view(
                batch_size,
                49
            )


            # Mean L0 across image patches
            #
            # Gives one number per image

            image_l0 = patch_l0.mean(
                dim=1
            )


            results[block].append(
                image_l0.cpu()
            )


    # Concatenate all batches

    for block in selected_blocks:

        results[block] = torch.cat(
            results[block]
        )


    return results




print("\nCalculating CLEAN L0...")

clean_results = get_l0_for_images(
    clean_images
)




print("Calculating PGD L0...")

pgd_results = get_l0_for_images(
    adv_images
)




for handle in hooks:

    handle.remove()


# --------------------------------------------------
# 13. Final results
# --------------------------------------------------

print("\n")
print("=" * 78)
print("CLEAN vs PGD SAE FEATURE ACTIVITY")
print("=" * 78)

print(
    f"{'Block':<10}"
    f"{'Clean L0':<15}"
    f"{'PGD L0':<15}"
    f"{'Difference':<15}"
    f"{'PGD/Clean':<15}"
)

print("-" * 70)


for block in selected_blocks:

    clean_l0 = (
        clean_results[block]
        .mean()
        .item()
    )

    pgd_l0 = (
        pgd_results[block]
        .mean()
        .item()
    )

    difference = (
        pgd_l0 - clean_l0
    )

    ratio = (
        pgd_l0 / clean_l0
    )


    print(
        f"{block:<10}"
        f"{clean_l0:<15.2f}"
        f"{pgd_l0:<15.2f}"
        f"{difference:<15.2f}"
        f"{ratio:<15.3f}"
    )




print("\n")
print("=" * 78)
print("PER-IMAGE DIRECTION OF CHANGE")
print("=" * 78)


for block in selected_blocks:

    clean = clean_results[block]
    pgd = pgd_results[block]

    increases = (
        pgd > clean
    ).sum().item()

    decreases = (
        pgd < clean
    ).sum().item()

    equal = (
        pgd == clean
    ).sum().item()


    print(
        f"\nBlock {block}"
    )

    print(
        "Images with increased L0:",
        increases,
        "/",
        len(clean)
    )

    print(
        "Images with decreased L0:",
        decreases,
        "/",
        len(clean)
    )

    print(
        "Images unchanged:",
        equal,
        "/",
        len(clean)
    )



torch.save(
    {
        "clean_l0": clean_results,
        "pgd_l0": pgd_results
    },
    "clean_vs_pgd_l0_results.pt"
)


print(
    "\nSaved clean_vs_pgd_l0_results.pt"
)