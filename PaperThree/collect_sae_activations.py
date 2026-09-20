# 2000 CIFAR-100 training images

import torch
import open_clip

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100




device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model_name = "ViT-B-32"
pretrained = "datacomp_xl_s13b_b90k"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

model = model.to(device)
model.eval()




selected_blocks = [2, 5, 8, 11]

print("Selected blocks:", selected_blocks)




dataset = CIFAR100(
    root="./data",
    train=True,
    download=False,
    transform=preprocess
)

num_images = 2000

subset = Subset(
    dataset,
    range(num_images)
)

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)




current_activations = {}

all_activations = {
    block: []
    for block in selected_blocks
}

hooks = []




def make_hook(block_index):

    def hook(module, inputs, output):

        # OpenCLIP activation:
        #
        # [batch, 50, 768]
        #
        # token 0     = CLS
        # tokens 1:50 = image patches

        patch_tokens = output[:, 1:, :]

        # Convert:
        #
        # [B, 49, 768]
        #
        # ->
        #
        # [B * 49, 768]

        patch_tokens = patch_tokens.reshape(
            -1,
            patch_tokens.shape[-1]
        )

        # Move to CPU and store as float16 to
        # reduce disk / RAM usage.

        current_activations[block_index] = (
            patch_tokens
            .detach()
            .cpu()
            .half()
        )

    return hook



for block_index in selected_blocks:

    block = model.visual.transformer.resblocks[
        block_index
    ]

    handle = block.register_forward_hook(
        make_hook(block_index)
    )

    hooks.append(handle)



processed = 0

print("\nCollecting clean patch activations...")


with torch.no_grad():

    for images, _ in loader:

        images = images.to(device)

        # Forward pass triggers hooks
        model.encode_image(images)

        for block_index in selected_blocks:

            all_activations[
                block_index
            ].append(
                current_activations[
                    block_index
                ]
            )

        processed += images.size(0)

        print(
            f"Processed "
            f"{processed}/{num_images}"
        )




for handle in hooks:
    handle.remove()




print("\n" + "=" * 60)
print("ACTIVATION DATASETS")
print("=" * 60)


for block_index in selected_blocks:

    activations = torch.cat(
        all_activations[block_index],
        dim=0
    )

    print(
        f"\nBlock {block_index}:",
        activations.shape
    )

    # Convert to float32 for accurate statistics
    activations_float = activations.float()

    mean = activations_float.mean(
        dim=0
    )

    std = activations_float.std(
        dim=0
    )

    # Avoid division-by-zero later
    std = torch.clamp(
        std,
        min=1e-6
    )

    print(
        "Mean activation magnitude:",
        activations_float.abs().mean().item()
    )

    print(
        "Average feature std:",
        std.mean().item()
    )


    # --------------------------------------------------
    # Save RAW clean activations
    # --------------------------------------------------

    torch.save(
        activations,
        f"sae_train_block{block_index}.pt"
    )


    # --------------------------------------------------
    # Save standardization statistics
    # --------------------------------------------------

    torch.save(
        {
            "mean": mean,
            "std": std
        },
        f"sae_stats_block{block_index}.pt"
    )


print("\nFinished.")

print(
    "Expected activation shape per block: "
    "[98000, 768]"
)