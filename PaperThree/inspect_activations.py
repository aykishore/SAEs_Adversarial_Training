import torch
import open_clip

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



# use CIFAR-100 for now because it is small
# and easy to work with.
#
# OpenCLIP's preprocess automatically resizes
# the image to 224 x 224.


dataset = CIFAR100(
    root="./data",
    train=False,
    download=True
)

image, label = dataset[0]

print("\nOriginal image size:")
print(image.size)

print("\nCIFAR-100 label:")
print(label)




image_tensor = preprocess(image)

print("\nAfter CLIP preprocessing:")
print(image_tensor.shape)

# Add batch dimension:
#
# [3, 224, 224]
#       ->
# [1, 3, 224, 224]

image_tensor = image_tensor.unsqueeze(0).to(device)

print("\nInput batch shape:")
print(image_tensor.shape)




selected_blocks = [2, 5, 8, 11]

activations = {}

hooks = []




def make_hook(block_index):

    def hook(module, inputs, output):

        # Detach so PyTorch doesn't keep the
        # computation graph around.
        activations[block_index] = output.detach()

    return hook




for block_index in selected_blocks:

    block = model.visual.transformer.resblocks[
        block_index
    ]

    handle = block.register_forward_hook(
        make_hook(block_index)
    )

    hooks.append(handle)




with torch.no_grad():

    image_embedding = model.encode_image(
        image_tensor
    )


print("\nFinal CLIP embedding:")
print(image_embedding.shape)




print("\n" + "=" * 60)
print("CAPTURED TRANSFORMER ACTIVATIONS")
print("=" * 60)


for block_index in selected_blocks:

    activation = activations[block_index]

    print(
        f"\nBlock {block_index} raw shape:",
        activation.shape
    )




example = activations[selected_blocks[0]]

print("\nExample raw activation shape:")
print(example.shape)


# OpenCLIP versions may represent transformer
# activations as either:
#
# [batch, tokens, hidden]
#
# or:
#
# [tokens, batch, hidden]
#
# We detect which format we have.

if example.shape[0] == 1:

    layout = "batch_first"

elif example.shape[1] == 1:

    layout = "sequence_first"

else:

    raise ValueError(
        "Could not determine activation layout."
    )


print("\nDetected activation layout:")
print(layout)




print("\n" + "=" * 60)
print("PATCH TOKEN ACTIVATIONS")
print("=" * 60)


for block_index in selected_blocks:

    activation = activations[block_index]


    if layout == "batch_first":

        # Shape:
        # [1, 50, 768]
        #
        # Token 0 = CLS
        # Tokens 1:50 = patches

        cls_token = activation[:, 0, :]

        patch_tokens = activation[:, 1:, :]


    else:

        # Shape:
        # [50, 1, 768]

        cls_token = activation[0, :, :]

        patch_tokens = activation[1:, :, :]

        # Convert to:
        # [1, 49, 768]

        patch_tokens = patch_tokens.permute(
            1, 0, 2
        )


    print(
        f"\nBlock {block_index}"
    )

    print(
        "CLS shape:",
        cls_token.shape
    )

    print(
        "Patch-token shape:",
        patch_tokens.shape
    )


    # Flatten so every patch becomes one
    # independent 768-dimensional example.

    flat_patches = patch_tokens.reshape(
        -1,
        patch_tokens.shape[-1]
    )


    print(
        "Flattened patches:",
        flat_patches.shape
    )




for handle in hooks:

    handle.remove()


print("\nFinished.")