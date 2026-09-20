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




tokenizer = open_clip.get_tokenizer(
    model_name
)



dataset = CIFAR100(
    root="./data",
    train=False,
    download=False,
    transform=preprocess
)

class_names = dataset.classes

print("\nNumber of classes:")
print(len(class_names))

print("\nFirst 10 class names:")
print(class_names[:10])



prompts = []

for class_name in class_names:

    readable_name = class_name.replace(
        "_",
        " "
    )

    prompt = f"a photo of a {readable_name}"

    prompts.append(prompt)


print("\nExample prompts:")

for prompt in prompts[:10]:
    print(prompt)




text_tokens = tokenizer(
    prompts
).to(device)


with torch.no_grad():

    text_features = model.encode_text(
        text_tokens
    )


# Normalize embeddings
text_features = (
    text_features
    / text_features.norm(
        dim=-1,
        keepdim=True
    )
)


print("\nText feature shape:")
print(text_features.shape)

# Expected:
# [100, 512]


# start with 500 images

num_test_images = 500

subset = Subset(
    dataset,
    range(num_test_images)
)


loader = DataLoader(
    subset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)




correct = 0
total = 0


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)


        # ------------------------------------------
        # Image embeddings
        # ------------------------------------------

        image_features = model.encode_image(
            images
        )


        # Normalize
        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )


        # ------------------------------------------
        # Image-text similarity
        # ------------------------------------------

        logits = (
            image_features
            @ text_features.T
        )


        predictions = logits.argmax(
            dim=-1
        )


        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)




accuracy = correct / total


print("\n" + "=" * 50)
print("CLEAN ZERO-SHOT RESULTS")
print("=" * 50)

print("Images evaluated:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)