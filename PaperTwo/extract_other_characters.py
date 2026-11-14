import torch
import random

from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# Cross-character test set.
#
# FERG-DB images are animation frames, so a random split of Mery's
# images puts near-identical frames in both train and test. To test
# whether the directions learned on Mery generalize, we evaluate them
# on the other five FERG-DB characters instead, which share no frames
# with Mery.


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()


characters = ["aia", "bonnie", "jules", "malcolm", "ray"]

classes = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

ferg_root = Path("data/FERG_DB/FERG_DB_256")
out_dir = Path("cross_character")
out_dir.mkdir(exist_ok=True)

num_test_per_class = 64
batch_size = 32


def get_all_cls_reps(images):

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )

    # hidden_states[1] ... hidden_states[12]
    # correspond to transformer blocks 0 ... 11
    #
    # [12, batch, 768]
    return torch.stack(
        [outputs.hidden_states[i][:, 0, :] for i in range(1, 13)]
    )


for character in characters:

    for class_name in classes:

        paths = sorted(
            (ferg_root / character / f"{character}_{class_name}").glob("*.png")
        )

        random.seed(42)
        random.shuffle(paths)

        paths = paths[:num_test_per_class]

        reps = []

        for start in range(0, len(paths), batch_size):

            images = [
                Image.open(p).convert("RGB")
                for p in paths[start:start + batch_size]
            ]

            reps.append(get_all_cls_reps(images).cpu())

        # [12, 64, 768]
        reps = torch.cat(reps, dim=1)

        torch.save(
            reps,
            out_dir / f"{character}_{class_name}_reps_all_layers.pt"
        )

        print(f"{character} {class_name}: {tuple(reps.shape)}")


print("\nSaved cross-character representations to", out_dir)
