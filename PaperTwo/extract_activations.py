import torch
import random

from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor




device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()




emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

mery_root = Path(
    "data/FERG_DB/FERG_DB_256/mery"
)




def get_all_cls_reps(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )

    cls_reps = []

    # hidden_states[1] ... hidden_states[12]
    # correspond to transformer blocks 0 ... 11

    for hidden_state_index in range(1, 13):

        hidden_state = outputs.hidden_states[
            hidden_state_index
        ]

        cls_rep = hidden_state[:, 0, :].squeeze(0)

        cls_reps.append(cls_rep)

    # [12, 768]
    return torch.stack(cls_reps)




def get_reps_from_paths(paths, label):

    reps = []

    for i, path in enumerate(paths):

        image = Image.open(path).convert("RGB")

        # [12, 768]
        image_reps = get_all_cls_reps(image)

        reps.append(image_reps)

        if (i + 1) % 25 == 0:
            print(
                f"{label}: "
                f"processed {i + 1}/{len(paths)}"
            )

    # [num_images, 12, 768]
    reps = torch.stack(reps)

    # Change to:
    # [12, num_images, 768]
    reps = reps.permute(1, 0, 2)

    return reps




num_train_per_class = 256
num_test_per_class = 64




neutral_paths = list(
    (mery_root / "mery_neutral").glob("*.png")
)

random.seed(42)
random.shuffle(neutral_paths)

neutral_train = neutral_paths[
    :num_train_per_class
]

neutral_test = neutral_paths[
    num_train_per_class:
    num_train_per_class + num_test_per_class
]

print(
    "Neutral:",
    len(neutral_train),
    "train,",
    len(neutral_test),
    "test"
)



print("\nExtracting neutral training representations...")

neutral_train_reps = get_reps_from_paths(
    neutral_train,
    "Neutral train"
)


print("\nExtracting neutral test representations...")

neutral_test_reps = get_reps_from_paths(
    neutral_test,
    "Neutral test"
)


print(
    "Neutral train reps:",
    neutral_train_reps.shape
)

print(
    "Neutral test reps:",
    neutral_test_reps.shape
)


torch.save(
    neutral_train_reps.cpu(),
    "neutral_train_reps_all_layers.pt"
)

torch.save(
    neutral_test_reps.cpu(),
    "neutral_test_reps_all_layers.pt"
)




for emotion in emotions:

    print("\n")
    print("=" * 60)
    print("Processing emotion:", emotion)
    print("=" * 60)

    emotion_dir = (
        mery_root / f"mery_{emotion}"
    )

    emotion_paths = list(
        emotion_dir.glob("*.png")
    )

    print(
        f"Total {emotion} images:",
        len(emotion_paths)
    )

    # Reset seed so each emotion has a reproducible split
    random.seed(42)
    random.shuffle(emotion_paths)




    emotion_train = emotion_paths[
        :num_train_per_class
    ]

    emotion_test = emotion_paths[
        num_train_per_class:
        num_train_per_class + num_test_per_class
    ]

    print(
        f"{emotion} train:",
        len(emotion_train)
    )

    print(
        f"{emotion} test:",
        len(emotion_test)
    )




    print(
        f"\nExtracting {emotion} training representations..."
    )

    emotion_train_reps = get_reps_from_paths(
        emotion_train,
        f"{emotion} train"
    )


    print(
        f"\nExtracting {emotion} test representations..."
    )

    emotion_test_reps = get_reps_from_paths(
        emotion_test,
        f"{emotion} test"
    )


    print(
        f"{emotion} train reps:",
        emotion_train_reps.shape
    )

    print(
        f"{emotion} test reps:",
        emotion_test_reps.shape
    )



    difference_vectors = []

    for i in range(num_train_per_class):

        # Each:
        # [12, 768]

        emotion_rep = emotion_train_reps[
            :, i, :
        ]

        neutral_rep = neutral_train_reps[
            :, i, :
        ]


        # Alternate sign of contrastive pairs
        #
        # pair 0:
        # emotion - neutral
        #
        # pair 1:
        # neutral - emotion
        #
        # etc.

        if i % 2 == 0:

            difference = (
                emotion_rep - neutral_rep
            )

        else:

            difference = (
                neutral_rep - emotion_rep
            )


        # Normalize separately for every layer
        norms = difference.norm(
            dim=-1,
            keepdim=True
        )

        difference = difference / norms

        difference_vectors.append(
            difference
        )


    # [256, 12, 768]
    differences = torch.stack(
        difference_vectors
    )

    # -> [12, 256, 768]
    differences = differences.permute(
        1, 0, 2
    )


    print(
        f"{emotion} differences:",
        differences.shape
    )


   

    torch.save(
        differences.cpu(),
        f"{emotion}_differences_all_layers.pt"
    )

    torch.save(
        emotion_train_reps.cpu(),
        f"{emotion}_train_reps_all_layers.pt"
    )

    torch.save(
        emotion_test_reps.cpu(),
        f"{emotion}_test_reps_all_layers.pt"
    )

    print(
        f"Saved tensors for {emotion}."
    )


print("\n")
print("=" * 60)
print("FINISHED ALL EMOTIONS")
print("=" * 60)