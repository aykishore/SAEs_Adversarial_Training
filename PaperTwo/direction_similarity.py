import torch
import torch.nn.functional as F


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

layer = 11

directions = {}


# ---------------------------------------------
# Load each emotion direction at SAME layer
# ---------------------------------------------

for emotion in emotions:

    all_directions = torch.load(
        f"{emotion}_directions_random_pairs.pt"
    )

    directions[emotion] = all_directions[layer]


# ---------------------------------------------
# Print cosine similarity matrix
# ---------------------------------------------

print(f"\nDirection cosine similarity at block {layer}\n")

header = f"{'':<12}"

for emotion in emotions:
    header += f"{emotion:<11}"

print(header)
print("-" * 80)


for emotion1 in emotions:

    row = f"{emotion1:<12}"

    for emotion2 in emotions:

        similarity = F.cosine_similarity(
            directions[emotion1].unsqueeze(0),
            directions[emotion2].unsqueeze(0)
        ).item()

        row += f"{similarity:<11.4f}"

    print(row)