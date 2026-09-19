import torch


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


neutral_test_reps = torch.load(
    "neutral_test_reps_all_layers.pt"
)


# --------------------------------------------------
# Store results
# --------------------------------------------------

results = {}


for direction_emotion in emotions:

    print("\n")
    print("=" * 70)
    print("Direction learned for:", direction_emotion)
    print("=" * 70)

    directions = torch.load(
        f"{direction_emotion}_directions_random_pairs.pt"
    )

    pca_means = torch.load(
        f"{direction_emotion}_pca_means_random_pairs.pt"
    )


    # Use the best block we previously found for
    # each learned direction.
    #
    # From your random-pair results:
    layer = 11

    direction = directions[layer]
    pca_mean = pca_means[layer]


    results[direction_emotion] = {}


    # --------------------------------------------------
    # Compare this direction against EVERY emotion
    # --------------------------------------------------

    for test_emotion in emotions:

        emotion_test_reps = torch.load(
            f"{test_emotion}_test_reps_all_layers.pt"
        )


        emotion_scores = (
            emotion_test_reps[layer] - pca_mean
        ) @ direction


        neutral_scores = (
            neutral_test_reps[layer] - pca_mean
        ) @ direction


        emotion_correct = (
            emotion_scores > 0
        ).sum().item()

        neutral_correct = (
            neutral_scores <= 0
        ).sum().item()


        accuracy = (
            emotion_correct + neutral_correct
        ) / 128


        results[direction_emotion][
            test_emotion
        ] = accuracy


        print(
            f"{direction_emotion} direction "
            f"on {test_emotion}: "
            f"{accuracy:.4f}"
        )


# --------------------------------------------------
# Print matrix
# --------------------------------------------------

print("\n")
print("=" * 90)
print("CROSS-EMOTION ACCURACY MATRIX")
print("=" * 90)


header = f"{'Direction':<12}"

for emotion in emotions:
    header += f"{emotion:<11}"

print(header)

print("-" * 80)


for direction_emotion in emotions:

    row = f"{direction_emotion:<12}"

    for test_emotion in emotions:

        accuracy = results[
            direction_emotion
        ][test_emotion]

        row += f"{accuracy:<11.4f}"

    print(row)