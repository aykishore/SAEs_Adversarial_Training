import torch
from sklearn.decomposition import PCA


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


for emotion in emotions:

    print("\n")
    print("=" * 60)
    print("Learning LAT directions for:", emotion)
    print("=" * 60)

    differences = torch.load(
        f"{emotion}_differences_all_layers.pt"
    )

    emotion_train_reps = torch.load(
        f"{emotion}_train_reps_all_layers.pt"
    )

    neutral_train_reps = torch.load(
        "neutral_train_reps_all_layers.pt"
    )

    print("Differences:", differences.shape)

    directions = []
    difference_means = []

    for layer in range(12):

        # [256, 768]
        X = differences[layer]

        # Mean used for RepE-style recentering
        diff_mean = X.mean(dim=0)

        # PCA
        pca = PCA(n_components=1)
        pca.fit(X.numpy())

        direction = torch.tensor(
            pca.components_[0],
            dtype=emotion_train_reps.dtype
        )

        # --------------------------------------
        # Orient the direction
        #
        # Higher score should mean:
        # "more of this emotion"
        # --------------------------------------

        emotion_scores = (
            emotion_train_reps[layer] - diff_mean
        ) @ direction

        neutral_scores = (
            neutral_train_reps[layer] - diff_mean
        ) @ direction

        proportion_emotion_higher = (
            emotion_scores > neutral_scores
        ).float().mean().item()

        if proportion_emotion_higher < 0.5:
            direction = -direction

        directions.append(direction)
        difference_means.append(diff_mean)

        print(
            f"Block {layer}: "
            f"explained variance = "
            f"{pca.explained_variance_ratio_[0]:.4f}"
        )

    # [12, 768]
    directions = torch.stack(directions)

    # [12, 768]
    difference_means = torch.stack(
        difference_means
    )

    # --------------------------------------
    # Save
    # --------------------------------------

    torch.save(
        directions,
        f"{emotion}_directions_all_layers.pt"
    )

    torch.save(
        difference_means,
        f"{emotion}_difference_means_all_layers.pt"
    )

    print(
        f"Saved LAT directions for {emotion}."
    )


print("\n")
print("=" * 60)
print("FINISHED LAT FOR ALL EMOTIONS")
print("=" * 60)