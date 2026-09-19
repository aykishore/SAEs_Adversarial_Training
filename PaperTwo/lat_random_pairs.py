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


neutral_train_reps = torch.load(
    "neutral_train_reps_all_layers.pt"
)


for emotion_index, emotion in enumerate(emotions):

    print("\n")
    print("=" * 60)
    print("Random-pair LAT:", emotion)
    print("=" * 60)

    emotion_train_reps = torch.load(
        f"{emotion}_train_reps_all_layers.pt"
    )

    # --------------------------------------------------
    # 1. Combine the 512 training images
    #
    # 256 emotion + 256 neutral
    # --------------------------------------------------

    all_reps = torch.cat(
        [
            emotion_train_reps,
            neutral_train_reps
        ],
        dim=1
    )

    # [512]
    labels = torch.cat(
        [
            torch.ones(256),
            torch.zeros(256)
        ]
    )

    print("Combined reps:", all_reps.shape)


    # --------------------------------------------------
    # 2. Randomly shuffle the 512 stimuli
    #
    # IMPORTANT:
    # Labels are NOT used to decide the pairs.
    # --------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(42 + emotion_index)

    permutation = torch.randperm(
        512,
        generator=generator
    )

    shuffled_reps = all_reps[:, permutation, :]
    shuffled_labels = labels[permutation]


    # --------------------------------------------------
    # 3. Pair adjacent random stimuli
    #
    # image 0 vs image 1
    # image 2 vs image 3
    # ...
    #
    # -> 256 random pairs
    # --------------------------------------------------

    first = shuffled_reps[:, 0::2, :]
    second = shuffled_reps[:, 1::2, :]

    differences = first - second


    # --------------------------------------------------
    # 4. Normalize each difference vector
    # --------------------------------------------------

    norms = differences.norm(
        dim=-1,
        keepdim=True
    )

    differences = differences / norms

    print(
        "Random differences:",
        differences.shape
    )

    # Expected:
    # [12, 256, 768]


    # --------------------------------------------------
    # 5. PCA separately at each layer
    # --------------------------------------------------

    directions = []
    pca_means = []
    explained_variances = []


    for layer in range(12):

        X = differences[layer]

        pca = PCA(n_components=1)
        pca.fit(X.numpy())

        direction = torch.tensor(
            pca.components_[0],
            dtype=emotion_train_reps.dtype
        )


        # --------------------------------------------------
        # 6. Labels are used ONLY NOW
        #    to orient the sign.
        # --------------------------------------------------

        emotion_scores = (
            emotion_train_reps[layer]
            @ direction
        )

        neutral_scores = (
            neutral_train_reps[layer]
            @ direction
        )

        if emotion_scores.mean() < neutral_scores.mean():
            direction = -direction


        directions.append(direction)

        pca_means.append(
            torch.tensor(
                pca.mean_,
                dtype=emotion_train_reps.dtype
            )
        )

        explained_variances.append(
            pca.explained_variance_ratio_[0]
        )


        print(
            f"Block {layer}: "
            f"explained variance = "
            f"{pca.explained_variance_ratio_[0]:.4f}"
        )


    directions = torch.stack(directions)
    pca_means = torch.stack(pca_means)


    # --------------------------------------------------
    # 7. Save separately from old LAT
    # --------------------------------------------------

    torch.save(
        directions,
        f"{emotion}_directions_random_pairs.pt"
    )

    torch.save(
        pca_means,
        f"{emotion}_pca_means_random_pairs.pt"
    )


    print(
        f"Saved random-pair LAT for {emotion}."
    )


print("\nFINISHED RANDOM-PAIR LAT")