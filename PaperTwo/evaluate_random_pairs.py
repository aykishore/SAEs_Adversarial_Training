import torch
import numpy as np


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


paper_results = {
    "joy": 0.742,
    "sadness": 0.617,
    "anger": 0.727,
    "fear": 0.734,
    "surprise": 0.688,
    "disgust": 0.609
}


neutral_test_reps = torch.load(
    "neutral_test_reps_all_layers.pt"
)


final_results = []


for emotion in emotions:

    print("\n")
    print("=" * 60)
    print("Evaluating random-pair LAT:", emotion)
    print("=" * 60)

    directions = torch.load(
        f"{emotion}_directions_random_pairs.pt"
    )

    pca_means = torch.load(
        f"{emotion}_pca_means_random_pairs.pt"
    )

    emotion_test_reps = torch.load(
        f"{emotion}_test_reps_all_layers.pt"
    )


    best_layer = None
    best_accuracy = 0.0
    best_correlation = None


    for layer in range(12):

        direction = directions[layer]
        pca_mean = pca_means[layer]


        # --------------------------------------------------
        # 1. Recenter using PCA training mean
        # --------------------------------------------------

        emotion_centered = (
            emotion_test_reps[layer] - pca_mean
        )

        neutral_centered = (
            neutral_test_reps[layer] - pca_mean
        )


        # --------------------------------------------------
        # 2. Project onto LAT direction
        # --------------------------------------------------

        emotion_scores = (
            emotion_centered @ direction
        )

        neutral_scores = (
            neutral_centered @ direction
        )


        # --------------------------------------------------
        # 3. Individual-image classification
        #
        # No learned midpoint threshold.
        #
        # positive -> emotion
        # <= 0     -> neutral
        # --------------------------------------------------

        emotion_correct = (
            emotion_scores > 0
        ).sum().item()

        neutral_correct = (
            neutral_scores <= 0
        ).sum().item()

        accuracy = (
            emotion_correct + neutral_correct
        ) / 128


        # --------------------------------------------------
        # 4. Also measure correlation
        #
        # This is NOT the paper's reported "accuracy".
        # It is just useful diagnostic information.
        # --------------------------------------------------

        all_scores = torch.cat(
            [
                emotion_scores,
                neutral_scores
            ]
        ).numpy()

        labels = np.concatenate(
            [
                np.ones(64),
                np.zeros(64)
            ]
        )

        correlation = np.corrcoef(
            all_scores,
            labels
        )[0, 1]


        print(
            f"Block {layer}: "
            f"{emotion} {emotion_correct}/64, "
            f"neutral {neutral_correct}/64, "
            f"accuracy = {accuracy:.4f}, "
            f"corr = {correlation:.4f}"
        )


        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_layer = layer
            best_correlation = correlation


    final_results.append(
        {
            "emotion": emotion,
            "best_layer": best_layer,
            "accuracy": best_accuracy,
            "correlation": best_correlation
        }
    )


# --------------------------------------------------
# FINAL TABLE
# --------------------------------------------------

print("\n")
print("=" * 80)
print("RANDOM-PAIR LAT vs PAPER")
print("=" * 80)

print(
    f"{'Emotion':<12}"
    f"{'Best Block':<14}"
    f"{'Our Acc':<12}"
    f"{'Paper Acc':<12}"
    f"{'Corr':<12}"
)

print("-" * 62)


for result in final_results:

    emotion = result["emotion"]

    print(
        f"{emotion:<12}"
        f"{result['best_layer']:<14}"
        f"{result['accuracy']:<12.4f}"
        f"{paper_results[emotion]:<12.4f}"
        f"{result['correlation']:<12.4f}"
    )