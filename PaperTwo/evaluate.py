import torch


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]


# Paper's reported best-layer accuracies
paper_results = {
    "joy": 0.742,
    "sadness": 0.617,
    "anger": 0.727,
    "fear": 0.734,
    "surprise": 0.688,
    "disgust": 0.609
}


neutral_train_reps = torch.load(
    "neutral_train_reps_all_layers.pt"
)

neutral_test_reps = torch.load(
    "neutral_test_reps_all_layers.pt"
)


final_results = []


for emotion in emotions:

    print("\n")
    print("=" * 60)
    print("Evaluating:", emotion)
    print("=" * 60)

    directions = torch.load(
        f"{emotion}_directions_all_layers.pt"
    )

    difference_means = torch.load(
        f"{emotion}_difference_means_all_layers.pt"
    )

    emotion_train_reps = torch.load(
        f"{emotion}_train_reps_all_layers.pt"
    )

    emotion_test_reps = torch.load(
        f"{emotion}_test_reps_all_layers.pt"
    )


    best_layer = None
    best_accuracy = 0.0

    best_emotion_correct = None
    best_neutral_correct = None


    for layer in range(12):

        direction = directions[layer]

        diff_mean = difference_means[layer]


        # --------------------------------------------------
        # 1. TRAINING SCORES
        # --------------------------------------------------

        emotion_train_centered = (
            emotion_train_reps[layer] - diff_mean
        )

        neutral_train_centered = (
            neutral_train_reps[layer] - diff_mean
        )

        emotion_train_scores = (
            emotion_train_centered @ direction
        )

        neutral_train_scores = (
            neutral_train_centered @ direction
        )


        # --------------------------------------------------
        # 2. Learn threshold ONLY from training data
        # --------------------------------------------------

        emotion_train_mean = (
            emotion_train_scores.mean()
        )

        neutral_train_mean = (
            neutral_train_scores.mean()
        )

        threshold = (
            emotion_train_mean
            + neutral_train_mean
        ) / 2


        # --------------------------------------------------
        # 3. TEST SCORES
        # --------------------------------------------------

        emotion_test_centered = (
            emotion_test_reps[layer] - diff_mean
        )

        neutral_test_centered = (
            neutral_test_reps[layer] - diff_mean
        )

        emotion_test_scores = (
            emotion_test_centered @ direction
        )

        neutral_test_scores = (
            neutral_test_centered @ direction
        )


        # --------------------------------------------------
        # 4. Individual-image classification
        # --------------------------------------------------

        emotion_correct = (
            emotion_test_scores > threshold
        ).sum().item()

        neutral_correct = (
            neutral_test_scores <= threshold
        ).sum().item()

        total_correct = (
            emotion_correct
            + neutral_correct
        )

        accuracy = total_correct / 128


        print(
            f"Block {layer}: "
            f"{emotion} {emotion_correct}/64, "
            f"neutral {neutral_correct}/64, "
            f"accuracy = {accuracy:.4f}"
        )


        # --------------------------------------------------
        # 5. Track best layer
        # --------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_layer = layer

            best_emotion_correct = emotion_correct
            best_neutral_correct = neutral_correct


    # Save summary for this emotion

    final_results.append(
        {
            "emotion": emotion,
            "best_layer": best_layer,
            "accuracy": best_accuracy,
            "emotion_correct": best_emotion_correct,
            "neutral_correct": best_neutral_correct
        }
    )

    print("\nBest result for", emotion)
    print("Best block:", best_layer)
    print(
        "Best accuracy:",
        f"{best_accuracy:.4f}"
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n")
print("=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print(
    f"{'Emotion':<12}"
    f"{'Best Block':<14}"
    f"{'Our Acc':<12}"
    f"{'Paper Acc':<12}"
)

print("-" * 50)


for result in final_results:

    emotion = result["emotion"]
    best_layer = result["best_layer"]
    accuracy = result["accuracy"]

    paper_accuracy = paper_results[emotion]

    print(
        f"{emotion:<12}"
        f"{best_layer:<14}"
        f"{accuracy:<12.4f}"
        f"{paper_accuracy:<12.4f}"
    )