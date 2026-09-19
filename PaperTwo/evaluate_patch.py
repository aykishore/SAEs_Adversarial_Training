import torch


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


# Your previous CLS-token results
cls_results = {
    "joy": 0.9922,
    "sadness": 1.0000,
    "anger": 1.0000,
    "fear": 1.0000,
    "surprise": 1.0000,
    "disgust": 1.0000
}


neutral_train_reps = torch.load(
    "neutral_train_reps_patch_all_layers.pt"
)

neutral_test_reps = torch.load(
    "neutral_test_reps_patch_all_layers.pt"
)


final_results = []


for emotion in emotions:

    print("\n")
    print("=" * 60)
    print("Evaluating PATCH representations:", emotion)
    print("=" * 60)

    directions = torch.load(
        f"{emotion}_directions_patch_all_layers.pt"
    )

    difference_means = torch.load(
        f"{emotion}_difference_means_patch_all_layers.pt"
    )

    emotion_train_reps = torch.load(
        f"{emotion}_train_reps_patch_all_layers.pt"
    )

    emotion_test_reps = torch.load(
        f"{emotion}_test_reps_patch_all_layers.pt"
    )


    best_layer = None
    best_accuracy = 0.0

    best_emotion_correct = None
    best_neutral_correct = None


    for layer in range(12):

        direction = directions[layer]
        diff_mean = difference_means[layer]


        # --------------------------------------------------
        # 1. Training scores
        # --------------------------------------------------

        emotion_train_scores = (
            emotion_train_reps[layer] - diff_mean
        ) @ direction

        neutral_train_scores = (
            neutral_train_reps[layer] - diff_mean
        ) @ direction


        # --------------------------------------------------
        # 2. Threshold learned ONLY from training data
        # --------------------------------------------------

        emotion_train_mean = emotion_train_scores.mean()
        neutral_train_mean = neutral_train_scores.mean()

        threshold = (
            emotion_train_mean + neutral_train_mean
        ) / 2


        # --------------------------------------------------
        # 3. Test scores
        # --------------------------------------------------

        emotion_test_scores = (
            emotion_test_reps[layer] - diff_mean
        ) @ direction

        neutral_test_scores = (
            neutral_test_reps[layer] - diff_mean
        ) @ direction


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
            emotion_correct + neutral_correct
        )

        accuracy = total_correct / 128


        print(
            f"Block {layer}: "
            f"{emotion} {emotion_correct}/64, "
            f"neutral {neutral_correct}/64, "
            f"accuracy = {accuracy:.4f}"
        )


        # --------------------------------------------------
        # 5. Best layer
        # --------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_layer = layer

            best_emotion_correct = emotion_correct
            best_neutral_correct = neutral_correct


    final_results.append(
        {
            "emotion": emotion,
            "best_layer": best_layer,
            "patch_accuracy": best_accuracy,
            "emotion_correct": best_emotion_correct,
            "neutral_correct": best_neutral_correct
        }
    )


    print("\nBest PATCH result for", emotion)
    print("Best block:", best_layer)
    print(
        "Best accuracy:",
        f"{best_accuracy:.4f}"
    )


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n")
print("=" * 85)
print("CLS vs PATCH vs PAPER")
print("=" * 85)

print(
    f"{'Emotion':<12}"
    f"{'Patch Block':<14}"
    f"{'CLS Acc':<12}"
    f"{'Patch Acc':<12}"
    f"{'Paper Acc':<12}"
)

print("-" * 62)


for result in final_results:

    emotion = result["emotion"]
    best_layer = result["best_layer"]
    patch_accuracy = result["patch_accuracy"]

    print(
        f"{emotion:<12}"
        f"{best_layer:<14}"
        f"{cls_results[emotion]:<12.4f}"
        f"{patch_accuracy:<12.4f}"
        f"{paper_results[emotion]:<12.4f}"
    )