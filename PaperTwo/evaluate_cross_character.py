import json
import math

import torch
from sklearn.metrics import roc_auc_score


# Evaluates the emotion directions learned on Mery on the other five
# FERG-DB characters (run extract_other_characters.py first).
#
# Layer selection: the block is chosen on one held-out character (aia),
# by highest accuracy with ties broken by the larger score separation,
# and accuracy is reported on the remaining four. Mery's own test images
# can't be used for this: near-duplicate frames put them at ~100% at
# almost every block, so they don't distinguish the blocks.
#
# Two methods, as in the rest of this folder:
#   paired LAT       lat.py:              score = (rep - diff_mean) @ direction,
#                                         threshold = midpoint of Mery train means
#   random-pair LAT  lat_random_pairs.py: score = (rep - pca_mean) @ direction,
#                                         threshold = 0 (the RepE paper's method)


emotions = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

validation_character = "aia"
characters = ["bonnie", "jules", "malcolm", "ray"]

paper_results = {
    "joy": 0.742,
    "sadness": 0.617,
    "anger": 0.727,
    "fear": 0.734,
    "surprise": 0.688,
    "disgust": 0.609
}


def load_cross(character, class_name):
    return torch.load(f"cross_character/{character}_{class_name}_reps_all_layers.pt")


neutral_train = torch.load("neutral_train_reps_all_layers.pt")
neutral_test = torch.load("neutral_test_reps_all_layers.pt")


def make_scorer(method, emotion):

    if method == "paired":
        directions = torch.load(f"{emotion}_directions_all_layers.pt")
        centers = torch.load(f"{emotion}_difference_means_all_layers.pt")
    else:
        directions = torch.load(f"{emotion}_directions_random_pairs.pt")
        centers = torch.load(f"{emotion}_pca_means_random_pairs.pt")

    emotion_train = torch.load(f"{emotion}_train_reps_all_layers.pt")

    thresholds = []

    for layer in range(12):

        if method == "paired":
            emotion_mean = ((emotion_train[layer] - centers[layer]) @ directions[layer]).mean()
            neutral_mean = ((neutral_train[layer] - centers[layer]) @ directions[layer]).mean()
            thresholds.append(((emotion_mean + neutral_mean) / 2).item())
        else:
            thresholds.append(0.0)

    def score(reps, layer):
        return (reps[layer] - centers[layer]) @ directions[layer]

    return score, thresholds


def evaluate(score, threshold, layer, emotion_reps, neutral_reps):

    emotion_scores = score(emotion_reps, layer)
    neutral_scores = score(neutral_reps, layer)

    correct = (
        (emotion_scores > threshold).sum().item()
        + (neutral_scores <= threshold).sum().item()
    )

    n = len(emotion_scores) + len(neutral_scores)

    labels = [1] * len(emotion_scores) + [0] * len(neutral_scores)
    scores = torch.cat([emotion_scores, neutral_scores]).tolist()

    # separation in units of pooled standard deviation, used to break ties
    pooled_std = torch.cat([emotion_scores, neutral_scores]).std().item()
    separation = (emotion_scores.mean() - neutral_scores.mean()).item() / max(pooled_std, 1e-8)

    return {
        "correct": correct,
        "n": n,
        "accuracy": correct / n,
        "auc": roc_auc_score(labels, scores),
        "separation": separation
    }


def ci95(correct, n):
    # normal approximation to the binomial
    p = correct / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


results = {}

for method in ["random_pairs", "paired"]:

    print("\n")
    print("=" * 92)
    print("Method:", "random-pair LAT (RepE paper's method)" if method == "random_pairs" else "paired LAT (our variant)")
    print("=" * 92)

    print(
        f"{'Emotion':<10}{'Block':<7}{'Mery test':<11}{'Aia (val)':<11}"
        f"{'Test chars acc [95% CI]':<27}{'Test chars AUC':<16}{'Paper (Mery)':<12}"
    )
    print("-" * 92)

    results[method] = {}

    for emotion in emotions:

        score, thresholds = make_scorer(method, emotion)
        emotion_test = torch.load(f"{emotion}_test_reps_all_layers.pt")

        # Mery's own held-out images, for reference only
        mery = [
            evaluate(score, thresholds[layer], layer, emotion_test, neutral_test)
            for layer in range(12)
        ]

        # 1. choose the block on the validation character
        val = [
            evaluate(score, thresholds[layer], layer,
                     load_cross(validation_character, emotion), load_cross(validation_character, "neutral"))
            for layer in range(12)
        ]

        block = max(range(12), key=lambda l: (val[l]["accuracy"], val[l]["separation"]))

        # 2. evaluate every block on the test characters (for the per-block table),
        #    and report the chosen one
        per_block = []

        for layer in range(12):

            per_character = {
                c: evaluate(score, thresholds[layer], layer, load_cross(c, emotion), load_cross(c, "neutral"))
                for c in characters
            }

            correct = sum(r["correct"] for r in per_character.values())
            n = sum(r["n"] for r in per_character.values())

            per_block.append({
                "accuracy": correct / n,
                "correct": correct,
                "n": n,
                "mean_auc": sum(r["auc"] for r in per_character.values()) / len(characters),
                "per_character": {c: {"accuracy": r["accuracy"], "auc": r["auc"]} for c, r in per_character.items()}
            })

        chosen = per_block[block]
        low, high = ci95(chosen["correct"], chosen["n"])

        accuracy_with_ci = f"{chosen['accuracy']:.3f} [{low:.3f}, {high:.3f}]"

        print(
            f"{emotion:<10}{block:<7}{mery[block]['accuracy']:<11.3f}{val[block]['accuracy']:<11.3f}"
            f"{accuracy_with_ci:<27}{chosen['mean_auc']:<16.3f}{paper_results[emotion]:<12.3f}"
        )

        results[method][emotion] = {
            "chosen_block": block,
            "mery_test_accuracy": mery[block]["accuracy"],
            "validation_accuracy": val[block]["accuracy"],
            "cross_character_accuracy": chosen["accuracy"],
            "cross_character_accuracy_ci95": [low, high],
            "cross_character_mean_auc": chosen["mean_auc"],
            "per_character": chosen["per_character"],
            "per_block_cross_character_accuracy": [b["accuracy"] for b in per_block],
            "per_block_cross_character_auc": [b["mean_auc"] for b in per_block],
            "paper_accuracy": paper_results[emotion]
        }

    print("\nTest-character accuracy by block (all 6 emotions averaged):")
    print("  " + "  ".join(
        f"{layer}:{sum(results[method][e]['per_block_cross_character_accuracy'][layer] for e in emotions) / 6:.2f}"
        for layer in range(12)
    ))


with open("results_cross_character.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results_cross_character.json")
