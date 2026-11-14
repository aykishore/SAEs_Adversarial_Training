"""Layer-selection probe for where to apply the RR loss.

Why not divergence: RR has to make model_cb treat adversarial inputs differently from
clean ones, so it needs a layer where adversarial inputs are *recognizable* -- not merely
where an attack moves the representation the most. For a targeted attack these two
pull in opposite directions: the attack's goal is to make a cat image represent "dog",
so at layers where it succeeds, rep(x_adv) looks like a genuine dog image, and rerouting it
directly fights L_r on real dogs.

For every vision-transformer block, on the frozen model, this measures (for both the CLS
token and the mean of the 49 patch tokens):

  divergence        mean cos_sim(rep(x_clean), rep(x_adv)) -- the old criterion, for reference
  separability      held-out ROC-AUC of a linear probe (L2-regularized logistic regression)
                    for clean vs. adversarial; train/test split is by image, so a clean
                    image and its own adversarial twin never land on opposite sides
  target-likeness   fraction of adversarial reps whose nearest class centroid (built from
                    separate, genuine clean images) is the attacker's target class
  centroid acc      same nearest-centroid rule on clean reps vs. their true label, to show
                    how class-informative the layer is at all

A good RR layer has high separability and low target-likeness. The printed candidate
(argmax of AUC - target-likeness) is a heuristic starting point for the layer sweep in
run_layer_sweep.py, not a final answer. Only successful attacks on correctly classified
images are used, matching train_rr.py's circuit-breaker set."""
import argparse
import json

import torch
import torch.nn.functional as F

from attacks import pgd_targeted
from common import AllLayerCapture, NormalizingCLIPWrapper, build_zero_shot_classifier, load_clip, \
    sample_wrong_targets
from data import ZERO_SHOT_TEMPLATE, build_dataset

KINDS = ("cls", "patch_mean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="ViT-B-32")
    p.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--classnames-json", default=None)
    p.add_argument("--num-candidates", type=int, default=512,
                    help="images to attack; only clean-correct + attack-successful pairs are kept")
    p.add_argument("--centroid-images-per-class", type=int, default=10)
    p.add_argument("--attack-batch-size", type=int, default=32)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=1 / 255)
    p.add_argument("--pgd-steps", type=int, default=50)
    p.add_argument("--probe-train-frac", type=float, default=0.5)
    p.add_argument("--probe-steps", type=int, default=300)
    p.add_argument("--probe-l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-prefix", default="layer_separability")
    return p.parse_args()


@torch.no_grad()
def layer_features(wrapped, capture, num_layers, images, batch_size, device):
    """Returns {kind: [num_layers, N, D]} on CPU for the CLS token and the patch-token mean."""
    feats = {kind: [[] for _ in range(num_layers)] for kind in KINDS}
    for i in range(0, images.shape[0], batch_size):
        wrapped.encode_image(images[i:i + batch_size].to(device))
        for layer in range(num_layers):
            h = capture.get(layer).float()
            feats["cls"][layer].append(h[:, 0, :].cpu())
            feats["patch_mean"][layer].append(h[:, 1:, :].mean(dim=1).cpu())
    return {kind: torch.stack([torch.cat(chunks) for chunks in per_layer]) for kind, per_layer in feats.items()}


def roc_auc(scores_pos, scores_neg):
    diff = scores_pos[:, None] - scores_neg[None, :]
    return ((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item()


def linear_probe_auc(clean, adv, train_idx, test_idx, steps, l2, seed):
    x_tr = torch.cat([clean[train_idx], adv[train_idx]])
    y_tr = torch.cat([torch.zeros(len(train_idx)), torch.ones(len(train_idx))])
    mean, std = x_tr.mean(0), x_tr.std(0).clamp(min=1e-6)
    x_tr = (x_tr - mean) / std

    torch.manual_seed(seed)
    w = torch.zeros(x_tr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(x_tr @ w + b, y_tr) + l2 * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        score = lambda x: ((x - mean) / std) @ w + b
        return roc_auc(score(adv[test_idx]), score(clean[test_idx]))


def nearest_centroid(reps, centroids):
    return (F.normalize(reps, dim=-1) @ centroids.T).argmax(dim=-1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    model, resize_crop, tokenizer = load_clip(args.model_name, args.pretrained, device)
    wrapped = NormalizingCLIPWrapper(model, device).to(device)
    num_layers = len(model.visual.transformer.resblocks)

    dataset = build_dataset(args.dataset, args.data_root, True, resize_crop, args.classnames_json)
    classnames = dataset.classnames
    num_classes = len(classnames)
    text_embeds = build_zero_shot_classifier(model, tokenizer, classnames, ZERO_SHOT_TEMPLATE, device)

    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    candidate_idx = order[:args.num_candidates]

    # genuine clean images for class centroids, disjoint from the attacked candidates
    per_class = {c: [] for c in range(num_classes)}
    for i in order[args.num_candidates:]:
        label = dataset.base.targets[i] if hasattr(dataset.base, "targets") else dataset[i][1]
        if len(per_class[label]) < args.centroid_images_per_class:
            per_class[label].append(i)
        if all(len(v) == args.centroid_images_per_class for v in per_class.values()):
            break
    centroid_idx = [i for idxs in per_class.values() for i in idxs]
    centroid_labels = torch.tensor([dataset[i][1] for i in centroid_idx])
    centroid_images = torch.stack([dataset[i][0] for i in centroid_idx])

    print(f"Attacking {len(candidate_idx)} candidates (eps={args.eps * 255:g}/255, steps={args.pgd_steps})...")
    clean_chunks, adv_chunks, label_chunks, target_chunks = [], [], [], []
    n_correct = 0
    for start in range(0, len(candidate_idx), args.attack_batch_size):
        idx = candidate_idx[start:start + args.attack_batch_size]
        images = torch.stack([dataset[i][0] for i in idx]).to(device)
        labels = torch.tensor([dataset[i][1] for i in idx], device=device)
        with torch.no_grad():
            correct = (wrapped.encode_image(images) @ text_embeds.T).argmax(dim=-1) == labels
        images, labels = images[correct], labels[correct]
        n_correct += int(correct.sum())
        if images.shape[0] == 0:
            continue
        targets = sample_wrong_targets(labels, num_classes, args.seed + start).to(device)
        adv = pgd_targeted(wrapped, images, text_embeds, targets, wrapped.logit_scale,
                           args.eps, args.pgd_alpha, args.pgd_steps)
        with torch.no_grad():
            success = (wrapped.encode_image(adv) @ text_embeds.T).argmax(dim=-1) == targets
        clean_chunks.append(images[success].cpu())
        adv_chunks.append(adv[success].cpu())
        label_chunks.append(labels[success].cpu())
        target_chunks.append(targets[success].cpu())
    x_clean, x_adv = torch.cat(clean_chunks), torch.cat(adv_chunks)
    labels, targets = torch.cat(label_chunks), torch.cat(target_chunks)
    n = x_clean.shape[0]
    print(f"{n_correct}/{len(candidate_idx)} clean-correct, {n} successful attacks kept")

    capture = AllLayerCapture(model)
    f_clean = layer_features(wrapped, capture, num_layers, x_clean, args.attack_batch_size, device)
    f_adv = layer_features(wrapped, capture, num_layers, x_adv, args.attack_batch_size, device)
    f_cent = layer_features(wrapped, capture, num_layers, centroid_images, args.attack_batch_size, device)
    capture.remove()

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(args.probe_train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    results = {"args": vars(args), "num_pairs": n, "num_probe_test_pairs": len(test_idx), "layers": []}
    for layer in range(num_layers):
        row = {"layer": layer}
        for kind in KINDS:
            clean, adv, cent = f_clean[kind][layer], f_adv[kind][layer], f_cent[kind][layer]
            centroids = torch.stack([F.normalize(cent[centroid_labels == c], dim=-1).mean(0)
                                     for c in range(num_classes)])
            centroids = F.normalize(centroids, dim=-1)
            row[kind] = {
                "divergence_cos": F.cosine_similarity(clean, adv, dim=-1).mean().item(),
                "probe_auc": linear_probe_auc(clean, adv, train_idx, test_idx, args.probe_steps, args.probe_l2,
                                              args.seed),
                "target_likeness": (nearest_centroid(adv, centroids) == targets).float().mean().item(),
                "clean_centroid_acc": (nearest_centroid(clean, centroids) == labels).float().mean().item(),
            }
        results["layers"].append(row)

    print(f"\n{'':6s}" + "".join(f"{kind + ': cos / AUC / tgt-like / cent-acc':>46s}" for kind in KINDS))
    for row in results["layers"]:
        line = f"{row['layer']:4d}  "
        for kind in KINDS:
            m = row[kind]
            line += (f"{m['divergence_cos']:>16.3f} {m['probe_auc']:>7.3f} {m['target_likeness']:>10.3f} "
                     f"{m['clean_centroid_acc']:>10.3f}")
        print(line)

    results["candidate_layer"] = {}
    for kind in KINDS:
        best = max(results["layers"], key=lambda r: r[kind]["probe_auc"] - r[kind]["target_likeness"])
        results["candidate_layer"][kind] = best["layer"]
        print(f"candidate layer for {kind} (argmax AUC - target-likeness): {best['layer']}")
    print("This is a starting point for run_layer_sweep.py, not a definitive choice.")

    with open(args.out_prefix + ".json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out_prefix}.json")

    try:
        import matplotlib.pyplot as plt
        layers = list(range(num_layers))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        panels = [("divergence_cos", "cos_sim(clean, adv)  [old criterion]"),
                  ("probe_auc", "held-out linear-probe AUC (clean vs adv)"),
                  ("target_likeness", "adv nearest centroid = target class")]
        for ax, (metric, title) in zip(axes, panels):
            for kind in KINDS:
                ax.plot(layers, [r[kind][metric] for r in results["layers"]], marker="o", label=kind)
            if metric == "target_likeness":
                for kind in KINDS:
                    ax.plot(layers, [r[kind]["clean_centroid_acc"] for r in results["layers"]],
                            linestyle=":", label=f"{kind} clean centroid acc")
            ax.set_title(title)
            ax.set_xlabel("vision-transformer block")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_prefix + ".png")
        print(f"Saved {args.out_prefix}.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
