"""Layer sweep: trains and evaluates one RR checkpoint per (target layer, token mode), then
aggregates the eval JSONs into a summary table, summary JSON and plot.

Resumable: a configuration whose checkpoint / eval JSON already exists in --out-dir is
skipped, so an interrupted sweep can just be re-run. model_orig's adversarial images are
cached once (--orig-cache) and reused by every evaluation.

Extra arguments for train_rr.py / evaluate.py are passed through verbatim, e.g.
    python run_layer_sweep.py --data-root ../../PaperThree/data \
        --train-args "--total-steps 300" --eval-args "--num-eval-images 128"
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="3,5,8,11")
    p.add_argument("--rep-tokens", default="cls,all")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--capability-data-root", default="./data")
    p.add_argument("--out-dir", default="runs/layer_sweep")
    p.add_argument("--cache-dir", default=os.path.join(tempfile.gettempdir(), "clip_rr_cache"),
                    help="multi-GB caches (training sets, model_orig adversarial images); kept out of "
                         "the project folder by default since it's synced")
    # training budget for the sweep: smaller than train_rr.py's defaults. The paper trains RR for
    # 150 steps x batch 16 (~2.4k harmful samples); 200 x 32 is still ~2.7x that. If a config's
    # training log shows 1-adv_cos never growing, delete its checkpoint and re-run with
    # --train-args "--total-steps 500" (the sweep resumes and redoes only that config).
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--train-retain-batch", type=int, default=16)
    p.add_argument("--cb-pgd-steps", type=int, default=10,
                    help="PGD steps for building the circuit-breaker set; only successful attacks are kept")
    # without these, 5 of 8 configs in the first sweep never left the L_s saddle point (L_r pulled the
    # LoRA back to a no-op first); see train_rr.py --ls-warmup-frac. Set to 0 / 1e-3 for the paper's schedule.
    p.add_argument("--train-ls-warmup", type=float, default=0.2)
    p.add_argument("--train-b-init", type=float, default=1e-2)
    # evaluation budget for the sweep: cheaper than evaluate.py's defaults so all 8 configs fit
    # in about an hour on a laptop GPU. Re-run evaluate.py at full budget on the configs that matter.
    p.add_argument("--eval-images", type=int, default=128)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--eps-list", default="2,8")
    p.add_argument("--clean-images", type=int, default=2000)
    p.add_argument("--train-args", default="", help="extra arguments passed to train_rr.py")
    p.add_argument("--eval-args", default="", help="extra arguments passed to evaluate.py")
    return p.parse_args()


def run(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    token_modes = args.rep_tokens.split(",")
    os.makedirs(args.cache_dir, exist_ok=True)
    orig_cache = os.path.join(args.cache_dir, "orig_adv_cache.pt")

    for layer in layers:
        for tokens in token_modes:
            ckpt = os.path.join(out_dir, f"rr_L{layer}_{tokens}.pt")
            eval_json = os.path.join(out_dir, f"rr_L{layer}_{tokens}_eval.json")
            if not os.path.exists(ckpt):
                run([sys.executable, "train_rr.py", "--target-layer", str(layer), "--rep-tokens", tokens,
                     "--data-root", args.data_root, "--cache-dir", args.cache_dir, "--out", ckpt,
                     "--total-steps", str(args.train_steps), "--batch-size", str(args.train_batch),
                     "--retain-batch-size", str(args.train_retain_batch), "--pgd-steps", str(args.cb_pgd_steps),
                     "--ls-warmup-frac", str(args.train_ls_warmup), "--lora-b-init-std", str(args.train_b_init)]
                    + shlex.split(args.train_args))
            if not os.path.exists(eval_json):
                run([sys.executable, "evaluate.py", "--checkpoint", ckpt, "--data-root", args.data_root,
                     "--capability-data-root", args.capability_data_root, "--orig-cache", orig_cache,
                     "--num-eval-images", str(args.eval_images), "--pgd-steps", str(args.eval_steps),
                     "--eps-list", args.eps_list, "--num-clean-images", str(args.clean_images),
                     "--out-json", eval_json] + shlex.split(args.eval_args))

    rows = []
    for layer in layers:
        for tokens in token_modes:
            with open(os.path.join(out_dir, f"rr_L{layer}_{tokens}_eval.json")) as f:
                rows.append(json.load(f))

    eps_list = [a["eps_255"] for a in rows[0]["attacks"]]
    datasets = list(rows[0]["clean_accuracy"])
    header = f"{'layer':>5} {'tokens':>6} " + "".join(f"{'clean ' + d:>16}" for d in datasets) + \
        "".join(f"{f'ASR@{e:g} orig/cb/xfer':>26}" for e in eps_list) + \
        "".join(f"{f'rob@{e:g} orig/cb':>18}" for e in eps_list)
    print("\n" + header)
    summary = []
    for r in rows:
        line = f"{r['target_layer']:>5} {r['rep_tokens']:>6} "
        for d in datasets:
            acc = r["clean_accuracy"][d]
            line += f"{acc['model_cb']:.3f}({acc['model_cb'] - acc['model_orig']:+.3f})".rjust(16)
        line += "".join(f"{a['model_orig']['targeted_asr']:.2f}/{a['model_cb_adaptive']['targeted_asr']:.2f}/"
                        f"{a['model_cb_transfer']['targeted_asr']:.2f}{'!' if a['gradient_masking_suspected'] else ' '}"
                        .rjust(26) for a in r["attacks"])
        line += "".join(f"{a['model_orig']['robust_accuracy']:.2f}/{a['model_cb_adaptive']['robust_accuracy']:.2f}"
                        .rjust(18) for a in r["attacks"])
        print(line)
        summary.append({"target_layer": r["target_layer"], "rep_tokens": r["rep_tokens"],
                        "clean_accuracy": r["clean_accuracy"],
                        "clean_representation_drift": r["clean_representation_drift"], "attacks": r["attacks"]})
    print("('!' = transfer ASR > adaptive ASR, i.e. gradient masking suspected)")

    with open(os.path.join(out_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'sweep_summary.json')}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(eps_list) + 1, figsize=(4 * (len(eps_list) + 1), 3.6), squeeze=False)
        axes = axes[0]
        for tokens in token_modes:
            sub = sorted((r for r in rows if r["rep_tokens"] == tokens), key=lambda r: r["target_layer"])
            xs = [r["target_layer"] for r in sub]
            for ax, eps_i in zip(axes, range(len(eps_list))):
                ax.plot(xs, [r["attacks"][eps_i]["model_cb_adaptive"]["targeted_asr"] for r in sub],
                        marker="o", label=f"model_cb ({tokens})")
            axes[-1].plot(xs, [r["clean_accuracy"][datasets[0]]["model_cb"] for r in sub], marker="o",
                          label=f"{datasets[0]} ({tokens})")
            if len(datasets) > 1:
                axes[-1].plot(xs, [r["clean_accuracy"][datasets[1]]["model_cb"] for r in sub], marker="s",
                              linestyle="--", label=f"{datasets[1]} ({tokens})")
        for ax, eps_i, e in zip(axes, range(len(eps_list)), eps_list):
            ax.axhline(rows[0]["attacks"][eps_i]["model_orig"]["targeted_asr"], color="gray", linestyle=":",
                       label="model_orig")
            ax.set_title(f"adaptive targeted ASR, eps={e:g}/255")
            ax.set_xlabel("RR target layer")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=7)
        for i, d in enumerate(datasets):
            axes[-1].axhline(rows[0]["clean_accuracy"][d]["model_orig"], color="gray", linestyle=":" if i == 0 else "-.",
                             label=f"model_orig {d}")
        axes[-1].set_title("clean zero-shot accuracy")
        axes[-1].set_xlabel("RR target layer")
        axes[-1].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sweep_summary.png"))
        print(f"saved {os.path.join(out_dir, 'sweep_summary.png')}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
