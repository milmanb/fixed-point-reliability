"""Figures for one trained model, from the per-image table written by evaluate_models.py.

signal_vs_error_<model>.png
    Each reference-free signal against the true error, one column per corruption family.
    The family is highlighted over all other conditions in gray, so each panel shows whether
    the signal separates that family's errors from the rest.
failures_<model>.png
    Random examples (fixed seed) among images whose error is in the top 10% over all
    conditions while g is below its median: outputs that are stable but wrong.

    python scripts/plot_models.py --model dae_lam0_seed0
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from fpr.data import REPO_ROOT, load_fashion_mnist  # noqa: E402
from fpr.evaluation import CONDITIONS, observe  # noqa: E402
from fpr.models import load_restorer  # noqa: E402
from fpr.plotting import INK, INK_SECONDARY, SEQUENTIAL, family_scatter  # noqa: E402

SIGNAL_LABELS = {"g": "g  idempotence residual", "d": "d  displacement", "r_A": "r_A  measurement residual"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dae_lam0_seed0")
    parser.add_argument("--results", default="results/models", help="relative to the repository")
    parser.add_argument("--checkpoints", default="checkpoints", help="relative to the repository")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rho(a, b):
    return spearmanr(a, b).statistic


def select_failures(frame, count, seed):
    """Random rows with error in the top 10% and g below the median, both over all conditions."""
    frame = frame.assign(e_pct=frame["e"].rank(pct=True), g_pct=frame["g"].rank(pct=True))
    candidates = frame[(frame["e_pct"] >= 0.9) & (frame["g_pct"] <= 0.5)]
    return candidates.sample(n=min(count, len(candidates)), random_state=seed), candidates, len(frame)


def plot_failures(rows, restorer, x, seed, path, model, candidates, n_total):
    conditions = {c.label: c for c in CONDITIONS}
    error_cmap = LinearSegmentedColormap.from_list("error", SEQUENTIAL)
    fig, axes = plt.subplots(4, len(rows), figsize=(1.75 * len(rows), 7.3), squeeze=False)
    row_names = ["clean x", "observed y", "output f(y)", "|f(y) - x|"]
    observed = {}
    for j, (_, row) in enumerate(rows.iterrows()):
        index = int(row["image"])
        # The corruption of image i is the i-th slice of a draw for the whole evaluation batch.
        if row["condition"] not in observed:
            observed[row["condition"]] = observe(conditions[row["condition"]], x, seed)[0]
        y = observed[row["condition"]][index:index + 1]
        with torch.no_grad():
            fy = restorer(y)
        error = (fy - x[index:index + 1]).abs().mean().item()
        if not np.isclose(error, row["e"], rtol=1e-3):
            raise RuntimeError(f"recomputed e={error:.6f} differs from the table ({row['e']:.6f})")
        panels = [
            (x[index, 0], "gray", ""),
            (y[0, 0], "gray", row["condition"]),
            (fy[0, 0], "gray", f"e = {row['e']:.3f} (p{100 * row['e_pct']:.0f})\n"
                               f"g = {row['g']:.4f} (p{100 * row['g_pct']:.0f})"),
            ((fy - x[index:index + 1])[0, 0].abs(), error_cmap, ""),
        ]
        for i, (image, cmap, caption) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(np.clip(image.numpy(), 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(caption, fontsize=6.5, color=INK_SECONDARY, labelpad=2)
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=8, color=INK)
    shares = candidates["condition"].value_counts(normalize=True).head(3)
    composition = ", ".join(f"{share:.0%} {condition}" for condition, share in shares.items())
    fig.suptitle(f"{model}: stable but wrong\n"
                 f"Random picks among the {len(candidates):,} of {n_total:,} images with error in the top 10% "
                 f"and g below its median.\nCandidates: {composition}. "
                 f"p = percentile over all conditions.",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    results = REPO_ROOT / args.results
    run_info = json.loads((results / "run_info.json").read_text())
    per_image = pd.read_csv(results / "per_image.csv.gz")
    frame = per_image[per_image["model"] == args.model].reset_index(drop=True)
    if frame.empty:
        raise SystemExit(f"model {args.model!r} not found in {results / 'per_image.csv.gz'}")
    figures = results / "figures"
    figures.mkdir(exist_ok=True)

    family_scatter([(label, frame, signal) for signal, label in SIGNAL_LABELS.items()],
                   figures / f"signal_vs_error_{args.model}.png",
                   f"{args.model}: signal vs. true error per image; family in blue, all conditions in gray",
                   seed=args.seed)

    eval_seed = int(run_info["args"]["seed"])
    x_test, _ = load_fashion_mnist("test", dtype=torch.float64)
    x = x_test[:int(run_info["args"]["n_eval"])]
    restorer, _ = load_restorer(REPO_ROOT / args.checkpoints / f"{args.model}.pt", device="cpu")
    rows, candidates, n_total = select_failures(frame, args.examples, args.seed)
    print("Selected failures by condition:", rows["condition"].value_counts().to_dict())
    plot_failures(rows, restorer, x, eval_seed, figures / f"failures_{args.model}.png", args.model,
                  candidates, n_total)
    print(f"Wrote figures to {figures}")


if __name__ == "__main__":
    main()
