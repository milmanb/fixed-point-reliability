"""Preliminary smoke check (proposal, Sec. 5): reference-free signals on exact projectors.

Claim to reproduce: for exact radial and PCA projectors, g(y) ~ 0 while e(y) > 0.

For an exact projector this holds by construction (P(P(y)) = P(y) for all y), so
the check validates the pipeline rather than the hypothesis. The script also runs
the evaluation planned for the learned models (Spearman rho and worst-quartile
AUROC with image-level bootstrap CIs) on the same projectors, and adds two
references: the identity (loosest projector) and the nearest training neighbour
(a nonlinear projector whose outputs are always real training images).

    python scripts/smoke_projectors.py                              # full run
    python scripts/smoke_projectors.py --n-eval 2000 --n-boot 200   # quick run
"""

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from fpr.data import REPO_ROOT, load_fashion_mnist  # noqa: E402
from fpr.evaluation import CONDITIONS, observe  # noqa: E402
from fpr.metrics import rank_metrics  # noqa: E402
from fpr.plotting import INK, INK_SECONDARY, NA_FILL, SERIES, style_axes  # noqa: E402
from fpr.projectors import PCA, Identity, NearestNeighbor, Radial, principal_components  # noqa: E402
from fpr.signals import SIGNALS, compute_signals  # noqa: E402

EXAMPLE_CONDITIONS = ["noise(sigma=0.3)", "blur(std=1.5)", "pixel_mask(drop=0.5)", "box_mask(size=14)"]
SCATTER_CONDITIONS = ["noise(sigma=0.3)", "box_mask(size=14)"]

# Validated categorical slots 1-3; diverging blue <-> red around a gray midpoint.
SIGNAL_COLORS = dict(zip(SIGNALS, SERIES))
DIVERGING = LinearSegmentedColormap.from_list("rho", ["#e34948", NA_FILL, SERIES[0]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-eval", type=int, default=10_000, help="number of test images")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-k", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--example-index", type=int, default=0, help="test image shown in the example grid")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "smoke_projectors")
    parser.add_argument("--no-per-image", action="store_true", help="skip the per-image CSV dump")
    return parser.parse_args()


def build_projectors(x_train, pca_ks):
    mean, eigvecs, eigvals = principal_components(x_train)
    explained = (eigvals.cumsum(0) / eigvals.sum()).numpy()
    projectors = [Identity(), Radial.fit(x_train)]
    projectors += [PCA.from_components(mean, eigvecs, k) for k in pca_ks]
    projectors.append(NearestNeighbor(x_train))
    info = {
        "radial_radius": projectors[1].radius,
        "pca_explained_variance": {k: float(explained[k - 1]) for k in pca_ks},
    }
    return projectors, info


def run_conditions(projectors, x, labels, example_index, seed):
    frames, examples = [], {}
    for condition in CONDITIONS:
        y, A = observe(condition, x, seed)
        if condition.label in EXAMPLE_CONDITIONS:
            examples[condition.label] = {"x": x[example_index, 0].numpy(),
                                         "y": y[example_index, 0].numpy(), "outputs": {}}
        for projector in projectors:
            start = time.perf_counter()
            signals = compute_signals(projector, x, y, A)
            frame = pd.DataFrame(signals)
            frame.insert(0, "image", np.arange(len(x)))
            frame.insert(1, "class", labels.numpy())
            frame.insert(0, "projector", projector.name)
            frame.insert(0, "level", condition.level)
            frame.insert(0, "family", condition.family)
            frame.insert(0, "condition", condition.label)
            frames.append(frame)
            if condition.label in examples:
                fy = projector(y[example_index:example_index + 1])[0, 0].numpy()
                examples[condition.label]["outputs"][projector.name] = (
                    fy, {k: float(v[example_index]) for k, v in signals.items()})
            print(f"  {condition.label:<22} {projector.name:<9} "
                  f"max g={signals['g'].max():.1e}  mean e={signals['e'].mean():.4f}  "
                  f"({time.perf_counter() - start:.1f}s)", flush=True)
    return pd.concat(frames, ignore_index=True), examples


def evaluate(per_image, n_boot, seed):
    def score(frame, scope, group, clusters=None):
        rows = rank_metrics({s: frame[s].to_numpy() for s in SIGNALS}, frame["e"].to_numpy(),
                            clusters=clusters, n_boot=n_boot, seed=seed)
        return [{"scope": scope, "group": group, "projector": frame["projector"].iat[0], **row}
                for row in rows]

    rows = []
    for (condition, _), frame in per_image.groupby(["condition", "projector"], sort=False):
        rows += score(frame, "condition", condition)
    # Pooled over severities: the same test image appears once per level, so resample images.
    for (family, _), frame in per_image.groupby(["family", "projector"], sort=False):
        rows += score(frame, "family", family, clusters=frame["image"].to_numpy())
    for _, frame in per_image.groupby("projector", sort=False):
        rows += score(frame, "all", "all", clusters=frame["image"].to_numpy())
    return pd.DataFrame(rows)


def summarize(per_image):
    grouped = per_image.groupby(["condition", "projector"], sort=False)
    summary = grouped.agg(g_mean=("g", "mean"), g_max=("g", "max"), d_mean=("d", "mean"),
                          r_A_mean=("r_A", "mean"), e_mean=("e", "mean"), e_min=("e", "min"),
                          e_in_mean=("e_in", "mean"))
    summary["frac_improved"] = grouped.apply(lambda f: float((f["e"] < f["e_in"]).mean()),
                                             include_groups=False)
    return summary.reset_index()


def plot_examples(examples, projector_names, path):
    shown = [p for p in projector_names if p != "identity"]  # identity output equals y
    rows = ["clean x", "observed y", *shown]
    fig, axes = plt.subplots(len(rows), len(examples), figsize=(1.75 * len(examples), 1.62 * len(rows)))
    for j, (label, example) in enumerate(examples.items()):
        e_in = next(iter(example["outputs"].values()))[1]["e_in"]
        panels = [(example["x"], ""), (example["y"], f"e_in={e_in:.3f}")]
        for name in shown:
            fy, s = example["outputs"][name]
            g_text = "0" if s["g"] == 0 else f"{s['g']:.0e}"
            panels.append((fy, f"e={s['e']:.3f}  d={s['d']:.3f}\nr_A={s['r_A']:.3f}  g={g_text}"))
        for i, (image, caption) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(np.clip(image, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(caption, fontsize=6.5, color=INK_SECONDARY, labelpad=2)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=8, color=INK)
            if i == 0:
                ax.set_title(label, fontsize=8, color=INK)
    fig.suptitle("Exact projectors on one test image: g is zero at every output, e is not",
                 fontsize=9, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scatter(per_image, projector_names, path, max_points=2000, seed=0):
    fig, axes = plt.subplots(len(SCATTER_CONDITIONS), len(projector_names),
                             figsize=(2.3 * len(projector_names), 2.5 * len(SCATTER_CONDITIONS)),
                             squeeze=False)
    for i, condition in enumerate(SCATTER_CONDITIONS):
        signals = ("g", "d") if condition.startswith("noise") else SIGNALS  # r_A = d when A = I
        for j, name in enumerate(projector_names):
            ax = axes[i, j]
            style_axes(ax)
            frame = per_image[(per_image["condition"] == condition) & (per_image["projector"] == name)]
            frame = frame.sample(min(max_points, len(frame)), random_state=seed)
            for signal in signals:
                ax.scatter(frame["e"], frame[signal], s=4, color=SIGNAL_COLORS[signal], alpha=0.45,
                           linewidths=0, label=signal, rasterized=True)
            ax.set_ylim(bottom=-0.01)
            ax.set_title(name if i == 0 else "", fontsize=9, color=INK)
            ax.set_xlabel("true error e" if i == len(SCATTER_CONDITIONS) - 1 else "", fontsize=8,
                          color=INK_SECONDARY)
            if j == 0:
                ax.set_ylabel(f"{condition}\nsignal value", fontsize=8, color=INK_SECONDARY)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=SIGNAL_COLORS[s])
               for s in SIGNALS]
    fig.legend(handles, ["g  idempotence residual", "d  displacement", "r_A  measurement residual"],
               loc="upper right", bbox_to_anchor=(1, 1.0), ncol=3, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.suptitle("Signal vs. true error per image (for noise, r_A = d)", fontsize=9, color=INK,
                 x=0.01, y=0.985, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spearman_heatmap(metrics, projector_names, path):
    by_condition = metrics[metrics["scope"] == "condition"]
    columns = [c.label for c in CONDITIONS]
    cmap = DIVERGING.copy()
    cmap.set_bad(NA_FILL)
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(11, 1.1 + 0.33 * len(projector_names) * len(SIGNALS)),
                             sharex=True)
    for ax, signal in zip(axes, SIGNALS):
        table = (by_condition[by_condition["signal"] == signal]
                 .pivot(index="projector", columns="group", values="spearman")
                 .reindex(index=projector_names, columns=columns))
        values = table.to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                v = values[r, c]
                text = "n/a" if np.isnan(v) else f"{v:.2f}"
                color = "#ffffff" if np.isfinite(v) and abs(v) > 0.65 else INK_SECONDARY
                ax.text(c, r, text, ha="center", va="center", fontsize=6.5, color=color)
        ax.set_yticks(range(len(projector_names)), projector_names, fontsize=8, color=INK)
        ax.set_title(f"Spearman rho({signal}, e) within each corruption level", loc="left",
                     fontsize=9, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xticks(range(len(columns)), columns, rotation=35, ha="right", fontsize=8, color=INK)
    bar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=INK_SECONDARY, length=0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_report(summary, metrics, projector_names):
    print("\nSmoke check: maximum idempotence residual g vs. true error e over all conditions")
    print(f"  {'projector':<10}{'max g':>12}{'min mean e':>13}{'max mean e':>13}")
    for name in projector_names:
        rows = summary[summary["projector"] == name]
        print(f"  {name:<10}{rows['g_max'].max():>12.1e}{rows['e_mean'].min():>13.4f}"
              f"{rows['e_mean'].max():>13.4f}")

    print("\nSpearman rho(signal, e): median over the 13 conditions | pooled over all conditions")
    header = "".join(f"{s:>22}" for s in SIGNALS)
    print(f"  {'projector':<10}{header}")
    for name in projector_names:
        cells = []
        for signal in SIGNALS:
            per_condition = metrics[(metrics["scope"] == "condition") & (metrics["projector"] == name)
                                    & (metrics["signal"] == signal)]["spearman"]
            pooled = metrics[(metrics["scope"] == "all") & (metrics["projector"] == name)
                             & (metrics["signal"] == signal)]["spearman"].iat[0]
            median = per_condition.median() if per_condition.notna().any() else float("nan")
            cells.append(f"{median:>+10.2f} | {pooled:>+7.2f}   ")
        print(f"  {name:<10}{''.join(cells)}")


def _repo_relative(path):
    """Path relative to the repository when possible, so committed logs carry no local paths."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()

    x_train, _ = load_fashion_mnist("train", dtype=torch.float64)
    x_test, y_test = load_fashion_mnist("test", dtype=torch.float64)
    x, labels = x_test[:args.n_eval], y_test[:args.n_eval]
    projectors, fit_info = build_projectors(x_train, args.pca_k)
    names = [p.name for p in projectors]
    print(f"Evaluating {len(x)} test images, {len(CONDITIONS)} conditions, projectors: {names}")
    print(f"PCA explained variance: {fit_info['pca_explained_variance']}")

    per_image, examples = run_conditions(projectors, x, labels, args.example_index, args.seed)
    t_signals = time.perf_counter()
    metrics = evaluate(per_image, args.n_boot, args.seed)
    t_metrics = time.perf_counter()
    summary = summarize(per_image)

    summary.to_csv(args.out / "summary.csv", index=False, float_format="%.6g")
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.4f")
    if not args.no_per_image:
        per_image.to_csv(args.out / "per_image.csv.gz", index=False, float_format="%.6g")
    plot_examples(examples, names, args.out / "examples.png")
    plot_scatter(per_image, names, args.out / "scatter.png", seed=args.seed)
    plot_spearman_heatmap(metrics, names, args.out / "spearman_heatmap.png")

    arguments = vars(args) | {"out": _repo_relative(args.out)}
    run_info = {
        "args": arguments,
        "conditions": [c.label for c in CONDITIONS],
        "projectors": names,
        **fit_info,
        "seconds": {"signals": round(t_signals - t0, 1), "metrics": round(t_metrics - t_signals, 1),
                    "total": round(time.perf_counter() - t0, 1)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "nn_device": str(projectors[-1].device),
    }
    (args.out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print_report(summary, metrics, names)
    print(f"\nWrote results to {args.out} in {run_info['seconds']['total']}s")


if __name__ == "__main__":
    main()
