"""Compare groups of trained autoencoders, from the outputs of evaluate_models.py and train_dae.py.

A group is a model name without its seed, e.g. dae_lam0 or dae_lam1w5. Tables are written
to results/models/comparison/ and the key numbers are printed.

Figures (results/models/comparison/):
  training_curves.png    validation error and g per epoch, for every training log
  rho_by_condition.png   within-level Spearman and partial rho given brightness, per signal
  g_vs_error.png         g vs. true error per corruption family, one row per group (first seed)

    python scripts/compare_models.py --groups dae_lam0 dae_lam1w5
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fpr.data import REPO_ROOT  # noqa: E402
from fpr.evaluation import CONDITIONS  # noqa: E402
from fpr.plotting import (AXIS, FAMILIES, GRID, INK, INK_SECONDARY, SERIES,  # noqa: E402
                          family_scatter, pretty_condition, style_axes)

SIGNALS = ("g", "d", "r_A")
SIGNAL_LABELS = {"g": "g  idempotence residual", "d": "d  displacement", "r_A": "r_A  measurement residual"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=None,
                        help="model groups to compare (default: all autoencoder groups, at most 3)")
    parser.add_argument("--scatter-groups", nargs="*", default=None,
                        help="groups shown in g_vs_error.png (default: all compared groups)")
    parser.add_argument("--scatter-families", nargs="*", default=list(FAMILIES),
                        help="corruption families shown in g_vs_error.png")
    parser.add_argument("--scatter-name", default="g_vs_error",
                        help="file name (without extension) for the scatter figure")
    parser.add_argument("--results", default="results/models", help="relative to the repository")
    parser.add_argument("--logs", default="results/train", help="relative to the repository")
    parser.add_argument("--out-dir", default="comparison",
                        help="output folder inside --results (use a second name for ablations)")
    return parser.parse_args()


def model_group(name):
    return name.rsplit("_seed", 1)[0]


def group_label(group):
    """dae_lam1w5_inner -> lambda_id = 1, warm-up 5 ep., inner only (as mathtext)."""
    match = re.fullmatch(r"(dae|unet)_lam([\d.]+)(?:w([\d.]+))?(?:_(inner|outer))?", group)
    if not match:
        return group
    architecture, weight, warmup, routing = match.groups()
    label = rf"$\lambda_{{id}} = {weight}$"
    if architecture == "unet":
        label += ", skips"
    if warmup:
        label += f", warm-up {warmup} ep."
    elif float(weight) > 0:
        label += ", no warm-up"
    if routing:
        label += f", {routing} only"
    return label


def with_groups(frame, groups):
    frame = frame.assign(model_group=frame["model"].map(model_group))
    return frame[frame["model_group"].isin(groups)]


def error_table(summary, groups):
    return (with_groups(summary, groups)
            .groupby(["condition", "model_group"], sort=False)
            .agg(e=("e", "mean"), e_min=("e", "min"), e_max=("e", "max"),
                 frac_improved=("frac_improved", "mean"), g=("g", "mean"))
            .reset_index())


def within_level_table(metrics, groups):
    m = with_groups(metrics[(metrics["scope"] == "condition") & (metrics["target"] == "e")], groups)
    return (m.groupby(["group", "model_group", "signal"], sort=False)
            .agg(spearman=("spearman", "mean"), spearman_min=("spearman", "min"),
                 spearman_max=("spearman", "max"), partial=("partial", "mean"),
                 partial_min=("partial", "min"), partial_max=("partial", "max"), auroc=("auroc", "mean"))
            .reset_index().rename(columns={"group": "condition"}))


def pooled_table(metrics, groups):
    m = with_groups(metrics[(metrics["scope"] != "condition") & (metrics["target"] == "e")], groups)
    return (m.groupby(["scope", "group", "model_group", "signal"], sort=False)
            [["spearman", "partial", "auroc"]].mean().reset_index())


def blind_spot_table(per_image, groups):
    """Worst error quartile over all conditions: share per family, and recall of each signal's top quartile."""
    rows = []
    for model, frame in with_groups(per_image, groups).groupby("model", sort=False):
        worst = frame["e"] >= frame["e"].quantile(0.75)
        # A constant signal (a collapsed model has g = 0 everywhere) has no top quartile:
        # thresholding it would flag every image and report a recall of 1.
        flagged = {s: (frame[s] >= frame[s].quantile(0.75) if np.ptp(frame[s]) > 1e-9
                       else pd.Series(np.nan, index=frame.index)) for s in SIGNALS}
        parts = [("all", worst)] + [(family, worst & (frame["family"] == family))
                                    for family in frame["family"].unique()]
        for family, mask in parts:
            rows.append({"model_group": model_group(model), "family": family,
                         "share_of_worst": mask.sum() / worst.sum(),
                         **{f"recall_{s}": flagged[s][mask].mean() for s in SIGNALS}})
    return pd.DataFrame(rows).groupby(["model_group", "family"], sort=False).mean().reset_index()


def plot_training_curves(log_dir, path, groups):
    histories = {p.stem: pd.read_csv(p) for p in sorted(log_dir.glob("*_seed*.csv"))
                 if model_group(p.stem) in groups}
    groups = list(dict.fromkeys(model_group(name) for name in histories))
    if len(groups) > len(SERIES):
        raise SystemExit(f"training_curves: {len(groups)} groups, but only {len(SERIES)} colours")
    colors = dict(zip(groups, SERIES))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    panels = (("val_e@0.2", "validation error e"), ("val_g@0.2", "validation g"))
    for ax, (column, title) in zip(axes, panels):
        style_axes(ax)
        seen = set()
        for name, history in histories.items():
            group = model_group(name)
            ax.plot(history["epoch"], history[column], color=colors[group], linewidth=1.6,
                    solid_capstyle="round", label=None if group in seen else group_label(group))
            seen.add(group)
        ax.set_title(f"{title} (noise $\\sigma$ = 0.2)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=8, color=INK_SECONDARY)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rho_by_condition(within, groups, path):
    conditions = [c.label for c in CONDITIONS]
    families = np.array([c.family for c in CONDITIONS])
    x = np.arange(len(conditions))
    offsets = np.linspace(-0.18, 0.18, len(groups)) if len(groups) > 1 else np.zeros(1)
    colors = dict(zip(groups, SERIES))
    stats = (("spearman", r"Spearman $\rho$(signal, e)"), ("partial", r"partial $\rho$ given brightness b"))
    shown = within[within["signal"].isin(SIGNALS)]
    bottom = min(-0.1, np.floor(np.nanmin(shown[["spearman_min", "partial_min"]].to_numpy()) * 10) / 10)

    fig, axes = plt.subplots(len(SIGNALS), 2, figsize=(11, 8.4), sharex=True, sharey=True)
    for i, signal in enumerate(SIGNALS):
        for j, (stat, title) in enumerate(stats):
            ax = axes[i, j]
            style_axes(ax, grid=False)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.axhline(0, color=AXIS, linewidth=0.8)
            for boundary in np.flatnonzero(families[1:] != families[:-1]):
                ax.axvline(boundary + 0.5, color=GRID, linewidth=0.8)
            for k, group in enumerate(groups):
                rows = (shown[(shown["signal"] == signal) & (shown["model_group"] == group)]
                        .set_index("condition").reindex(conditions))
                ax.vlines(x + offsets[k], rows[f"{stat}_min"], rows[f"{stat}_max"], color=colors[group],
                          linewidth=1.2)
                ax.scatter(x + offsets[k], rows[stat], s=26, color=colors[group], edgecolors="white",
                           linewidths=0.8, zorder=3, label=group_label(group))
            ax.set_ylim(bottom, 1.02)
            if i == 0:
                ax.set_title(title, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(SIGNAL_LABELS[signal], fontsize=8, color=INK_SECONDARY)
    for ax in axes[-1]:
        ax.set_xticks(x, [pretty_condition(c) for c in conditions], rotation=40, ha="right",
                      fontsize=7.5, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(groups), frameon=False, fontsize=8,
               labelcolor=INK)
    fig.suptitle("Ranking quality within each corruption level: mean over seeds; bars span the seeds",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(errors, within, pooled, blind, groups):
    pd.set_option("display.width", 200)
    order = [c.label for c in CONDITIONS]

    def show(title, table):
        print(f"\n{title}\n{table.to_string(float_format=lambda v: f'{v:.3f}')}")

    show("Mean L1 error e (mean over seeds)",
         errors.pivot(index="condition", columns="model_group", values="e").reindex(index=order, columns=groups))
    show("Mean idempotence residual g",
         errors.pivot(index="condition", columns="model_group", values="g").reindex(index=order, columns=groups))
    for stat in ("spearman", "partial"):
        table = (within[within["signal"].isin(SIGNALS)]
                 .groupby(["model_group", "signal"])[stat].median().unstack().reindex(index=groups))
        show(f"Within-level {stat}, median over the 13 levels", table[list(SIGNALS)])
    everything = pooled[(pooled["scope"] == "all")]
    for stat in ("spearman", "auroc"):
        table = everything.pivot(index="signal", columns="model_group", values=stat).reindex(columns=groups)
        keep = [s for s in (*SIGNALS, *(f"{s}@level" for s in SIGNALS), "b") if s in table.index]
        show(f"Pooled over all 13 levels: {stat}", table.loc[keep])
    show("Worst error quartile over all conditions: share per family and recall of each signal",
         blind.set_index(["model_group", "family"]).reindex(groups, level=0))


def main():
    args = parse_args()
    results = REPO_ROOT / args.results
    out = results / args.out_dir
    out.mkdir(exist_ok=True)
    metrics = pd.read_csv(results / "metrics.csv")
    summary = pd.read_csv(results / "summary.csv")
    per_image = pd.read_csv(results / "per_image.csv.gz")

    available = list(dict.fromkeys(summary["model"].map(model_group)))
    groups = args.groups or [g for g in available if g.startswith("dae_")]
    missing = [g for g in groups if g not in available]
    if missing:
        raise SystemExit(f"groups not in the results: {missing}; available: {available}")
    if len(groups) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} groups can be compared, got {groups}")

    errors = error_table(summary, groups)
    within = within_level_table(metrics, groups)
    pooled = pooled_table(metrics, groups)
    blind = blind_spot_table(per_image, groups)
    for name, table in (("errors", errors), ("within_level", within), ("pooled", pooled),
                        ("blind_spot", blind)):
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")

    plot_training_curves(REPO_ROOT / args.logs, out / "training_curves.png", groups)
    plot_rho_by_condition(within, groups, out / "rho_by_condition.png")
    scatter_groups = [g for g in (args.scatter_groups or groups) if g in groups]
    first_model = {group: sorted(m for m in per_image["model"].unique() if model_group(m) == group)[0]
                   for group in scatter_groups}
    families = [f for f in args.scatter_families if f in FAMILIES]
    family_scatter([(f"g, {group_label(group)}", per_image[per_image["model"] == first_model[group]], "g")
                    for group in scatter_groups],
                   out / f"{args.scatter_name}.png",
                   "Idempotence residual g vs. true error; family in blue, other levels in gray",
                   families=families, width=2.7 * len(families) + 0.4)
    print_summary(errors, within, pooled, blind, groups)
    print(f"\nWrote tables and figures to {out}")


if __name__ == "__main__":
    main()
