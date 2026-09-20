"""Risk-coverage curves and selective risk for reference-free signals.

    python scripts/selective.py
    python scripts/selective.py --results results/models results/models_skip
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpr.data import REPO_ROOT
from fpr.plotting import style_axes
from fpr.selective import risk_coverage_curve, summarize

SIGNALS = ("g", "d", "r_A", "dis", "g2", "q", "b")
MAIN_GROUPS = ("dae_lam0", "dae_lam1w5", "unet_lam0", "unet_lam1w5")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", default=["results/models", "results/models_skip"])
    parser.add_argument("--out", default="results/selective")
    parser.add_argument("--groups", nargs="*", default=list(MAIN_GROUPS))
    return parser.parse_args()


def model_group(name):
    return name.rsplit("_seed", 1)[0]


def main():
    args = parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for folder in args.results:
        path = REPO_ROOT / folder / "per_image.csv.gz"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no per_image.csv.gz found; run evaluate_models.py first")
    per_image = pd.concat(frames, ignore_index=True)
    per_image = per_image[per_image["model"].str.contains("_seed")].copy()
    per_image["group"] = per_image["model"].map(model_group)
    if args.groups:
        per_image = per_image[per_image["group"].isin(args.groups)]

    rows = []
    for (group, model), frame in per_image.groupby(["group", "model"], sort=False):
        error = frame["e"].to_numpy()
        for signal in SIGNALS:
            if signal not in frame or frame[signal].isna().all():
                continue
            values = frame[signal].to_numpy()
            if np.ptp(values[np.isfinite(values)]) <= 1e-12:
                continue
            stats = summarize(values, error)
            rows.append({"group": group, "model": model, "signal": signal, **stats})
    table = pd.DataFrame(rows)
    table.to_csv(out / "selective.csv", index=False, float_format="%.4f")

    summary = (table.groupby(["group", "signal"], sort=False)
               [["aurc", "aurc_norm", "risk@0.8", "risk@0.5", "mean_error"]]
               .mean().reset_index())
    summary.to_csv(out / "selective_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Figure: risk-coverage for g vs dis vs oracle, one panel per main group (seed 0).
    groups = [g for g in args.groups if g in set(summary["group"])]
    fig, axes = plt.subplots(1, max(len(groups), 1), figsize=(3.2 * max(len(groups), 1), 3.0),
                             sharey=True, squeeze=False)
    for ax, group in zip(axes[0], groups):
        seed0 = [m for m in per_image.loc[per_image["group"] == group, "model"].unique()
                 if m.endswith("_seed0")]
        if not seed0:
            continue
        frame = per_image[per_image["model"] == seed0[0]]
        error = frame["e"].to_numpy()
        cov_o, risk_o = risk_coverage_curve(error, error)
        ax.plot(cov_o, risk_o, color="0.3", lw=1.2, label="oracle")
        ax.axhline(error.mean(), color="0.6", ls=":", lw=1, label="random")
        for signal, color in (("g", "C0"), ("dis", "C1"), ("d", "C2")):
            if signal not in frame or frame[signal].isna().all():
                continue
            cov, risk = risk_coverage_curve(frame[signal].to_numpy(), error)
            ax.plot(cov, risk, color=color, lw=1.4, label=signal)
        ax.set_xlabel("coverage")
        ax.set_title(group.replace("dae_", "").replace("unet_", "skip/"))
        ax.set_xlim(0, 1)
        style_axes(ax)
    axes[0, 0].set_ylabel("selective risk (mean $e$)")
    axes[0, -1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=150)
    plt.close(fig)

    (out / "selective.json").write_text(json.dumps({
        "groups": list(groups), "signals": list(SIGNALS),
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
