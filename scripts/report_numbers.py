"""Numbers the report and the README state that no other results file holds, from the per-image dumps.

Writes results/report_numbers.json. Each entry names the sentence it supports; every other number
in the report comes from results/*.csv, results/checks.json or the generated tables.

    python scripts/report_numbers.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist, per_image_dump
from fpr.ensemble import model_group
from fpr.evaluation import CONDITIONS, observe, stable_but_wrong
from fpr.models import load_restorer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_table import VARIANTS, flagged_share  # noqa: E402  (the definitions of Table 1 and Fig. 2)

COLUMNS = ["condition", "family", "model", "image", "e", "g", "g_lin"]
PRETRAINED = ("dncnn_b", "restormer")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/report_numbers.json", help="output file, relative to the repository")
    return parser.parse_args()


def load(folder):
    return pd.read_csv(per_image_dump(folder), usecols=COLUMNS,
                       dtype={"condition": "category", "family": "category", "model": "category"})


def seeds(frame, group):
    return sorted(m for m in frame["model"].cat.categories if model_group(m) == group)


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def per_model(values):
    return {model: float(values[model]) for model in PRETRAINED}


def pretrained_numbers(numbers):
    """Sec. 3.5 and the README's natural-image finding: DnCNN-B and Restormer on BSD68 crops."""
    summary = pd.read_csv(REPO_ROOT / "results/pretrained/summary.csv").set_index(["model", "condition"])
    blur = [f"blur(std={std:g})" for std in (0.5, 1, 1.5, 2)]
    share = {f"{model} {level}": float(summary.loc[(model, level), "e"] / summary.loc[(model, level), "e_in"])
             for model in PRETRAINED for level in blur}
    numbers["pretrained_blur_error_share_of_input"] = {
        "min": min(share.values()), "max": max(share.values()), "values": share,
        "claim": "Sec. 3.5: under blur their mean error is 78-91% of the input's (outputs clipped)",
    }
    growth = {model: summary.loc[(model, blur[-1]), "e"] / summary.loc[(model, blur[0]), "e"]
              for model in PRETRAINED}
    numbers["pretrained_blur_error_growth"] = {
        "values": per_model(growth),
        "claim": "Sec. 3.5: from blur width 0.5 to 2 their mean error grows 2.7-fold",
    }
    change = {model: summary.loc[(model, blur[-1]), "g"] / summary.loc[(model, blur[0]), "g"] - 1
              for model in PRETRAINED}
    numbers["pretrained_blur_g_change"] = {
        "values": per_model(change),
        "claim": "Sec. 3.5: over the same widths their mean g falls by 25% (DnCNN-B) and 11% (Restormer)",
    }

    metrics = {name: pd.read_csv(REPO_ROOT / folder / "metrics.csv")
               for name, folder in (("clipped", "results/pretrained"), ("unclipped", "results/pretrained_noclip"))}
    within = {name: frame[(frame["scope"] == "condition") & (frame["target"] == "e") & (frame["signal"] == "g")]
              for name, frame in metrics.items()}
    halfwidth = within["clipped"].assign(half=(within["clipped"]["spearman_hi"] - within["clipped"]["spearman_lo"]) / 2)
    numbers["pretrained_within_level_interval_halfwidth"] = {
        "values": per_model(halfwidth.groupby("model")["half"].median()),
        "claim": "Sec. 3.5: per-level bootstrap 95% intervals of rho(g, e), resampling source images, "
                 "are about +-0.12 (median half-width over the 13 levels)",
    }
    numbers["pretrained_unclipped_within_level_rho"] = {
        "values": per_model(within["unclipped"].groupby("model")["spearman"].median()),
        "claim": "Sec. 3.5: without the clip the median within-level rho(g, e) is 0.43 (DnCNN-B) and "
                 "0.11 (Restormer)",
    }

    unclipped = pd.read_csv(per_image_dump("results/pretrained_noclip"), usecols=["model", "family", "e", "g"])
    numbers["pretrained_unclipped_flagged_box_share"] = {
        "values": per_model(flagged_share(unclipped, "box_mask", "g")),
        "claim": "Sec. 3.5, Fig. 2 caption and README: without the clip g flags 94% of Restormer's "
                 "worst-quartile box-mask errors (DnCNN-B: 1%)",
    }
    clipped = pd.read_csv(per_image_dump("results/pretrained"), usecols=["model", "e", "g"])
    counts = {model: len(stable_but_wrong(rows)) for model, rows in clipped.groupby("model")}
    numbers["pretrained_stable_but_wrong"] = {
        "values": {model: counts[model] for model in PRETRAINED},
        "rows": int(len(clipped) / len(PRETRAINED)),
        "claim": "README: 16 (DnCNN-B) and 2 (Restormer) of their 5,304 crops and levels are stable but "
                 "wrong (top-10% error, below-median g; outputs clipped)",
    }


def skip_box_numbers(numbers, x):
    """Sec. 3.3: the skip model leaves the box empty, and a second pass keeps it empty.

    On the evaluation's observations (seed 0, all test images) of both box levels and every seed:
    the mean output inside the box against the clean image there, and the residual inside the box
    against the residual outside it.
    """
    boxes = [c for c in CONDITIONS if c.family == "box_mask"]
    rows = {}
    for seed in range(3):
        restorer, _ = load_restorer(REPO_ROOT / "checkpoints" / f"unet_lam0_seed{seed}.pt", device="cpu")
        for condition in boxes:
            y, A = observe(condition, x, 0)
            box = A(torch.ones_like(x)) == 0
            with torch.no_grad():
                fy = restorer(y)
                residual = (restorer(fy) - fy).abs()
            rows[f"unet_lam0_seed{seed} {condition.label}"] = {
                "output_in_box": float(fy[box].mean()), "clean_in_box": float(x[box].mean()),
                "residual_in_box": float(residual[box].mean()), "residual_outside": float(residual[~box].mean()),
            }
    numbers["skip_box_left_empty"] = {
        "max_output_in_box": max(r["output_in_box"] for r in rows.values()),
        "min_clean_in_box": min(r["clean_in_box"] for r in rows.values()),
        "residual_smaller_in_box": all(r["residual_in_box"] < r["residual_outside"] for r in rows.values()),
        "values": rows,
        "claim": "Sec. 3.3: the skip model leaves the box empty (mean output in the box under 0.01, against "
                 "about 0.4 in the clean image), and a second pass keeps it empty (its residual is smaller "
                 "inside the box than outside)",
    }


def displacement_numbers(numbers):
    """Discussion: d and r_A flag few of the worst blur and box-mask errors of every Table 1 model."""
    groups = {group for group, _, training in VARIANTS if "no warm-up" not in training}
    shares = {}
    for folder in ("results/models", "results/models_skip", "results/pretrained"):
        frame = pd.read_csv(per_image_dump(folder), usecols=["model", "family", "e", "d", "r_A"])
        frame = frame[frame["model"].map(model_group).isin(groups)]
        for signal in ("d", "r_A"):
            for family in ("blur", "box_mask"):
                for model, share in flagged_share(frame, family, signal).dropna().items():
                    shares[f"{model} {signal} {family}"] = float(share)
    table = pd.Series(shares)
    keys = table.index.str.split(" ")
    group_means = table.groupby([keys.str[0].map(model_group), keys.str[1], keys.str[2]]).mean()
    numbers["displacement_flagged_blur_box"] = {
        "max_seed_mean": float(group_means.max()), "max_single_run": float(table.max()),
        "values": {" ".join(k): float(v) for k, v in group_means.items()},
        "claim": "Discussion: averaged over seeds, d and r_A flag at most 2% of the worst blur and box-mask "
                 "errors of each model in Table 1 (DnCNN-B has no worst-quartile blur errors)",
    }


def collapse_numbers(numbers):
    """Sec. 3.4: last-epoch validation errors at sigma 0.2, collapsed runs against the warm-up runs."""
    def last(run):
        return float(pd.read_csv(REPO_ROOT / "results" / "train" / f"{run}.csv").iloc[-1]["val_e@0.2"])

    warm = {group: [last(f"{group}_seed{seed}") for seed in range(3)] for group in ("dae_lam1w5", "unet_lam1w5")}
    numbers["collapse_validation_sigma02"] = {
        "values": {"dae_lam1_seed0": last("dae_lam1_seed0"), "unet_lam1_seed0": last("unet_lam1_seed0"),
                   **{f"{group} (seeds)": runs for group, runs in warm.items()}},
        "claim": "Sec. 3.4: at sigma 0.2 the collapsed runs' validation errors are 0.21 and 0.29, against 0.05 "
                 "and 0.04 for lambda 1 with the warm-up (every seed)",
    }


def main():
    args = parse_args()
    numbers = {}

    bottleneck = load("results/models")
    noise = bottleneck["condition"].astype(str).str.startswith("noise")
    projectors = bottleneck[noise & bottleneck["model"].isin(["radial", "pca64"])]
    numbers["projector_max_g_lin_noise"] = {
        "value": float(projectors["g_lin"].max()),
        "claim": "README, exact projectors: the linearized term (radial and PCA-64 under noise) is at most 1e-12",
    }

    taylor = []
    for model in seeds(bottleneck, "dae_lam0"):
        for sigma in (0.1, 0.2, 0.3):
            rows = bottleneck[(bottleneck["model"] == model) & (bottleneck["condition"] == f"noise(sigma={sigma})")]
            taylor.append(spearman(rows["g"], rows["g_lin"]))
    numbers["taylor_rho_g_glin_bottleneck"] = {
        "min": min(taylor), "max": max(taylor), "values": taylor,
        "claim": "Sec. 2, why the residual can be blind: Spearman rho between g and its linearization, "
                 "bottleneck lambda 0, 3 seeds x sigma in {0.1, 0.2, 0.3}: 0.81 to 0.97",
    }
    taylor_high = []
    for model in seeds(bottleneck, "dae_lam0"):
        rows = bottleneck[(bottleneck["model"] == model) & (bottleneck["condition"] == "noise(sigma=0.5)")]
        taylor_high.append(spearman(rows["g"], rows["g_lin"]))
    numbers["taylor_rho_g_glin_bottleneck_sigma05"] = {
        "min": min(taylor_high), "max": max(taylor_high), "values": taylor_high,
        "claim": "Sec. 2, why the residual can be blind: the same at sigma 0.5, 3 seeds: 0.20 to 0.81",
    }

    worst_pixel = []
    for model in seeds(bottleneck, "dae_lam0"):
        rows = bottleneck[bottleneck["model"] == model]
        worst = rows[rows["e"] >= rows["e"].quantile(0.75)]
        worst_pixel.append(float((worst["family"] == "pixel_mask").mean()))
    numbers["pixel_mask_share_of_worst_quartile_bottleneck"] = {
        "mean": float(np.mean(worst_pixel)), "values": worst_pixel,
        "claim": "Sec. 3.3: pixel masks hold 60% of the worst-quartile errors (bottleneck lambda 0)",
    }

    picked = stable_but_wrong(bottleneck[bottleneck["model"] == "dae_lam0_seed0"])
    numbers["figure1_pixel_mask_share"] = {
        "value": float((picked["family"] == "pixel_mask").mean()),
        "claim": "Fig. 1 caption: 98% of top-10%-error, below-median-g images are pixel masks "
                 "(bottleneck lambda 0, seed 0)",
    }
    del bottleneck

    skip = load("results/models_skip")
    boxes, blur_above_median = [], []
    for model in seeds(skip, "unet_lam0"):
        rows = skip[skip["model"] == model]
        picked = stable_but_wrong(rows)
        boxes.append(float((picked["condition"] == "box_mask(size=14)").mean()))
        worst_blur = rows[(rows["e"] >= rows["e"].quantile(0.75)) & (rows["family"] == "blur")]
        blur_above_median.append(float((worst_blur["g"] >= rows["g"].median()).mean()))
    numbers["skip_large_box_share_of_stable_but_wrong"] = {
        "min": min(boxes), "max": max(boxes), "values": boxes,
        "claim": "Sec. 3.3: 83-87% of the skip model's top-10%-error, below-median-g images are "
                 "boxes of side 14 (per seed)",
    }
    numbers["skip_worst_blur_above_median_g"] = {
        "mean": float(np.mean(blur_above_median)), "values": blur_above_median,
        "claim": "Fig. 2 caption: the skip model's worst-quartile blur errors are mostly not stable, "
                 "their g is above its median (lambda 0, per seed; median over all levels)",
    }
    del skip

    pretrained_numbers(numbers)
    displacement_numbers(numbers)
    collapse_numbers(numbers)

    x, _ = load_fashion_mnist("test", dtype=torch.float64)
    skip_box_numbers(numbers, x)
    numbers["clean_image_spread_test_set"] = {
        "value": float(x.std(dim=0).mean()),
        "claim": "README, collapse without warm-up: clean images vary by 0.27 across inputs (per-pixel "
                 "std, mean over pixels), the reference for the initialization spreads in results/checks.json",
    }

    out = REPO_ROOT / args.out
    out.write_text(json.dumps(numbers, indent=2) + "\n")
    for key, entry in numbers.items():
        shown = {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in entry.items()
                 if k in ("value", "min", "max", "mean") or k.startswith(("max_", "min_", "residual_"))}
        if not shown:
            shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry["values"].items()}
        print(f"{key}: {shown}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
