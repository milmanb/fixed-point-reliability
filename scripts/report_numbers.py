"""Numbers the report states that no other results file holds, computed from the per-image dumps.

Writes results/report_numbers.json. Each entry names the report sentence it supports; every other
number in the report comes from results/*.csv, results/checks.json or the generated tables.

    python scripts/report_numbers.py
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist

COLUMNS = ["condition", "family", "model", "image", "e", "g", "g_lin"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/report_numbers.json")
    return parser.parse_args()


def load(folder):
    return pd.read_csv(REPO_ROOT / folder / "per_image.csv.gz", usecols=COLUMNS,
                       dtype={"condition": "category", "family": "category", "model": "category"})


def seeds(frame, group):
    return sorted(m for m in frame["model"].cat.categories if m.rsplit("_seed", 1)[0] == group)


def stable_but_wrong(rows):
    """Images with top-10% error and below-median g, both percentiles over all levels."""
    return rows[(rows["e"] >= rows["e"].quantile(0.9)) & (rows["g"] < rows["g"].median())]


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    args = parse_args()
    numbers = {}

    bottleneck = load("results/models")
    noise = bottleneck["condition"].astype(str).str.startswith("noise")
    projectors = bottleneck[noise & bottleneck["model"].isin(["radial", "pca64"])]
    numbers["projector_max_g_lin_noise"] = {
        "value": float(projectors["g_lin"].max()),
        "claim": "Sec. 3.1: the right side of Eq. (1) (radial and PCA-64 under noise) is at most 1e-12",
    }

    taylor = []
    for model in seeds(bottleneck, "dae_lam0"):
        for sigma in (0.1, 0.2, 0.3):
            rows = bottleneck[(bottleneck["model"] == model) & (bottleneck["condition"] == f"noise(sigma={sigma})")]
            taylor.append(spearman(rows["g"], rows["g_lin"]))
    numbers["taylor_rho_g_glin_bottleneck"] = {
        "min": min(taylor), "max": max(taylor), "values": taylor,
        "claim": "Sec. 3.2: Spearman rho between g and its linearization, bottleneck lambda 0, "
                 "3 seeds x sigma in {0.1, 0.2, 0.3}: 0.81 to 0.97",
    }

    worst_pixel = []
    for model in seeds(bottleneck, "dae_lam0"):
        rows = bottleneck[bottleneck["model"] == model]
        worst = rows[rows["e"] >= rows["e"].quantile(0.75)]
        worst_pixel.append(float((worst["family"] == "pixel_mask").mean()))
    numbers["pixel_mask_share_of_worst_quartile_bottleneck"] = {
        "mean": float(np.mean(worst_pixel)), "values": worst_pixel,
        "claim": "Sec. 3.2: pixel masks hold 60% of the worst-quartile errors (bottleneck lambda 0)",
    }

    picked = stable_but_wrong(bottleneck[bottleneck["model"] == "dae_lam0_seed0"])
    numbers["figure1_pixel_mask_share"] = {
        "value": float((picked["family"] == "pixel_mask").mean()),
        "claim": "Fig. 1 caption: 98% of top-10%-error, below-median-g images are pixel masks "
                 "(bottleneck lambda 0, seed 0)",
    }
    del bottleneck

    skip = load("results/models_skip")
    boxes = []
    for model in seeds(skip, "unet_lam0"):
        picked = stable_but_wrong(skip[skip["model"] == model])
        boxes.append(float((picked["condition"] == "box_mask(size=14)").mean()))
    numbers["skip_large_box_share_of_stable_but_wrong"] = {
        "min": min(boxes), "max": max(boxes), "values": boxes,
        "claim": "Sec. 3.2: 83-87% of the skip model's top-10%-error, below-median-g images are "
                 "large boxes (per seed)",
    }
    del skip

    x, _ = load_fashion_mnist("test", dtype=torch.float64)
    numbers["clean_image_spread_test_set"] = {
        "value": float(x.std(dim=0).mean()),
        "claim": "Sec. 3.3: clean images vary by 0.27 across inputs (per-pixel std, mean over pixels), "
                 "the reference for the initialization spreads in results/checks.json",
    }

    out = REPO_ROOT / args.out
    out.write_text(json.dumps(numbers, indent=2) + "\n")
    for key, entry in numbers.items():
        shown = {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in entry.items()
                 if k in ("value", "min", "max", "mean")}
        print(f"{key}: {shown}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
