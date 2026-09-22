"""Split conformal error bars on reference-free reliability signals.

Writes coverage, width, shift and SURE-self-calibration tables under results/conformal/.

    python scripts/conformal.py
    python scripts/conformal.py --results results/models --out results/conformal
"""

import argparse
import json

import pandas as pd

from fpr.conformal import evaluate_signals, shift_coverage, sure_self_calibrate
from fpr.data import REPO_ROOT

SIGNALS = ("g", "d", "r_A", "dis", "g2", "q")
ALPHA = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", default=["results/models", "results/models_skip"],
                        help="result folders containing per_image.csv.gz")
    parser.add_argument("--out", default="results/conformal")
    parser.add_argument("--models", nargs="*", default=None,
                        help="default: every autoencoder with a seed in its name")
    parser.add_argument("--alpha", type=float, default=ALPHA)
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
    models = args.models or [m for m in per_image["model"].unique() if "_seed" in m]

    coverage_rows, shift_rows, sure_rows = [], [], []
    for model in models:
        frame = per_image[per_image["model"] == model].reset_index(drop=True)
        available = [s for s in SIGNALS if s in frame and frame[s].notna().all()]
        # `group` here is the corruption level/family from evaluate_signals; keep it distinct
        # from the model group, or the per-family breakdown is lost.
        table = evaluate_signals(frame, signals=available, alpha=args.alpha)
        table = table.assign(model=model, model_group=model_group(model))
        coverage_rows.append(table)

        for signal in available:
            shift = shift_coverage(frame, signal, alpha=args.alpha)
            shift_rows.append({"model": model, "model_group": model_group(model), "signal": signal,
                               "coverage": shift["coverage"], "width": shift["width"],
                               "q_hat": shift["q_hat"], "n_fit": shift["n_fit"],
                               "n_cal": shift["n_cal"], "n_deploy": shift["n_deploy"]})
            sure = sure_self_calibrate(frame, signal=signal, alpha=args.alpha)
            sure_rows.append({"model": model, "model_group": model_group(model), "signal": signal,
                              **sure})

    coverage = pd.concat(coverage_rows, ignore_index=True)
    shift = pd.DataFrame(shift_rows)
    sure = pd.DataFrame(sure_rows)

    # Compact pooled summary: one row per (model group, signal, setting) averaged over seeds.
    pooled = (coverage[coverage["setting"].isin(("normalized", "marginal", "severity"))]
              .groupby(["model_group", "signal", "setting"], sort=False)[["coverage", "width"]]
              .mean().reset_index())
    pooled.to_csv(out / "coverage.csv", index=False, float_format="%.4f")
    coverage.to_csv(out / "coverage_detail.csv", index=False, float_format="%.4f")
    shift.to_csv(out / "shift.csv", index=False, float_format="%.4f")
    sure.to_csv(out / "sure.csv", index=False, float_format="%.4f")

    # Per-family coverage of the normalized bound: where does the guarantee break?
    family = (coverage[coverage["setting"] == "normalized/family"]
              .groupby(["model_group", "signal", "group"], sort=False)[["coverage", "width"]]
              .mean().reset_index())
    family.to_csv(out / "coverage_family.csv", index=False, float_format="%.4f")

    width = pooled.pivot_table(index=["model_group", "signal"], columns="setting", values="width")
    width.to_csv(out / "width.csv", float_format="%.4f")

    print("Pooled coverage / width (mean over seeds), alpha =", args.alpha)
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nPer-family coverage of the normalized bound")
    print(family.pivot_table(index=["model_group", "signal"], columns="group", values="coverage")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nShift: calibrate on noise sigma in {0.1, 0.2}, deploy on held-out images elsewhere")
    shift_summary = (shift.groupby(["model_group", "signal"])[["coverage", "width"]]
                     .mean().reset_index())
    print(shift_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSURE self-calibration on the noise family (target e_mse), vs. the supervised bound")
    print(sure.groupby(["model_group", "signal"])
          [["coverage", "width", "coverage_supervised", "width_supervised"]]
          .mean().to_string(float_format=lambda v: f"{v:.3f}"))

    (out / "conformal.json").write_text(json.dumps({
        "alpha": args.alpha,
        "models": models,
        "signals": list(SIGNALS),
        "settings": ["normalized", "marginal", "severity"],
        "split": "image % 4 == 0 fits u_hat, image % 4 == 2 calibrates, odd images are held out",
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
