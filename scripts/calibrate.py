"""Per-operator calibration: can a reference-free signal be made comparable across corruptions?

Pooled scores are weak because the scale of a signal differs between noise, blur and masks, not
because the signal is uninformative inside a corruption. Here we map each signal to an error
prediction with isotonic regression fitted on half the test images, and score the predictions on
the other half. Three settings:

    per level   the corruption and its severity are known at test time
    per family  only the operator is known (noise, blur, pixel mask, box mask)
    severity    predict the calibration-half mean error of the level; no per-image information

Calibration needs clean images offline, which the reliability signals themselves do not.

    python scripts/calibrate.py
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from fpr.data import REPO_ROOT

SIGNALS = ("g", "d", "r_A")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/models", help="relative to the repository")
    parser.add_argument("--out", default="results/calibration", help="relative to the repository")
    parser.add_argument("--models", nargs="*", default=None, help="default: every autoencoder")
    return parser.parse_args()


def isotonic_predictions(frame, calibration, signal, by):
    """Fit e ~ signal on the calibration half of each group and predict for the whole group."""
    predictions = pd.Series(np.nan, index=frame.index)
    for _, rows in frame.groupby(by, sort=False):
        fitted = rows[calibration.loc[rows.index]]
        model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        model.fit(fitted[signal].to_numpy(), fitted["e"].to_numpy())
        predictions.loc[rows.index] = model.predict(rows[signal].to_numpy())
    return predictions


def score(predictions, error, worst):
    return {"spearman": spearmanr(predictions, error).statistic,
            "auroc": roc_auc_score(worst, predictions)}


def evaluate_model(frame):
    """Pooled scores on the held-out half for the raw signal and three calibrations."""
    calibration = (frame["image"] % 2 == 0)
    held_out = ~calibration
    error = frame.loc[held_out, "e"].to_numpy()
    worst = error >= np.quantile(error, 0.75)
    level_mean = frame[calibration].groupby("condition")["e"].mean()

    rows = []
    for signal in SIGNALS:
        predictions = {
            "raw": frame[signal],
            "per level": isotonic_predictions(frame, calibration, signal, "condition"),
            "per family": isotonic_predictions(frame, calibration, signal, "family"),
        }
        for setting, values in predictions.items():
            rows.append({"signal": signal, "setting": setting,
                         **score(values[held_out].to_numpy(), error, worst)})
    severity = frame.loc[held_out, "condition"].map(level_mean).to_numpy()
    rows.append({"signal": "-", "setting": "severity only", **score(severity, error, worst)})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results = REPO_ROOT / args.results
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_image = pd.read_csv(results / "per_image.csv.gz")
    models = args.models or [m for m in per_image["model"].unique() if "_seed" in m]

    tables = []
    for model in models:
        table = evaluate_model(per_image[per_image["model"] == model].reset_index(drop=True))
        tables.append(table.assign(model=model, group=model.rsplit("_seed", 1)[0]))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "calibration.csv", index=False, float_format="%.4f")

    summary = table.groupby(["group", "setting", "signal"])[["spearman", "auroc"]].mean()
    print("Pooled over all 13 levels, scored on the held-out half, mean over seeds\n")
    print(summary.unstack("setting").to_string(float_format=lambda v: f"{v:.2f}"))
    (out / "calibration.json").write_text(json.dumps(
        {"models": models, "settings": ["raw", "per level", "per family", "severity only"]}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
