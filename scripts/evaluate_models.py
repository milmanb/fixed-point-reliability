"""Evaluate trained autoencoders and reference projectors on the shared corruption grid.

Signals: g, d, r_A (proposal); div and g_lin (first-order); sure (Gaussian noise only);
b, the brightness of y, as a model-free baseline.
Every signal is scored against the L1 error e, as in the proposal, with bootstrap CIs,
and against the per-pixel MSE e_mse, the quantity SURE estimates (point estimates only).
Partial Spearman correlations given b show what each signal adds beyond image brightness.

    python scripts/evaluate_models.py                                # all checkpoints/dae_*.pt
    python scripts/evaluate_models.py --n-eval 2000 --n-boot 200     # quick run
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist
from fpr.evaluation import CONDITIONS, per_image_signals, score
from fpr.models import load_restorer
from fpr.projectors import PCA, Radial, principal_components

SIGNALS = ("b", "g", "d", "r_A", "div", "g_lin", "sure")
SUMMARY_COLUMNS = ("e", "e_mse", "e_in", "b", "g", "d", "r_A", "div", "g_lin", "sure")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="checkpoint directory, relative to the repository")
    parser.add_argument("--pattern", type=str, default="dae_*.pt")
    parser.add_argument("--projectors", nargs="*", default=["radial", "pca64"])
    parser.add_argument("--n-eval", type=int, default=10_000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--probes", type=int, default=4, help="Hutchinson probes for div")
    parser.add_argument("--jacobian-families", nargs="*", default=["noise"],
                        help="corruption families that get the first-order signals div, g_lin, sure")
    parser.add_argument("--device", type=str, default=None,
                        help="device for the networks (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/models",
                        help="output directory, relative to the repository")
    parser.add_argument("--no-per-image", action="store_true")
    return parser.parse_args()


def build_models(args):
    models, train_args = {}, {}
    if args.projectors:
        x_train, _ = load_fashion_mnist("train", dtype=torch.float64)
        mean, eigvecs, _ = principal_components(x_train)
        for name in args.projectors:
            if name == "radial":
                models[name] = Radial.fit(x_train)
            elif name.startswith("pca"):
                models[name] = PCA.from_components(mean, eigvecs, int(name[3:]))
            else:
                raise ValueError(f"unknown projector {name!r}")
    for path in sorted((REPO_ROOT / args.checkpoints).glob(args.pattern)):
        restorer, checkpoint = load_restorer(path, device=args.device)
        models[restorer.name] = restorer
        train_args[restorer.name] = checkpoint["train_args"]
    return models, train_args


def model_group(name):
    """dae_lam0_seed2 -> dae_lam0; projectors keep their name."""
    return name.rsplit("_seed", 1)[0]


def summarize(per_image):
    columns = [c for c in SUMMARY_COLUMNS if c in per_image]
    grouped = per_image.groupby(["condition", "model"], sort=False)
    summary = grouped[columns].mean()
    summary["frac_improved"] = grouped.apply(lambda f: float((f["e"] < f["e_in"]).mean()),
                                             include_groups=False)
    return summary.reset_index()


def print_report(summary, metrics, per_image):
    order = list(dict.fromkeys(summary["model"]))
    print("\nMean L1 error e per condition")
    table = summary.pivot(index="condition", columns="model", values="e")
    print(table.reindex(index=[c.label for c in CONDITIONS], columns=order).to_string(float_format="%.4f"))

    by_level = metrics[(metrics["scope"] == "condition") & (metrics["target"] == "e")].copy()
    by_level["model_group"] = by_level["model"].map(model_group)
    noise = by_level["group"].str.startswith("noise")
    for stat, title, rows in (("spearman", "all levels", by_level),
                              ("spearman", "noise levels", by_level[noise]),
                              ("partial", "all levels", by_level)):
        name = "Spearman rho(signal, e)" if stat == "spearman" else "Partial rho(signal, e | b)"
        print(f"\n{name} within level: median over {title} (seeds pooled; "
              f"div, g_lin and sure exist only where first-order signals were computed)")
        pivot = rows.groupby(["model_group", "signal"])[stat].median().unstack()
        print(pivot.reindex(columns=[s for s in SIGNALS if s in pivot]).to_string(float_format="%+.2f"))

    print("\nTaylor check: Spearman rho(g, g_lin) within level, median over levels")
    rhos = []
    for (condition, model), frame in per_image.groupby(["condition", "model"], sort=False):
        if frame["g_lin"].notna().all() and np.ptp(frame["g"]) > 1e-9:
            rhos.append({"model_group": model_group(model),
                         "rho": frame["g"].rank().corr(frame["g_lin"].rank())})
    if rhos:
        print(pd.DataFrame(rhos).groupby("model_group")["rho"].median().to_string(float_format="%+.3f"))


def main():
    args = parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    x_test, y_test = load_fashion_mnist("test", dtype=torch.float64)
    x, labels = x_test[:args.n_eval], y_test[:args.n_eval]
    models, train_args = build_models(args)
    print(f"Evaluating {len(x)} test images, {len(CONDITIONS)} conditions, models: {list(models)}")

    per_image = per_image_signals(models, x, labels, seed=args.seed, probes=args.probes,
                                  jacobian_families=tuple(args.jacobian_families))
    t_signals = time.perf_counter()
    metrics = pd.concat([score(per_image, SIGNALS, target="e", n_boot=args.n_boot, seed=args.seed),
                         score(per_image, SIGNALS, target="e_mse", n_boot=0, seed=args.seed)],
                        ignore_index=True)
    t_metrics = time.perf_counter()
    summary = summarize(per_image)

    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    if not args.no_per_image:
        per_image.to_csv(out / "per_image.csv.gz", index=False, float_format="%.6g")
    run_info = {
        "args": vars(args),
        "conditions": [c.label for c in CONDITIONS],
        "models": list(models),
        "train_args": train_args,
        "seconds": {"signals": round(t_signals - t0, 1), "metrics": round(t_metrics - t_signals, 1),
                    "total": round(time.perf_counter() - t0, 1)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print_report(summary, metrics, per_image)
    print(f"\nWrote results to {out} in {run_info['seconds']['total']}s")


if __name__ == "__main__":
    main()
