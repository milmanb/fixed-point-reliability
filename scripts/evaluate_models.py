"""Evaluate trained autoencoders and reference projectors on the shared corruption grid.

Signals: g, d, r_A (proposal); div and g_lin (first-order); sure (Gaussian noise only);
g2, g3, q (multi-step residuals); dis (ensemble disagreement across seeds);
b, the brightness of y, as a model-free baseline.
Every signal is scored against the L1 error e, as in the proposal, with bootstrap CIs,
and against the per-pixel MSE e_mse, the quantity SURE estimates (point estimates only).
Partial Spearman correlations given b show what each signal adds beyond image brightness.

    python scripts/evaluate_models.py                                # all checkpoints/dae_*.pt
    python scripts/evaluate_models.py --n-eval 2000 --n-boot 200     # quick run
    python scripts/evaluate_models.py --ensemble --no-metrics        # per-image only
    python scripts/evaluate_models.py --signals dis g2 g3 q --append # score new signals
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
from fpr.ensemble import attach_disagreement
from fpr.evaluation import CONDITIONS, per_image_signals, score
from fpr.models import load_restorer
from fpr.projectors import PCA, Radial, principal_components

SIGNALS = ("b", "g", "d", "r_A", "div", "g_lin", "sure", "dis", "g2", "g3", "q")
SUMMARY_COLUMNS = ("e", "e_mse", "e_in", "b", "g", "d", "r_A", "div", "g_lin", "sure",
                   "dis", "g2", "g3", "q")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=str, default="checkpoints",
                        help="checkpoint directory, relative to the repository")
    parser.add_argument("--pattern", nargs="+", default=["dae_*.pt"],
                        help="one or more checkpoint globs, e.g. dae_lam0_*.pt dae_lam1w5_*.pt")
    parser.add_argument("--projectors", nargs="*", default=["radial", "pca64"])
    parser.add_argument("--n-eval", type=int, default=10_000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--probes", type=int, default=4, help="Hutchinson probes for div")
    parser.add_argument("--jacobian-families", nargs="*", default=["noise"],
                        help="corruption families that get the first-order signals div, g_lin, sure")
    parser.add_argument("--iterate-steps", type=int, default=3,
                        help="applications of f beyond the first for multi-step residuals")
    parser.add_argument("--ensemble", action="store_true",
                        help="compute ensemble disagreement dis across seeds of each group")
    parser.add_argument("--device", type=str, default=None,
                        help="device for the networks (default: cuda if available)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="worker processes: one model per worker for signals, one group per "
                             "worker for scores (CPU only)")
    parser.add_argument("--threads", type=int, default=None,
                        help="PyTorch threads per signal worker (default: PyTorch's choice)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/models",
                        help="output directory, relative to the repository")
    parser.add_argument("--no-per-image", action="store_true")
    parser.add_argument("--no-metrics", action="store_true",
                        help="write per-image and summary only; skip bootstrap scoring")
    parser.add_argument("--signals", nargs="*", default=None,
                        help="signals to score (default: all present); useful with --append")
    parser.add_argument("--append", action="store_true",
                        help="add these models to the results already in --out, replacing re-run models")
    parser.add_argument("--from-per-image", action="store_true",
                        help="skip signal computation; score an existing per_image.csv.gz in --out")
    parser.add_argument("--models", nargs="*", default=None,
                        help="with --from-per-image, restrict scoring to these models")
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
    paths = {path for pattern in args.pattern for path in (REPO_ROOT / args.checkpoints).glob(pattern)}
    for path in sorted(paths):
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

    if metrics is None or metrics.empty:
        return
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

    if "g_lin" not in per_image:
        return
    print("\nTaylor check: Spearman rho(g, g_lin) within level, median over levels")
    rhos = []
    for (condition, model), frame in per_image.groupby(["condition", "model"], sort=False):
        if frame["g_lin"].notna().all() and np.ptp(frame["g"]) > 1e-9:
            rhos.append({"model_group": model_group(model),
                         "rho": frame["g"].rank().corr(frame["g_lin"].rank())})
    if rhos:
        print(pd.DataFrame(rhos).groupby("model_group")["rho"].median().to_string(float_format="%+.3f"))


def merge_with_existing(out, summary, metrics, per_image, run_info):
    """Add this run's models to an earlier run in the same folder, replacing rows of re-run models.

    Valid because every row is computed from one model alone with fixed seeds, so a separate run
    produces the same rows a joint run would.
    """
    new_models = set(run_info["models"])
    previous = json.loads((out / "run_info.json").read_text())
    if previous["args"]["n_eval"] != run_info["args"]["n_eval"] or \
            previous["args"]["seed"] != run_info["args"]["seed"] or \
            previous["args"]["n_boot"] != run_info["args"]["n_boot"]:
        raise SystemExit("--append needs the same --n-eval, --seed and --n-boot as the earlier run")

    def combine(name, frame, reader):
        old = reader(out / name)
        return pd.concat([old[~old["model"].isin(new_models)], frame], ignore_index=True)

    summary = combine("summary.csv", summary, pd.read_csv)
    if metrics is not None and (out / "metrics.csv").exists():
        metrics = combine("metrics.csv", metrics, pd.read_csv)
    per_image = combine("per_image.csv.gz", per_image, pd.read_csv)
    kept = [m for m in previous["models"] if m not in new_models]
    run_info = previous | {
        "models": kept + run_info["models"],
        "train_args": previous["train_args"] | run_info["train_args"],
        "appended_runs": previous.get("appended_runs", []) + [
            {"models": run_info["models"], "seconds": run_info["seconds"], "args": run_info["args"]}],
    }
    return summary, metrics, per_image, run_info


def append_metrics_only(out, metrics, new_signals):
    """Replace metric rows for (model, signal) pairs that were re-scored; keep the rest."""
    old = pd.read_csv(out / "metrics.csv")
    models = set(metrics["model"])
    drop = old["model"].isin(models) & old["signal"].isin(new_signals)
    # Also drop severity baselines of those signals.
    drop |= old["model"].isin(models) & old["signal"].isin({f"{s}@level" for s in new_signals})
    return pd.concat([old[~drop], metrics], ignore_index=True)


def main():
    args = parse_args()
    if args.jobs > 1 and args.device != "cpu":
        raise SystemExit("--jobs > 1 needs --device cpu: CUDA models cannot be sent to worker processes")
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    x_test, y_test = load_fashion_mnist("test", dtype=torch.float64)
    x, labels = x_test[:args.n_eval], y_test[:args.n_eval]

    if args.from_per_image:
        per_image = pd.read_csv(out / "per_image.csv.gz")
        if args.n_eval != per_image["image"].nunique():
            raise SystemExit(f"--from-per-image has {per_image['image'].nunique()} images, "
                             f"but --n-eval={args.n_eval}")
        if args.models:
            per_image = per_image[per_image["model"].isin(args.models)].reset_index(drop=True)
            if per_image.empty:
                raise SystemExit(f"none of {args.models} are in {out / 'per_image.csv.gz'}")
        models, train_args = {}, {}
        t_signals = time.perf_counter()
        run_models = list(dict.fromkeys(per_image["model"]))
    else:
        models, train_args = build_models(args)
        print(f"Evaluating {len(x)} test images, {len(CONDITIONS)} conditions, "
              f"models: {list(models)}")
        per_image = per_image_signals(models, x, labels, seed=args.seed, probes=args.probes,
                                      jacobian_families=tuple(args.jacobian_families),
                                      jobs=args.jobs, threads=args.threads,
                                      iterate_steps=args.iterate_steps)
        if args.ensemble:
            print("Computing ensemble disagreement...", flush=True)
            per_image = attach_disagreement(per_image, models, x, seed=args.seed)
        t_signals = time.perf_counter()
        run_models = list(models)

    metrics = None
    if not args.no_metrics:
        signal_list = tuple(args.signals) if args.signals else tuple(
            s for s in SIGNALS if s in per_image.columns)
        metrics = pd.concat([score(per_image, signal_list, target="e", n_boot=args.n_boot,
                                   seed=args.seed, jobs=args.jobs),
                             score(per_image, signal_list, target="e_mse", n_boot=0,
                                   seed=args.seed, jobs=args.jobs)],
                            ignore_index=True)
    t_metrics = time.perf_counter()
    summary = summarize(per_image)
    run_info = {
        "args": vars(args),
        "conditions": [c.label for c in CONDITIONS],
        "models": run_models,
        "train_args": train_args,
        "seconds": {"signals": round(t_signals - t0, 1), "metrics": round(t_metrics - t_signals, 1),
                    "total": round(time.perf_counter() - t0, 1)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    if args.append and not args.from_per_image:
        summary, metrics, per_image, run_info = merge_with_existing(out, summary, metrics,
                                                                    per_image, run_info)
    elif args.from_per_image and args.append and metrics is not None and (out / "metrics.csv").exists():
        metrics = append_metrics_only(out, metrics, tuple(args.signals or SIGNALS))
        previous = json.loads((out / "run_info.json").read_text())
        run_info = previous | {
            "appended_runs": previous.get("appended_runs", []) + [
                {"models": run_models, "seconds": run_info["seconds"], "args": run_info["args"],
                 "mode": "from_per_image"}],
        }

    # --from-per-image only re-scores: it must not shrink the summary or rewrite the dump.
    if not args.from_per_image:
        summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
        if not args.no_per_image:
            per_image.to_csv(out / "per_image.csv.gz", index=False, float_format="%.6g")
    if metrics is not None:
        metrics.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print_report(summary, metrics if metrics is not None else pd.DataFrame(), per_image)
    print(f"\nWrote results to {out} in {run_info['seconds']['total']}s")


if __name__ == "__main__":
    main()
