"""Add g2, g3, q and dis to a stored per-image dump (the "add_signals" step in run_info.json).

The committed results were produced this way: the per-image dumps of the original evaluation were
kept as stored, and the new columns were computed with the pipeline's own functions,
fpr.signals.compute_signals(iterate_steps=3) and fpr.ensemble.disagreement, on the same
observations (seed 0). As a check, the recomputed g, d and e are compared with the stored values
(written with 6 significant digits). A full `scripts/evaluate_models.py --ensemble` run computes the
same columns from scratch.

    python scripts/add_signals.py --folder results/models --pattern "dae_*.pt" --projectors radial pca64 --threads 7
    python scripts/add_signals.py --folder results/models_skip --pattern "unet_*.pt" --threads 7

Afterwards, re-score the new signals with
`scripts/evaluate_models.py --from-per-image --signals dis g2 g3 q --out <folder>`.
"""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist
from fpr.ensemble import disagreement, model_group
from fpr.evaluation import CONDITIONS, observe
from fpr.models import load_restorer
from fpr.projectors import PCA, Radial, principal_components
from fpr.signals import compute_signals

NEW = ("g2", "g3", "q", "dis")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--projectors", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", default=None, help="default: overwrite <folder>/per_image.csv.gz")
    return parser.parse_args()


def build_models(pattern, projectors):
    models = {}
    if projectors:
        x_train, _ = load_fashion_mnist("train", dtype=torch.float64)
        mean, eigvecs, _ = principal_components(x_train)
        for name in projectors:
            models[name] = (Radial.fit(x_train) if name == "radial"
                            else PCA.from_components(mean, eigvecs, int(name[3:])))
    for path in sorted((REPO_ROOT / "checkpoints").glob(pattern)):
        restorer, _ = load_restorer(path, device="cpu")
        models[restorer.name] = restorer
    return models


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    folder = REPO_ROOT / args.folder
    t0 = time.perf_counter()
    stored = pd.read_csv(folder / "per_image.csv.gz",
                         dtype={"condition": "category", "family": "category", "model": "category"})
    for column in NEW:
        if column in stored:
            raise SystemExit(f"{column} is already in {folder / 'per_image.csv.gz'}")
    n = stored["image"].nunique()
    x, _ = load_fashion_mnist("test", dtype=torch.float64)
    x = x[:n]

    models = build_models(args.pattern, args.projectors)
    if set(models) != set(stored["model"].cat.categories):
        raise SystemExit(f"models differ: built {sorted(models)}, stored "
                         f"{sorted(stored['model'].cat.categories)}")
    groups = {}
    for name in models:
        groups.setdefault(model_group(name) if "_seed" in name else name, []).append(name)

    new = {c: np.full(len(stored), np.nan) for c in NEW}
    check = {"rows": 0, "max_rel": {"g": 0.0, "d": 0.0, "e": 0.0}, "exact": {"g": 0, "d": 0, "e": 0}}
    dis_checked = set()
    for condition in CONDITIONS:
        y, A = observe(condition, x, 0)
        for group, names in groups.items():
            start = time.perf_counter()
            for name in names:
                rows = np.flatnonzero((stored["condition"] == condition.label).to_numpy()
                                      & (stored["model"] == name).to_numpy())
                images = stored["image"].to_numpy()[rows]
                if not np.array_equal(images, np.arange(n)):
                    raise SystemExit(f"unexpected row order for {condition.label} / {name}")
                signals = compute_signals(models[name], x, y, A, iterate_steps=3)
                for column in ("g", "d", "e"):
                    ours = np.array([float(f"{v:.6g}") for v in signals[column]])
                    theirs = stored[column].to_numpy()[rows]
                    rel = np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-300)
                    check["max_rel"][column] = max(check["max_rel"][column], float(rel.max()))
                    check["exact"][column] += int((ours == theirs).sum())
                check["rows"] += len(rows)
                for column in ("g2", "g3", "q"):
                    new[column][rows] = signals[column]
            if len(names) >= 2:
                values = disagreement([models[name] for name in names], y)
                for name in names:
                    rows = np.flatnonzero((stored["condition"] == condition.label).to_numpy()
                                          & (stored["model"] == name).to_numpy())
                    new["dis"][rows] = values
                dis_checked.add(group)
            print(f"  {condition.label:<22} {group:<18} {time.perf_counter() - start:6.1f}s", flush=True)

    for column in NEW:
        stored[column] = new[column]
    out = REPO_ROOT / args.out if args.out else folder / "per_image.csv.gz"
    stored.to_csv(out, index=False, float_format="%.6g")
    info = {"folder": args.folder, "pattern": args.pattern, "projectors": args.projectors,
            "models": list(models), "dis_groups": sorted(dis_checked), "threads": args.threads,
            "seconds": round(time.perf_counter() - t0, 1), "check": check,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__}}
    record = REPO_ROOT / ".cache" / f"add_new_signals_{folder.name}.json"  # git-ignored
    record.parent.mkdir(exist_ok=True)
    record.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
