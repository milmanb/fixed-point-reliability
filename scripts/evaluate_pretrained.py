"""Evaluate two pretrained standard denoisers on natural images: DnCNN-B and Restormer on BSD68.

Both are blind Gaussian denoisers trained on natural images (src/fpr/pretrained.py), so this checks
whether the residual's behaviour on Fashion-MNIST carries over to standard models and data. The 68
BSD68 test images, held out from both training sets, are cut into six 128 x 128 crops each, and
every crop is corrupted at the 13 levels of the main grid (box sides scaled to the crop, blur
reflected at its border). Signals and rank metrics are computed as in evaluate_models.py, without
the first-order signals and the scores of the squared error. The bootstrap resamples whole source
images, also within a level, since the six crops of an image are not independent. Writes
per_image.csv.gz, summary.csv, metrics.csv and
run_info.json to results/pretrained. Outputs are clipped to [0, 1], as when they are saved as
images; --no-clip keeps the raw outputs (results/pretrained_noclip), since the clip can turn
outputs that leave [0, 1] into exact fixed points.

    python scripts/download_pretrained.py
    python scripts/evaluate_pretrained.py
    python scripts/evaluate_pretrained.py --no-clip
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fpr.data import DUMP_COMPRESSION, REPO_ROOT, load_bsd68_crops
from fpr.download import sha256
from fpr.evaluation import natural_conditions, per_image_signals, score
from fpr.pretrained import MODELS, load_denoiser
from fpr.signals import SIGNALS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_models import summarize  # noqa: E402  (same summary as the main runs)

BATCH = {"dncnn_b": 32, "restormer": 2}  # crops per forward pass; small, to fit a 4 GB GPU


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--size", type=int, default=128, help="crop side in pixels")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterate-steps", type=int, default=2,
                        help="applications of f after the first (2 adds g2 and q)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-clip", action="store_true",
                        help="keep the networks' raw outputs instead of clipping them to [0, 1]")
    parser.add_argument("--out", default=None,
                        help="relative to the repository (default: results/pretrained, or "
                             "results/pretrained_noclip with --no-clip)")
    args = parser.parse_args()
    args.out = args.out or ("results/pretrained_noclip" if args.no_clip else "results/pretrained")
    return args


def deepinv_version():
    """The installed deepinv version, which Restormer's architecture comes from (None if absent)."""
    try:
        return importlib.metadata.version("deepinv")
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    args = parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    x, sources = load_bsd68_crops(size=args.size, dtype=torch.float64)
    conditions = natural_conditions(args.size)
    models = {name: load_denoiser(name, device=args.device, batch_size=BATCH[name], clip=not args.no_clip)
              for name in args.models}
    print(f"Evaluating {len(x)} crops of {sources.max().item() + 1} images, {len(conditions)} "
          f"conditions, models: {list(models)}", flush=True)
    per_image = per_image_signals(models, x, sources, seed=args.seed, probes=0,
                                  conditions=conditions, iterate_steps=args.iterate_steps)
    per_image = per_image.rename(columns={"class": "source"})
    t_signals = time.perf_counter()

    signals = tuple(s for s in (*SIGNALS, "b", "g2") if s in per_image.columns)
    metrics = score(per_image, signals, target="e", n_boot=args.n_boot, seed=args.seed,
                    cluster="source", cluster_levels=True)
    summary = summarize(per_image)
    run_info = {
        "args": vars(args),
        "conditions": [c.label for c in conditions],
        "models": list(models),
        "weights": {name: {"file": MODELS[name][0],
                           "sha256": sha256(REPO_ROOT / "data" / MODELS[name][0])}
                    for name in models},
        "crops": len(x), "source_images": int(sources.max().item()) + 1,
        "seconds": {"signals": round(t_signals - t0, 1),
                    "total": round(time.perf_counter() - t0, 1)},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "deepinv": deepinv_version()},
    }
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    per_image.to_csv(out / "per_image.csv.gz", index=False, float_format="%.6g", compression=DUMP_COMPRESSION)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))

    within = metrics[metrics["scope"] == "condition"].groupby(["model", "signal"])["spearman"]
    print("\nSpearman rho(signal, e) within a level, median over the 13 levels")
    print(within.median().unstack().to_string(float_format="%+.2f"))
    print("\nMean error and residual per condition")
    print(summary.pivot(index="condition", columns="model", values=["e", "g"])
          .reindex([c.label for c in conditions]).to_string(float_format="%.4f"))
    print(f"\nWrote {out} in {run_info['seconds']['total']}s")


if __name__ == "__main__":
    main()
