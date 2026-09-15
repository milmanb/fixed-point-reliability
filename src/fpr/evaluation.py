"""Shared evaluation protocol for exact projectors and trained models.

The corruption grid and the per-condition seeds live here, so every model is scored on
the same observations y of the same test images.
"""

import time
import zlib

import numpy as np
import pandas as pd
import torch

from fpr.degradations import BoxMask, GaussianBlur, GaussianNoise, PixelMask
from fpr.metrics import rank_metrics
from fpr.signals import compute_signals, jacobian_signals

CONDITIONS = [
    *(GaussianNoise(sigma) for sigma in (0.1, 0.2, 0.3, 0.5)),
    *(GaussianBlur(std) for std in (0.5, 1.0, 1.5, 2.0)),
    *(PixelMask(drop) for drop in (0.25, 0.5, 0.75)),
    *(BoxMask(size) for size in (8, 14)),
]


def observe(condition, x, seed):
    """Corrupt x under `condition`; the random draw depends only on `seed` and the condition."""
    gen = torch.Generator().manual_seed(seed + zlib.crc32(condition.label.encode()))
    return condition(x, gen)


def per_image_signals(models, x, labels, seed=0, probes=0, jacobian_families=("noise",),
                      conditions=CONDITIONS, verbose=True):
    """Long-format table with one row per (condition, model, test image).

    models: dict name -> callable f. Models with `differentiable = False` skip first-order signals.
    probes: Hutchinson probes for `div`; 0 disables the first-order signals.
    jacobian_families: corruption families that get first-order signals. The default is noise
        only: SURE is defined for Gaussian denoising, and the probes dominate the run time.
    `sure` is added for Gaussian-noise conditions, where its assumptions hold.
    """
    frames = []
    for condition in conditions:
        y, A = observe(condition, x, seed)
        for name, f in models.items():
            start = time.perf_counter()
            signals = compute_signals(f, x, y, A)
            if (probes and condition.family in jacobian_families
                    and getattr(f, "differentiable", True)):
                signals |= jacobian_signals(f, y, probes=probes, seed=seed)
                if condition.family == "noise":
                    var = condition.sigma ** 2
                    signals["sure"] = signals["d_mse"] - var + 2.0 * var * signals["div"]
            frame = pd.DataFrame(signals)
            frame.insert(0, "class", labels.numpy())
            frame.insert(0, "image", np.arange(len(x)))
            frame.insert(0, "model", name)
            frame.insert(0, "level", condition.level)
            frame.insert(0, "family", condition.family)
            frame.insert(0, "condition", condition.label)
            frames.append(frame)
            if verbose:
                print(f"  {condition.label:<22} {name:<16} mean e={signals['e'].mean():.4f}  "
                      f"mean g={signals['g'].mean():.1e}  ({time.perf_counter() - start:.1f}s)",
                      flush=True)
    return pd.concat(frames, ignore_index=True)


def score(per_image, signals, target="e", n_boot=1000, seed=0, severity_baseline=True):
    """Rank metrics per condition, pooled per corruption family, and pooled over all conditions.

    A signal enters a group only if it is defined on every row of the group (`sure` exists for
    noise only). In pooled groups, `<signal>@level` replaces each value by the median of the
    signal over its corruption level. It keeps the differences between levels and removes all
    per-image information, so it measures how much of a pooled score is severity detection.
    """
    rows = []

    def add(frame, scope, group, clusters=None):
        present = [s for s in signals if s in frame and frame[s].notna().all()]
        if not present:
            return
        error = frame[target].to_numpy()
        results = rank_metrics({s: frame[s].to_numpy() for s in present}, error,
                               clusters=clusters, n_boot=n_boot, seed=seed)
        if severity_baseline and scope != "condition":
            medians = frame.groupby("condition")[present].transform("median")
            results += rank_metrics({f"{s}@level": medians[s].to_numpy() for s in present}, error,
                                    clusters=clusters, n_boot=0, seed=seed)
        base = {"scope": scope, "group": group, "model": frame["model"].iat[0], "target": target}
        rows.extend(base | result for result in results)

    for (condition, _), frame in per_image.groupby(["condition", "model"], sort=False):
        add(frame, "condition", condition)
    # Pooled groups hold several levels of the same test image, so resample whole images.
    for (family, _), frame in per_image.groupby(["family", "model"], sort=False):
        add(frame, "family", family, clusters=frame["image"].to_numpy())
    for _, frame in per_image.groupby("model", sort=False):
        add(frame, "all", "all", clusters=frame["image"].to_numpy())
    return pd.DataFrame(rows)
