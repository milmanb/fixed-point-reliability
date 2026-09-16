"""Checks behind two claims in the report, written to results/checks.json.

collapse: the model trained with lambda_id = 1 and no warm-up outputs one constant image. We
    compare its validation error with that of the per-pixel median training image, which is the
    best constant under an L1 loss, and measure how much its output varies with the input.
finite_differences: the pipeline estimates Jacobian-vector products with finite differences.
    We compare them with exact forward-mode products on the trained checkpoints.

    python scripts/checks.py
"""

import json

import numpy as np
import torch

from fpr.data import REPO_ROOT, load_fashion_mnist
from fpr.evaluation import CONDITIONS, observe
from fpr.models import load_restorer
from fpr.signals import jacobian_signals

COLLAPSED = "dae_lam1_seed0"
TRAINED = ("dae_lam0_seed0", "dae_lam1w5_seed0")


def collapse_check(val_size=5000, sigma=0.2, seed=0):
    images, _ = load_fashion_mnist("train", dtype=torch.float64)
    x_train, x_val = images[:-val_size], images[-val_size:]
    median = x_train.median(dim=0).values
    row = {"median_image_error": (median - x_val).abs().mean().item(),
           "mean_image_error": (x_train.mean(dim=0) - x_val).abs().mean().item()}

    path = REPO_ROOT / "checkpoints" / f"{COLLAPSED}.pt"
    if path.exists():
        restorer, _ = load_restorer(path, device="cpu")
        y = x_val + sigma * torch.randn(x_val.shape, generator=torch.Generator().manual_seed(seed),
                                        dtype=x_val.dtype)
        with torch.no_grad():
            fy = torch.cat([restorer(part) for part in y.split(1000)])
        row |= {
            "model": COLLAPSED,
            "model_error": (fy - x_val).abs().mean().item(),
            # How far the output moves when the input changes: 0 means a constant map.
            "output_spread": fy.std(dim=0).max().item(),
            "distance_to_median_image": (fy.mean(dim=0) - median).abs().mean().item(),
        }
    return row


def finite_difference_check(n_eval=200, probes=4):
    x, _ = load_fashion_mnist("test", dtype=torch.float64)
    y, _ = observe(CONDITIONS[1], x[:n_eval], 0)
    rows = {}
    for name in TRAINED:
        path = REPO_ROOT / "checkpoints" / f"{name}.pt"
        if not path.exists():
            continue
        restorer, _ = load_restorer(path, device="cpu")
        approximate = jacobian_signals(restorer, y, probes=probes, method="fd")
        exact = jacobian_signals(restorer, y, probes=probes, method="jvp")
        rows[name] = {key: {"mean_relative_error": float(np.mean(np.abs(approximate[key] - exact[key])
                                                                / np.abs(exact[key]))),
                            "max_relative_error": float(np.max(np.abs(approximate[key] - exact[key])
                                                               / np.abs(exact[key])))}
                      for key in ("div", "g_lin")}
    return rows


def main():
    out = {"collapse": collapse_check(), "finite_differences": finite_difference_check()}
    path = REPO_ROOT / "results" / "checks.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
