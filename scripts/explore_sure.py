"""Exploration: does SURE's divergence term fix displacement's ranking under Gaussian noise?

Stein's unbiased risk estimate for denoising (A = I, n ~ N(0, sigma^2 I)):
    SURE(y) = ||f(y) - y||^2 - D sigma^2 + 2 sigma^2 div f(y),   E[SURE] = E||f(y) - x||^2.
Its first term is the (squared, L2) displacement d. The divergence is
    PCA-k  : k, a constant           -> SURE ranks images exactly like d
    radial : R (D - 1) / ||y - mu||  -> varies per image
    nn     : 0 almost everywhere, but f is discontinuous, so Stein's lemma fails
The Monte-Carlo divergence (Ramani, Blu & Unser, 2008) is checked against the exact value,
since it is what we would use for a black-box CNN.

    python scripts/explore_sure.py
"""

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from fpr.data import load_fashion_mnist
from fpr.projectors import PCA, NearestNeighbor, Radial, principal_components


def mc_divergence(f, y, eps=1e-3, probes=4, seed=0):
    """Hutchinson-style estimate b^T (f(y + eps b) - f(y)) / eps, averaged over probes."""
    gen = torch.Generator().manual_seed(seed)
    fy = f(y).flatten(1)
    total = 0.0
    for _ in range(probes):
        b = torch.randn(y.shape, generator=gen, dtype=y.dtype)
        total = total + (b.flatten(1) * (f(y + eps * b).flatten(1) - fy)).sum(1) / eps
    return (total / probes).numpy()


def main(n_eval=2000, seed=123):
    x_train, _ = load_fashion_mnist("train", dtype=torch.float64)
    x_test, _ = load_fashion_mnist("test", dtype=torch.float64)
    x = x_test[:n_eval]
    dim = x[0].numel()
    mean, eigvecs, _ = principal_components(x_train)
    radial = Radial.fit(x_train)
    projectors = {"radial": radial, "pca16": PCA.from_components(mean, eigvecs, 16),
                  "pca256": PCA.from_components(mean, eigvecs, 256), "nn": NearestNeighbor(x_train)}

    def exact_divergence(name, y):
        if name.startswith("pca"):
            return np.full(len(y), float(projectors[name].basis.shape[1]))
        if name == "radial":
            return radial.radius * (dim - 1) / (y.flatten(1) - radial.center).norm(dim=1).numpy()
        return np.zeros(len(y))

    print(f"{'sigma':>5} {'proj':>7} | {'rho(d1,e1)':>10} {'rho(d2,e2)':>10} {'rho(SURE,e2)':>12} "
          f"| {'mean e2':>8} {'mean SURE':>9} | {'MC-div rel. err':>15}")
    for sigma in (0.1, 0.2, 0.3, 0.5):
        y = x + sigma * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed), dtype=x.dtype)
        for name, f in projectors.items():
            fy = f(y)
            d1 = (fy - y).abs().flatten(1).mean(1).numpy()
            e1 = (fy - x).abs().flatten(1).mean(1).numpy()
            d2 = (fy - y).pow(2).flatten(1).sum(1).numpy()
            e2 = (fy - x).pow(2).flatten(1).sum(1).numpy()
            div = exact_divergence(name, y)
            sure = d2 - dim * sigma**2 + 2 * sigma**2 * div
            if name == "nn":
                rel = float("nan")
            else:
                mc = mc_divergence(f, y[:200])
                rel = np.abs(mc - div[:200]).mean() / np.abs(div[:200]).mean()
            print(f"{sigma:>5} {name:>7} | {spearmanr(d1, e1).statistic:>10.3f} "
                  f"{spearmanr(d2, e2).statistic:>10.3f} {spearmanr(sure, e2).statistic:>12.3f} "
                  f"| {e2.mean():>8.2f} {sure.mean():>9.2f} | {rel:>15.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    cli = parser.parse_args()
    main(n_eval=cli.n_eval, seed=cli.seed)
