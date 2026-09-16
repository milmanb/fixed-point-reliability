"""Reference-free reliability signals and the offline error (proposal, Sec. 3).

Zeroth-order signals, per-image mean absolute values over pixels, (1/|Omega|) ||.||_1:

    g    idempotence residual    |f(f(y)) - f(y)|   uses f, y
    d    displacement            |f(y) - y|         uses f, y
    r_A  measurement residual    |A f(y) - y|       uses f, y, A
    e    true error (offline)    |f(y) - x|         needs the clean image x
    e_in input error (offline)   |y - x|            corruption strength

plus the per-pixel squared versions d_mse and e_mse. For denoising (A = I), r_A and d coincide.

    b    input brightness        mean(y)            uses y only

b is a model-free baseline, not a reliability signal. Under masking and strong blur the error
grows with the amount of content in the image, so any signal that tracks brightness ranks
images well there without detecting model failures.

First-order signals (`jacobian_signals`), from Jacobian-vector products:

    div    (1/D) tr J_f(y)               Hutchinson estimate with Rademacher probes
    g_lin  (1/D) |J_f(y) (f(y) - y)|_1   first-order Taylor approximation of g, since
                                         f(f(y)) = f(y + delta) ~ f(y) + J_f(y) delta

For Gaussian noise with known sigma, Stein's unbiased estimate of the per-pixel MSE is
    sure = d_mse - sigma^2 + 2 sigma^2 div.
"""

import numpy as np
import torch

SIGNALS = ("g", "d", "r_A")


def _batched(f, z, batch_size):
    return torch.cat([f(part) for part in z.split(batch_size)])


def _mean_abs(z):
    return z.abs().flatten(1).mean(1).detach().cpu().numpy()


def _mean_sq(z):
    return z.pow(2).flatten(1).mean(1).detach().cpu().numpy()


@torch.no_grad()
def compute_signals(f, x, y, A, batch_size=2048):
    """Return a dict of per-image numpy arrays; A acts on the full batch, f in chunks."""
    fy = _batched(f, y, batch_size)
    ffy = _batched(f, fy, batch_size)
    return {
        "g": _mean_abs(ffy - fy),
        "d": _mean_abs(fy - y),
        "r_A": _mean_abs(A(fy) - y),
        "e": _mean_abs(fy - x),
        "e_in": _mean_abs(y - x),
        "b": y.flatten(1).mean(1).detach().cpu().numpy(),
        "d_mse": _mean_sq(fy - y),
        "e_mse": _mean_sq(fy - x),
    }


def jacobian_signals(f, y, probes=8, batch_size=1024, seed=0, method="fd", eps=1e-3):
    """Per-image `div` and `g_lin` (see module docstring) from Jacobian-vector products.

    method="fd" takes the products as finite differences, (f(y + eps v) - f(y)) / eps, which is
    the estimator of Ramani et al. and needs only forward passes. method="jvp" uses forward-mode
    automatic differentiation, which is exact but crashes inside spawned worker processes with
    this PyTorch build on Windows, so it is used for tests rather than for the pipeline.
    """
    if method not in ("fd", "jvp"):
        raise ValueError(f"method must be 'fd' or 'jvp', got {method!r}")
    gen = torch.Generator().manual_seed(seed)
    div, g_lin = [], []
    for part in y.split(batch_size):
        with torch.no_grad():
            fy = f(part)

        def product(tangent, fy=fy, part=part):
            """The Jacobian-vector product J_f(y) v."""
            if method == "jvp":
                return torch.func.jvp(f, (part,), (tangent,))[1]
            with torch.no_grad():
                return (f(part + eps * tangent) - fy) / eps

        g_lin.append(_mean_abs(product(fy - part)))
        total = np.zeros(len(part))
        for _ in range(probes):
            probe = (2 * torch.randint(0, 2, part.shape, generator=gen) - 1).to(part)
            total += (probe * product(probe)).flatten(1).mean(1).detach().cpu().double().numpy()
        div.append(total / probes)
    return {"div": np.concatenate(div), "g_lin": np.concatenate(g_lin)}
