"""Reference-free reliability signals and the offline error (proposal, Sec. 3).

Zeroth-order signals, per-image mean absolute values over pixels, (1/|Omega|) ||.||_1:

    g    idempotence residual    |f(f(y)) - f(y)|   uses f, y
    d    displacement            |f(y) - y|         uses f, y
    r_A  measurement residual    |A f(y) - y|       uses f, y, A
    e    true error (offline)    |f(y) - x|         needs the clean image x
    e_in input error (offline)   |y - x|            corruption strength

plus the per-pixel squared versions d_mse and e_mse. For denoising (A = I), r_A and d coincide.

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
        "d_mse": _mean_sq(fy - y),
        "e_mse": _mean_sq(fy - x),
    }


def jacobian_signals(f, y, probes=8, batch_size=1024, seed=0):
    """Per-image `div` and `g_lin` (see module docstring) via forward-mode AD.

    Uses torch.func.jvp, so f must be built from differentiable torch operations.
    """
    gen = torch.Generator().manual_seed(seed)
    div, g_lin = [], []
    for part in y.split(batch_size):
        with torch.no_grad():
            delta = f(part) - part
        _, j_delta = torch.func.jvp(f, (part,), (delta,))
        g_lin.append(_mean_abs(j_delta))
        total = np.zeros(len(part))
        for _ in range(probes):
            probe = (2 * torch.randint(0, 2, part.shape, generator=gen) - 1).to(part)
            _, j_probe = torch.func.jvp(f, (part,), (probe,))
            total += (probe * j_probe).flatten(1).mean(1).detach().cpu().double().numpy()
        div.append(total / probes)
    return {"div": np.concatenate(div), "g_lin": np.concatenate(g_lin)}
