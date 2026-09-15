"""Reference-free reliability signals and the offline error (proposal, Sec. 3).

All quantities are per-image mean absolute values over pixels, (1/|Omega|) ||.||_1:

    g    idempotence residual    |f(f(y)) - f(y)|   uses f, y
    d    displacement            |f(y) - y|         uses f, y
    r_A  measurement residual    |A f(y) - y|       uses f, y, A
    e    true error (offline)    |f(y) - x|         needs the clean image x
    e_in input error (offline)   |y - x|            corruption strength

Note that for denoising (A = I) r_A and d coincide exactly.
"""

import torch

SIGNALS = ("g", "d", "r_A")


def _batched(f, z, batch_size):
    return torch.cat([f(part) for part in z.split(batch_size)])


def _mean_abs(z):
    return z.abs().flatten(1).mean(1).cpu().numpy()


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
    }
