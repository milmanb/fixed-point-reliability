import numpy as np
import torch

from fpr.projectors import PCA, principal_components
from fpr.signals import compute_signals, jacobian_signals


def _images(n, seed, side=6):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((n, 1, side, side), generator=gen, dtype=torch.float64)


def test_divergence_is_exact_for_diagonal_maps():
    """Rademacher probes satisfy b * b = 1, so Hutchinson is exact when J is diagonal."""
    scale = torch.linspace(0.1, 0.9, 36, dtype=torch.float64).view(1, 1, 6, 6)
    out = jacobian_signals(lambda y: scale * y, _images(10, 0), probes=3)
    assert np.allclose(out["div"], scale.mean().item(), rtol=0, atol=1e-15)


def test_linearized_idempotence_residual_is_exact_for_affine_maps():
    train = _images(200, 1)
    mean, eigvecs, _ = principal_components(train)
    shrink = torch.linspace(1.0, 0.2, 36, dtype=torch.float64)
    weights = eigvecs @ torch.diag(shrink) @ eigvecs.T  # affine, not idempotent

    def f(y):
        return (mean + (y.flatten(1) - mean) @ weights).view_as(y)

    y = _images(20, 2)
    exact = compute_signals(f, y, y, lambda z: z)["g"]
    assert np.allclose(jacobian_signals(f, y, probes=1)["g_lin"], exact, rtol=1e-10, atol=1e-14)


def test_sure_is_unbiased_for_a_pca_projector():
    train, x = _images(500, 3), _images(4000, 4)
    mean, eigvecs, _ = principal_components(train)
    projector = PCA.from_components(mean, eigvecs, 8)
    sigma = 0.2
    y = x + sigma * torch.randn(x.shape, generator=torch.Generator().manual_seed(5), dtype=x.dtype)
    signals = compute_signals(projector, x, y, lambda z: z)
    div = jacobian_signals(projector, y, probes=4)["div"]
    sure = signals["d_mse"] - sigma**2 + 2 * sigma**2 * div
    assert abs(sure.mean() - signals["e_mse"].mean()) < 0.05 * signals["e_mse"].mean()
    assert abs(div.mean() - 8 / 36) < 0.02
