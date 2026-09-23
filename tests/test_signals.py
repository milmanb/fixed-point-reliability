import numpy as np
import torch

from fpr.degradations import PixelMask, gaussian_blur
from fpr.models import ConvAutoencoder
from fpr.projectors import PCA, principal_components
from fpr.signals import compute_signals, jacobian_signals


def _images(n, seed, side=6):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((n, 1, side, side), generator=gen, dtype=torch.float64)


def test_divergence_is_exact_for_diagonal_maps():
    """Rademacher probes satisfy b * b = 1, so Hutchinson is exact when J is diagonal."""
    scale = torch.linspace(0.1, 0.9, 36, dtype=torch.float64).view(1, 1, 6, 6)
    out = jacobian_signals(lambda y: scale * y, _images(10, 0), probes=3, method="jvp")
    assert np.allclose(out["div"], scale.mean().item(), rtol=0, atol=1e-15)


def test_finite_differences_agree_with_forward_mode_ad():
    """The pipeline uses finite differences; they must match exact products on a smooth model."""
    torch.manual_seed(0)
    net = ConvAutoencoder(width=4, latent=8).double()
    y = _images(8, 7, side=28)
    with torch.no_grad():
        exact = jacobian_signals(net, y, probes=2, method="jvp")
        approximate = jacobian_signals(net, y, probes=2, method="fd")
    for key in ("div", "g_lin"):
        assert np.allclose(exact[key], approximate[key], rtol=1e-4, atol=1e-8), key


def test_linearized_idempotence_residual_is_exact_for_affine_maps():
    train = _images(200, 1)
    mean, eigvecs, _ = principal_components(train)
    shrink = torch.linspace(1.0, 0.2, 36, dtype=torch.float64)
    weights = eigvecs @ torch.diag(shrink) @ eigvecs.T  # affine, not idempotent

    def f(y):
        return (mean + (y.flatten(1) - mean) @ weights).view_as(y)

    y = _images(20, 2)
    exact = compute_signals(f, y, y, lambda z: z)["g"]
    linearized = jacobian_signals(f, y, probes=1, method="jvp")["g_lin"]
    assert np.allclose(linearized, exact, rtol=1e-10, atol=1e-14)


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


def test_compute_signals_matches_the_definitions_under_a_mask():
    """Every per-image signal against its formula, with an operator for which r_A differs from d."""
    x = _images(8, 10)
    y, A = PixelMask(0.5)(x, torch.Generator().manual_seed(11))

    def f(z):
        return 0.5 * z + 0.1

    signals = compute_signals(f, x, y, A)
    fy = f(y)

    def mean_abs(z):
        return z.abs().flatten(1).mean(1).numpy()

    def mean_sq(z):
        return z.pow(2).flatten(1).mean(1).numpy()

    expected = {"g": mean_abs(f(fy) - fy), "d": mean_abs(fy - y), "r_A": mean_abs(A(fy) - y),
                "e": mean_abs(fy - x), "e_in": mean_abs(y - x), "b": y.flatten(1).mean(1).numpy(),
                "d_mse": mean_sq(fy - y), "e_mse": mean_sq(fy - x)}
    for key, value in expected.items():
        assert np.allclose(signals[key], value, rtol=1e-12, atol=0), key
    assert not np.allclose(signals["r_A"], signals["d"])


def test_finite_differences_match_forward_mode_on_a_curved_map():
    """Unlike an untrained network, this map has an O(0.1) Jacobian, so the comparison has teeth."""
    def f(z):
        return 0.6 * torch.tanh(2 * gaussian_blur(z, 1.0)) + 0.2

    y = _images(4, 12, side=12)
    fd = jacobian_signals(f, y, probes=8, seed=3, method="fd")
    jvp = jacobian_signals(f, y, probes=8, seed=3, method="jvp")
    for key in ("div", "g_lin"):
        assert np.all(np.abs(jvp[key]) > 0.01), key
        assert np.allclose(fd[key], jvp[key], rtol=1e-3, atol=0), key
