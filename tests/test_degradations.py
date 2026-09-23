import math
from dataclasses import replace

import pytest
import torch

from fpr.degradations import (BoxMask, GaussianBlur, GaussianNoise, PixelMask, gaussian_blur,
                              gaussian_kernel1d)
from fpr.evaluation import CONDITIONS

DEGRADATIONS = [GaussianNoise(0.1), GaussianBlur(1.5), PixelMask(0.5), BoxMask(8)]


@pytest.fixture
def x():
    return torch.rand((16, 1, 28, 28), generator=torch.Generator().manual_seed(0),
                      dtype=torch.float64)


@pytest.mark.parametrize("degradation", DEGRADATIONS, ids=lambda d: d.label)
def test_seeded_and_linear(degradation, x):
    y1, A = degradation(x, torch.Generator().manual_seed(1))
    y2, _ = degradation(x, torch.Generator().manual_seed(1))
    assert y1.shape == x.shape
    assert torch.equal(y1, y2)
    u, v = torch.randn_like(x), torch.randn_like(x)
    assert torch.allclose(A(2 * u - 3 * v), 2 * A(u) - 3 * A(v))


@pytest.mark.parametrize("degradation", DEGRADATIONS, ids=lambda d: d.label)
def test_noise_free_observation_is_A_x(degradation, x):
    y, A = replace(degradation, sigma=0.0)(x, torch.Generator().manual_seed(2))
    assert torch.allclose(y, A(x))


def test_blur_operator_is_symmetric():
    gen = torch.Generator().manual_seed(3)
    u = torch.randn((1, 1, 12, 12), generator=gen, dtype=torch.float64)
    v = torch.randn((1, 1, 12, 12), generator=gen, dtype=torch.float64)
    assert torch.allclose((gaussian_blur(u, 1.0) * v).sum(), (u * gaussian_blur(v, 1.0)).sum())


def test_box_mask_removes_exactly_size_squared_pixels(x):
    _, A = BoxMask(8, sigma=0.0)(x, torch.Generator().manual_seed(4))
    removed = (A(torch.ones_like(x)) == 0).flatten(1).sum(1)
    assert (removed == 64).all()


def _batch(seed, n=64):
    return torch.rand((n, 1, 28, 28), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@pytest.mark.parametrize("drop", [0.25, 0.75])
def test_pixel_mask_drops_the_stated_share(drop):
    x = _batch(5)
    _, A = PixelMask(drop, sigma=0.0)(x, torch.Generator().manual_seed(6))
    share = (A(torch.ones_like(x)) == 0).double().mean().item()
    assert share == pytest.approx(drop, abs=0.01)


@pytest.mark.parametrize("degradation", [PixelMask(0.5), BoxMask(14)], ids=lambda d: d.label)
def test_masked_pixels_are_exactly_zero_despite_the_noise(degradation):
    """The weak noise is added before zeroing, so a masked pixel carries no signal at all."""
    x = _batch(7)
    y, A = degradation(x, torch.Generator().manual_seed(8))
    masked = A(torch.ones_like(x)) == 0
    assert masked.any() and (y[masked] == 0).all()


@pytest.mark.parametrize("degradation", [GaussianNoise(0.2), GaussianBlur(1.0), PixelMask(0.5)],
                         ids=lambda d: d.label)
def test_noise_has_the_stated_standard_deviation(degradation):
    x = _batch(9)
    y, A = degradation(x, torch.Generator().manual_seed(10))
    kept = A(torch.ones_like(x)) != 0
    assert (y - A(x))[kept].std().item() == pytest.approx(degradation.sigma, rel=0.02)


@pytest.mark.parametrize("std", [0.5, 1.0, 1.5, 2.0])
def test_blur_kernel_spans_three_standard_deviations(std):
    kernel = gaussian_kernel1d(std)
    assert kernel.numel() == 2 * max(1, math.ceil(3 * std)) + 1
    assert kernel.sum().item() == pytest.approx(1.0)


def test_corruption_grid_matches_the_report():
    assert [c.label for c in CONDITIONS] == [
        "noise(sigma=0.1)", "noise(sigma=0.2)", "noise(sigma=0.3)", "noise(sigma=0.5)",
        "blur(std=0.5)", "blur(std=1)", "blur(std=1.5)", "blur(std=2)",
        "pixel_mask(drop=0.25)", "pixel_mask(drop=0.5)", "pixel_mask(drop=0.75)",
        "box_mask(size=8)", "box_mask(size=14)"]
    assert all(c.sigma == 0.02 for c in CONDITIONS if c.family != "noise")
