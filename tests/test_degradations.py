from dataclasses import replace

import pytest
import torch

from fpr.degradations import BoxMask, GaussianBlur, GaussianNoise, PixelMask, gaussian_blur

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
