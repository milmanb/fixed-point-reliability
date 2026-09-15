"""Checks that any of the three gradient routings in fpr.losses.idempotence_loss must pass.

The tests skip while the function is still a TODO.
"""

import pytest
import torch
from torch import nn

from fpr.losses import idempotence_loss


class Projection(nn.Module):
    """Orthogonal projection onto a 5-dimensional subspace, so f(f(y)) = f(y) exactly."""

    def __init__(self):
        super().__init__()
        basis, _ = torch.linalg.qr(torch.randn(36, 5, generator=torch.Generator().manual_seed(0),
                                               dtype=torch.float64))
        self.basis = nn.Parameter(basis)

    def forward(self, y):
        v = y.flatten(1)
        return ((v @ self.basis) @ self.basis.T).view_as(y)


class Shrink(nn.Module):
    """f(y) = 0.3 y, so |f(f(y)) - f(y)| = 0.21 |y|."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

    def forward(self, y):
        return self.scale * y


def _loss(model, y):
    try:
        return idempotence_loss(model, model(y))
    except NotImplementedError:
        pytest.skip("fpr.losses.idempotence_loss is not implemented yet")


def _images():
    return torch.rand((4, 1, 6, 6), generator=torch.Generator().manual_seed(1), dtype=torch.float64)


def test_zero_for_an_idempotent_model():
    loss = _loss(Projection(), _images())
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_value_and_gradient_for_a_non_idempotent_model():
    model, y = Shrink(), _images()
    loss = _loss(model, y)
    assert loss.item() == pytest.approx(0.21 * y.abs().mean().item(), rel=1e-12)
    loss.backward()
    assert model.scale.grad is not None and model.scale.grad.abs().item() > 0
