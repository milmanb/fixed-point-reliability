"""Checks that any of the three gradient routings in fpr.losses.idempotence_loss must pass."""

import pytest
import torch
from torch import nn

from fpr.losses import idempotence_loss
from fpr.models import ConvAutoencoder


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
    return idempotence_loss(model, model(y))


def _images():
    return torch.rand((4, 1, 6, 6), generator=torch.Generator().manual_seed(1), dtype=torch.float64)


def test_zero_for_an_idempotent_model():
    loss = _loss(Projection(), _images())
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_routings_share_the_value_and_split_the_gradient():
    """"both" routes gradients through both applications, so its gradient is "inner" + "outer"."""
    torch.manual_seed(0)
    model = ConvAutoencoder(width=4, latent=8).double()
    y = torch.rand((4, 1, 28, 28), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    gradients, values = {}, {}
    for routing in ("both", "inner", "outer"):
        loss = idempotence_loss(model, model(y), routing=routing)
        model.zero_grad(set_to_none=True)
        loss.backward()
        values[routing] = loss.item()
        gradients[routing] = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert values["inner"] == pytest.approx(values["both"], rel=1e-12)
    assert values["outer"] == pytest.approx(values["both"], rel=1e-12)
    assert torch.allclose(gradients["both"], gradients["inner"] + gradients["outer"],
                          rtol=1e-9, atol=1e-12)


def test_value_and_gradient_for_a_non_idempotent_model():
    model, y = Shrink(), _images()
    loss = _loss(model, y)
    assert loss.item() == pytest.approx(0.21 * y.abs().mean().item(), rel=1e-12)
    loss.backward()
    assert model.scale.grad is not None and model.scale.grad.abs().item() > 0
