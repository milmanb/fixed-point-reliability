import torch

from fpr.models import ConvAutoencoder, Restorer
from fpr.signals import jacobian_signals


def test_autoencoder_size_shape_and_range():
    net = ConvAutoencoder()
    assert sum(p.numel() for p in net.parameters()) < 1_000_000
    y = torch.randn(4, 1, 28, 28)
    out = net(y)
    assert out.shape == y.shape
    assert out.min() >= 0 and out.max() <= 1


def test_restorer_keeps_dtype_and_supports_jvp():
    torch.manual_seed(0)
    restorer = Restorer(ConvAutoencoder(width=8, latent=8), name="tiny", device="cpu")
    y = torch.rand(3, 1, 28, 28, dtype=torch.float64)
    assert restorer(y).dtype == torch.float64
    out = jacobian_signals(restorer, y, probes=2, batch_size=2)
    assert out["div"].shape == (3,) and out["g_lin"].shape == (3,)
