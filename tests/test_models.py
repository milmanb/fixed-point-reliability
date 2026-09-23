import importlib.util

import pytest
import torch

from fpr.data import REPO_ROOT
from fpr.models import ARCHITECTURES, ConvAutoencoder, Restorer, load_restorer, save_checkpoint
from fpr.signals import jacobian_signals


@pytest.mark.parametrize("architecture, count", [("bottleneck", 602_049), ("skip", 230_497)])
def test_architectures_have_the_reported_size_shape_and_range(architecture, count):
    net = ARCHITECTURES[architecture]()
    assert sum(p.numel() for p in net.parameters()) == count
    y = torch.randn(4, 1, 28, 28)
    out = net(y)
    assert out.shape == y.shape
    assert out.min() >= 0 and out.max() <= 1


def test_restorer_keeps_dtype_and_supports_forward_mode():
    torch.manual_seed(0)
    restorer = Restorer(ConvAutoencoder(width=8, latent=8), name="tiny", device="cpu")
    y = torch.rand(3, 1, 28, 28, dtype=torch.float64)
    assert restorer(y).dtype == torch.float64
    for method in ("fd", "jvp"):
        out = jacobian_signals(restorer, y, probes=2, batch_size=2, method=method)
        assert out["div"].shape == (3,) and out["g_lin"].shape == (3,)


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_checkpoint_round_trip(architecture, tmp_path):
    torch.manual_seed(1)
    net = ARCHITECTURES[architecture]().eval()
    path = tmp_path / f"{architecture}_test_seed0.pt"
    save_checkpoint(path, net, {"lambda_id": 0.0}, [])
    restorer, checkpoint = load_restorer(path, device="cpu")
    y = torch.rand(2, 1, 28, 28)
    assert checkpoint["architecture"] == architecture and restorer.name == path.stem
    with torch.no_grad():
        assert torch.equal(restorer(y), net(y))


def test_checkpoints_without_an_architecture_load_as_the_bottleneck_model(tmp_path):
    net = ConvAutoencoder()
    path = tmp_path / "old_seed0.pt"
    torch.save({"model_config": net.config, "state_dict": net.state_dict(), "train_args": {},
                "history": []}, path)
    restorer, _ = load_restorer(path, device="cpu")
    assert isinstance(restorer.net, ConvAutoencoder)


def test_warm_up_ramps_linearly_then_holds():
    spec = importlib.util.spec_from_file_location("train_dae", REPO_ROOT / "scripts" / "train_dae.py")
    train_dae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_dae)
    lambda_at = train_dae.lambda_at
    assert lambda_at(0, 1.0, 1075) == 0.0
    assert lambda_at(537, 1.0, 1075) == pytest.approx(537 / 1075)
    assert lambda_at(1075, 1.0, 1075) == 1.0 and lambda_at(5000, 1.0, 1075) == 1.0
    assert lambda_at(0, 1.0, 0) == 1.0  # no warm-up: full weight from the first step
