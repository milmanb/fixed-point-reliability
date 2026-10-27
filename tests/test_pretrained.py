"""Tests for the natural-image check: BSD68 crops, the scaled corruption grid and the denoisers."""

import importlib.util
import re

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fpr.data import REPO_ROOT, load_bsd68_crops
from fpr.degradations import GaussianBlur, gaussian_blur
from fpr.evaluation import CONDITIONS, natural_conditions, score
from fpr.pretrained import DnCNN, Denoiser, ReflectPadded


def test_natural_grid_scales_only_the_boxes():
    natural = natural_conditions(128)
    assert len(natural) == len(CONDITIONS) == 13
    for ours, theirs in zip(CONDITIONS, natural, strict=True):
        assert ours.family == theirs.family
        if ours.family == "box_mask":
            assert theirs.size == round(ours.size * 128 / 28)
        else:
            assert ours.label == theirs.label
    assert [c.size for c in natural if c.family == "box_mask"] == [37, 64]
    assert all(c.padding == "reflect" for c in natural if c.family == "blur")


def test_reflected_blur_keeps_a_flat_image_flat_and_stays_linear():
    flat = torch.full((1, 1, 16, 16), 0.6, dtype=torch.float64)
    assert torch.allclose(gaussian_blur(flat, 2.0, padding="reflect"), flat)
    assert gaussian_blur(flat, 2.0)[0, 0, 0, 0] < 0.5  # zero padding darkens the border
    y, A = GaussianBlur(1.5, sigma=0.0, padding="reflect")(flat, torch.Generator().manual_seed(0))
    assert torch.allclose(y, A(flat))
    gen = torch.Generator().manual_seed(1)
    u, v = (torch.randn((1, 1, 16, 16), generator=gen, dtype=torch.float64) for _ in range(2))
    assert torch.allclose(A(2 * u - 3 * v), 2 * A(u) - 3 * A(v))
    with pytest.raises(ValueError, match="padding"):
        gaussian_blur(flat, 1.0, padding="circular")


def test_bsd68_crops_follow_the_grid(tmp_path):
    rng = np.random.default_rng(0)
    for k in range(68):  # landscape and portrait, as in BSD68 (481 x 321)
        shape = (321, 481) if k % 2 == 0 else (481, 321)
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(tmp_path / f"test{k + 1:03d}.png")
    x, sources = load_bsd68_crops(tmp_path, size=128, dtype=torch.float64)
    assert x.shape == (408, 1, 128, 128) and sources.tolist() == [k for k in range(68) for _ in range(6)]
    first = np.asarray(Image.open(tmp_path / "test001.png"), dtype=np.float64) / 255.0
    top, left = (321 - 256) // 2, (481 - 384) // 2  # centred 2 x 3 grid of a landscape image
    assert np.allclose(x[0, 0].numpy(), first[top:top + 128, left:left + 128])
    assert np.allclose(x[5, 0].numpy(), first[top + 128:top + 256, left + 256:left + 384])
    (tmp_path / "test068.png").unlink()
    with pytest.raises(SystemExit, match="67 of the 68"):
        load_bsd68_crops(tmp_path)


def test_dncnn_matches_kair_layer_names_and_is_residual():
    net = DnCNN()
    names = [k for k in net.state_dict() if k.endswith("weight")]
    assert names == [f"model.{2 * i}.weight" for i in range(20)]
    assert sum(p.numel() for p in net.parameters()) == 665_921  # 0.67M, batch norm merged
    with torch.no_grad():
        net.model[-1].weight.zero_()
        net.model[-1].bias.zero_()
    y = torch.rand(2, 1, 9, 9)
    assert torch.equal(net(y), y)  # a zero residual branch returns its input


def test_reflect_padding_crops_back_to_the_input_size():
    class Stub(torch.nn.Module):
        def forward_restormer(self, z):
            assert z.shape[-2] % 8 == 0 and z.shape[-1] % 8 == 0
            return 2 * z

    y = torch.rand(1, 1, 13, 21)
    assert torch.equal(ReflectPadded(Stub())(y), 2 * y)


def test_denoiser_clips_keeps_dtype_and_runs_in_chunks():
    calls = []

    class Spread(torch.nn.Module):
        def forward(self, z):
            calls.append(len(z))
            return 3 * z - 1

    y = torch.rand(10, 1, 4, 4, dtype=torch.float64)
    out = Denoiser(Spread(), "spread", "cpu", batch_size=4)(y)
    assert calls == [4, 4, 2] and out.dtype == torch.float64
    assert out.min() >= 0 and out.max() <= 1
    assert torch.allclose(out, (3 * y - 1).clamp(0, 1), atol=1e-6)
    raw = Denoiser(Spread(), "spread", "cpu", batch_size=4, clip=False)(y)
    assert torch.allclose(raw, 3 * y - 1, atol=1e-6) and raw.min() < 0  # no clip: raw outputs


def test_bootstrap_by_source_widens_the_interval_of_correlated_crops():
    rng = np.random.default_rng(0)
    sources = np.repeat(np.arange(20), 6)
    # Crops of one image share most of their error and of the signal's noise.
    error = rng.normal(size=20)[sources] + 0.1 * rng.normal(size=120)
    signal = error + rng.normal(size=20)[sources] + 0.1 * rng.normal(size=120)
    frame = pd.DataFrame({"condition": "noise(sigma=0.1)", "family": "noise", "model": "m",
                          "image": np.arange(120), "source": sources, "e": error, "g": signal})
    rows = [score(frame, ("g",), n_boot=400, **options).query("scope == 'condition'").iloc[0]
            for options in ({}, {"cluster": "source", "cluster_levels": True})]
    plain, by_source = (row["spearman_hi"] - row["spearman_lo"] for row in rows)
    assert rows[0]["spearman"] == pytest.approx(rows[1]["spearman"])
    assert by_source > 1.5 * plain


def test_psnr_rounds_both_images_to_eight_bits():
    spec = importlib.util.spec_from_file_location(
        "check_pretrained", REPO_ROOT / "scripts" / "check_pretrained.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean = np.full((4, 4), 100 / 255)
    expected = 10 * np.log10(255.0 ** 2 / 100)  # every pixel off by 10 grey levels
    assert module.psnr(clean + 10 / 255, clean) == pytest.approx(expected)
    assert module.psnr(clean + 10.4 / 255, clean) == pytest.approx(expected)  # rounds to 10


def test_download_script_pins_every_file():
    spec = importlib.util.spec_from_file_location(
        "download_pretrained", REPO_ROOT / "scripts" / "download_pretrained.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = [line.split() for line in module.BSD68.split("\n") if line.strip()]
    assert [name for _, name in lines] == [f"test{k:03d}.png" for k in range(1, 69)]
    digests = [digest for digest, _ in lines] + [d for _, d in module.WEIGHTS.values()]
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests) and len(set(digests)) == 70
    assert all(url.startswith("https://") for url, _ in module.WEIGHTS.values())
