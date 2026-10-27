"""Tests for the natural-image check: BSD68 crops, the scaled corruption grid and the denoisers."""

import importlib.util
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fpr.data import REPO_ROOT, load_bsd68_crops
from fpr.degradations import GaussianBlur, gaussian_blur
from fpr.evaluation import CONDITIONS, natural_conditions, score
from fpr.pretrained import DnCNN, Denoiser, ReflectPadded, load_restormer


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bsd68_folder(folder, shape=(321, 481)):
    """68 random grey PNGs, landscape and portrait in turn, as in BSD68."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(68):
        size = shape if k % 2 == 0 else shape[::-1]
        Image.fromarray(rng.integers(0, 256, size, dtype=np.uint8)).save(folder / f"test{k + 1:03d}.png")


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
    assert torch.allclose(y, flat) and torch.allclose(A(flat), flat)  # the degradation reflects too
    gen = torch.Generator().manual_seed(1)
    u, v = (torch.randn((1, 1, 16, 16), generator=gen, dtype=torch.float64) for _ in range(2))
    assert torch.allclose(A(2 * u - 3 * v), 2 * A(u) - 3 * A(v))
    with pytest.raises(ValueError, match="padding"):
        gaussian_blur(flat, 1.0, padding="circular")


def test_bsd68_crops_follow_the_grid(tmp_path):
    _bsd68_folder(tmp_path)
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


def test_reflect_padding_mirrors_the_border_and_crops_back():
    padded = []

    class Stub(torch.nn.Module):
        def forward_restormer(self, z):
            padded.append(z)
            return 2 * z

    y = torch.rand(1, 1, 13, 18)
    assert torch.equal(ReflectPadded(Stub())(y), 2 * y)
    z = padded[0]
    assert z.shape[-2:] == (16, 24)  # rows and columns padded to multiples of 8, at the bottom and right
    # Reflection mirrors about the edge pixel; replication would repeat it.
    assert torch.equal(z[..., 13, :18], y[..., 11, :]) and torch.equal(z[..., :13, 18], y[..., :, 16])


def test_restormer_is_built_with_the_authors_configuration(tmp_path, monkeypatch):
    models = pytest.importorskip("deepinv.models")
    # Denoising/Options/GaussianGrayDenoising_Restormer.yml of the authors' repository.
    authors = models.Restormer(in_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8],
                               num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type="BiasFree", pretrained=None)
    path = tmp_path / "restormer.pth"
    torch.save({"params": authors.state_dict()}, path)
    net = load_restormer(path)  # a strict load: another configuration has other weights
    assert isinstance(net, ReflectPadded) and sum(p.numel() for p in net.parameters()) == 26_109_076
    monkeypatch.setitem(sys.modules, "deepinv.models", None)  # as if deepinv were not installed
    with pytest.raises(SystemExit, match="pip install"):
        load_restormer(path)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_denoiser_on_cuda_turns_off_tf32():
    previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        Denoiser(torch.nn.Identity(), "identity", "cuda")
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


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


def test_evaluation_script_bootstraps_source_images_and_sees_the_same_inputs(tmp_path, monkeypatch):
    evaluate = _script("evaluate_pretrained")
    crops = torch.rand((12, 1, 16, 16), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    seen = {"clip": []}

    def fake_crops(size, dtype):
        seen["size"], seen["dtype"] = size, dtype
        return crops.to(dtype), torch.arange(2).repeat_interleave(6)

    def fake_denoiser(name, device=None, batch_size=32, clip=True):
        seen["clip"].append(clip)
        net = torch.nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            net.weight.fill_(1 / 9)
            net.bias.zero_()
        return Denoiser(net, name, "cpu", batch_size=batch_size, clip=clip)

    def recording_score(*args, **kwargs):
        seen["score"] = kwargs
        return score(*args, **kwargs)

    monkeypatch.setattr(evaluate, "load_bsd68_crops", fake_crops)
    monkeypatch.setattr(evaluate, "load_denoiser", fake_denoiser)
    monkeypatch.setattr(evaluate, "score", recording_score)
    monkeypatch.setattr(evaluate, "sha256", lambda path: "0" * 64)
    for no_clip in (False, True):
        out = tmp_path / ("noclip" if no_clip else "clip")
        monkeypatch.setattr(sys, "argv", ["evaluate_pretrained.py", "--models", "dncnn_b", "--size", "16",
                                          "--n-boot", "20", "--device", "cpu", "--out", str(out)]
                            + ["--no-clip"] * no_clip)
        evaluate.main()
        info = json.loads((out / "run_info.json").read_text())
        assert info["args"]["no_clip"] is no_clip and seen["clip"][-1] is (not no_clip)
        assert info["crops"] == 12 and info["source_images"] == 2
        assert (out / "per_image.csv.gz").read_bytes()[4:8] == bytes(4)  # no gzip time stamp
    assert seen["dtype"] == torch.float64 and seen["size"] == 16
    assert seen["score"]["cluster"] == "source" and seen["score"]["cluster_levels"] is True
    clipped, raw = (pd.read_csv(tmp_path / name / "per_image.csv.gz") for name in ("clip", "noclip"))
    assert clipped["source"].tolist() == raw["source"].tolist()
    assert clipped["source"].value_counts().to_dict() == {0: 6 * 13, 1: 6 * 13}
    assert np.allclose(clipped["e_in"], raw["e_in"])  # both settings see the same observations


def test_psnr_check_adds_the_authors_noise_reseeded_for_every_image():
    check = _script("check_pretrained")
    image = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    state = np.random.get_state()
    try:
        np.random.seed(seed=0)  # as in KAIR's main_test_dncnn.py and Restormer's test script
        expected = image.copy()
        expected += np.random.normal(0, 25 / 255.0, image.shape)
    finally:
        np.random.set_state(state)
    first = check.noisy(image, 25)
    assert first.dtype == np.float32 and np.array_equal(first, expected)
    assert np.array_equal(check.noisy(image, 25), first)  # the same draw for every image


def test_psnr_check_clips_and_adds_rows_for_replication_padding(tmp_path, monkeypatch):
    check = _script("check_pretrained")
    _bsd68_folder(tmp_path / "data" / "bsd68", shape=(12, 16))

    class Stub(torch.nn.Module):  # Restormer's two entry points: forward (replication) and the network
        def forward(self, z):
            return 2 * z - 0.4

        def forward_restormer(self, z):
            return 2 * z - 0.5

    seen = {}

    def fake_denoiser(name, device=None, batch_size=32, clip=True):
        seen[name] = clip
        net = ReflectPadded(Stub()) if name == "restormer" else Stub()
        return Denoiser(net, name, "cpu", batch_size=batch_size, clip=clip)

    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "load_denoiser", fake_denoiser)
    monkeypatch.setattr(sys, "argv", ["check_pretrained.py", "--device", "cpu"])
    check.main()
    table = pd.read_csv(tmp_path / "results" / "pretrained" / "psnr.csv")
    assert seen == {"dncnn_b": True, "restormer": True}  # the papers score clipped outputs
    assert table["model"].tolist() == ["dncnn_b"] * 3 + ["restormer"] * 3 + ["restormer_replicate"] * 3
    assert table["published"].tolist()[6:] == table["published"].tolist()[3:6] == [31.95, 29.51, 26.62]
    psnr = table.set_index(["model", "sigma"])["psnr"]
    assert all(psnr["restormer_replicate", s] != psnr["restormer", s] for s in (15, 25, 50))  # deepinv's forward


PRETRAINED_FILES = [REPO_ROOT / "data" / "pretrained" / "dncnn" / "dncnn_gray_blind.pth",
                    REPO_ROOT / "data" / "pretrained" / "restormer" / "gaussian_gray_denoising_blind.pth",
                    REPO_ROOT / "data" / "bsd68" / "test001.png", REPO_ROOT / "results" / "pretrained" / "per_image.csv.gz"]


@pytest.mark.skipif(not all(p.exists() for p in PRETRAINED_FILES) or importlib.util.find_spec("deepinv") is None,
                    reason="needs the weights and BSD68 in data/, the BSD68 dump and deepinv")
def test_pretrained_denoisers_reproduce_a_row_of_the_dump():
    """The real weights, the evaluation's observations and the stored dump agree for the first crop."""
    from fpr.evaluation import observe
    from fpr.pretrained import MODELS, load_denoiser

    x, _ = load_bsd68_crops(dtype=torch.float64)
    dump = pd.read_csv(REPO_ROOT / "results" / "pretrained" / "per_image.csv.gz", usecols=["condition", "model", "image", "e", "g"])
    for condition in (natural_conditions(128)[1], natural_conditions(128)[-2]):  # noise 0.2 and the side-37 box
        y = observe(condition, x, 0)[0][:1]  # drawn for all crops, as in the evaluation
        for name in MODELS:
            denoiser = load_denoiser(name, device="cpu", batch_size=1)
            fy = denoiser(y)
            e, g = (fy - x[:1]).abs().mean().item(), (denoiser(fy) - fy).abs().mean().item()
            row = dump[(dump["model"] == name) & (dump["condition"] == condition.label) & (dump["image"] == 0)].iloc[0]
            assert e == pytest.approx(row["e"], rel=1e-4) and g == pytest.approx(row["g"], rel=1e-3)


def test_psnr_rounds_both_images_to_eight_bits():
    check = _script("check_pretrained")
    clean = np.full((4, 4), 100 / 255)
    expected = 10 * np.log10(255.0 ** 2 / 100)  # every pixel off by 10 grey levels
    assert check.psnr(clean + 10 / 255, clean) == pytest.approx(expected)
    assert check.psnr(clean + 10.4 / 255, clean) == pytest.approx(expected)  # rounds to 10


def test_download_script_pins_every_file():
    module = _script("download_pretrained")
    lines = [line.split() for line in module.BSD68.split("\n") if line.strip()]
    assert [name for _, name in lines] == [f"test{k:03d}.png" for k in range(1, 69)]
    digests = [digest for digest, _ in lines] + [d for _, d in module.WEIGHTS.values()]
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests) and len(set(digests)) == 70
    assert all(url.startswith("https://") for url, _ in module.WEIGHTS.values())
