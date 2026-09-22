"""Tests for multi-step residuals, ensemble disagreement, conformal bounds and selective risk."""

import numpy as np
import pandas as pd
import pytest
import torch

from fpr.conformal import (conformal_quantile, evaluate_signals, marginal_bound,
                           normalized_bound, split_masks, sure_self_calibrate)
from fpr.ensemble import disagreement, seed_groups
from fpr.projectors import Identity
from fpr.selective import aurc, normalized_aurc, selective_risk, summarize
from fpr.signals import compute_signals, iterate_signals


def test_iterate_signals_matches_g_for_one_step():
    f = Identity()
    y = torch.randn(32, 1, 8, 8)
    out = iterate_signals(f, y, steps=3)
    assert np.allclose(out["g"], 0.0)
    assert np.allclose(out["g2"], 0.0)
    assert np.allclose(out["g3"], 0.0)


def test_g2_equals_g_for_exact_projector():
    """An exact projector has g = g2 = 0; compute_signals agrees with iterate_signals."""
    f = Identity()
    x = torch.randn(16, 1, 8, 8)
    y = x + 0.1 * torch.randn_like(x)
    A = lambda z: z
    signals = compute_signals(f, x, y, A, iterate_steps=3)
    assert signals["g"].max() < 1e-12
    assert signals["g2"].max() < 1e-12
    assert signals["g3"].max() < 1e-12


def test_iterate_signals_on_contraction():
    """f(y) = 0.5 y has g_k = |f^{k+1}(y) - f^k(y)| = 0.5^(k+1) for y = 1, and q = 0.5."""
    def f(z):
        return 0.5 * z

    y = torch.ones(8, 1, 4, 4)
    out = iterate_signals(f, y, steps=3)
    assert np.allclose(out["g"], 0.25)  # |0.25 - 0.5|; |f(y) - y| = 0.5 is d, not g
    assert np.allclose(out["g2"], 0.125)
    assert np.allclose(out["g3"], 0.0625)
    assert np.allclose(out["q"], 0.5)


def test_iterate_signals_matches_compute_signals():
    """On a nonlinear map the two functions give the same g, g2, g3 and q."""
    def f(z):
        return z ** 2

    y = torch.rand(6, 1, 4, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ours = iterate_signals(f, y, steps=3)
    reference = compute_signals(f, torch.zeros_like(y), y, lambda z: z, iterate_steps=3)
    for key in ("g", "g2", "g3", "q"):
        assert np.allclose(ours[key], reference[key], rtol=1e-12, atol=0), key
    assert not np.allclose(reference["g"], reference["d"])
    with pytest.raises(ValueError):
        compute_signals(f, torch.zeros_like(y), y, lambda z: z, iterate_steps=0)


def test_disagreement_zero_for_identical_models():
    f = Identity()
    y = torch.randn(20, 1, 8, 8)
    values = disagreement([f, f, f], y)
    assert np.allclose(values, 0.0, atol=1e-7)


def test_disagreement_positive_for_distinct_models():
    def f1(z):
        return z

    def f2(z):
        return 2 * z

    y = torch.ones(10, 1, 4, 4)
    values = disagreement([f1, f2], y)
    # mean = 1.5 y, |1-1.5| and |2-1.5| each contribute 0.5; mean over models = 0.5
    assert np.allclose(values, 0.5)


def test_seed_groups_filters_singletons():
    models = {"dae_lam0_seed0": object(), "dae_lam0_seed1": object(),
              "dae_lam1_seed0": object(), "radial": object()}
    groups = seed_groups(models)
    assert set(groups) == {"dae_lam0"}
    assert len(groups["dae_lam0"]) == 2


def test_conformal_quantile_finite_sample():
    scores = np.arange(1, 101, dtype=float)
    # ceil((100 + 1) * 0.9) = 91, so the 91st smallest score
    assert conformal_quantile(scores, alpha=0.1) == 91
    assert conformal_quantile(np.arange(1, 6, dtype=float), alpha=0.1) == np.inf  # rank 6 > 5


def test_normalized_conformal_covers_with_separate_fit():
    """With u_hat fit on other images, held-out coverage meets 1 - alpha on average even for a
    small calibration set; fitting u_hat on the calibration images themselves under-covers."""
    rng = np.random.default_rng(0)
    image = np.arange(800)
    fit, calibration, held = split_masks(image)
    separate, shared = [], []
    for _ in range(300):
        signal = rng.exponential(1.0, 800)
        error = signal * rng.lognormal(0.0, 0.5, 800)  # informative but noisy
        separate.append(normalized_bound(signal, error, calibration, alpha=0.1,
                                         fit_mask=fit)["coverage"])
        shared.append(normalized_bound(signal, error, image % 2 == 0, alpha=0.1)["coverage"])
    assert np.mean(separate) >= 0.895
    assert np.mean(shared) < np.mean(separate)


def test_split_masks_partition_the_images():
    fit, calibration, held = split_masks(np.arange(12))
    assert not (fit & calibration).any() and not (fit & held).any() and not (calibration & held).any()
    assert (fit | calibration | held).all()
    assert held.sum() == 6 and fit.sum() == 3 and calibration.sum() == 3


def test_marginal_bound_covers():
    rng = np.random.default_rng(1)
    error = rng.exponential(0.1, size=1000)
    calibration = np.zeros(1000, dtype=bool)
    calibration[::2] = True
    result = marginal_bound(error, calibration, alpha=0.1)
    assert 0.85 <= result["coverage"] <= 0.98


def test_evaluate_signals_returns_baselines():
    rng = np.random.default_rng(2)
    n, n_img = 200, 50
    frame = pd.DataFrame({
        "image": np.tile(np.arange(n_img), n // n_img),
        "condition": np.repeat(["noise(sigma=0.1)", "noise(sigma=0.2)",
                                "blur(std=1)", "pixel_mask(drop=0.5)"], n // 4),
        "family": np.repeat(["noise", "noise", "blur", "pixel_mask"], n // 4),
        "e": rng.exponential(0.1, n),
        "g": rng.exponential(0.05, n),
        "d": rng.exponential(0.05, n),
        "r_A": rng.exponential(0.05, n),
    })
    table = evaluate_signals(frame, signals=("g",), alpha=0.1)
    settings = set(table["setting"])
    assert "normalized" in settings
    assert "marginal" in settings
    assert "severity" in settings


def test_sure_self_calibrate_covers_when_sure_tracks_e_mse():
    """SURE is a noisy but unbiased estimate of e_mse, so the bound it calibrates should
    cover e_mse near the nominal rate. It must never be scored against the L1 error e,
    which lives on a different scale."""
    rng = np.random.default_rng(3)
    n = 400
    signal = rng.exponential(0.05, n)
    e_mse = signal * 2.0 + rng.exponential(0.01, n)
    frame = pd.DataFrame({
        "image": np.tile(np.arange(100), 4),
        "family": "noise",
        "condition": np.repeat(["noise(sigma=0.1)", "noise(sigma=0.2)",
                                "noise(sigma=0.3)", "noise(sigma=0.5)"], 100),
        "e": np.sqrt(e_mse),
        "e_mse": e_mse,
        # unbiased for e_mse but noisy, and occasionally negative, as real SURE is
        "sure": e_mse + rng.normal(0.0, 0.01, n),
        "g": signal,
    })
    result = sure_self_calibrate(frame, signal="g", alpha=0.1)
    assert np.isfinite(result["coverage"]) and np.isfinite(result["width"])
    assert result["coverage"] > 0.7, result["coverage"]
    # the supervised bound on the same target is the reference point; 200 held-out
    # images leave a few points of sampling slack around the nominal 0.9
    assert abs(result["coverage_supervised"] - 0.9) < 0.1


def test_sure_self_calibrate_needs_e_mse():
    frame = pd.DataFrame({"image": np.arange(20), "family": "noise",
                          "e": np.linspace(0.1, 0.2, 20), "sure": np.linspace(0.1, 0.2, 20),
                          "g": np.linspace(0.01, 0.02, 20)})
    assert np.isnan(sure_self_calibrate(frame, signal="g")["coverage"])


def test_aurc_oracle_beats_random():
    rng = np.random.default_rng(4)
    error = rng.exponential(0.1, size=500)
    aurc_oracle = aurc(error, error)
    aurc_random = float(np.mean(error))
    assert aurc_oracle < aurc_random
    assert normalized_aurc(error, error) == pytest.approx(0.0, abs=1e-6)
    # A random permutation's AURC is near the mean-error baseline in expectation.
    perm = rng.permutation(error)
    assert abs(normalized_aurc(perm, error) - 1.0) < 0.15


def test_selective_risk_at_full_coverage_is_mean():
    rng = np.random.default_rng(5)
    error = rng.random(100)
    signal = rng.random(100)
    assert selective_risk(signal, error, coverage=1.0) == pytest.approx(error.mean())


def test_summarize_keys():
    rng = np.random.default_rng(6)
    error = rng.random(100)
    signal = error + 0.01 * rng.random(100)
    out = summarize(signal, error)
    assert "aurc" in out and "aurc_norm" in out
    assert "risk@0.8" in out and "risk@0.5" in out
