"""Tests for multi-step residuals, ensemble disagreement, conformal bounds and selective risk."""

import numpy as np
import pandas as pd
import pytest
import torch

from fpr.conformal import (conformal_quantile, evaluate_signals, group_coverage, marginal_bound,
                           normalized_bound, shift_coverage, split_masks, sure_self_calibrate)
from fpr.ensemble import disagreement, seed_groups
from fpr.projectors import Identity
from fpr.selective import aurc, normalized_aurc, selective_risk, summarize
from fpr.signals import compute_signals, iterate_signals


def test_iterate_signals_vanish_for_the_identity():
    f = Identity()
    y = torch.randn(32, 1, 8, 8)
    out = iterate_signals(f, y, steps=3)
    assert np.allclose(out["g"], 0.0)
    assert np.allclose(out["g2"], 0.0)
    assert np.allclose(out["g3"], 0.0)


def test_residuals_vanish_for_an_exact_projector_but_displacement_does_not():
    """An orthogonal projection has g = g2 = g3 = 0 up to round-off, and q is undefined (NaN),
    while y itself is off the range, so d > 0."""
    basis, _ = torch.linalg.qr(torch.randn(64, 5, generator=torch.Generator().manual_seed(0),
                                           dtype=torch.float64))

    def f(z):
        return ((z.flatten(1) @ basis) @ basis.T).view_as(z)

    x = torch.randn(16, 1, 8, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    y = x + 0.1 * torch.randn(x.shape, generator=torch.Generator().manual_seed(2), dtype=x.dtype)
    signals = compute_signals(f, x, y, lambda z: z, iterate_steps=3)
    assert signals["g"].max() < 1e-12
    assert signals["g2"].max() < 1e-12
    assert signals["g3"].max() < 1e-12
    assert np.isnan(signals["q"]).all()
    assert signals["d"].min() > 0.1


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


def _levels(conditions, n_img=80, seed=0):
    """The same n_img images under each condition, with an informative signal g."""
    rng = np.random.default_rng(seed)
    frames = []
    for condition in conditions:
        g = rng.exponential(0.05, n_img)
        frames.append(pd.DataFrame({"image": np.arange(n_img), "condition": condition,
                                    "family": condition.split("(")[0],
                                    "g": g, "e": g * rng.lognormal(0.0, 0.3, n_img)}))
    return pd.concat(frames, ignore_index=True)


def test_evaluate_signals_fits_and_calibrates_on_separate_images():
    frame = _levels(["noise(sigma=0.1)", "blur(std=1)"])
    row = evaluate_signals(frame, signals=("g",)).query("setting == 'normalized'").iloc[0]
    fit, calibration, _ = split_masks(frame["image"])
    reference = normalized_bound(frame["g"], frame["e"], calibration, fit_mask=fit)
    for key in ("q_hat", "coverage", "width"):
        assert row[key] == pytest.approx(reference[key], rel=1e-12), key


def test_severity_bound_uses_the_fitting_images_level_means():
    """Fitting images have e = 1 and calibration images e = 2, so the level means from the
    fitting images give scores of 2; means from the calibration images would give 1."""
    frame = _levels(["noise(sigma=0.1)", "blur(std=1)"])
    frame["e"] = np.where(frame["image"] % 4 == 0, 1.0, np.where(frame["image"] % 4 == 2, 2.0, 1.5))
    row = evaluate_signals(frame, signals=("g",)).query("setting == 'severity'").iloc[0]
    assert row["q_hat"] == pytest.approx(2.0)
    assert row["coverage"] == 1.0 and row["width"] == pytest.approx(2.0)


def test_shift_coverage_deploys_only_held_out_images_of_other_levels():
    frame = _levels(["noise(sigma=0.1)", "noise(sigma=0.2)", "noise(sigma=0.3)", "blur(std=1)"])
    result = shift_coverage(frame, "g", alpha=0.1)
    assert (result["n_fit"], result["n_cal"], result["n_deploy"]) == (40, 40, 80)
    expected = (frame["condition"].isin(["noise(sigma=0.3)", "blur(std=1)"])
                & (frame["image"] % 2 == 1)).to_numpy()
    assert np.array_equal(result["deploy_mask"], expected)
    e_hi, error = result["e_hi"], frame["e"].to_numpy()
    assert result["coverage"] == pytest.approx(np.mean(error[expected] <= e_hi[expected]))


def test_group_coverage_counts_only_held_out_rows():
    error = np.array([0.1, 0.1, 5.0, 5.0, 0.1, 5.0])
    e_hi = np.ones(6)
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    held = np.array([True, True, False, False, True, False])  # every miss is not held out
    table = group_coverage(error, e_hi, groups, held).set_index("group")
    assert table.loc["a", "coverage"] == 1.0 and table.loc["a", "n"] == 2
    assert table.loc["b", "coverage"] == 1.0 and table.loc["b", "n"] == 1


def test_sure_self_calibration_never_reads_the_clean_error_of_its_own_images():
    """Corrupting e_mse on the fitting and calibration images must not change the bound."""
    rng = np.random.default_rng(7)
    n = 400
    g = rng.exponential(0.05, n)
    e_mse = 2.0 * g + rng.exponential(0.01, n)
    frame = pd.DataFrame({"image": np.tile(np.arange(100), 4), "family": "noise",
                          "condition": np.repeat([f"noise(sigma={s})" for s in (0.1, 0.2, 0.3, 0.5)],
                                                 100),
                          "e": np.sqrt(e_mse), "e_mse": e_mse,
                          "sure": e_mse + rng.normal(0.0, 0.01, n), "g": g})
    clean = sure_self_calibrate(frame, signal="g")
    poisoned = frame.copy()
    poisoned.loc[poisoned["image"] % 2 == 0, "e_mse"] = 1e6
    corrupted = sure_self_calibrate(poisoned, signal="g")
    for key in ("q_hat", "width", "coverage"):
        assert corrupted[key] == clean[key], key
    assert corrupted["width_supervised"] != clean["width_supervised"]  # that one does use e_mse


def test_aurc_is_the_mean_selective_risk_over_all_coverages():
    signal = np.array([0.1, 0.2, 0.3, 0.4])
    error = np.array([1.0, 2.0, 3.0, 4.0])
    # accepted means at k = 1..4: 1, 1.5, 2, 2.5
    assert aurc(signal, error) == pytest.approx(1.75)
    assert selective_risk(signal, error, coverage=0.5) == pytest.approx(1.5)
    assert normalized_aurc(signal, error) == pytest.approx(0.0)
    # reversed: 4, 3.5, 3, 2.5 -> 3.25; (3.25 - 1.75) / (2.5 - 1.75) = 2
    assert aurc(-signal, error) == pytest.approx(3.25)
    assert normalized_aurc(-signal, error) == pytest.approx(2.0)
    # On a curved risk-coverage curve a trapezoid over the coverage grid is not exact:
    # accepted means 4, 2, 4/3, 1 -> 25/12
    assert aurc(signal, np.array([4.0, 0.0, 0.0, 0.0])) == pytest.approx(25 / 12)


def test_disagreement_of_three_members():
    """Constant outputs 0, 1 and 2: the mean is 1 and the mean absolute deviation 2/3."""
    y = torch.zeros(5, 1, 4, 4)
    members = [lambda z, c=c: torch.full_like(z, c) for c in (0.0, 1.0, 2.0)]
    assert np.allclose(disagreement(members, y), 2.0 / 3.0)
