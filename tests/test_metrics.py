import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from fpr.metrics import _Resamples, _TieGroups, bootstrap_counts, rank_metrics


def test_point_estimates_match_scipy_and_sklearn():
    rng = np.random.default_rng(0)
    e = rng.gamma(2.0, size=500)
    s = e + rng.normal(size=500)
    s[:100] = np.round(s[:100])  # ties
    row = rank_metrics({"s": s}, e, n_boot=0)[0]
    labels = e >= np.quantile(e, 0.75)
    assert row["spearman"] == pytest.approx(spearmanr(s, e).statistic, abs=1e-12)
    assert row["auroc"] == pytest.approx(roc_auc_score(labels, s), abs=1e-12)


def test_weighted_ranks_equal_explicit_resamples():
    rng = np.random.default_rng(1)
    n = 200
    e = rng.normal(size=n)
    s = np.round(e + rng.normal(size=n), 1)  # many ties
    labels = (e >= np.quantile(e, 0.75)).astype(float)[None, :]
    z = np.round(e + rng.normal(size=n), 1)
    counts = bootstrap_counts(n, 5, rng)
    resamples = _Resamples(counts.astype(float), _TieGroups(e), labels, _TieGroups(z))
    rho, auc, partial = resamples.score(_TieGroups(s))
    for b in range(len(counts)):
        idx = np.repeat(np.arange(n), counts[b])
        assert rho[b] == pytest.approx(spearmanr(s[idx], e[idx]).statistic, abs=1e-10)
        assert auc[b] == pytest.approx(roc_auc_score(labels[0, idx] > 0, s[idx]), abs=1e-10)
        assert partial[b] == pytest.approx(_partial_by_residuals(s[idx], e[idx], z[idx]), abs=1e-10)


def _partial_by_residuals(s, e, z):
    """Partial Spearman as the correlation of rank residuals after regressing out rank(z)."""
    rs, re, rz = rankdata(s), rankdata(e), rankdata(z)
    design = np.c_[np.ones_like(rz), rz]

    def residual(v):
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return np.corrcoef(residual(rs), residual(re))[0, 1]


def test_partial_spearman_matches_residualization():
    rng = np.random.default_rng(8)
    z = rng.normal(size=400)
    e = z + rng.normal(size=400)
    s = np.round(0.5 * z + e + rng.normal(size=400), 1)
    row = rank_metrics({"s": s}, e, control=z, n_boot=100)[0]
    assert row["partial"] == pytest.approx(_partial_by_residuals(s, e, z), abs=1e-12)
    assert row["partial_lo"] <= row["partial"] <= row["partial_hi"]


def test_partial_spearman_removes_a_confound():
    """A signal that only tracks the control has rho > 0 with e but partial rho near 0."""
    rng = np.random.default_rng(9)
    z = rng.normal(size=5000)
    e = z + 0.5 * rng.normal(size=5000)
    rows = rank_metrics({"proxy": z + 0.1 * rng.normal(size=5000), "control": z}, e,
                        control=z, n_boot=0)
    proxy, control = rows
    assert proxy["spearman"] > 0.8
    assert abs(proxy["partial"]) < 0.05
    assert np.isnan(control["partial"])


def test_no_control_keeps_the_output_schema():
    rng = np.random.default_rng(10)
    e = rng.normal(size=100)
    row = rank_metrics({"s": e + rng.normal(size=100)}, e, n_boot=10)[0]
    assert list(row) == ["signal", "constant", "spearman", "spearman_lo", "spearman_hi",
                         "auroc", "auroc_lo", "auroc_hi", "n"]


def test_no_tie_fast_path_is_bitwise_identical():
    rng = np.random.default_rng(7)
    values = rng.normal(size=500)
    weights = bootstrap_counts(500, 20, rng).astype(float)
    groups = _TieGroups(values)
    assert not groups.has_ties
    fast = groups.ranks(weights)
    groups.has_ties = True
    assert np.array_equal(fast, groups.ranks(weights))


def test_roundoff_sized_signal_is_treated_as_constant():
    rng = np.random.default_rng(2)
    e = rng.normal(size=300)
    g = 1e-17 * rng.normal(size=300)
    row = rank_metrics({"g": g}, e, n_boot=50)[0]
    assert row["constant"]
    assert np.isnan(row["spearman"])
    assert row["auroc"] == pytest.approx(0.5)


def test_bootstrap_counts_rows_sum_to_n():
    counts = bootstrap_counts(37, 11, np.random.default_rng(3))
    assert counts.shape == (11, 37)
    assert (counts.sum(1) == 37).all()


def test_cluster_bootstrap_ignores_duplicated_rows():
    """Duplicating every image must not shrink the CI when clusters are declared."""
    rng = np.random.default_rng(4)
    n = 400
    e = rng.normal(size=n)
    s = e + 1.5 * rng.normal(size=n)

    def width(signal, error, clusters=None):
        row = rank_metrics({"s": signal}, error, clusters=clusters, n_boot=2000, seed=5)[0]
        return row["spearman_hi"] - row["spearman_lo"], row["spearman"]

    base, rho = width(s, e)
    dup_s, dup_e, dup_c = np.tile(s, 2), np.tile(e, 2), np.tile(np.arange(n), 2)
    clustered, rho_dup = width(dup_s, dup_e, dup_c)
    naive, _ = width(dup_s, dup_e)
    assert rho_dup == pytest.approx(rho, abs=1e-12)
    assert clustered == pytest.approx(base, rel=0.2)
    assert naive < 0.85 * base


def test_chunked_bootstrap_gives_ordered_intervals():
    rng = np.random.default_rng(6)
    e = rng.normal(size=1000)
    rows = rank_metrics({"a": e + rng.normal(size=1000), "b": rng.normal(size=1000)}, e,
                        n_boot=300, max_cells=50_000)
    for row in rows:
        assert row["spearman_lo"] <= row["spearman"] <= row["spearman_hi"]
        assert row["auroc_lo"] <= row["auroc"] <= row["auroc_hi"]
