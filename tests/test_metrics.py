import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fpr.metrics import (_TieGroups, _weighted_auroc, _weighted_pearson, bootstrap_counts,
                         rank_metrics)


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
    counts = bootstrap_counts(n, 5, rng)
    w = counts.astype(float)
    s_ranks = _TieGroups(s).ranks(w)
    rho = _weighted_pearson(s_ranks, _TieGroups(e).ranks(w), w)
    auc = _weighted_auroc(s_ranks, labels, w)
    for b in range(len(counts)):
        idx = np.repeat(np.arange(n), counts[b])
        assert rho[b] == pytest.approx(spearmanr(s[idx], e[idx]).statistic, abs=1e-10)
        assert auc[b] == pytest.approx(roc_auc_score(labels[0, idx] > 0, s[idx]), abs=1e-10)


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
