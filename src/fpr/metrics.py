"""Ranking metrics for reliability signals, with image-level bootstrap CIs.

A signal s is useful when it orders images like the true error e. We report
  * Spearman rank correlation rho(s, e);
  * AUROC of s for detecting the worst-error quartile, e >= q_0.75(e).

Bootstrap. Resampling n items with replacement is the same as drawing a count
vector w ~ Multinomial(n, uniform). Every statistic here is a weighted rank
statistic, so each resample is evaluated from w and one global sort, instead of
re-sorting every resample. All signals scored against the same error share the
same draws, so their CIs are paired.

Pooled analyses (several corruption levels per image) pass `clusters`: rows
that come from the same test image are resampled together.
"""

import numpy as np


class _TieGroups:
    """Precomputed sort order and tie structure of a 1-D array."""

    def __init__(self, values):
        self.order = np.argsort(values, kind="stable")
        sorted_values = values[self.order]
        new_group = np.ones(len(values), dtype=bool)
        new_group[1:] = sorted_values[1:] != sorted_values[:-1]
        self.starts = np.flatnonzero(new_group)
        self.group_of_sorted = np.cumsum(new_group) - 1
        self.has_ties = len(self.starts) < len(values)

    def ranks(self, weights):
        """Average rank of every item within each weighted resample (rows of `weights`).

        A tie group holding T copies after `before` smaller copies occupies
        positions before+1 .. before+T, so each copy gets before + (T + 1) / 2.
        """
        w_sorted = weights[:, self.order]
        ranks = np.empty(weights.shape, dtype=np.float64)
        if not self.has_ties:
            # Groups of one item: before + (T + 1) / 2 = cumsum - (w - 1) / 2, exactly.
            ranks[:, self.order] = np.cumsum(w_sorted, axis=1) - (w_sorted - 1.0) / 2.0
            return ranks
        totals = np.add.reduceat(w_sorted, self.starts, axis=1)
        before = np.cumsum(totals, axis=1) - totals
        ranks[:, self.order] = (before + (totals + 1.0) / 2.0)[:, self.group_of_sorted]
        return ranks


class _Resamples:
    """Weights and error-side quantities shared by every signal scored on the same draws."""

    def __init__(self, weights, error_groups, labels):
        self.w = weights
        self.total = weights.sum(1, keepdims=True)
        error_ranks = error_groups.ranks(weights)
        self.b = error_ranks - (weights * error_ranks).sum(1, keepdims=True) / self.total
        self.wbb = (weights * self.b * self.b).sum(1)
        self.w_labels = weights * labels
        self.n_pos = self.w_labels.sum(1)
        self.n_neg = weights.sum(1) - self.n_pos

    def score(self, groups):
        """Weighted Spearman rho with the error and Mann-Whitney AUROC for the worst quartile.

        In the AUROC, ties between a positive and a negative count as 1/2.
        """
        ranks = groups.ranks(self.w)
        a = ranks - (self.w * ranks).sum(1, keepdims=True) / self.total
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (self.w * a * self.b).sum(1) / np.sqrt((self.w * a * a).sum(1) * self.wbb)
            u = (self.w_labels * ranks).sum(1) - self.n_pos * (self.n_pos + 1.0) / 2.0
            auc = u / (self.n_pos * self.n_neg)
        return rho, auc


def bootstrap_counts(n_clusters, n_boot, rng):
    """Resample counts, shape (n_boot, n_clusters); each row sums to n_clusters."""
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    offsets = (np.arange(n_boot) * n_clusters)[:, None]
    counts = np.bincount((draws + offsets).ravel(), minlength=n_boot * n_clusters)
    return counts.reshape(n_boot, n_clusters)


def rank_metrics(signals, error, clusters=None, n_boot=1000, seed=0, quantile=0.75,
                 atol=1e-9, max_cells=2e7):
    """Spearman rho and worst-quartile AUROC for each signal, with 95% percentile CIs.

    signals : dict name -> (n,) array
    error   : (n,) array, the true error e
    clusters: optional (n,) int array in [0, C); rows sharing a cluster are resampled together
    atol    : a signal whose range is below atol is treated as exactly constant. This keeps
              round-off (e.g. g ~ 1e-17 for exact projectors) from being ranked as if it were data.

    Returns a list of dicts, one per signal.
    """
    error = np.asarray(error, dtype=np.float64)
    n = error.shape[0]
    labels = (error >= np.quantile(error, quantile)).astype(np.float64)[None, :]
    n_clusters = n if clusters is None else int(np.max(clusters)) + 1

    prepared = {}
    for name, values in signals.items():
        values = np.asarray(values, dtype=np.float64)
        constant = bool(np.ptp(values) <= atol)
        prepared[name] = (_TieGroups(np.zeros(n) if constant else values), constant)
    error_groups = _TieGroups(error)

    def evaluate(weights):
        resamples = _Resamples(weights, error_groups, labels)
        return {name: resamples.score(groups) for name, (groups, _) in prepared.items()}

    point = evaluate(np.ones((1, n)))
    boot = {name: ([], []) for name in prepared}
    rng = np.random.default_rng(seed)
    per_chunk = max(1, int(max_cells // n))
    for start in range(0, n_boot, per_chunk):
        counts = bootstrap_counts(n_clusters, min(per_chunk, n_boot - start), rng)
        weights = counts if clusters is None else counts[:, clusters]
        for name, (rho, auc) in evaluate(weights.astype(np.float64)).items():
            boot[name][0].append(rho)
            boot[name][1].append(auc)

    rows = []
    for name, (_, constant) in prepared.items():
        rho_boot = np.concatenate(boot[name][0]) if n_boot else np.array([np.nan])
        auc_boot = np.concatenate(boot[name][1]) if n_boot else np.array([np.nan])
        rows.append({
            "signal": name,
            "constant": constant,
            "spearman": float(point[name][0][0]),
            "spearman_lo": _percentile(rho_boot, 2.5),
            "spearman_hi": _percentile(rho_boot, 97.5),
            "auroc": float(point[name][1][0]),
            "auroc_lo": _percentile(auc_boot, 2.5),
            "auroc_hi": _percentile(auc_boot, 97.5),
            "n": n,
        })
    return rows


def _percentile(values, q):
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")
