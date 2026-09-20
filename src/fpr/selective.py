"""Risk-coverage curves and selective risk for reference-free reliability signals.

A practitioner who rejects the top fraction of a signal and keeps the rest cares about
the mean error of the accepted set. The risk-coverage curve plots that mean against the
kept fraction (coverage). The area under it (AURC) summarizes the curve; selective risk
at a fixed coverage (e.g. 80%) is the operational number.

Normalization: report (aurc - aurc_oracle) / (aurc_random - aurc_oracle), so 0 is perfect
ranking by true error and 1 is a constant score (cf. selective conformal risk control,
arXiv:2512.12844).
"""

import numpy as np


def risk_coverage_curve(signal, error, n_points=50):
    """Mean error of the accepted set vs. fraction kept, rejecting high signal first.

    Returns coverage in (0, 1] and the corresponding selective risk (mean e of accepted).
    Coverage 1 keeps everything; coverage near 0 keeps only the lowest-signal images.
    """
    signal = np.asarray(signal, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    n = len(error)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(signal, kind="stable")  # lowest signal first = most trusted
    error_sorted = error[order]
    cumsum = np.cumsum(error_sorted)
    # Keep the first k images (lowest signal); coverage = k/n
    ks = np.unique(np.round(np.linspace(1, n, n_points)).astype(int))
    coverage = ks / n
    risk = cumsum[ks - 1] / ks
    return coverage, risk


def aurc(signal, error, n_points=200):
    """Trapezoidal area under the risk-coverage curve."""
    coverage, risk = risk_coverage_curve(signal, error, n_points=n_points)
    if len(coverage) < 2:
        return float("nan")
    return float(np.trapz(risk, coverage))


def selective_risk(signal, error, coverage=0.8):
    """Mean error among the lowest-signal fraction `coverage` of images."""
    signal = np.asarray(signal, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    n = len(error)
    k = max(1, int(round(coverage * n)))
    order = np.argsort(signal, kind="stable")
    return float(error[order[:k]].mean())


def normalized_aurc(signal, error, n_points=200):
    """AURC scaled so 0 = oracle (rank by e) and 1 = random (constant score)."""
    aurc_s = aurc(signal, error, n_points=n_points)
    aurc_oracle = aurc(error, error, n_points=n_points)
    aurc_random = float(np.mean(error))  # constant score: risk = mean e at every coverage
    denom = aurc_random - aurc_oracle
    if abs(denom) < 1e-15:
        return float("nan")
    return float((aurc_s - aurc_oracle) / denom)


def summarize(signal, error, coverages=(0.8, 0.5), n_points=200):
    """Dict with AURC, normalized AURC, and selective risk at each coverage."""
    out = {
        "aurc": aurc(signal, error, n_points=n_points),
        "aurc_norm": normalized_aurc(signal, error, n_points=n_points),
        "mean_error": float(np.mean(error)),
    }
    for c in coverages:
        out[f"risk@{c:g}"] = selective_risk(signal, error, coverage=c)
        out[f"risk_oracle@{c:g}"] = selective_risk(error, error, coverage=c)
    return out
