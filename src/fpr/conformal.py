"""Normalized split conformal bounds on the per-image reconstruction error.

Given a reference-free signal s and the offline error e, fit an isotonic map
u_hat = Isotonic(s -> e) on a calibration half, form nonconformity scores
s_i = e_i / (u_hat(y_i) + eps), take the split-conformal quantile
q_hat = quantile_{ceil((n+1)(1-alpha))/n}(s_cal), and report the upper bound
e_hi(y) = q_hat * u_hat(y). Coverage is P(e <= e_hi) >= 1 - alpha under exchangeability.

Baselines that ignore the per-image signal:
  marginal   one constant bound = the (1-alpha) quantile of e on the calibration half
  severity   isotonic of e on the calibration-half mean e of each corruption level

SURE self-calibration (noise family only): replace e by the already-computed sure column
when forming scores, so no clean image is used at calibration time
(cf. arXiv:2502.05127).
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def conformal_quantile(scores, alpha):
    """Empirical (1-alpha) quantile with the split-conformal finite-sample correction."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def isotonic_fit_predict(signal_cal, error_cal, signal_all, eps=1e-6):
    """Fit e ~ signal on the calibration half; predict for every row."""
    model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    model.fit(np.asarray(signal_cal, dtype=np.float64),
              np.asarray(error_cal, dtype=np.float64))
    return np.maximum(model.predict(np.asarray(signal_all, dtype=np.float64)), eps)


def normalized_bound(signal, error, calibration_mask, alpha=0.1, eps=1e-6):
    """Return (e_hi, coverage_on_held_out, mean_width_on_held_out, q_hat)."""
    signal = np.asarray(signal, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    cal = np.asarray(calibration_mask, dtype=bool)
    held = ~cal
    u_hat = isotonic_fit_predict(signal[cal], error[cal], signal, eps=eps)
    scores = error[cal] / u_hat[cal]
    q_hat = conformal_quantile(scores, alpha)
    e_hi = q_hat * u_hat
    return {
        "e_hi": e_hi,
        "coverage": float(np.mean(error[held] <= e_hi[held])),
        "width": float(np.mean(e_hi[held])),
        "q_hat": q_hat,
        "u_hat": u_hat,
    }


def marginal_bound(error, calibration_mask, alpha=0.1):
    """Constant (1-alpha) quantile of e on the calibration half."""
    error = np.asarray(error, dtype=np.float64)
    cal = np.asarray(calibration_mask, dtype=bool)
    held = ~cal
    bound = conformal_quantile(error[cal], alpha)
    e_hi = np.full_like(error, bound)
    return {
        "e_hi": e_hi,
        "coverage": float(np.mean(error[held] <= bound)),
        "width": float(bound),
        "q_hat": bound,
    }


def severity_bound(frame, calibration_mask, alpha=0.1, eps=1e-6):
    """Bound from the calibration-half mean error of each corruption level."""
    cal = np.asarray(calibration_mask, dtype=bool)
    level_mean = frame.loc[cal].groupby("condition")["e"].mean()
    u_hat = frame["condition"].map(level_mean).to_numpy(dtype=np.float64)
    u_hat = np.maximum(u_hat, eps)
    error = frame["e"].to_numpy(dtype=np.float64)
    scores = error[cal] / u_hat[cal]
    q_hat = conformal_quantile(scores, alpha)
    e_hi = q_hat * u_hat
    held = ~cal
    return {
        "e_hi": e_hi,
        "coverage": float(np.mean(error[held] <= e_hi[held])),
        "width": float(np.mean(e_hi[held])),
        "q_hat": q_hat,
        "u_hat": u_hat,
    }


def group_coverage(error, e_hi, groups, held_out):
    """Coverage within each group, restricted to the held-out half."""
    frame = pd.DataFrame({"group": np.asarray(groups),
                          "error": np.asarray(error, dtype=np.float64),
                          "e_hi": np.asarray(e_hi, dtype=np.float64),
                          "held": np.asarray(held_out, dtype=bool)})
    rows = []
    for name, rows_g in frame[frame["held"]].groupby("group", sort=False):
        rows.append({"group": name,
                     "coverage": float((rows_g["error"] <= rows_g["e_hi"]).mean()),
                     "n": len(rows_g),
                     "width": float(rows_g["e_hi"].mean())})
    return pd.DataFrame(rows)


def evaluate_signals(frame, signals=("g", "d", "r_A"), alpha=0.1, target="e"):
    """Pooled coverage and width for each signal, plus marginal and severity baselines.

    Split: even image indices calibrate, odd indices are held out (matches calibrate.py).
    """
    work = frame.copy()
    if target != "e":
        work["e"] = frame[target]
    calibration = (work["image"] % 2 == 0).to_numpy()
    error = work["e"].to_numpy(dtype=np.float64)
    rows = []
    for signal in signals:
        if signal not in work or work[signal].isna().any():
            continue
        result = normalized_bound(work[signal].to_numpy(), error, calibration, alpha=alpha)
        rows.append({"signal": signal, "setting": "normalized",
                     "coverage": result["coverage"], "width": result["width"],
                     "q_hat": result["q_hat"]})
        for scope, key in (("condition", "condition"), ("family", "family")):
            for _, r in group_coverage(error, result["e_hi"], work[key], ~calibration).iterrows():
                rows.append({"signal": signal, "setting": f"normalized/{scope}",
                             "group": r["group"], "coverage": r["coverage"],
                             "width": r["width"], "n": r["n"]})
    marginal = marginal_bound(error, calibration, alpha=alpha)
    rows.append({"signal": "-", "setting": "marginal",
                 "coverage": marginal["coverage"], "width": marginal["width"],
                 "q_hat": marginal["q_hat"]})
    severity = severity_bound(work, calibration, alpha=alpha)
    rows.append({"signal": "-", "setting": "severity",
                 "coverage": severity["coverage"], "width": severity["width"],
                 "q_hat": severity["q_hat"]})
    return pd.DataFrame(rows)


def shift_coverage(frame, signal, alpha=0.1,
                   calibrate_families=("noise",),
                   calibrate_levels=None):
    """Calibrate on a subset of conditions; report coverage on the complement.

    calibrate_levels: optional iterable of condition labels; if given, takes precedence.
    Default: all noise(sigma=0.1) and noise(sigma=0.2) rows with even image indices.
    """
    if calibrate_levels is None:
        calibrate_levels = [c for c in frame["condition"].unique()
                            if any(c.startswith(f) for f in ("noise(sigma=0.1)",
                                                             "noise(sigma=0.2)"))]
    cal_cond = frame["condition"].isin(calibrate_levels)
    # Use even images within the calibration conditions as the conformal calibration set.
    calibration = cal_cond & (frame["image"] % 2 == 0)
    deploy = ~cal_cond
    if calibration.sum() < 10 or deploy.sum() < 10:
        return {"coverage": float("nan"), "width": float("nan"), "q_hat": float("nan"),
                "n_cal": int(calibration.sum()), "n_deploy": int(deploy.sum())}
    error = frame["e"].to_numpy(dtype=np.float64)
    signal_vals = frame[signal].to_numpy(dtype=np.float64)
    u_hat = isotonic_fit_predict(signal_vals[calibration], error[calibration],
                                 signal_vals)
    q_hat = conformal_quantile(error[calibration] / u_hat[calibration], alpha)
    e_hi = q_hat * u_hat
    return {
        "coverage": float(np.mean(error[deploy] <= e_hi[deploy])),
        "width": float(np.mean(e_hi[deploy])),
        "q_hat": q_hat,
        "n_cal": int(calibration.sum()),
        "n_deploy": int(deploy.sum()),
        "e_hi": e_hi,
        "deploy_mask": deploy.to_numpy() if hasattr(deploy, "to_numpy") else np.asarray(deploy),
    }


def sure_self_calibrate(frame, signal="g", alpha=0.1, eps=1e-6):
    """Calibrate nonconformity with SURE instead of the true error (noise family only).

    SURE estimates the per-pixel squared error, so the bound lives on the `e_mse` scale and
    coverage is measured against `e_mse`. No clean image enters the calibration step: the
    isotonic fit and the conformal quantile both use `sure`, computed from y alone. The
    supervised bound on the same target is returned alongside as the reference.
    """
    noise = frame[frame["family"] == "noise"]
    if len(noise) == 0 or "sure" not in noise or "e_mse" not in noise \
            or noise["sure"].isna().any() or noise[signal].isna().any():
        return {"coverage": float("nan"), "width": float("nan"), "q_hat": float("nan"),
                "coverage_supervised": float("nan"), "width_supervised": float("nan")}
    calibration = (noise["image"] % 2 == 0).to_numpy()
    held = ~calibration
    truth = noise["e_mse"].to_numpy(dtype=np.float64)
    signal_vals = noise[signal].to_numpy(dtype=np.float64)
    # SURE is unbiased but not non-negative on individual images; clip before using it as a scale.
    sure = np.maximum(noise["sure"].to_numpy(dtype=np.float64), eps)

    u_hat = isotonic_fit_predict(signal_vals[calibration], sure[calibration], signal_vals, eps=eps)
    q_hat = conformal_quantile(sure[calibration] / u_hat[calibration], alpha)
    e_hi = q_hat * u_hat

    supervised = normalized_bound(signal_vals, truth, calibration, alpha=alpha, eps=eps)
    return {
        "coverage": float(np.mean(truth[held] <= e_hi[held])),
        "width": float(np.mean(e_hi[held])),
        "q_hat": q_hat,
        "coverage_supervised": supervised["coverage"],
        "width_supervised": supervised["width"],
    }
