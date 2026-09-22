"""Normalized split conformal bounds on the per-image reconstruction error.

Test images are split by index: i % 4 == 0 fits, i % 4 == 2 calibrates, odd i are held out.
Given a reference-free signal s and the offline error e, fit an isotonic map
u_hat = Isotonic(s -> e) on the fitting images, form nonconformity scores
s_i = e_i / max(u_hat(y_i), eps) on the calibration images, take the split-conformal quantile
q_hat = the ceil((n+1)(1-alpha))-th smallest of the n calibration scores, and report the upper
bound e_hi(y) = q_hat * u_hat(y). Because u_hat never sees the calibration scores, coverage is
P(e <= e_hi) >= 1 - alpha under exchangeability of calibration and held-out images.

Baselines that ignore the per-image signal:
  marginal   one constant bound = the conformal quantile of e on the calibration images
  severity   the mean e of each corruption level on the fitting images, scaled like u_hat

SURE self-calibration (noise family only): replace e by the already-computed sure column
when forming scores, so no clean image is used at calibration time
(cf. arXiv:2502.05127).
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def split_masks(image):
    """(fit, calibration, held_out) row masks from test-image indices; see the module docstring."""
    image = np.asarray(image)
    return image % 4 == 0, image % 4 == 2, image % 2 == 1


def conformal_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score: the split-conformal finite-sample quantile.

    Infinite when that rank exceeds n (too few scores for the requested level).
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def isotonic_fit_predict(signal_fit, error_fit, signal_all, eps=1e-6):
    """Fit e ~ signal on the fitting rows; predict for every row."""
    model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    model.fit(np.asarray(signal_fit, dtype=np.float64),
              np.asarray(error_fit, dtype=np.float64))
    return np.maximum(model.predict(np.asarray(signal_all, dtype=np.float64)), eps)


def normalized_bound(signal, error, calibration_mask, alpha=0.1, eps=1e-6, fit_mask=None):
    """Return (e_hi, coverage_on_held_out, mean_width_on_held_out, q_hat).

    u_hat is fit on `fit_mask` and the scores come from `calibration_mask`; all other rows are
    held out. Without `fit_mask`, u_hat is fit on the calibration rows themselves, which makes
    their scores optimistic and the coverage guarantee only approximate.
    """
    signal = np.asarray(signal, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    cal = np.asarray(calibration_mask, dtype=bool)
    fit = cal if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    held = ~(cal | fit)
    u_hat = isotonic_fit_predict(signal[fit], error[fit], signal, eps=eps)
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


def marginal_bound(error, calibration_mask, alpha=0.1, held_out=None):
    """Constant conformal quantile of e on the calibration rows."""
    error = np.asarray(error, dtype=np.float64)
    cal = np.asarray(calibration_mask, dtype=bool)
    held = ~cal if held_out is None else np.asarray(held_out, dtype=bool)
    bound = conformal_quantile(error[cal], alpha)
    e_hi = np.full_like(error, bound)
    return {
        "e_hi": e_hi,
        "coverage": float(np.mean(error[held] <= bound)),
        "width": float(bound),
        "q_hat": bound,
    }


def severity_bound(frame, calibration_mask, alpha=0.1, eps=1e-6, fit_mask=None):
    """Bound from each corruption level's mean error on the fitting rows (default: calibration)."""
    cal = np.asarray(calibration_mask, dtype=bool)
    fit = cal if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    level_mean = frame.loc[fit].groupby("condition")["e"].mean()
    u_hat = frame["condition"].map(level_mean).to_numpy(dtype=np.float64)
    u_hat = np.maximum(u_hat, eps)
    error = frame["e"].to_numpy(dtype=np.float64)
    scores = error[cal] / u_hat[cal]
    q_hat = conformal_quantile(scores, alpha)
    e_hi = q_hat * u_hat
    held = ~(cal | fit)
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

    Split by image index (split_masks); the held-out odd images are those calibrate.py scores.
    """
    work = frame.copy()
    if target != "e":
        work["e"] = frame[target]
    fit, calibration, held = split_masks(work["image"].to_numpy())
    error = work["e"].to_numpy(dtype=np.float64)
    rows = []
    for signal in signals:
        if signal not in work or work[signal].isna().any():
            continue
        result = normalized_bound(work[signal].to_numpy(), error, calibration, alpha=alpha,
                                  fit_mask=fit)
        rows.append({"signal": signal, "setting": "normalized",
                     "coverage": result["coverage"], "width": result["width"],
                     "q_hat": result["q_hat"]})
        for scope, key in (("condition", "condition"), ("family", "family")):
            for _, r in group_coverage(error, result["e_hi"], work[key], held).iterrows():
                rows.append({"signal": signal, "setting": f"normalized/{scope}",
                             "group": r["group"], "coverage": r["coverage"],
                             "width": r["width"], "n": r["n"]})
    marginal = marginal_bound(error, calibration, alpha=alpha, held_out=held)
    rows.append({"signal": "-", "setting": "marginal",
                 "coverage": marginal["coverage"], "width": marginal["width"],
                 "q_hat": marginal["q_hat"]})
    severity = severity_bound(work, calibration, alpha=alpha, fit_mask=fit)
    rows.append({"signal": "-", "setting": "severity",
                 "coverage": severity["coverage"], "width": severity["width"],
                 "q_hat": severity["q_hat"]})
    return pd.DataFrame(rows)


def shift_coverage(frame, signal, alpha=0.1, calibrate_levels=None):
    """Fit and calibrate on some conditions; report coverage on the others.

    calibrate_levels: condition labels to calibrate on (default: noise at sigma 0.1 and 0.2,
    inside the training range). Within them the images split as in split_masks, and the bound
    is deployed on the held-out (odd) images of every other condition, so no deployed image
    was used for fitting or calibration.
    """
    if calibrate_levels is None:
        calibrate_levels = [c for c in frame["condition"].unique()
                            if any(c.startswith(f) for f in ("noise(sigma=0.1)",
                                                             "noise(sigma=0.2)"))]
    cal_cond = frame["condition"].isin(calibrate_levels).to_numpy()
    fit, calibration, held = split_masks(frame["image"].to_numpy())
    fit, calibration, deploy = cal_cond & fit, cal_cond & calibration, ~cal_cond & held
    if fit.sum() < 10 or calibration.sum() < 10 or deploy.sum() < 10:
        return {"coverage": float("nan"), "width": float("nan"), "q_hat": float("nan"),
                "n_fit": int(fit.sum()), "n_cal": int(calibration.sum()),
                "n_deploy": int(deploy.sum())}
    error = frame["e"].to_numpy(dtype=np.float64)
    signal_vals = frame[signal].to_numpy(dtype=np.float64)
    u_hat = isotonic_fit_predict(signal_vals[fit], error[fit], signal_vals)
    q_hat = conformal_quantile(error[calibration] / u_hat[calibration], alpha)
    e_hi = q_hat * u_hat
    return {
        "coverage": float(np.mean(error[deploy] <= e_hi[deploy])),
        "width": float(np.mean(e_hi[deploy])),
        "q_hat": q_hat,
        "n_fit": int(fit.sum()),
        "n_cal": int(calibration.sum()),
        "n_deploy": int(deploy.sum()),
        "e_hi": e_hi,
        "deploy_mask": deploy,
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
    fit, calibration, held = split_masks(noise["image"].to_numpy())
    truth = noise["e_mse"].to_numpy(dtype=np.float64)
    signal_vals = noise[signal].to_numpy(dtype=np.float64)
    # SURE is unbiased but not non-negative on individual images; clip before using it as a scale.
    sure = np.maximum(noise["sure"].to_numpy(dtype=np.float64), eps)

    u_hat = isotonic_fit_predict(signal_vals[fit], sure[fit], signal_vals, eps=eps)
    q_hat = conformal_quantile(sure[calibration] / u_hat[calibration], alpha)
    e_hi = q_hat * u_hat

    supervised = normalized_bound(signal_vals, truth, calibration, alpha=alpha, eps=eps,
                                  fit_mask=fit)
    return {
        "coverage": float(np.mean(truth[held] <= e_hi[held])),
        "width": float(np.mean(e_hi[held])),
        "q_hat": q_hat,
        "coverage_supervised": supervised["coverage"],
        "width_supervised": supervised["width"],
    }
