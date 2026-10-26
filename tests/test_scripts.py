"""Tests for the helpers in scripts/ that merge result folders and build the report's tables."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from fpr.data import REPO_ROOT


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(models, signals, value):
    return pd.DataFrame([{"scope": "condition", "group": "noise(sigma=0.1)", "model": model,
                          "target": "e", "signal": signal, "spearman": value}
                         for model in models for signal in signals])


def test_rescoring_replaces_only_the_rescored_signals_of_the_rescored_models(tmp_path):
    evaluate_models = _script("evaluate_models")
    _metrics(["m1", "m2"], ["g", "g@level", "d", "d@level"], 0.0).to_csv(tmp_path / "metrics.csv",
                                                                         index=False)
    new = _metrics(["m1"], ["g", "g@level"], 1.0)
    merged = evaluate_models.append_metrics_only(tmp_path, new, ["g"])
    value = merged.set_index(["model", "signal"])["spearman"]
    assert len(merged) == 8 and not merged.duplicated(["model", "signal"]).any()
    assert value["m1", "g"] == 1.0 and value["m1", "g@level"] == 1.0
    assert value["m1", "d"] == 0.0 and value["m1", "d@level"] == 0.0
    assert value["m2", "g"] == 0.0 and value["m2", "g@level"] == 0.0


def _run_info(models, n_boot=1000):
    return {"models": models, "train_args": {m: {} for m in models}, "seconds": 1.0,
            "versions": {}, "args": {"n_eval": 10, "seed": 0, "n_boot": n_boot, "probes": 8,
                                     "jacobian_families": ["noise"], "iterate_steps": 3}}


def _write_run(folder, models):
    run_info = _run_info(models)
    frame = pd.DataFrame({"model": models, "e": 0.0})
    frame.to_csv(folder / "summary.csv", index=False)
    frame.to_csv(folder / "per_image.csv.gz", index=False)
    _metrics(models, ["g", "dis"], 0.0).to_csv(folder / "metrics.csv", index=False)
    (folder / "run_info.json").write_text(json.dumps(run_info))
    return run_info


def test_appending_keeps_earlier_models_and_replaces_rerun_ones(tmp_path):
    evaluate_models = _script("evaluate_models")
    _write_run(tmp_path, ["a_seed0", "b_seed0"])
    run_info = _run_info(["b_seed0", "c_seed0"])
    frame = pd.DataFrame({"model": run_info["models"], "e": 1.0})

    summary, metrics, per_image, info = evaluate_models.merge_with_existing(
        tmp_path, frame, _metrics(run_info["models"], ["dis"], 1.0), frame, run_info,
        scored=["dis"])
    assert summary.set_index("model")["e"].to_dict() == {"a_seed0": 0.0, "b_seed0": 1.0,
                                                         "c_seed0": 1.0}
    assert info["models"] == ["a_seed0", "b_seed0", "c_seed0"]
    assert len(info["appended_runs"]) == 1
    value = metrics.set_index(["model", "signal"])["spearman"]
    assert value["b_seed0", "dis"] == 1.0 and value["b_seed0", "g"] == 0.0  # g was not re-scored
    assert value["a_seed0", "dis"] == 0.0

    run_info["args"]["n_boot"] = 200
    with pytest.raises(SystemExit, match="n-boot"):
        evaluate_models.merge_with_existing(tmp_path, frame, None, frame, run_info)


def test_flagged_share_takes_both_quartiles_over_all_levels():
    """Rows 6 and 7 hold the worst quartile of e (both pixel masks); g flags rows 3 and 6.
    Quartiles taken within the pixel-mask rows would give 0 instead of 1/2."""
    report_table = _script("report_table")
    per_image = pd.DataFrame({
        "model": "m",
        "family": ["noise"] * 4 + ["pixel_mask"] * 4,
        "e": np.arange(1.0, 9.0),
        "g": [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.8, 0.6],
    })
    constant = per_image.assign(model="constant", g=0.5)
    partial = per_image.assign(model="partial", g=per_image["g"].where(per_image.index != 6))
    shares = report_table.flagged_share(pd.concat([per_image, constant, partial]), "pixel_mask")
    assert shares.to_dict() == {"m": 0.5}  # constant or partly missing signals are skipped


def test_missing_dump_names_the_download_script(tmp_path):
    from fpr.data import per_image_dump
    with pytest.raises(SystemExit, match="download_dumps.py"):
        per_image_dump(tmp_path)
    (tmp_path / "per_image.csv.gz").write_bytes(b"")
    assert per_image_dump(tmp_path) == tmp_path / "per_image.csv.gz"


def test_download_checks_the_digest_and_keeps_identical_files(tmp_path):
    download_dumps = _script("download_dumps")
    source = tmp_path / "asset.csv.gz"
    source.write_bytes(b"per-image rows")
    digest = download_dumps.sha256(source)
    target = tmp_path / "results" / "per_image.csv.gz"
    assert download_dumps.fetch(source.as_uri(), target, digest) == "downloaded"
    assert target.read_bytes() == b"per-image rows"
    assert download_dumps.fetch(source.as_uri(), target, digest) == "present"
    with pytest.raises(SystemExit, match="differs"):
        download_dumps.fetch(source.as_uri(), target, "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        download_dumps.fetch(source.as_uri(), target, "0" * 64, force=True)
    assert target.read_bytes() == b"per-image rows"  # a failed download leaves the file alone
    assert not list(target.parent.glob("*.part"))
    with pytest.raises(SystemExit, match="could not download"):
        download_dumps.fetch((tmp_path / "missing.csv.gz").as_uri(), tmp_path / "new" / "per_image.csv.gz", digest)
    assert not list((tmp_path / "new").glob("*"))


def test_add_signals_appends_its_record_to_run_info(tmp_path):
    add_signals = _script("add_signals")
    path = tmp_path / "run_info.json"
    path.write_text(json.dumps({"models": ["a"], "appended_runs": [{"models": ["b"]}]}))
    add_signals.append_run_record(path, {"mode": "add_signals"})
    add_signals.append_run_record(tmp_path / "new.json", {"mode": "add_signals"})
    assert json.loads(path.read_text()) == {
        "models": ["a"], "appended_runs": [{"models": ["b"]}, {"mode": "add_signals"}]}
    assert json.loads((tmp_path / "new.json").read_text()) == {"appended_runs": [{"mode": "add_signals"}]}


def test_per_seed_median_takes_the_median_level_of_each_seed():
    report_table = _script("report_table")
    levels = ["noise(sigma=0.1)", "noise(sigma=0.5)", "blur(std=1)"]
    metrics = pd.DataFrame([{"scope": "condition", "target": "e", "signal": "g", "model": model,
                             "group": level, "spearman": value}
                            for model, values in (("m_seed0", (0.1, 0.5, 0.9)), ("m_seed1", (0.3, 0.2, 0.4)))
                            for level, value in zip(levels, values)])
    other = metrics.assign(target="e_mse", spearman=-1.0)  # ignored: another target
    medians = report_table.per_seed_median(pd.concat([metrics, other]), "g", "spearman")
    assert medians.to_dict() == {"m_seed0": 0.5, "m_seed1": 0.3}
    noise = report_table.per_seed_median(metrics, "g", "spearman", families="noise")
    assert noise.round(6).to_dict() == {"m_seed0": 0.3, "m_seed1": 0.25}


def test_coverage_table_averages_the_conformal_and_selective_results(tmp_path, monkeypatch):
    report_table = _script("report_table")
    monkeypatch.setattr(report_table, "REPO_ROOT", tmp_path)
    (tmp_path / "results" / "conformal").mkdir(parents=True)
    (tmp_path / "results" / "selective").mkdir(parents=True)
    (tmp_path / "report").mkdir()
    pd.DataFrame([
        {"model_group": "dae_lam0", "signal": "g", "setting": "normalized", "coverage": 0.9, "width": 0.2},
        {"model_group": "dae_lam0", "signal": "-", "setting": "marginal", "coverage": 0.9, "width": 0.3},
        {"model_group": "dae_lam0", "signal": "-", "setting": "severity", "coverage": 0.9, "width": 0.1},
    ]).to_csv(tmp_path / "results" / "conformal" / "coverage.csv", index=False)
    pd.DataFrame([
        {"model_group": "dae_lam0", "signal": "g", "group": "pixel_mask", "coverage": 0.6},
        {"model_group": "dae_lam0", "signal": "g", "group": "box_mask", "coverage": 0.95},
    ]).to_csv(tmp_path / "results" / "conformal" / "coverage_family.csv", index=False)
    pd.DataFrame([{"model_group": "dae_lam0", "signal": "g", "coverage": c} for c in (0.4, 0.6)]
                 ).to_csv(tmp_path / "results" / "conformal" / "shift.csv", index=False)
    pd.DataFrame([{"group": "dae_lam0", "model": f"dae_lam0_seed{k}", "signal": s, "aurc_norm": v}
                  for k in range(2) for s, v in (("g", 0.4 + 0.1 * k), ("dis", 0.5))]
                 ).to_csv(tmp_path / "results" / "selective" / "selective.csv", index=False)
    report_table.coverage_table(groups=("dae_lam0",))
    row = pd.read_csv(tmp_path / "results" / "report_coverage.csv").iloc[0]
    assert (row["width_g"], row["width_marginal"], row["width_severity"]) == (0.2, 0.3, 0.1)
    assert (row["cov_pixel_g"], row["cov_box_g"], row["shift_g"]) == (0.6, 0.95, 0.5)
    assert (row["aurc_g"], row["aurc_dis"]) == (0.45, 0.5) and np.isnan(row["cov_pixel_dis"])
    tex = (tmp_path / "report" / "table_coverage.tex").read_text()
    assert r"bottleneck, $\lid{=}0$ & 0.200 & 0.100 & 0.60 & 0.95 & 0.50 & 0.45 & 0.50 \\" in tex


def test_blind_spot_shares_follow_the_flagged_rule(tmp_path):
    plot_blind_spots = _script("plot_blind_spots")
    rows = []
    for seed in range(2):
        # Four images per family; the worst errors are the pixel masks and one box mask.
        for family, errors, g in (("noise", (1, 2, 3, 4), (1, 2, 3, 4)),
                                  ("pixel_mask", (13, 14, 15, 16), (1, 1, 1, 1)),
                                  ("box_mask", (9, 10, 11, 12), (9, 10, 11, 12)),
                                  ("blur", (5, 6, 7, 8), (5, 6, 7, 8))):
            rows += [{"model": f"dae_lam0_seed{seed}", "family": family, "e": e, "g": v, "dis": v}
                     for e, v in zip(errors, g)]
    pd.DataFrame(rows).to_csv(tmp_path / "per_image.csv.gz", index=False)
    table = pd.DataFrame(plot_blind_spots.family_shares("bottleneck", tmp_path, "dae_lam0"))
    g = table[table["signal"] == "g"].set_index("family")
    assert g.loc["pixel_mask", "flagged"] == 0.0 and g.loc["pixel_mask", "worst_quartile_share"] == 1.0
    assert g.loc["box_mask", "worst_quartile_share"] == 0.0 and (g["seeds"] == 2).all()


def test_stable_but_wrong_takes_both_percentiles_over_all_levels():
    report_numbers = _script("report_numbers")
    rows = pd.DataFrame({"e": np.arange(20.0), "g": np.r_[np.full(10, 5.0), np.arange(10.0)]})
    picked = report_numbers.stable_but_wrong(rows)
    # Top-10% error: rows 18 and 19 (e >= 17.1); their g (8 and 9) is not below the median (5).
    assert picked.empty
    rows.loc[19, "g"] = 0.0
    assert report_numbers.stable_but_wrong(rows).index.tolist() == [19]


def test_selective_risk_scores_only_complete_varying_signals():
    selective = _script("selective")
    assert selective.scorable(np.array([0.1, 0.2, 0.3]))
    assert not selective.scorable(np.array([0.1, np.nan, 0.3]))
    assert not selective.scorable(np.full(3, 0.2))


def _calibration_frame(n=200, seed=0):
    """Two noise levels on different error scales. Within each level g ranks e perfectly, but
    between levels g runs the wrong way, so raw g fails pooled and per-level calibration fixes it."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, offset, scale in (("noise(sigma=0.1)", 0.0, 1.0), ("noise(sigma=0.5)", 1.0, -1.0)):
        e = offset + rng.random(n)
        rows.append(pd.DataFrame({"image": np.arange(n), "condition": condition, "family": "noise",
                                  "e": e, "g": 5.0 - offset + 0.1 * scale * e}))
    frame = pd.concat(rows, ignore_index=True)
    return frame.assign(d=frame["g"], r_A=frame["g"])


def test_calibration_fits_on_even_images_and_reverses_where_needed():
    calibrate = _script("calibrate")
    frame = _calibration_frame()
    calibration = frame["image"] % 2 == 0
    predictions = calibrate.isotonic_predictions(frame, calibration, "g", "condition")
    changed = frame.assign(e=frame["e"].where(calibration, 0.0))  # alter only held-out errors
    assert predictions.equals(calibrate.isotonic_predictions(changed, calibration, "g", "condition"))
    held = frame[~calibration]
    reversed_level = held["condition"] == "noise(sigma=0.5)"  # g falls as e grows there
    assert predictions[held.index][reversed_level].corr(held["e"][reversed_level], "spearman") > 0.99


def test_per_level_calibration_beats_raw_scores_and_the_level_mean():
    calibrate = _script("calibrate")
    table = calibrate.evaluate_model(_calibration_frame()).set_index(["signal", "setting"])
    raw, per_level = table.loc[("g", "raw"), "spearman"], table.loc[("g", "per level"), "spearman"]
    severity = table.loc[("-", "severity only"), "spearman"]
    assert raw < 0 < severity < per_level and per_level > 0.99


def test_report_table_rows_and_percent_formatting():
    report_table = _script("report_table")
    base = {"error": 0.1234, "residual": 0.0567, "rho_g": -0.05, "rho_g_noise": 0.7,
            "partial_g": 0.5, "pooled_g": 0.6, "auroc_g": 0.8, "rho_d": 0.77, "rho_rA": 0.8,
            "rho_dis": np.nan}
    table = pd.DataFrame([
        base | {"architecture": "bottleneck", "training": "none",
                "flagged_pixel_g": 0.004, "flagged_box_g": 0.005},
        base | {"architecture": "skip", "training": "lambda 1",
                "flagged_pixel_g": 0.0, "flagged_box_g": 0.61},
        base | {"architecture": "skip", "training": "lambda 1, no warm-up",
                "flagged_pixel_g": np.nan, "flagged_box_g": np.nan},
    ])
    lines = report_table.latex_rows(table).splitlines()
    bottleneck = next(line for line in lines if line.startswith("bottleneck"))
    skip = next(line for line in lines if line.startswith("skip"))
    assert bottleneck.startswith(r"bottleneck, $\lid{=}0$ & 0.123 & 0.057 & $-$0.05 & 0.70")
    assert bottleneck.endswith(r"& -- & $<$1\% & $<$1\% \\")  # dis missing; shares under 1%
    assert skip.startswith(r"skip, $\lid{=}1$ &")
    assert skip.endswith(r"& 0\% & 61\% \\")
    assert not any("no warm-up" in line for line in lines)
