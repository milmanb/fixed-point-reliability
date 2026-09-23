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
    shares = report_table.flagged_share(pd.concat([per_image, constant]), "pixel_mask")
    assert shares.to_dict() == {"m": 0.5}  # a constant signal flags nothing and is skipped


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
