"""End-to-end runs of the training, evaluation and check scripts on a few images.

They need Fashion-MNIST in data/ (downloaded by the first script that uses it) and are skipped
without it, so the rest of the suite runs offline.
"""

import importlib.util
import json
import sys

import pandas as pd
import pytest
import torch

from fpr.data import REPO_ROOT

pytestmark = pytest.mark.skipif(not (REPO_ROOT / "data" / "FashionMNIST").exists(),
                                reason="needs Fashion-MNIST in data/")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(name, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
    _script(name).main()


def test_training_writes_a_checkpoint_and_its_curve(tmp_path, monkeypatch):
    # 500 training images and a tiny network: one epoch with the warm-up ramp.
    _run("train_dae", monkeypatch, "--epochs", 1, "--val-size", 59500, "--width", 4, "--latent", 4,
         "--lambda-id", 1, "--lambda-warmup", 1, "--device", "cpu", "--seed", 3,
         "--checkpoints", tmp_path / "ckpt", "--logs", tmp_path / "logs")
    checkpoint = torch.load(tmp_path / "ckpt" / "dae_lam1w1_seed3.pt", weights_only=True)
    assert checkpoint["architecture"] == "bottleneck" and checkpoint["model_config"] == {"width": 4, "latent": 4}
    assert checkpoint["train_args"]["lambda_id"] == 1 and checkpoint["train_args"]["device"] == "cpu"
    curve = pd.read_csv(tmp_path / "logs" / "dae_lam1w1_seed3.csv")
    assert len(curve) == 1 and {"val_e@0.1", "val_e@0.5", "val_g@0.2", "train_idem"} <= set(curve.columns)
    assert 0 < curve["lambda"].iat[0] <= 1  # ramped up during the epoch


def test_evaluation_writes_signals_scores_and_provenance(tmp_path, monkeypatch):
    _run("evaluate_models", monkeypatch, "--pattern", "dae_lam0_seed0.pt", "--projectors",
         "--n-eval", 20, "--n-boot", 5, "--device", "cpu", "--out", tmp_path)
    per_image = pd.read_csv(tmp_path / "per_image.csv.gz")
    assert len(per_image) == 13 * 20 and per_image["image"].nunique() == 20
    assert {"g", "d", "r_A", "e", "b", "g2", "q"} <= set(per_image.columns)
    noise = per_image["family"] == "noise"
    assert per_image.loc[noise, "sure"].notna().all() and per_image.loc[~noise, "sure"].isna().all()
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert set(metrics["scope"]) == {"condition", "family", "all"}
    assert set(metrics.loc[metrics["scope"] == "condition", "group"]) == set(per_image["condition"])
    run_info = json.loads((tmp_path / "run_info.json").read_text())
    assert run_info["models"] == ["dae_lam0_seed0"] and run_info["args"]["n_eval"] == 20


def test_add_signals_completes_a_dump_and_records_the_run(tmp_path, monkeypatch):
    # A dump as the committed results were first written: no deeper iterates, no disagreement.
    _run("evaluate_models", monkeypatch, "--pattern", "dae_lam0_seed*.pt", "--projectors", "--n-eval", 10,
         "--iterate-steps", 1, "--no-metrics", "--device", "cpu", "--out", tmp_path)
    before = pd.read_csv(tmp_path / "per_image.csv.gz")
    assert not {"g2", "g3", "q", "dis"} & set(before.columns)
    # No gzip time stamp, so a rewrite with the same rows gives the same bytes and digest.
    assert (tmp_path / "per_image.csv.gz").read_bytes()[4:8] == bytes(4)
    _run("add_signals", monkeypatch, "--folder", tmp_path, "--pattern", "dae_lam0_seed*.pt", "--threads", 1)
    assert (tmp_path / "per_image.csv.gz").read_bytes()[4:8] == bytes(4)
    after = pd.read_csv(tmp_path / "per_image.csv.gz")
    assert {"g2", "g3", "q", "dis"} <= set(after.columns) and after[["g2", "g3", "q", "dis"]].notna().all().all()
    assert after[["g", "d", "e"]].equals(before[["g", "d", "e"]])  # the stored values are kept
    record = json.loads((tmp_path / "run_info.json").read_text())["appended_runs"][-1]
    assert record["mode"] == "add_signals" and record["check"]["rows"] == len(after)
    assert max(record["check"]["max_rel"].values()) < 1e-5  # recomputed g, d, e agree with the dump


def test_checks_reproduce_the_reported_relations():
    if not (REPO_ROOT / "checkpoints" / "dae_lam1_seed0.pt").exists():
        pytest.skip("needs checkpoints/dae_lam1_seed0.pt, the run that collapsed")
    checks = _script("checks")
    initialization = checks.initialization_check(n=64, seeds=1)
    assert (initialization["skip"]["output_spread_over_inputs"]
            > 100 * initialization["bottleneck"]["output_spread_over_inputs"])
    collapse = checks.collapse_check(val_size=1000)
    # The collapsed model outputs the per-pixel median image, so it inherits that image's error.
    assert abs(collapse["model_error"] - collapse["median_image_error"]) < 1e-3
    assert collapse["output_spread"] < 1e-3
    finite = checks.finite_difference_check(n_eval=8, probes=2)
    assert set(finite) == set(checks.TRAINED)
    assert all(errors[key]["max_relative_error"] < 0.05 for errors in finite.values() for key in errors)
