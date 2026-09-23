import numpy as np
import pandas as pd
import pytest
import torch

from fpr.ensemble import attach_disagreement
from fpr.evaluation import CONDITIONS, observe, per_image_signals, score
from fpr.metrics import rank_metrics
from fpr.models import ConvAutoencoder, Restorer
from fpr.projectors import PCA, Radial, principal_components


def test_process_pool_reproduces_serial_results():
    gen = torch.Generator().manual_seed(0)
    train = torch.rand((200, 1, 28, 28), generator=gen, dtype=torch.float64)
    x = torch.rand((60, 1, 28, 28), generator=gen, dtype=torch.float64)
    labels = torch.randint(0, 10, (60,), generator=gen)
    mean, eigvecs, _ = principal_components(train)
    torch.manual_seed(0)
    models = {"radial": Radial.fit(train), "pca8": PCA.from_components(mean, eigvecs, 8),
              "tiny": Restorer(ConvAutoencoder(width=4, latent=4), name="tiny", device="cpu")}
    conditions = [CONDITIONS[1], CONDITIONS[5], CONDITIONS[12]]  # noise, blur, box mask
    kwargs = dict(labels=labels, seed=3, probes=2, conditions=conditions, verbose=False)

    serial = per_image_signals(models, x, **kwargs)
    pooled = per_image_signals(models, x, jobs=2, threads=1, **kwargs)
    assert serial[["condition", "model", "image"]].equals(pooled[["condition", "model", "image"]])
    numeric = serial.select_dtypes("number").columns
    assert np.allclose(serial[numeric], pooled[numeric], rtol=1e-5, atol=1e-7, equal_nan=True)

    signals = ("b", "g", "d", "r_A", "div", "sure")
    pd.testing.assert_frame_equal(score(serial, signals, n_boot=20, seed=1),
                                  score(serial, signals, n_boot=20, seed=1, jobs=2))


def _per_image(levels, n=120, seed=0):
    """A synthetic per-image table: one model, the same n images at every (condition, family)."""
    rng = np.random.default_rng(seed)
    frames = []
    for condition, family in levels:
        e = rng.gamma(2.0, 0.05, n)
        frames.append(pd.DataFrame({"family": family, "condition": condition, "model": "m",
                                    "image": np.arange(n), "e": e, "b": rng.random(n),
                                    "s": e + rng.normal(0, 0.05, n)}))
    return pd.concat(frames, ignore_index=True)


def test_a_signal_with_a_missing_value_leaves_only_the_groups_it_misses():
    """A signal enters a group only if it is defined on every row of the group."""
    frame = _per_image([("noise(sigma=0.1)", "noise"), ("noise(sigma=0.2)", "noise"),
                        ("blur(std=1)", "blur")])
    frame["hole"] = frame["s"]
    frame.loc[3, "hole"] = np.nan  # one image of noise(sigma=0.1)
    table = score(frame, ("s", "hole"), n_boot=0)
    has_hole = set(table.loc[table["signal"] == "hole", "group"])
    assert has_hole == {"noise(sigma=0.2)", "blur(std=1)", "blur"}
    assert set(table.loc[table["signal"] == "s", "group"]) == {
        "noise(sigma=0.1)", "noise(sigma=0.2)", "blur(std=1)", "noise", "blur", "all"}


def test_level_baseline_ranks_the_median_of_each_level():
    """The harder level has the higher median signal; the easier one, through ten outliers, the
    higher mean. Only the median ranks the levels correctly."""
    frame = _per_image([("noise(sigma=0.1)", "noise"), ("noise(sigma=0.3)", "noise")])
    rng = np.random.default_rng(1)
    easy = (frame["condition"] == "noise(sigma=0.1)").to_numpy()
    frame.loc[~easy, "e"] *= 3
    frame.loc[easy, "s"] = 0.01 * rng.random(easy.sum()) + 100.0 * (frame.loc[easy, "image"] < 10)
    frame.loc[~easy, "s"] = 0.5 + 0.01 * rng.random((~easy).sum())
    table = score(frame, ("s",), n_boot=0)
    row = table[(table["group"] == "noise") & (table["signal"] == "s@level")].iloc[0]
    medians = frame.groupby("condition")["s"].transform("median").to_numpy()
    expected = rank_metrics({"s@level": medians}, frame["e"].to_numpy(),
                            clusters=frame["image"].to_numpy(), n_boot=0,
                            control=frame["b"].to_numpy())[0]
    for key in ("spearman", "auroc", "partial"):
        assert row[key] == pytest.approx(expected[key], rel=1e-12)
    assert row["spearman"] > 0.5
    assert set(table.loc[table["signal"] == "s@level", "scope"]) == {"family", "all"}


def test_pooled_groups_resample_whole_images():
    """Two identical copies of a level carry no more information than one. Resampling images
    keeps the pooled interval equal to the single level's; resampling rows would narrow it."""
    frame = _per_image([("noise(sigma=0.1)", "noise")])
    copy = frame.assign(condition="noise(sigma=0.2)")
    table = score(pd.concat([frame, copy], ignore_index=True), ("s",), n_boot=200, seed=5)
    single = table[(table["group"] == "noise(sigma=0.1)") & (table["signal"] == "s")].iloc[0]
    pooled = table[(table["group"] == "noise") & (table["signal"] == "s")].iloc[0]
    for key in ("spearman", "spearman_lo", "spearman_hi", "auroc_lo", "auroc_hi"):
        assert pooled[key] == pytest.approx(single[key], rel=1e-9), key


def test_disagreement_is_attached_to_each_members_own_images():
    """Two members, y and y / 2, disagree by mean|y| / 4 on every image. Rows are shuffled, so
    only a lookup by image index puts each value on the right row."""
    x = torch.rand((12, 1, 28, 28), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    models = {"a_seed0": lambda y: y, "a_seed1": lambda y: 0.5 * y, "b_seed0": lambda y: y}
    condition, other = CONDITIONS[0], CONDITIONS[1]
    rows = [{"condition": c.label, "model": m, "image": i}
            for c in (condition, other) for m in models for i in range(12)]
    per_image = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = attach_disagreement(per_image, models, x, seed=7, conditions=[condition], verbose=False)

    y, _ = observe(condition, x, 7)
    expected = 0.25 * y.abs().flatten(1).mean(1).numpy()
    for member in ("a_seed0", "a_seed1"):
        rows = out[(out["condition"] == condition.label) & (out["model"] == member)]
        assert np.allclose(rows["dis"].to_numpy(), expected[rows["image"].to_numpy()], rtol=1e-12)
    assert out.loc[out["model"] == "b_seed0", "dis"].isna().all()  # no second seed
    assert out.loc[out["condition"] == other.label, "dis"].isna().all()  # not re-observed
    assert "dis" not in per_image  # the input is not modified
