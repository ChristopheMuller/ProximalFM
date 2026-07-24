"""Tests for the cross-fitted OOF regression helper.

Uses the fast ``linear`` / ``rf`` models only; TabICL is exercised at analysis
level, not unit level.
"""

import numpy as np
import pytest

from analysis_prior.regression import (
    cached_oof,
    fold_indices,
    leave_one_out_r2,
    oof_predictions,
)


def test_fold_indices_are_a_pure_function_of_arguments():
    a = fold_indices(200, 3, 0)
    b = fold_indices(200, 3, 0)
    assert len(a) == 3
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_folds_partition_every_row_once():
    folds = fold_indices(200, 4, 1)
    covered = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(covered), np.arange(200))


def test_different_seed_gives_different_folds():
    a = fold_indices(200, 3, 0)[0][1]
    b = fold_indices(200, 3, 1)[0][1]
    assert not np.array_equal(a, b)


def test_oof_predictions_same_length_as_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=120)
    pred = oof_predictions(X, y, model="linear", n_splits=3)
    assert pred.shape == y.shape


def test_empty_predictor_set_predicts_train_fold_mean():
    rng = np.random.default_rng(0)
    y = rng.normal(size=90).astype(np.float32)
    pred = oof_predictions(np.zeros((90, 0)), y, model="linear", n_splits=3)
    # Each test row gets its train fold's mean; overall R2 <= 0.
    for train, test in fold_indices(90, 3, 0):
        assert np.allclose(pred[test], y[train].mean(), atol=1e-5)


def test_leave_one_out_ignores_a_pure_noise_block():
    """A block unrelated to the target has ~0 unique contribution."""
    rng = np.random.default_rng(1)
    n = 600
    signal = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 2))
    y = signal[:, 0] * 1.5 + signal[:, 1] * -0.8 + rng.normal(scale=0.2, size=n)
    r2 = leave_one_out_r2(
        {"signal": signal, "noise": noise},
        y,
        tag_prefix="noise-test",
        model="linear",
        n_splits=3,
    )
    assert r2["full"] - r2["noise"] == pytest.approx(0.0, abs=0.02)
    assert r2["full"] - r2["signal"] > 0.5


def test_cached_oof_round_trips_and_hits(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=100)

    first = cached_oof("tag", X, y, model="linear", cache_dir=tmp_path)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1

    # A second call must read the cache, not recompute: identical result and
    # no new file.
    second = cached_oof("tag", X, y, model="linear", cache_dir=tmp_path)
    np.testing.assert_array_equal(first["pred"], second["pred"])
    assert first["r2"] == second["r2"]
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_cache_key_distinguishes_model_and_tag(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=80)
    cached_oof("a", X, y, model="linear", cache_dir=tmp_path)
    cached_oof("b", X, y, model="linear", cache_dir=tmp_path)
    cached_oof("a", X, y, model="rf", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3
