"""Tests for the cross-fitted OOF regression helper.

Uses the fast ``linear`` / ``rf`` models only; TabICL is exercised at analysis
level, not unit level.
"""

import numpy as np
import pytest

from analysis_prior.regression import (
    cached_oof,
    cached_t_learner,
    clip_to_train_range,
    fold_indices,
    leave_one_out_r2,
    oof_predictions,
    t_learner_mu,
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


def test_clip_to_train_range_bounds_extrapolation():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 1e8], dtype=np.float64)
    clipped = clip_to_train_range(pred, y, n_splits=3, seed=0)
    # Every clipped value sits within the global target range; the 1e8 blow-up
    # is pulled back to at most max(y).
    assert clipped.max() <= y.max() + 1e-9
    assert clipped.min() >= y.min() - 1e-9


def test_clipping_keeps_r2_finite_under_a_blowup(tmp_path, monkeypatch):
    """A regressor that extrapolates to 1e8 must not yield an absurd R²."""
    import analysis_prior.regression as reg

    y = np.linspace(-3, 3, 60).astype(np.float32)
    X = y.reshape(-1, 1)

    def _blowup_factory(model, seed):
        class _Blowup:
            def fit(self, X, y):
                self._max = float(np.max(y))

            def predict(self, X):
                out = X.ravel().astype(np.float64)
                out[0] = 1e8  # one catastrophic extrapolation
                return out

        return lambda: _Blowup()

    monkeypatch.setattr(reg, "_model_factory", _blowup_factory)
    result = cached_oof("blowup", X, y, model="linear", n_splits=3, cache_dir=tmp_path)
    # Without clipping this R² would be ~-1e15; clipped it stays sane.
    assert result["r2"] > -5.0
    assert np.isfinite(result["pred"]).all()
    assert result["pred"].max() <= y.max() + 1e-6


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


def _confounded(n=600, seed=0, effect=2.0):
    """(X, A, Y, true ATE) where X fully explains both assignment and outcome.

    Adjusting on X therefore identifies the effect, so a correct T-learner must
    recover ``effect`` even though the naive contrast is badly biased.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    A = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-2.0 * X[:, 0]))).astype(float)
    y0 = 1.5 * X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=n)
    Y = y0 + effect * A
    return X, A, Y, effect


def test_t_learner_predicts_every_row_in_both_arms():
    X, A, Y, _ = _confounded()
    mu0, mu1 = t_learner_mu(X, A, Y, model="linear", n_splits=3)
    assert mu0.shape == mu1.shape == Y.shape
    assert np.isfinite(mu0).all() and np.isfinite(mu1).all()


def test_t_learner_recovers_the_effect_the_naive_contrast_misses():
    X, A, Y, effect = _confounded()
    mu0, mu1 = t_learner_mu(X, A, Y, model="linear", n_splits=3)
    # Tolerance, not exactness: predictions are clipped to the training arm's
    # target range, which under this fixture's deliberately skewed assignment
    # truncates a few extrapolated counterfactuals and leaves a ~2% residual.
    # That conservatism is the price of the blow-up guard, and it is an order of
    # magnitude below the confounding bias the audit has to resolve.
    assert float(np.mean(mu1 - mu0)) == pytest.approx(effect, abs=0.1)
    # The confounded contrast X was there to remove.
    naive = Y[A == 1].mean() - Y[A == 0].mean()
    assert abs(naive - effect) > 0.5


def test_same_arm_rows_are_predicted_out_of_fold():
    """The factual half must not be fitted in-sample: a memorising regressor
    gets its own arm's rows wrong, while the counterfactual half is unaffected."""

    class _Memoriser:
        def fit(self, X, y):
            self._seen = {tuple(row): value for row, value in zip(X, y)}

        def predict(self, X):
            return np.array([self._seen.get(tuple(row), 0.0) for row in X])

    import analysis_prior.regression as reg

    n = 60
    X = np.arange(n, dtype=float).reshape(-1, 1)
    A = (np.arange(n) % 2).astype(float)
    Y = np.arange(n, dtype=float) + 100.0

    original = reg._model_factory
    try:
        reg._model_factory = lambda model, seed: _Memoriser
        mu0, _ = t_learner_mu(X, A, Y, model="linear", n_splits=3)
    finally:
        reg._model_factory = original

    control = A == 0
    # Held-out rows are unseen by the memoriser, so its prediction is the
    # clipped 0.0 -> the arm's minimum, never the row's own Y.
    assert not np.allclose(mu0[control], Y[control])


def test_t_learner_falls_back_to_the_arm_mean_when_an_arm_is_tiny():
    n = 40
    X = np.random.default_rng(0).normal(size=(n, 2))
    Y = np.arange(n, dtype=float)
    A = np.zeros(n)
    A[:3] = 1.0  # too few treated rows to cross-fit
    mu0, mu1 = t_learner_mu(X, A, Y, model="linear", n_splits=3)
    assert np.allclose(mu1, Y[A == 1].mean())


def test_t_learner_returns_nan_for_an_empty_arm():
    X = np.random.default_rng(0).normal(size=(30, 2))
    Y = np.arange(30, dtype=float)
    mu0, mu1 = t_learner_mu(X, np.zeros(30), Y, model="linear", n_splits=3)
    assert np.isnan(mu1).all()
    assert np.isfinite(mu0).all()


def test_cached_t_learner_round_trips_and_hits(tmp_path):
    X, A, Y, _ = _confounded(n=200)
    first = cached_t_learner("tag", X, A, Y, model="linear", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second = cached_t_learner("tag", X, A, Y, model="linear", cache_dir=tmp_path)
    np.testing.assert_array_equal(first["mu0"], second["mu0"])
    assert first["ate"] == second["ate"]
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_t_learner_cache_does_not_collide_with_oof(tmp_path):
    """Same tag, different helper: the entries must be distinct."""
    X, A, Y, _ = _confounded(n=200)
    cached_oof("same", X, Y, model="linear", cache_dir=tmp_path)
    cached_t_learner("same", X, A, Y, model="linear", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_cache_key_distinguishes_model_and_tag(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + rng.normal(scale=0.1, size=80)
    cached_oof("a", X, y, model="linear", cache_dir=tmp_path)
    cached_oof("b", X, y, model="linear", cache_dir=tmp_path)
    cached_oof("a", X, y, model="rf", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3
