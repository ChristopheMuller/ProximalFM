"""Proxy-marginalised TabICL diagnostic baseline."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import benchmarks.methods as methods
from benchmarks.compute import ComputeConfig
from benchmarks.methods import build_method, method_estimand
from benchmarks.methods.tabicl_gcomp import TabICLGComp


class RecordingOutcomeModel:
    """Deterministic outcome surface for testing the wrapper, not TabICL."""

    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.predict_calls = []

    def fit(self, X, y):
        self.fit_X = np.array(X, copy=True)
        self.fit_y = np.array(y, copy=True)
        return self

    def predict(self, X):
        X = np.asarray(X)
        self.predict_calls.append(np.array(X, copy=True))
        # Tests below use one column in each of X, W and Z.  The treatment
        # contrast is therefore 2 W + 3 Z; the X term cancels between arms.
        return 0.5 * X[:, 0] + X[:, 3] * (2.0 * X[:, 1] + 3.0 * X[:, 2])


def episode(n=5):
    X = np.linspace(-2.0, 2.0, n)[:, None]
    W = (10.0 + np.arange(n))[:, None]
    Z = (100.0 + 2.0 * np.arange(n))[:, None]
    A = (np.arange(n) % 2).astype(float)
    Y = np.arange(n, dtype=float)
    return X, W, Z, A, Y


def fitted(*, n=5, top_k=3, batch_size=4096):
    X, W, Z, A, Y = episode(n)
    model = RecordingOutcomeModel()
    estimator = TabICLGComp(
        model,
        proxy_top_k=top_k,
        kernel_scale=1.0,
        predict_batch_size=batch_size,
    )
    estimator.fit(X, A, Y, W=W, Z=Z)
    return estimator, model


def test_fit_and_weighted_prediction_match_manual_calculation():
    X, W, Z, A, Y = episode()
    model = RecordingOutcomeModel()
    estimator = TabICLGComp(model, proxy_top_k=3)
    estimator.fit(X, A, Y, W=W, Z=Z)

    query = np.array([[0.25], [1.75]])
    actual = estimator.predict(query)

    distances = cdist(X, X)
    sigma = 2.0 * np.median(distances) ** 2
    sq_distance = (query - X.T) ** 2 / sigma
    indices = np.argsort(sq_distance, axis=1, kind="stable")[:, :3]
    selected = np.take_along_axis(sq_distance, indices, axis=1)
    weights = np.exp(-(selected - selected[:, :1]))
    weights /= weights.sum(axis=1, keepdims=True)
    atom_effect = 2.0 * W[:, 0] + 3.0 * Z[:, 0]
    expected = np.sum(atom_effect[indices] * weights, axis=1)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(model.fit_X, np.hstack([X, W, Z, A[:, None]]))
    np.testing.assert_array_equal(model.fit_y, Y)


def test_prediction_preserves_aligned_proxy_pairs_and_chunks_calls():
    X, W, Z, A, Y = episode(n=4)
    model = RecordingOutcomeModel()
    estimator = TabICLGComp(model, proxy_top_k=4, predict_batch_size=3)
    estimator.fit(X, A, Y, W=W, Z=Z)
    estimator.predict(np.array([[0.0]]))

    synthetic = np.vstack(model.predict_calls)
    assert all(len(call) <= 3 for call in model.predict_calls)
    assert len(synthetic) == 8  # four aligned atoms for each of two arms
    observed_pairs = {tuple(row) for row in np.hstack([W, Z])}
    assert {tuple(row[1:3]) for row in synthetic} == observed_pairs


def test_only_top_32_atoms_contribute_and_weights_renormalise():
    X = np.arange(40, dtype=float)[:, None]
    W = np.arange(40, dtype=float)[:, None]
    Z = -np.arange(40, dtype=float)[:, None]
    A = np.arange(40) % 2
    Y = np.zeros(40)
    estimator = TabICLGComp(RecordingOutcomeModel())
    estimator.fit(X, A, Y, W=W, Z=Z)

    indices, weights = estimator._topk_atoms(np.array([[0.0]]))

    assert indices.shape == weights.shape == (1, 32)
    np.testing.assert_array_equal(indices[0], np.arange(32))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights > 0)


def test_query_proxies_are_ignored_and_output_is_per_query():
    estimator, _ = fitted()
    X_query = np.array([[-0.5], [0.5]])

    first = estimator.predict(X_query, W=np.full((2, 1), -1e6), Z=np.full((2, 1), 1e6))
    second = estimator.predict(X_query, W=np.full((2, 1), 7.0), Z=np.full((2, 1), 11.0))

    assert first.shape == (2,)
    np.testing.assert_array_equal(first, second)


def test_fewer_than_32_rows_constant_x_and_extreme_query_stay_finite():
    X = np.zeros((4, 1))
    W = np.arange(4, dtype=float)[:, None]
    Z = (10 + np.arange(4, dtype=float))[:, None]
    estimator = TabICLGComp(RecordingOutcomeModel())
    estimator.fit(X, np.array([0, 1, 0, 1]), np.zeros(4), W=W, Z=Z)

    prediction = estimator.predict(np.array([[1e300], [-1e300]]))
    indices, weights = estimator._topk_atoms(np.array([[1e300]]))

    assert indices.shape == weights.shape == (1, 4)
    np.testing.assert_allclose(weights, 0.25)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("missing", ["W", "Z"])
def test_fit_requires_both_proxy_blocks(missing):
    X, W, Z, A, Y = episode()
    kwargs = {"W": W, "Z": Z}
    kwargs[missing] = None
    with pytest.raises(ValueError, match="requires both W and Z"):
        TabICLGComp(RecordingOutcomeModel()).fit(X, A, Y, **kwargs)


def test_predict_before_fit_is_refused():
    with pytest.raises(ValueError, match=r"Call fit\(\) before predict"):
        TabICLGComp(RecordingOutcomeModel()).predict(np.zeros((1, 1)))


def test_config_validation():
    with pytest.raises(ValueError, match="proxy_top_k"):
        TabICLGComp(RecordingOutcomeModel(), proxy_top_k=0)
    with pytest.raises(ValueError, match="kernel_scale"):
        TabICLGComp(RecordingOutcomeModel(), kernel_scale=0)
    with pytest.raises(ValueError, match="predict_batch_size"):
        TabICLGComp(RecordingOutcomeModel(), predict_batch_size=0)


def test_factory_defaults_overrides_device_and_estimand(monkeypatch):
    class FakeTabICL:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(methods, "TabICLRegressor", FakeTabICL)
    monkeypatch.setattr(methods, "_TABICL_AVAILABLE", True)
    compute = ComputeConfig(device="mps", execution="inline")
    cfg = {
        "name": "gcomp",
        "cls": "TabICLGComp",
        "proxy_top_k": 7,
        "kernel_scale": 2.5,
        "predict_batch_size": 13,
        # This stale value must lose to the resolved ComputeConfig.
        "tabicl": {"kv_cache": "repr", "device": "cuda"},
    }

    name, estimator = build_method(cfg, compute)

    assert name == "gcomp"
    assert estimator.proxy_top_k == 7
    assert estimator.kernel_scale == 2.5
    assert estimator.predict_batch_size == 13
    assert estimator.outcome_model.kwargs["kv_cache"] == "repr"
    assert estimator.outcome_model.kwargs["device"] == "mps"
    assert method_estimand(cfg) == "cate(x)"

    _, defaults = build_method({"name": "defaults", "cls": "TabICLGComp"})
    assert defaults.proxy_top_k == 32
    assert defaults.kernel_scale == 1.0
    assert defaults.predict_batch_size == 4096
