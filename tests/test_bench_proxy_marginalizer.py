"""Generic proxy marginalisation for proxy-conditional CATE estimators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import benchmarks.methods as methods
from benchmarks.compute import ComputeConfig
from benchmarks.methods import build_method, method_estimand
from benchmarks.methods.proxy_marginalizer import (
    IntegrationSpec,
    ProxyMarginalizer,
)
from benchmarks.runners import method_output_names


class RecordingCATEEstimator:
    def __init__(self):
        self.fit_args = None
        self.predict_calls = []

    def fit(self, X, A, Y, *, W=None, Z=None):
        self.fit_args = tuple(np.array(v, copy=True) for v in (X, A, Y, W, Z))
        return self

    def predict(self, X, *, W=None, Z=None):
        self.predict_calls.append(tuple(np.array(v, copy=True) for v in (X, W, Z)))
        return 0.5 * X[:, 0] + 2.0 * W[:, 0] + 3.0 * Z[:, 0]

    def diagnostics(self):
        return {"nested_score": 7.0}


def episode(n=8):
    X = np.linspace(-2.0, 2.0, n)[:, None]
    W = (10.0 + np.arange(n))[:, None]
    Z = (100.0 + 2.0 * np.arange(n))[:, None]
    A = (np.arange(n) % 2).astype(float)
    Y = np.arange(n, dtype=float)
    return X, W, Z, A, Y


def multi_outputs():
    return [
        {"name": "knn-2", "integration": {"method": "knn", "n_points": 2}},
        {"name": "knn-4", "integration": {"method": "knn", "n_points": 4}},
        {
            "name": "kernel-2",
            "integration": {
                "method": "kernel",
                "n_points": 2,
                "kernel_scale": 1.0,
            },
        },
        {
            "name": "kernel-4",
            "integration": {
                "method": "kernel",
                "n_points": 4,
                "kernel_scale": 2.0,
            },
        },
    ]


def fit_wrapper(*, n=8, outputs=None, integration=None, batch_size=4096):
    X, W, Z, A, Y = episode(n)
    nested = RecordingCATEEstimator()
    wrapper = ProxyMarginalizer(
        nested,
        outputs=outputs,
        integration=integration,
        predict_batch_size=batch_size,
    )
    wrapper.fit(X, A, Y, W=W, Z=Z)
    return wrapper, nested


def manual_neighbours(X_context, X_query, n_points):
    sigma = 2.0 * np.median(cdist(X_context, X_context)) ** 2
    sq_distance = (X_query - X_context.T) ** 2 / sigma
    indices = np.argsort(sq_distance, axis=1, kind="stable")[:, :n_points]
    selected = np.take_along_axis(sq_distance, indices, axis=1)
    return indices, selected


def test_knn_and_kernel_match_manual_aggregation():
    X, W, Z, _, _ = episode()
    wrapper, _ = fit_wrapper(outputs=multi_outputs())
    query = np.array([[0.25], [1.75]])

    actual = wrapper.predict_variants(query)

    indices, selected = manual_neighbours(X, query, 4)
    atom_effect = 0.5 * query + 2.0 * W[indices, 0] + 3.0 * Z[indices, 0]
    np.testing.assert_allclose(actual["knn-2"], atom_effect[:, :2].mean(axis=1))
    np.testing.assert_allclose(actual["knn-4"], atom_effect.mean(axis=1))

    weights2 = np.exp(-(selected[:, :2] - selected[:, :1]))
    weights2 /= weights2.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        actual["kernel-2"], np.sum(atom_effect[:, :2] * weights2, axis=1)
    )
    weights4 = np.exp(-(selected - selected[:, :1]) / 2.0)
    weights4 /= weights4.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        actual["kernel-4"], np.sum(atom_effect * weights4, axis=1)
    )


def test_one_maximum_support_prediction_is_shared_and_pairs_stay_aligned():
    wrapper, nested = fit_wrapper(outputs=multi_outputs(), batch_size=5)
    query = np.array([[-0.25], [0.75]])

    wrapper.predict_variants(query)

    # max(n_points)=4, so two queries create eight predictions total. Chunking
    # changes calls, not the set of evaluated atoms.
    assert sum(len(call[0]) for call in nested.predict_calls) == 8
    assert all(len(call[0]) <= 5 for call in nested.predict_calls)
    predicted_wz = np.vstack(
        [np.hstack([call[1], call[2]]) for call in nested.predict_calls]
    )
    context_wz = {tuple(row) for row in np.hstack(episode()[1:3])}
    assert {tuple(row) for row in predicted_wz} <= context_wz


def test_empirical_uses_one_seeded_random_aligned_sample_for_every_query():
    X, W, Z, _, _ = episode()
    wrapper, nested = fit_wrapper(
        integration={"method": "empirical", "n_points": 3, "seed": 7}, batch_size=4
    )
    query = np.array([[-1.25], [0.75]])

    actual = wrapper.predict(query)

    indices = np.random.default_rng(7).choice(len(X), size=3, replace=False)
    expected = 0.5 * query[:, 0] + np.mean(2.0 * W[indices, 0] + 3.0 * Z[indices, 0])
    np.testing.assert_allclose(actual, expected)
    predicted_wz = np.vstack(
        [np.hstack([call[1], call[2]]) for call in nested.predict_calls]
    ).reshape(len(query), 3, 2)
    expected_atoms = np.hstack([W[indices], Z[indices]])
    np.testing.assert_array_equal(
        predicted_wz, np.broadcast_to(expected_atoms, predicted_wz.shape)
    )


def test_empirical_fit_does_not_compute_pairwise_context_distances(monkeypatch):
    import benchmarks.methods.proxy_marginalizer as marginalizer

    def fail_if_called(*args, **kwargs):
        raise AssertionError("empirical integration does not need context distances")

    monkeypatch.setattr(marginalizer, "cdist", fail_if_called)
    wrapper, _ = fit_wrapper(
        integration={"method": "empirical", "n_points": 3, "seed": 7}
    )
    assert wrapper.predict(np.array([[0.0]])).shape == (1,)


def test_empirical_support_caps_at_context_size():
    wrapper, nested = fit_wrapper(
        n=4, integration={"method": "empirical", "n_points": 128}
    )

    prediction = wrapper.predict(np.array([[0.0], [1.0]]))

    assert prediction.shape == (2,)
    assert sum(len(call[0]) for call in nested.predict_calls) == 8


def test_mixed_empirical_and_local_outputs_keep_declared_order():
    outputs = [
        {"name": "local", "integration": {"method": "knn", "n_points": 2}},
        {
            "name": "global",
            "integration": {"method": "empirical", "n_points": 3, "seed": 4},
        },
    ]
    wrapper, _ = fit_wrapper(outputs=outputs)

    result = wrapper.predict_variants(np.array([[0.0]]))

    assert list(result) == ["local", "global"]


def test_fit_is_forwarded_unchanged_and_diagnostics_are_forwarded():
    X, W, Z, A, Y = episode()
    wrapper, nested = fit_wrapper(integration={"method": "knn", "n_points": 2})

    for actual, expected in zip(nested.fit_args, (X, A, Y, W, Z)):
        np.testing.assert_array_equal(actual, expected)
    assert wrapper.diagnostics() == {"nested_score": 7.0}


def test_query_proxies_are_ignored_and_single_output_uses_none_key():
    wrapper, _ = fit_wrapper(integration={"method": "knn", "n_points": 2})
    query = np.array([[-0.5], [0.5]])

    first = wrapper.predict_variants(
        query, W=np.full((2, 1), -1e6), Z=np.full((2, 1), 1e6)
    )
    second = wrapper.predict_variants(
        query, W=np.full((2, 1), 7.0), Z=np.full((2, 1), 11.0)
    )

    assert list(first) == [None]
    assert first[None].shape == (2,)
    np.testing.assert_array_equal(first[None], second[None])


def test_support_caps_at_context_and_extreme_constant_x_stays_finite():
    X = np.zeros((4, 1))
    W = np.arange(4, dtype=float)[:, None]
    Z = (10 + np.arange(4, dtype=float))[:, None]
    nested = RecordingCATEEstimator()
    wrapper = ProxyMarginalizer(
        nested,
        outputs=[
            {"name": "knn", "integration": {"method": "knn", "n_points": 128}},
            {
                "name": "kernel",
                "integration": {"method": "kernel", "n_points": 128},
            },
        ],
    )
    wrapper.fit(X, np.array([0, 1, 0, 1]), np.zeros(4), W=W, Z=Z)

    prediction = wrapper.predict_variants(np.array([[1e300], [-1e300]]))

    assert sum(len(call[0]) for call in nested.predict_calls) == 8
    assert all(np.isfinite(value).all() for value in prediction.values())
    np.testing.assert_allclose(prediction["knn"], prediction["kernel"])


@pytest.mark.parametrize("missing", ["W", "Z"])
def test_fit_requires_both_proxy_blocks(missing):
    X, W, Z, A, Y = episode()
    kwargs = {"W": W, "Z": Z}
    kwargs[missing] = None
    wrapper = ProxyMarginalizer(
        RecordingCATEEstimator(), integration={"method": "knn", "n_points": 2}
    )
    with pytest.raises(ValueError, match="requires both W and Z"):
        wrapper.fit(X, A, Y, **kwargs)


def test_predict_before_fit_is_refused():
    wrapper = ProxyMarginalizer(
        RecordingCATEEstimator(), integration={"method": "knn", "n_points": 2}
    )
    with pytest.raises(ValueError, match=r"Call fit\(\) before predict"):
        wrapper.predict(np.zeros((1, 1)))


@pytest.mark.parametrize(
    "integration,match",
    [
        ({}, "method"),
        ({"method": "sideways", "n_points": 2}, "method"),
        ({"method": "knn", "n_points": 0}, "n_points"),
        ({"method": "knn", "n_points": 2.0}, "n_points"),
        ({"method": "knn", "n_points": 2, "kernel_scale": 1}, "kernel_scale"),
        ({"method": "knn", "n_points": 2, "seed": 1}, "seed"),
        ({"method": "kernel", "n_points": 2, "seed": 1}, "seed"),
        ({"method": "empirical", "n_points": 2, "seed": -1}, "seed"),
        ({"method": "empirical", "n_points": 2, "seed": 1.5}, "seed"),
        (
            {"method": "empirical", "n_points": 2, "kernel_scale": 1},
            "kernel_scale",
        ),
        ({"method": "kernel", "n_points": 2, "kernel_scale": 0}, "kernel_scale"),
        ({"method": "kernel", "n_points": 2, "extra": 1}, "unknown"),
    ],
)
def test_integration_validation(integration, match):
    with pytest.raises(ValueError, match=match):
        IntegrationSpec.from_dict(integration, where="integration")


def test_output_and_constructor_validation():
    nested = RecordingCATEEstimator()
    with pytest.raises(ValueError, match="either integration or outputs"):
        ProxyMarginalizer(
            nested,
            integration={"method": "knn", "n_points": 2},
            outputs=multi_outputs(),
        )
    with pytest.raises(ValueError, match="requires integration or outputs"):
        ProxyMarginalizer(nested)
    with pytest.raises(ValueError, match="duplicate output"):
        ProxyMarginalizer(
            nested,
            outputs=[
                {"name": "same", "integration": {"method": "knn", "n_points": 2}},
                {"name": "same", "integration": {"method": "knn", "n_points": 4}},
            ],
        )
    with pytest.raises(ValueError, match="predict_batch_size"):
        ProxyMarginalizer(
            nested,
            integration={"method": "knn", "n_points": 2},
            predict_batch_size=0,
        )
    with pytest.raises(ValueError, match="outputs must be a list"):
        ProxyMarginalizer(nested, outputs={"name": "not-a-list"})


def wrapper_cfg(cls="S_Learner", **estimator_overrides):
    estimator = {
        "cls": cls,
        "base": "linear",
        "use_W": True,
        "use_Z": True,
        **estimator_overrides,
    }
    return {
        "name": f"wrapped-{cls}",
        "cls": "ProxyMarginalizer",
        "estimator": estimator,
        "outputs": multi_outputs(),
    }


@pytest.mark.parametrize("cls", ["S_Learner", "T_Learner", "X_Learner"])
def test_factory_built_backdoor_learners_fit_and_emit_cate_x(cls):
    cfg = wrapper_cfg(cls)
    _, wrapper = build_method(cfg)
    X, W, Z, A, Y = episode(n=20)

    wrapper.fit(X, A, Y, W=W, Z=Z)
    predictions = wrapper.predict_variants(np.array([[-0.5], [0.5]]))

    assert list(predictions) == [output["name"] for output in cfg["outputs"]]
    assert all(value.shape == (2,) for value in predictions.values())
    assert all(np.isfinite(value).all() for value in predictions.values())
    assert method_estimand(cfg) == "cate(x)"
    assert [method_estimand(cfg, output) for output in cfg["outputs"]] == [
        "cate(x)"
    ] * 4
    assert method_output_names(cfg) == [output["name"] for output in cfg["outputs"]]


@pytest.mark.parametrize(
    "estimator,match",
    [
        ({"cls": "P2SLS"}, "proxy-conditional"),
        (
            {"cls": "S_Learner", "base": "linear", "use_W": False, "use_Z": False},
            "proxy-conditional",
        ),
        ({"cls": "ProxyMarginalizer"}, "recursively"),
        (
            {"cls": "S_Learner", "base": "linear", "outputs": []},
            "may not contain",
        ),
        (
            {"cls": "S_Learner", "base": "linear", "compute": {"n_jobs": 2}},
            "may not contain",
        ),
        (
            {"cls": "S_Learner", "base": "linear", "is_oracle": True},
            "may not contain",
        ),
        (
            {"cls": "S_Learner", "base": "linear", "estimand": "cate(x,w)"},
            "may not contain",
        ),
    ],
)
def test_factory_rejects_ineligible_nested_estimators(estimator, match):
    cfg = {
        "name": "bad",
        "cls": "ProxyMarginalizer",
        "estimator": estimator,
        "integration": {"method": "knn", "n_points": 2},
    }
    with pytest.raises(ValueError, match=match):
        build_method(cfg)


def test_nested_tabicl_receives_outer_compute_device(monkeypatch):
    class FakeTabICL:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(methods, "TabICLRegressor", FakeTabICL)
    monkeypatch.setattr(methods, "_TABICL_AVAILABLE", True)
    cfg = wrapper_cfg("S_Learner", base="tabicl", tabicl={"device": "cuda"})
    compute = ComputeConfig(device="mps", execution="inline")

    _, wrapper = build_method(cfg, compute)

    assert wrapper.estimator.model.kwargs["device"] == "mps"


def test_oracle_tag_keeps_u_after_proxy_marginalisation():
    cfg = {**wrapper_cfg(), "is_oracle": True}
    assert method_estimand(cfg) == "cate(x,u)"
    assert all(method_estimand(cfg, output) == "cate(x,u)" for output in cfg["outputs"])
