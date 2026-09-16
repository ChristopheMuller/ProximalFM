"""Behavioural checks for the same-context Causal Chambers ATE protocol."""

import sys

import numpy as np
import pandas as pd
import torch
import yaml

from benchmarks.causal_chambers import run_ate
from benchmarks.methods import (
    CausalFM_Estimator,
    CausalPFN_Estimator,
    DoPFN_Estimator,
    FeatureBuilder,
    S_Learner,
    T_Learner,
    X_Learner,
)


class _ZeroATE:
    """Dummy method whose native scalar ATE is always zero."""

    def __init__(self, fitted_contexts):
        self.fitted_contexts = fitted_contexts

    def fit(self, X, A, Y, *, W, Z):
        self.fitted_contexts.append(
            {
                "X": np.asarray(X).copy(),
                "A": np.asarray(A).copy(),
                "Y": np.asarray(Y).copy(),
                "W": np.asarray(W).copy(),
                "Z": np.asarray(Z).copy(),
            }
        )
        return self

    def predict_ate(self):
        return 0.0


class _AdditiveOutcomeModel:
    """Toy f(X, W, Z, A) whose treatment contrast is exactly one."""

    def fit(self, X, Y):
        self.fit_X = np.asarray(X).copy()
        self.fit_Y = np.asarray(Y).copy()
        return self

    def predict(self, X):
        self.predict_X = np.asarray(X).copy()
        # FeatureBuilder puts A in the last column when interactions are off.
        return self.predict_X[:, 0] + self.predict_X[:, -1]


class _MeanModel:
    """Toy regressor returning the mean target on which it was fitted."""

    def fit(self, X, Y):
        self.mean_ = float(np.mean(Y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class _FakeCausalPFN:
    """Minimal direct-CATE API used to exercise the adapter without a checkpoint."""

    def fit(self, X, A, Y):
        self.fit_X = np.asarray(X).copy()
        self.fit_A = np.asarray(A).copy()
        self.fit_Y = np.asarray(Y).copy()

    def estimate_cate(self, X):
        self.predict_X = np.asarray(X).copy()
        return self.predict_X[:, 0]


class _FakeDoPFN:
    """Minimal outcome-prediction API whose outcome equals the A feature."""

    def fit(self, X, Y):
        self.fit_X = np.asarray(X).copy()
        self.fit_Y = np.asarray(Y).copy()

    def predict(self, X):
        self.predict_X = X.detach().cpu().numpy().copy()
        return X[:, -1]


class _FakeCausalFM:
    """Minimal direct-CATE API used to exercise the CausalFM adapter."""

    def estimate_cate(self, ctx_x, ctx_a, ctx_y, x_test):
        self.ctx_x = ctx_x.detach().cpu().numpy().copy()
        self.ctx_a = ctx_a.detach().cpu().numpy().copy()
        self.ctx_y = ctx_y.detach().cpu().numpy().copy()
        self.predict_x = x_test.detach().cpu().numpy().copy()
        return {"cate": x_test[:, 0]}


def test_zero_ate_uses_the_same_prefix_for_fit_and_reference(tmp_path, monkeypatch):
    """A zero estimator has |error| equal to the paired prefix ATE.

    The signed error convention is estimate minus reference, hence it is
    ``-ate_ref``.  This checks the key physical-benchmark property directly:
    each growing context has its own paired reference rather than sharing a
    fixed held-out target.
    """
    n_rows = 8
    tau = np.arange(1, n_rows + 1, dtype=float)
    factual_y = 100 + np.arange(n_rows, dtype=float)
    frame = pd.DataFrame(
        {
            "x": np.arange(n_rows, dtype=float),
            "w": 10 + np.arange(n_rows, dtype=float),
            "z": 20 + np.arange(n_rows, dtype=float),
            "a": np.tile([0, 1], n_rows // 2),
            "y": factual_y,
            "y0": np.zeros(n_rows),
            "y1": tau,
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame.to_csv(data_dir / "condition.csv", index=False)

    output_dir = tmp_path / "results"
    config = {
        "run": {
            "seed": 17,
            "compute": {
                "device": "cpu",
                "n_workers": 1,
                "torch_threads": 1,
                "blas_threads": 1,
                "n_jobs": 1,
            },
        },
        "ate": {"context_sizes": [3, 5], "n_permutations": 2},
        "data": {
            "csv_dir": str(data_dir),
            "columns": {
                "X": "x",
                "W": "w",
                "Z": "z",
                "A": "a",
                "Y": "y",
                "Y0": "y0",
                "Y1": "y1",
            },
        },
        "output_dir": str(output_dir),
        "methods": [{"name": "Zero", "cls": "test-only"}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    fitted_contexts = []
    monkeypatch.setattr(
        run_ate,
        "build_method",
        lambda cfg, compute: ("Zero", _ZeroATE(fitted_contexts)),
    )
    monkeypatch.setattr(run_ate, "apply_compute", lambda compute: None)
    monkeypatch.setattr(sys, "argv", ["run_ate.py", "--config", str(config_path)])

    run_ate.main()

    results = pd.read_csv(output_dir / "results.csv")
    assert len(results) == 4  # 2 permutations × 2 nested context sizes
    for row in results.itertuples(index=False):
        prefix = run_ate._permutation(17, 0, row.permutation, n_rows)[: row.n_ctx]
        expected_ate = tau[prefix].mean()
        np.testing.assert_allclose(row.ate_hat, 0.0)
        np.testing.assert_allclose(row.ate_ref, expected_ate)
        np.testing.assert_allclose(row.error, -expected_ate)
        np.testing.assert_allclose(row.abs_error, expected_ate)

    # Every fit saw the same prefix used to calculate that result's reference,
    # and received factual Y rather than either potential-outcome column.
    for fit, row in zip(fitted_contexts, results.itertuples(index=False)):
        prefix = run_ate._permutation(17, 0, row.permutation, n_rows)[: row.n_ctx]
        np.testing.assert_array_equal(fit["X"].ravel(), frame["x"].to_numpy()[prefix])
        np.testing.assert_array_equal(fit["Y"], factual_y[prefix])

    # A second identical invocation must resume from the four scalar cache
    # entries rather than construct or fit another estimator.
    monkeypatch.setattr(
        run_ate,
        "build_method",
        lambda *_: (_ for _ in ()).throw(AssertionError("cache was not reused")),
    )
    run_ate.main()
    pd.testing.assert_frame_equal(results, pd.read_csv(output_dir / "results.csv"))


def test_s_learner_predict_ate_averages_counterfactuals_on_its_fitted_context():
    X = np.array([[1.0], [3.0], [7.0]])
    W = np.array([[10.0], [11.0], [12.0]])
    Z = np.array([[20.0], [21.0], [22.0]])
    A = np.array([0.0, 1.0, 0.0])
    Y = np.array([100.0, 101.0, 102.0])
    base = _AdditiveOutcomeModel()
    learner = S_Learner(base, use_W=True, use_Z=True)

    learner.fit(X, A, Y, W=W, Z=Z)
    got = learner.predict_ate()

    # f(x, w, z, 1) - f(x, w, z, 0) = 1 for every stored context row.
    assert got == 1.0
    queried = base.predict_X
    n = len(X)
    np.testing.assert_array_equal(queried[:n, 0], X[:, 0])
    np.testing.assert_array_equal(queried[n:, 0], X[:, 0])
    np.testing.assert_array_equal(queried[:n, -1], np.ones(n))
    np.testing.assert_array_equal(queried[n:, -1], np.zeros(n))


def test_t_and_x_learners_predict_ate_average_their_fitted_context():
    X = np.array([[1.0], [3.0], [7.0], [9.0]])
    W = np.array([[10.0], [11.0], [12.0], [13.0]])
    Z = np.array([[20.0], [21.0], [22.0], [23.0]])
    A = np.array([0.0, 1.0, 0.0, 1.0])
    Y = np.array([1.0, 3.0, 2.0, 4.0])

    for learner_cls in (T_Learner, X_Learner):
        learner = learner_cls(_MeanModel, use_W=True, use_Z=True)
        learner.fit(X, A, Y, W=W, Z=Z)

        # Both learners recover the two-unit arm difference and average it over
        # their stored fit context without accepting query covariates.
        np.testing.assert_allclose(learner.predict_ate(), 2.0)


def test_causalpfn_predict_ate_averages_direct_cate_on_its_fitted_context():
    X = np.array([[2.0], [4.0], [8.0]])
    W = np.array([[10.0], [11.0], [12.0]])
    Z = np.array([[20.0], [21.0], [22.0]])
    A = np.array([0.0, 1.0, 0.0])
    Y = np.array([100.0, 101.0, 102.0])
    fake = _FakeCausalPFN()

    # Bypass the installed-model constructor: only the adapter protocol is
    # under test here, not CausalPFN's external checkpoint loading.
    learner = object.__new__(CausalPFN_Estimator)
    learner.model = fake
    learner.fb = FeatureBuilder(use_W=True, use_Z=True)
    learner.fit(X, A, Y, W=W, Z=Z)

    np.testing.assert_allclose(learner.predict_ate(), X[:, 0].mean())
    np.testing.assert_array_equal(fake.fit_A, A)
    np.testing.assert_array_equal(fake.predict_X[:, 0], X[:, 0])


def test_dopfn_predict_ate_averages_two_arm_contrast_on_its_fitted_context():
    X = np.array([[2.0], [4.0], [8.0]])
    W = np.array([[10.0], [11.0], [12.0]])
    Z = np.array([[20.0], [21.0], [22.0]])
    A = np.array([0.0, 1.0, 0.0])
    Y = np.array([100.0, 101.0, 102.0])
    fake = _FakeDoPFN()

    # Bypass the external-model constructor: this tests the adapter's two-arm
    # outcome-prediction protocol, not DoPFN's checkpoint loading.
    learner = object.__new__(DoPFN_Estimator)
    learner.model = fake
    learner.fb = FeatureBuilder(use_W=True, use_Z=True)
    learner.fit(X, A, Y, W=W, Z=Z)

    np.testing.assert_allclose(learner.predict_ate(), 1.0)
    np.testing.assert_array_equal(fake.fit_X[:, 0], X[:, 0])
    np.testing.assert_array_equal(fake.predict_X[:, -1], np.zeros(len(X)))


def test_causalfm_predict_ate_averages_direct_cate_on_its_fitted_context():
    X = np.array([[2.0], [4.0], [8.0]])
    W = np.array([[10.0], [11.0], [12.0]])
    Z = np.array([[20.0], [21.0], [22.0]])
    A = np.array([0.0, 1.0, 0.0])
    Y = np.array([100.0, 101.0, 102.0])
    fake = _FakeCausalFM()

    # Bypass the external checkpoint constructor: this tests the in-context
    # adapter and its scalar aggregation protocol.
    learner = object.__new__(CausalFM_Estimator)
    learner.model = fake
    learner.fb = FeatureBuilder(use_W=True, use_Z=True)
    learner.device = torch.device("cpu")
    learner.ctx_x = learner.ctx_a = learner.ctx_y = None
    learner.fit(X, A, Y, W=W, Z=Z)

    np.testing.assert_allclose(learner.predict_ate(), X[:, 0].mean())
    np.testing.assert_array_equal(fake.ctx_a.ravel(), A)
    np.testing.assert_array_equal(fake.predict_x[:, 0], X[:, 0])
