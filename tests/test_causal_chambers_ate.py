"""Behavioural checks for the same-context Causal Chambers ATE protocol."""

import sys

import numpy as np
import pandas as pd
import yaml

from benchmarks.causal_chambers import run_ate


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
