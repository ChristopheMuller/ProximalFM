"""Contracts for the self-contained nonlinear proxy-strength study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis_synthetic_proxy_sweep_nonlinear.generate_nonlinear_sweep import (
    TRANSFORM_ALPHA,
    TRANSFORM_SCALE,
    StudyConfig,
    generate_sweep,
    inverse_nonlinear_transform,
    nonlinear_transform,
)
import analysis_synthetic_proxy_sweep_nonlinear.run_ablation as runner


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_cubic_transform_is_monotone_invertible_and_gaussian_standardized():
    values = np.linspace(-8.0, 8.0, 2001)
    transformed = nonlinear_transform(values)

    assert np.all(np.diff(transformed) > 0)
    np.testing.assert_allclose(
        inverse_nonlinear_transform(transformed), values, atol=2e-13, rtol=2e-13
    )
    assert TRANSFORM_SCALE**2 == 1 + 6 * TRANSFORM_ALPHA + 15 * TRANSFORM_ALPHA**2

    # Gauss-Hermite integrates this degree-six Gaussian moment exactly.
    nodes, weights = np.polynomial.hermite.hermgauss(4)
    normal_nodes = np.sqrt(2.0) * nodes
    mean = np.sum(weights * nonlinear_transform(normal_nodes)) / np.sqrt(np.pi)
    second_moment = np.sum(weights * nonlinear_transform(normal_nodes) ** 2) / np.sqrt(
        np.pi
    )
    assert abs(mean) < 1e-15
    assert abs(second_moment - 1.0) < 1e-14


def test_generator_builds_own_data_and_preserves_grid_pairing(tmp_path):
    output = tmp_path / "nonlinear"
    config = StudyConfig(n_rows=200, n_levels=3, dimensions=(1,), seed=17)

    manifest = generate_sweep(output, config)

    assert manifest["status"] == "complete"
    assert manifest["n_datasets"] == 3
    assert "source_linear_data" not in manifest["recipe"]
    with (output / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3

    reference_observed = load_npz(output / rows[0]["observed_path"])
    reference_truth = load_npz(output / rows[0]["truth_path"])
    for row in rows:
        observed = load_npz(output / row["observed_path"])
        truth = load_npz(output / row["truth_path"])
        for name in ("row_id", "split", "X", "A", "Y"):
            np.testing.assert_array_equal(observed[name], reference_observed[name])
        for name in reference_truth:
            np.testing.assert_array_equal(truth[name], reference_truth[name])

        metadata = json.loads((output / row["metadata_path"]).read_text())
        diagnostics = metadata["diagnostics"]
        assert diagnostics["inverse_roundtrip_max_error"] < 1e-12
        assert diagnostics["nonlinear_bridge_residual_identity_max_error"] < 1e-12
        assert metadata["observed_scale_ols_r2_is_controlled"] is False
        assert metadata["recipe"]["transform"]["invertible"] is True

    assert not np.array_equal(
        load_npz(output / rows[0]["observed_path"])["W"],
        load_npz(output / rows[-1]["observed_path"])["W"],
    )


def test_small_config_controls_rows_levels_and_dimensions(tmp_path):
    output = tmp_path / "small"
    config = StudyConfig(n_rows=240, n_levels=2, dimensions=(1, 2), seed=23)

    manifest = generate_sweep(output, config)

    assert manifest["n_datasets"] == 4
    assert manifest["n_rows_per_dataset"] == 240
    assert manifest["dimensions"] == [1, 2]
    assert len(manifest["r2_grid"]) == 2
    with (output / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {(int(row["dimension"]), int(row["level_id"])) for row in rows} == {
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    }
    assert {int(row["n_rows"]) for row in rows} == {240}


def test_generation_is_deterministic_and_rejects_a_changed_recipe(tmp_path):
    config = StudyConfig(n_rows=200, n_levels=2, dimensions=(1,), seed=29)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_sweep(first, config)
    generate_sweep(second, config)

    with (first / "index.csv").open(newline="") as stream:
        first_rows = list(csv.DictReader(stream))
    with (second / "index.csv").open(newline="") as stream:
        second_rows = list(csv.DictReader(stream))
    for first_row, second_row in zip(first_rows, second_rows):
        for field in ("observed_path", "truth_path"):
            assert (first / first_row[field]).read_bytes() == (
                second / second_row[field]
            ).read_bytes()

    with pytest.raises(ValueError, match="fresh output directory"):
        generate_sweep(
            first,
            StudyConfig(n_rows=200, n_levels=3, dimensions=(1,), seed=29),
        )


def make_runner_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n_context, n_query = 12, 5
    n = n_context + n_query
    rng = np.random.default_rng(4)
    observed = {
        "row_id": np.arange(n),
        "split": np.r_[
            np.zeros(n_context, dtype=np.uint8),
            np.ones(n_query, dtype=np.uint8),
        ],
        "X": rng.normal(size=(n, 2)),
        "W": rng.normal(size=(n, 1)),
        "Z": rng.normal(size=(n, 1)),
        "A": (rng.random(n) < 0.5).astype(np.uint8),
        "Y": rng.normal(size=n),
    }
    truth = {"row_id": observed["row_id"], "CATE": np.ones(n)}
    np.savez(data_dir / "observed.npz", **observed)
    np.savez(data_dir / "truth.npz", **truth)
    (data_dir / "index.csv").write_text(
        "dimension,joint_r2_target,level_id,observed_path,truth_path\n"
        "2,0.5,0,observed.npz,truth.npz\n"
    )
    return data_dir


def runner_args(tmp_path: Path, data_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=data_dir,
        output_dir=tmp_path / "results",
        checkpoint=tmp_path / "missing.ckpt",
        device="cpu",
        query_batch_size=2,
        dimensions=None,
        levels=None,
        limit=None,
        context_sizes=[8, 10],
        force=False,
        fail_fast=True,
    )


def test_runner_executes_exactly_three_methods_and_reuses_cached_contexts(
    tmp_path, monkeypatch
):
    data_dir = make_runner_data(tmp_path)
    args = runner_args(tmp_path, data_dir)
    calls = []

    def fake_fit(method, config, observed, **kwargs):
        n_train = int(np.sum(observed["split"] == 0))
        n_test = int(np.sum(observed["split"] == 1))
        calls.append((method, n_train, config["cls"]))
        return np.zeros(n_test), {
            "status": "ok",
            "n_train": n_train,
            "n_test": n_test,
            "fit_seconds": 0.0,
            "tuning_seconds": 0.0,
            "prediction_seconds": 0.0,
            "prediction_max_abs": 0.0,
            "diagnostics_json": "{}",
        }

    monkeypatch.setattr(runner, "fit_and_predict", fake_fit)

    assert runner.run(args) == 0
    assert len(calls) == 6
    assert {method for method, _, _ in calls} == set(runner.METHODS)
    assert {context for _, context, _ in calls} == {8, 10}

    args.context_sizes = [8, 10, 11]
    assert runner.run(args) == 0
    assert len(calls) == 9
    assert {context for _, context, _ in calls[-3:]} == {11}
    with (args.output_dir / "run_metadata.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["cache"]["hits"] == 6
    assert metadata["cache"]["computed"] == 3


def test_declared_method_configs_pin_rf_bridge_and_empirical_tabicl(tmp_path):
    configs = runner.method_configs(tmp_path / "model.ckpt")

    assert tuple(configs) == runner.METHODS
    assert configs[runner.METHOD_P_RF]["bridge"] == "kernel"
    assert configs[runner.METHOD_P_RF]["base"] == "rf"
    assert configs[runner.METHOD_P_RF]["n_splits"] == 4
    tabicl = configs[runner.METHOD_TABICL]
    assert tabicl["estimator"]["base"] == "tabicl"
    assert tabicl["estimator"]["tabicl"]["n_estimators"] == 1
    assert tabicl["integration"] == {
        "method": "empirical",
        "n_points": 64,
        "seed": 0,
    }
