"""The sweep chooses context and query rows when it runs, not when it builds."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.analysis_synthetic_proxy_sweep_linear.generate_synthetic_sweep import (
    StudyConfig,
    generate_sweep,
)
from analysis.analysis_synthetic_proxy_sweep_linear.run_ablation import (
    _compute_setting,
    restrict_aligned,
    restrict_context,
    run,
)


def test_generated_rows_can_be_split_at_run_time(tmp_path):
    data_dir = tmp_path / "data"
    generate_sweep(data_dir, StudyConfig(n_rows=200, n_levels=2, dimensions=(1,)))
    with (data_dir / "index.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert int(row["n_rows"]) == 200
    assert "n_train" not in row and "n_test" not in row
    with np.load(data_dir / row["observed_path"]) as data:
        observed = {key: data[key] for key in data.files}
    assert "split" not in observed

    selected = restrict_context(observed, 80, 40)
    expected_ids = np.r_[np.arange(80), np.arange(160, 200)]
    np.testing.assert_array_equal(selected["row_id"], expected_ids)
    np.testing.assert_array_equal(selected["split"], np.r_[np.zeros(80), np.ones(40)])
    np.testing.assert_array_equal(
        restrict_aligned(np.arange(200), 80, 40), expected_ids
    )
    np.testing.assert_array_equal(
        restrict_context(observed, 120, 40)["row_id"][-40:], expected_ids[-40:]
    )

    with np.load(data_dir / row["truth_path"]) as data:
        expected_ate = data["CATE"][-40:].mean()
    args = SimpleNamespace(
        data_dir=data_dir,
        oracle=False,
        context_sizes=[80, 120],
        n_query=40,
        fail_fast=True,
    )
    fits = _compute_setting(args, row, ["backdoor_ols_x"], {})
    assert [fit["status"] for fit in fits] == ["ok", "ok"]
    assert [fit["n_test"] for fit in fits] == [40, 40]
    assert all(fit["true_test_ate"] == pytest.approx(expected_ate) for fit in fits)


def test_run_rejects_oversized_context_and_query_before_fitting(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dimension",
                "joint_r2_target",
                "level_id",
                "observed_path",
                "truth_path",
                "n_rows",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dimension": 1,
                "joint_r2_target": 0.5,
                "level_id": 0,
                "observed_path": "observed.npz",
                "truth_path": "truth.npz",
                "n_rows": 200,
            }
        )
    args = SimpleNamespace(
        data_dir=data_dir,
        dimensions=None,
        levels=None,
        limit=None,
        context_sizes=[128],
        n_query=73,
    )
    with pytest.raises(ValueError, match="128.*73.*200 total rows"):
        run(args)
