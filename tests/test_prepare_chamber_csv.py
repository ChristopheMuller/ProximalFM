"""Tests for benchmarks/causal_chambers/prepare_chamber_csv.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "causal_chambers"
    / "prepare_chamber_csv.py"
)
_spec = importlib.util.spec_from_file_location("prepare_chamber_csv", _MODULE_PATH)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def raw_frame(T=20, a_levels=(0.0, 2047.0)):
    return pd.DataFrame(
        {
            "green": np.arange(T, dtype=float),
            "red": np.arange(T, dtype=float) * 2,
            "pol_1": np.arange(T, dtype=float) * 3,
            "led_2_uv": np.arange(T, dtype=float) * 5,
            "led_3_uv": np.tile(np.array(a_levels), T // len(a_levels)),
            "ir_3": np.arange(T, dtype=float) + 0.5,
            "Y0": np.arange(T, dtype=float) + 100,
            "Y1": np.arange(T, dtype=float) + 200,
        }
    )


def test_canonical_schema_and_binary_treatment():
    out, diag = prep.prepare(raw_frame(), prep.CHAMBER_DEFAULTS)
    assert list(out.columns) == ["U_1", "X_1", "W_1", "Z_1", "A", "Y", "Y0", "Y1"]
    assert set(np.unique(out["A"])) == {0.0, 1.0}
    assert diag["n_rows"] == 20


def test_y_equals_potential_outcome_under_assigned_treatment():
    df = raw_frame()
    out, _ = prep.prepare(df, prep.CHAMBER_DEFAULTS, y_source="potential")
    expected = np.where(out["A"] > 0.5, out["Y1"], out["Y0"])
    np.testing.assert_array_equal(out["Y"].to_numpy(), expected)


def test_y_source_observed_keeps_measured_column():
    df = raw_frame()
    out, _ = prep.prepare(df, prep.CHAMBER_DEFAULTS, y_source="observed")
    np.testing.assert_array_equal(out["Y"].to_numpy(), df["ir_3"].to_numpy())


def test_multi_level_treatment_raises_instead_of_rescaling():
    # Three levels must fail loudly: min-max would leave an intermediate value
    # that T/X-Learner arm masks (A == 0 / A == 1) silently drop.
    df = raw_frame(T=21, a_levels=(0.0, 1000.0, 2047.0))
    with pytest.raises(ValueError, match="binary treatment is required"):
        prep.prepare(df, prep.CHAMBER_DEFAULTS)


def test_diagnostics_flag_absent_heterogeneity():
    df = raw_frame()
    # Y1 - Y0 is exactly 100 for every row => zero heterogeneity.
    _, diag = prep.prepare(df, prep.CHAMBER_DEFAULTS)
    assert diag["ate_true"] == pytest.approx(100.0)
    assert diag["sd_tau"] == pytest.approx(0.0)
    assert diag["sd_tau_over_ate"] == pytest.approx(0.0)


def test_diagnostics_report_naive_confounding_bias():
    df = raw_frame()
    # Assign treatment to the high-Y0 half so the naive contrast is biased.
    df["led_3_uv"] = np.where(np.arange(20) >= 10, 2047.0, 0.0)
    _, diag = prep.prepare(df, prep.CHAMBER_DEFAULTS)
    assert diag["naive_bias_pct_of_ate"] > 0  # upward-biased by construction


def test_read_raw_csv_drops_unnamed_index(tmp_path):
    df = raw_frame()
    with_index = tmp_path / "with_index.csv"
    without_index = tmp_path / "without_index.csv"
    df.to_csv(with_index)  # leading unnamed index column
    df.to_csv(without_index, index=False)

    a = prep.read_raw_csv(with_index)
    b = prep.read_raw_csv(without_index)
    assert list(a.columns) == list(df.columns)
    assert list(b.columns) == list(df.columns)
