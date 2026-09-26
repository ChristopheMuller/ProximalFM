"""The physical-benchmark CSV adapter preserves its causal columns."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


path = Path(__file__).parents[1] / "benchmarks/causal_chambers/prepare_chamber_csv.py"
spec = importlib.util.spec_from_file_location("prepare_chamber_csv", path)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _raw(levels=(0.0, 2047.0)):
    n = 20
    return pd.DataFrame(
        {
            "green": np.arange(n),
            "red": np.arange(n),
            "pol_1": np.arange(n),
            "led_2_uv": np.arange(n),
            "led_3_uv": np.resize(levels, n),
            "ir_3": np.arange(n),
            "Y0": np.arange(n),
            "Y1": np.arange(n) + 2,
        }
    )


def test_prepare_emits_binary_treatment_and_factual_potential_outcome():
    out, _ = prep.prepare(_raw(), prep.CHAMBER_DEFAULTS)
    assert set(out["A"].unique()) == {0.0, 1.0}
    np.testing.assert_array_equal(out["Y"], np.where(out["A"], out["Y1"], out["Y0"]))


def test_prepare_refuses_nonbinary_treatment():
    with pytest.raises(ValueError, match="binary treatment"):
        prep.prepare(_raw((0.0, 1000.0, 2047.0)), prep.CHAMBER_DEFAULTS)
