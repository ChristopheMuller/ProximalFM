"""Tests for the confounding-dim (share dial) analysis.

The analysis is analytic -- it reads the spec's configuration, not the generated
data -- so most checks are pure functions of a spec. One end-to-end run confirms
it writes the promised figure and table.
"""

import numpy as np
import pytest
from scipy import stats

from analysis_prior import generate as gen
from analysis_prior.analyses.confounding import (
    ConfoundingDimAnalysis,
    _conversion_constants,
    _share_beta,
    representative_widths,
)
from analysis_prior.config import PriorSpec
from proximalfm.prior._dataset import SCMPrior


# --------------------------------------------------------------------------
# Pure config-driven helpers
# --------------------------------------------------------------------------


def _spec(**overrides) -> PriorSpec:
    base = dict(
        name="conftest",
        n_datasets=2,
        seq_len=60,
        mcmc_cate=2,
        min_dim_U=1,
        max_dim_U=10,
        min_dim_W=1,
        max_dim_W=10,
        min_dim_Z=1,
        max_dim_Z=10,
        min_dim_X=1,
        max_dim_X=10,
        seed=0,
    )
    base.update(overrides)
    return PriorSpec(**base)


def test_conversion_constants_match_generator_defaults():
    # The figure must use the exact floor/cap the generator uses.
    floor, cap = _conversion_constants()
    sig = SCMPrior.confounding_input_dim
    assert floor == 2 and cap == 0.75
    # And introspection stays pinned to the real signature defaults.
    assert sig(0.0, 20) == floor
    assert sig(0.999, 20) == 3 * 20  # cap -> k = 3n


def test_representative_widths_training_scale():
    # n = d_X + |proxy*| + 1: low = min_dim_X+1 (floored at 4), high =
    # max_dim_X + max_dim_W + 1 = 21, mid = round((4+21)/2) = 12.
    assert representative_widths(_spec()) == [4, 12, 21]


def test_representative_widths_dedupes_when_narrow():
    # A narrow prior can collapse low/mid/high; the result stays sorted+unique.
    widths = representative_widths(_spec(min_dim_X=2, max_dim_X=3, max_dim_W=2))
    assert widths == sorted(set(widths))
    assert all(w >= 4 for w in widths)


def test_share_beta_reads_default():
    assert _share_beta(_spec()) == (1.2, 2.8)


def test_share_beta_falls_back_when_pinned_non_beta():
    # Ablating the dial to a fixed share (uniform) must not break the
    # definitional figure: it documents the shipped Beta law.
    spec = _spec(
        sampled_hp_overrides={
            "confounding_share_of_U_in_Y": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 0.5,
            }
        }
    )
    assert _share_beta(spec) == (1.2, 2.8)


# --------------------------------------------------------------------------
# compute() scalars
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cache():
    return gen.generate(_spec(), n_jobs=1, progress=False)


def test_compute_scalars(tiny_cache):
    results = ConfoundingDimAnalysis().compute(tiny_cache)
    assert results.beta_a == 1.2 and results.beta_b == 2.8
    assert results.mean_share == pytest.approx(1.2 / (1.2 + 2.8))
    assert (results.floor, results.cap) == (2, 0.75)
    assert results.n_values == [4, 12, 21]


def test_compute_floor_atom_matches_analytic(tiny_cache):
    results = ConfoundingDimAnalysis().compute(tiny_cache)
    dist = stats.beta(results.beta_a, results.beta_b)
    for n in results.n_values:
        thr = (results.floor + 0.5) / (n + results.floor + 0.5)
        assert results.per_n[n]["floor_atom"] == pytest.approx(dist.cdf(thr))
        assert results.per_n[n]["floor_share"] == pytest.approx(
            results.floor / (results.floor + n)
        )


# --------------------------------------------------------------------------
# End-to-end render
# --------------------------------------------------------------------------


def test_run_writes_expected_artifacts(tiny_cache, tmp_path):
    artifacts = ConfoundingDimAnalysis().run(tiny_cache, tmp_path)

    names = {p.name for p in artifacts.figures + artifacts.tables}
    assert "confounding_dim_curve.pdf" in names
    assert "confounding_dim_curve.png" in names
    assert "confounding_dim_control.tex" in names
    assert "confounding_dim_control.csv" in names
    for path in artifacts.figures + artifacts.tables:
        assert path.exists()


def test_run_summary_consistent(tiny_cache, tmp_path):
    summary = ConfoundingDimAnalysis().run(tiny_cache, tmp_path).summary
    assert summary["mean_share"] == pytest.approx(0.30)
    assert summary["p_share_gt_cap"] == pytest.approx(
        float(stats.beta(1.2, 2.8).sf(0.75))
    )
    assert set(summary["floor_atom_by_n"]) == set(summary["n_values"])
