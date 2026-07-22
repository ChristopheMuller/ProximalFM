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
    k_pmf,
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
# k_pmf: exact induced pmf of the U-derived column count
# --------------------------------------------------------------------------


def test_k_pmf_sums_to_one():
    k, p = k_pmf(1.2, 2.8, n=12)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert k[0] == 2  # floor
    assert k[-1] == 36  # cap: k = 3n at n=12


def test_k_pmf_floor_matches_closed_form():
    a, b, n = 1.2, 2.8, 10
    k, p = k_pmf(a, b, n)
    thr = 2.5 / (n + 2.5)
    assert k[0] == 2
    assert p[0] == pytest.approx(stats.beta(a, b).cdf(thr))


def test_k_pmf_cap_matches_closed_form():
    a, b, n = 1.2, 2.8, 10
    k, p = k_pmf(a, b, n)
    thr = (3 * n - 0.5) / (n + 3 * n - 0.5)
    assert k[-1] == 3 * n
    assert p[-1] == pytest.approx(1.0 - stats.beta(a, b).cdf(thr))


def test_k_pmf_matches_monte_carlo():
    # Cross-check the closed form against direct sampling through the actual
    # generator function, so the analysis can never quietly drift from what
    # SCMPrior.confounding_input_dim does.
    a, b, n = 1.2, 2.8, 12
    k_vals, probs = k_pmf(a, b, n)

    rng = np.random.default_rng(0)
    s_draws = stats.beta(a, b).rvs(200_000, random_state=rng)
    k_draws = np.array([SCMPrior.confounding_input_dim(s, n) for s in s_draws])

    empirical = {int(k): (k_draws == k).mean() for k in k_vals}
    for k, p in zip(k_vals, probs):
        assert empirical[int(k)] == pytest.approx(p, abs=5e-3)


def test_k_pmf_degenerate_when_cap_collapses_onto_floor():
    # A tiny n can push k_max = round(3n) below/at the floor; the pmf must
    # still be a valid (single-point) distribution rather than an empty range.
    k, p = k_pmf(1.2, 2.8, n=1, floor=5)
    assert k.tolist() == [5]
    assert p.tolist() == [1.0]


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
    # canonical_n is the middle representative width, the one the figure uses.
    assert results.canonical_n == 12


def test_compute_pmf_matches_k_pmf_and_per_n(tiny_cache):
    results = ConfoundingDimAnalysis().compute(tiny_cache)
    expected_k, expected_p = k_pmf(
        results.beta_a,
        results.beta_b,
        results.canonical_n,
        floor=results.floor,
        cap=results.cap,
    )
    np.testing.assert_array_equal(results.pmf_k, expected_k)
    np.testing.assert_allclose(results.pmf_p, expected_p)
    # The pmf's floor mass must agree with the per-n floor_atom already
    # reported for canonical_n -- same quantity, computed two ways.
    assert results.pmf_p[0] == pytest.approx(
        results.per_n[results.canonical_n]["floor_atom"]
    )


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
    assert summary["canonical_n"] == 12
    assert summary["pmf_floor_prob"] == pytest.approx(summary["floor_atom_by_n"][12])
    assert summary["pmf_cap_prob"] == pytest.approx(
        float(stats.beta(1.2, 2.8).sf(0.75)), abs=1e-3
    )
