"""Tests for the dimensions / completeness analysis.

The completeness math is tested exactly on hand-built arrays; the full analysis
is exercised once end-to-end on a tiny generated cache.
"""

import numpy as np
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.dimensions import (
    DimensionsAnalysis,
    completeness_stats,
    joint_counts,
)
from analysis_prior.config import PriorSpec


# --------------------------------------------------------------------------
# Pure completeness math
# --------------------------------------------------------------------------


def test_completeness_uses_both_proxy_dimensions_and_handles_unconfounded_data():
    stats = completeness_stats(
        d_U=np.array([2, 3]), d_W=np.array([3, 3]), d_Z=np.array([2, 4])
    )
    assert stats["overall_fraction"] == 1.0
    assert stats["holds"].tolist() == [True, True]

    stats = completeness_stats(
        d_U=np.array([2, 2]), d_W=np.array([3, 1]), d_Z=np.array([3, 3])
    )
    assert stats["holds"].tolist() == [True, False]
    assert stats["overall_fraction"] == 0.5

    stats = completeness_stats(d_U=np.array([3]), d_W=np.array([5]), d_Z=np.array([2]))
    assert stats["holds"].tolist() == [False]

    stats = completeness_stats(d_U=np.array([0]), d_W=np.array([0]), d_Z=np.array([0]))
    assert stats["holds"].tolist() == [True]


def test_completeness_by_d_U_breakdown():
    stats = completeness_stats(
        d_U=np.array([2, 2, 3]),
        d_W=np.array([3, 1, 3]),  # second violates (d_U=2), third holds (d_U=3)
        d_Z=np.array([3, 3, 5]),
    )
    table = stats["by_d_U"].set_index("d_U")
    assert table.loc[2, "n_datasets"] == 2
    assert table.loc[2, "n_hold"] == 1
    assert table.loc[2, "fraction_hold"] == 0.5
    assert table.loc[3, "fraction_hold"] == 1.0


def test_completeness_rejects_invalid_inputs():
    cases = [
        ((np.array([1, 2]), np.array([1]), np.array([1, 2])), "equal length"),
        ((np.array([]), np.array([]), np.array([])), "at least one"),
    ]
    for args, message in cases:
        with pytest.raises(ValueError, match=message):
            completeness_stats(*args)


# --------------------------------------------------------------------------
# joint_counts
# --------------------------------------------------------------------------


def test_joint_counts_layout_and_explicit_bounds():
    grid = joint_counts(np.array([1, 1, 2]), np.array([2, 2, 3]))
    assert grid.shape == (3, 4)
    assert grid[1, 2] == 2
    assert grid[2, 3] == 1
    assert grid.sum() == 3

    grid = joint_counts(np.array([0, 1]), np.array([1, 1]))
    assert grid[0, 1] == 1
    assert grid[1, 1] == 1

    grid = joint_counts(np.array([1]), np.array([1]), row_max=5, col_max=7)
    assert grid.shape == (6, 8)
    assert grid[1, 1] == 1


def test_joint_counts_rejects_invalid_inputs():
    cases = [
        ((np.array([4]), np.array([1])), {"row_max": 2}, "must be >="),
        ((np.array([1, 2]), np.array([1])), {}, "equal length"),
        ((np.array([]), np.array([])), {}, "at least one"),
        ((np.array([-1]), np.array([1])), {}, "non-negative"),
    ]
    for args, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            joint_counts(*args, **kwargs)


# --------------------------------------------------------------------------
# End-to-end on a tiny cache
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cache():
    spec = PriorSpec(
        name="dimtest",
        n_datasets=3,
        seq_len=60,
        mc_cate=3,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=4,
        min_dim_Z=2,
        max_dim_Z=4,
        min_dim_X=2,
        max_dim_X=4,
        seed=0,
    )
    return gen.generate(spec, n_jobs=1, progress=False)


def test_compute_returns_per_dataset_dimensions(tiny_cache):
    results = DimensionsAnalysis().compute(tiny_cache)
    assert results.n_datasets == len(tiny_cache)
    for key in ("d_U", "d_X", "d_W", "d_Z"):
        arr = getattr(results, key)
        assert arr.shape == (len(tiny_cache),)
        # Must match the widths the cache actually delivered.
        expected = [r.dims[key] for r in tiny_cache]
        assert arr.tolist() == expected


def test_run_writes_expected_artifacts(tiny_cache, tmp_path):
    artifacts = DimensionsAnalysis().run(tiny_cache, tmp_path)

    names = {p.name for p in artifacts.figures + artifacts.tables}
    assert "dimension_histograms.pdf" in names
    assert "dimension_histograms.png" in names
    assert "dimension_joint_heatmaps.pdf" in names
    assert "dimension_joint_heatmaps.png" in names
    assert "completeness_heuristic.tex" in names
    assert "completeness_heuristic.csv" in names
    for path in artifacts.figures + artifacts.tables:
        assert path.exists()


def test_summary_scalars_are_consistent(tiny_cache, tmp_path):
    artifacts = DimensionsAnalysis().run(tiny_cache, tmp_path)
    summary = artifacts.summary

    assert summary["n_datasets"] == len(tiny_cache)
    assert 0.0 <= summary["completeness_overall_fraction"] <= 1.0
    assert summary["completeness_violation_fraction"] == pytest.approx(
        1.0 - summary["completeness_overall_fraction"]
    )
    # Reported means must match the raw dims.
    d_U = np.array([r.dims["d_U"] for r in tiny_cache])
    assert summary["mean_d_U"] == pytest.approx(d_U.mean())
    assert summary["n_unconfounded"] == int((d_U == 0).sum())


# --------------------------------------------------------------------------
# Analytic pmfs (exact)
# --------------------------------------------------------------------------


def test_dim_U_pmf_triangular_closed_form():
    from analysis_prior.analyses.dimensions import dim_U_pmf

    pmf = dim_U_pmf(1, 10, "triangular")
    # p(d) = (11 - d) / 55, mean 4.0
    for d in range(1, 11):
        assert pmf[d] == pytest.approx((11 - d) / 55)
    assert sum(pmf.values()) == pytest.approx(1.0)


def test_dim_U_pmf_uniform():
    from analysis_prior.analyses.dimensions import dim_U_pmf

    pmf = dim_U_pmf(2, 5, "uniform")
    assert pmf == {2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}


def test_dim_U_pmf_rejects_unknown_scheme():
    from analysis_prior.analyses.dimensions import dim_U_pmf

    with pytest.raises(ValueError, match="Unknown dim_U distribution"):
        dim_U_pmf(1, 10, "nope")


def test_proxy_marginal_is_exactly_linear_under_triangular():
    # The design fact behind Option B: triangular d_U x conditional-uniform
    # proxy => the proxy marginal is the exact linear ramp p(w) = w / 55.
    from analysis_prior.analyses.dimensions import dim_U_pmf, proxy_marginal_pmf

    pmf = proxy_marginal_pmf(dim_U_pmf(1, 10, "triangular"), 1, 10, 10)
    for w in range(1, 11):
        assert pmf[w] == pytest.approx(w / 55)
    assert sum(pmf.values()) == pytest.approx(1.0)


def test_proxy_marginal_is_harmonic_under_uniform():
    # And under the old uniform d_U it is the harmonic pmf
    # p(w) = (H_10 - H_{10-w}) / 10, spiking at the top (p(10) ~ 0.293).
    from analysis_prior.analyses.dimensions import dim_U_pmf, proxy_marginal_pmf

    pmf = proxy_marginal_pmf(dim_U_pmf(1, 10, "uniform"), 1, 10, 10)
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 11))])
    for w in range(1, 11):
        assert pmf[w] == pytest.approx((H[10] - H[10 - w]) / 10)
    assert pmf[10] == pytest.approx(0.2929, abs=1e-4)


def test_proxy_marginal_respects_min_proxy():
    from analysis_prior.analyses.dimensions import dim_U_pmf, proxy_marginal_pmf

    # With min_proxy=3, no mass can land below 3 even when d_U is small.
    pmf = proxy_marginal_pmf(dim_U_pmf(1, 5, "triangular"), 3, 5, 5)
    assert min(pmf) == 3
    assert sum(pmf.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# DimensionComparisonAnalysis (smoke)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_family_caches():
    """Three tiny variants: baseline, uniform d_U, enforcement off."""
    from analysis_prior.config import SpecFamily

    base = PriorSpec(
        name="cmp",
        n_datasets=3,
        seq_len=60,
        mc_cate=1,
        min_dim_U=1,
        max_dim_U=3,
        min_dim_W=1,
        max_dim_W=3,
        min_dim_Z=1,
        max_dim_Z=3,
        min_dim_X=1,
        max_dim_X=3,
        seed=0,
    )
    family = SpecFamily(
        name="cmp_family",
        base=base,
        interventions={
            "triangular": {},
            "uniform": {"fixed_hp_overrides": {"dim_U_distribution": "uniform"}},
            "unenforced": {
                "fixed_hp_overrides": {"enforce_completeness_heuristic": False}
            },
        },
    )
    return {
        label: gen.generate(spec, n_jobs=1, progress=False)
        for label, spec in family.resolve().items()
    }


def test_comparison_compute_reads_schemes_from_specs(tiny_family_caches):
    from analysis_prior.analyses.dimensions import DimensionComparisonAnalysis

    results = DimensionComparisonAnalysis().compute(tiny_family_caches)

    assert list(results.variants) == ["triangular", "uniform", "unenforced"]
    assert results.baseline == "triangular"
    assert results.variants["triangular"].scheme == "triangular"
    assert results.variants["uniform"].scheme == "uniform"
    assert results.variants["triangular"].enforced
    assert not results.variants["unenforced"].enforced
    # Theory pmfs are attached per variant and cover all four blocks.
    assert set(results.variants["triangular"].theory) == {"d_U", "d_W", "d_Z", "d_X"}


def test_comparison_run_writes_expected_artifacts(tiny_family_caches, tmp_path):
    from analysis_prior.analyses.dimensions import DimensionComparisonAnalysis

    artifacts = DimensionComparisonAnalysis().run(tiny_family_caches, tmp_path)

    names = {p.name for p in artifacts.figures + artifacts.tables}
    assert "dimension_marginals.pdf" in names
    assert "dimension_marginals.png" in names
    assert "dimension_joint_heatmaps_comparison.pdf" in names
    assert "dimension_sampling.tex" in names
    assert "dimension_sampling.csv" in names
    for path in artifacts.figures + artifacts.tables:
        assert path.exists()

    # Tables carry a \label so the appendix can \input and \Cref them.
    sampling_tex = (tmp_path / "dimension_sampling.tex").read_text()
    assert "\\label{tab:prior_dim_sampling}" in sampling_tex
    assert "\\caption" in sampling_tex
    # Comparison provenance: one spec per variant.
    import json

    with open(tmp_path / "specs.json") as f:
        assert set(json.load(f)) == {"triangular", "uniform", "unenforced"}


def test_comparison_summary_is_per_variant(tiny_family_caches, tmp_path):
    from analysis_prior.analyses.dimensions import DimensionComparisonAnalysis

    summary = DimensionComparisonAnalysis().run(tiny_family_caches, tmp_path).summary

    assert summary["baseline"] == "triangular"
    for label in ("triangular", "uniform", "unenforced"):
        stats = summary[label]
        assert stats["n_datasets"] == 3
        assert 0.0 <= stats["violation_fraction"] <= 1.0
    # Enforced variants cannot deliver violations or unconfounded datasets.
    assert summary["triangular"]["violation_fraction"] == 0.0
    assert summary["triangular"]["n_unconfounded"] == 0
