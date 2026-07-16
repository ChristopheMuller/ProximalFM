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
)
from analysis_prior.config import PriorSpec


# --------------------------------------------------------------------------
# Pure completeness math
# --------------------------------------------------------------------------


def test_completeness_all_hold():
    stats = completeness_stats(
        d_U=np.array([2, 3]), d_W=np.array([3, 3]), d_Z=np.array([2, 4])
    )
    assert stats["overall_fraction"] == 1.0
    assert stats["holds"].tolist() == [True, True]


def test_completeness_violation_detected():
    # d_W=1 < d_U=2 in the second dataset: min(d_W,d_Z)=1 < 2.
    stats = completeness_stats(
        d_U=np.array([2, 2]), d_W=np.array([3, 1]), d_Z=np.array([3, 3])
    )
    assert stats["holds"].tolist() == [True, False]
    assert stats["overall_fraction"] == 0.5


def test_completeness_uses_the_min_of_both_proxies():
    # Fails through Z rather than W.
    stats = completeness_stats(d_U=np.array([3]), d_W=np.array([5]), d_Z=np.array([2]))
    assert stats["holds"].tolist() == [False]


def test_completeness_unconfounded_datasets_trivially_hold():
    # d_U == 0 (unconfounded draw): min(d_W, d_Z) >= 0 always.
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


def test_completeness_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        completeness_stats(np.array([1, 2]), np.array([1]), np.array([1, 2]))


def test_completeness_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        completeness_stats(np.array([]), np.array([]), np.array([]))


# --------------------------------------------------------------------------
# End-to-end on a tiny cache
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cache():
    spec = PriorSpec(
        name="dimtest",
        n_datasets=3,
        seq_len=60,
        mcmc_cate=3,
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
