"""Tests for the heterogeneity-dial figure (appendix B.2.6).

Three layers, mirroring the treatment-walkthrough tests:

- ``heterogeneity_frame`` -- the scale-free statistics, checked against
  hand-built potential outcomes where the right answer is known in closed form
  (notably the ``gamma = 0`` anchor, where ``corr(Y0, Y1)`` must be exactly 1).
- ``select_contrast_pair`` -- picks one low- and one high-gamma dataset with
  *matched* SCM dispersion, so the illustrated contrast is about gamma.
- ``HeterogeneitySpreadAnalysis`` -- an end-to-end render on a tiny cache, plus
  the delivered-prior invariant that dispersion tracks the delivered gamma.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.outcome import (
    GAMMA_BINS,
    HeterogeneitySpreadAnalysis,
    heterogeneity_frame,
    select_contrast_pair,
)
from analysis_prior.config import PriorSpec
from analysis_prior.generate import PriorDatasetRecord


def _record(index: int, gamma, y0: np.ndarray, y1: np.ndarray) -> PriorDatasetRecord:
    """A record carrying only what this analysis reads."""
    n = len(y0)
    zeros = np.zeros((n, 1), dtype=np.float32)
    flat = np.zeros(n, dtype=np.float32)
    return PriorDatasetRecord(
        index=index,
        seed=1000 + index,
        U=zeros,
        X=zeros,
        W=zeros,
        Z=zeros,
        A=flat,
        A_probs=flat,
        Y_obs=np.asarray(y0, dtype=np.float32),
        Y0=np.asarray(y0, dtype=np.float32),
        Y1=np.asarray(y1, dtype=np.float32),
        MU0=np.asarray(y0, dtype=np.float32),
        MU1=np.asarray(y1, dtype=np.float32),
        gamma_hetero=gamma,
    )


@pytest.fixture(scope="module")
def tiny_spec():
    return PriorSpec(
        name="tiny-outcome",
        n_datasets=4,
        seq_len=200,
        mc_cate=1,
        min_dim_U=2,
        max_dim_U=3,
        min_dim_W=2,
        max_dim_W=3,
        min_dim_Z=2,
        max_dim_Z=3,
        min_dim_X=2,
        max_dim_X=3,
        seed=0,
    )


@pytest.fixture(scope="module")
def tiny_cache(tiny_spec):
    return gen.generate(tiny_spec, n_jobs=1, progress=False)


# --------------------------------------------------------------------------
# heterogeneity_frame
# --------------------------------------------------------------------------


def test_frame_statistics_match_closed_form():
    rng = np.random.default_rng(0)
    y0 = rng.normal(size=512)
    tau = rng.normal(size=512)
    y1 = y0 + tau

    frame = heterogeneity_frame([_record(0, 0.5, y0, y1)])
    row = frame.iloc[0]

    scale = np.std(np.concatenate([y0, y1]))
    assert row["sd_ite"] == pytest.approx(np.std(tau), rel=1e-5)
    assert row["scale"] == pytest.approx(scale, rel=1e-5)
    assert row["dispersion"] == pytest.approx(np.std(tau) / scale, rel=1e-5)
    assert row["ate"] == pytest.approx(np.mean(tau), abs=1e-5)
    assert row["corr_y0y1"] == pytest.approx(np.corrcoef(y0, y1)[0, 1], rel=1e-5)
    # c = dispersion / gamma, the gamma-free dispersion constant of the SCM.
    assert row["raw_dispersion"] == pytest.approx(row["dispersion"] / 0.5, rel=1e-5)


def test_homogeneous_effect_is_the_exact_anchor():
    """At gamma = 0 the remix collapses the effect to the constant tau_bar, so
    Y1 = Y0 + tau_bar and the outcome coupling is exactly 1 -- the analytic
    anchor panel (b) draws its reference line at."""
    rng = np.random.default_rng(1)
    y0 = rng.normal(size=256)
    y1 = y0 + 1.7

    row = heterogeneity_frame([_record(0, 0.0, y0, y1)]).iloc[0]
    assert row["corr_y0y1"] == pytest.approx(1.0, abs=1e-6)
    assert row["dispersion"] == pytest.approx(0.0, abs=1e-6)
    # gamma below the floor: the dispersion constant is 0/0 and must not be
    # reported as a number.
    assert np.isnan(row["raw_dispersion"])


def test_dispersion_is_proportional_to_gamma():
    """The relation the whole figure rests on: with one SCM's raw effect held
    fixed, sd(tau) = gamma * sd(tau_raw), so dispersion is linear in gamma."""
    rng = np.random.default_rng(2)
    y0 = rng.normal(size=512)
    tau_raw = rng.normal(size=512)
    tau_bar = float(tau_raw.mean())

    records = []
    gammas = [0.2, 0.5, 0.9]
    for i, gamma in enumerate(gammas):
        tau = gamma * tau_raw + (1 - gamma) * tau_bar
        records.append(_record(i, gamma, y0, y0 + tau))

    frame = heterogeneity_frame(records)
    ratios = frame["sd_ite"] / frame["gamma"]
    assert ratios.std() == pytest.approx(0.0, abs=1e-4)
    # The ATE is preserved at every gamma -- the remix anchors at tau_bar.
    assert frame["ate"].to_numpy() == pytest.approx(tau_bar, abs=1e-5)


def test_frame_rejects_records_without_gamma():
    rng = np.random.default_rng(3)
    y0 = rng.normal(size=32)
    with pytest.raises(ValueError, match="gamma_hetero"):
        heterogeneity_frame([_record(0, None, y0, y0 + 1.0)])


def test_constant_arm_gives_nan_correlation_not_a_crash():
    y0 = np.zeros(64)
    y1 = np.arange(64, dtype=float)
    row = heterogeneity_frame([_record(0, 0.5, y0, y1)]).iloc[0]
    assert np.isnan(row["corr_y0y1"])


# --------------------------------------------------------------------------
# select_contrast_pair
# --------------------------------------------------------------------------


def test_contrast_pair_matches_dispersion_constant():
    """Among candidates near each target gamma, the pair chosen must be the one
    whose SCM dispersion constants agree -- otherwise the illustrated contrast
    would confound gamma with SCM luck."""
    frame = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "gamma": [0.10, 0.10, 0.90, 0.90],
            # Dataset 1 and 2 share c = 0.5; 0 and 3 are decoys.
            "raw_dispersion": [0.05, 0.50, 0.50, 4.0],
        }
    )
    low, high = select_contrast_pair(frame)
    assert (low, high) == (1, 2)


def test_contrast_pair_needs_two_usable_datasets():
    frame = pd.DataFrame(
        {"index": [0], "gamma": [0.5], "raw_dispersion": [1.0]},
    )
    with pytest.raises(ValueError, match="at least two datasets"):
        select_contrast_pair(frame)


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_bins_drop_sparse_cells_and_keep_gamma_one():
    analysis = HeterogeneitySpreadAnalysis()
    rng = np.random.default_rng(4)
    # 40 datasets spread over [0, 1], plus the gamma = 1 edge case.
    gammas = np.concatenate([rng.uniform(0.0, 1.0, 40), [1.0]])
    records = []
    y0 = rng.normal(size=128)
    tau_raw = rng.normal(size=128)
    for i, gamma in enumerate(gammas):
        tau = gamma * tau_raw + (1 - gamma) * float(tau_raw.mean())
        records.append(_record(i, float(gamma), y0, y0 + tau))

    results = analysis.compute(gen.PriorCache(spec=PriorSpec(), records=records))
    bins = results.binned()
    assert bins, "expected at least one populated gamma bin"
    assert all(b["n"] >= 5 for b in bins)
    # gamma = 1 lands in the top bin rather than falling off the edge.
    assert bins[-1]["hi"] == GAMMA_BINS[-1]
    assert sum(b["n"] for b in bins) >= len(gammas) - 4 * (5 - 1)
    # Medians march upward with gamma: the population form of the dial.
    medians = [b["median"] for b in bins]
    assert medians == sorted(medians)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_analysis_runs_on_a_real_cache(tiny_cache, tmp_path):
    analysis = HeterogeneitySpreadAnalysis()
    artifacts = analysis.run(tiny_cache, tmp_path)

    assert [p.name for p in artifacts.figures] == [
        "heterogeneity_spread.pdf",
        "heterogeneity_spread.png",
    ]
    assert all(p.exists() for p in artifacts.figures)
    assert (tmp_path / "summary.json").exists()
    assert artifacts.summary["n_datasets"] == len(tiny_cache)
    # A four-dataset cache populates no bin (MIN_BIN_COUNT = 5); the render
    # must degrade to an empty panel rather than raise.
    assert artifacts.summary["dispersion_median_by_gamma_bin"] == {}
    for label in ("low", "high"):
        assert 0.0 <= artifacts.summary[f"contrast_{label}_gamma"] <= 1.0


def test_delivered_gamma_tracks_delivered_spread(tiny_cache):
    """The claim, measured on real prior output rather than on hand-built
    outcomes: within one cache, the dataset with the larger delivered gamma
    carries the wider (outcome-scaled) effect distribution."""
    frame = heterogeneity_frame(tiny_cache.records).dropna(subset=["dispersion"])
    ordered = frame.sort_values("gamma")
    lowest, highest = ordered.iloc[0], ordered.iloc[-1]
    assert highest["gamma"] > lowest["gamma"]
    assert highest["dispersion"] > lowest["dispersion"]
