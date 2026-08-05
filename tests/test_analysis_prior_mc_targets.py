"""Tests for the Monte Carlo CATE-target analysis (appendix B.2.6).

The load-bearing claim is that the two reliability estimators recover the true
signal/noise split of a replicate structure, so the central test builds one
where ``Var_x(Q)`` and ``E[sigma^2]`` are known by construction and checks the
estimates against them. The rest covers the leave-one-out identity, the
degenerate cases, and an end-to-end render on a real (tiny) cache.
"""

import numpy as np
import pytest

from analysis_prior import generate as gen
from analysis_prior.analyses.mc_targets import (
    K_GRID,
    SHARE_PANEL_K,
    McTargetsAnalysis,
    leave_one_out_cate,
    signal_share,
    target_frame,
)
from analysis_prior.config import PRESETS, PriorSpec
from analysis_prior.generate import PriorDatasetRecord


def _record(index: int, y0: np.ndarray, y1: np.ndarray, mu0, mu1, gamma=0.7):
    n = len(y0)
    zeros = np.zeros((n, 1), dtype=np.float32)
    flat = np.zeros(n, dtype=np.float32)
    return PriorDatasetRecord(
        index=index,
        seed=2000 + index,
        U=zeros,
        X=zeros,
        W=zeros,
        Z=zeros,
        A=flat,
        A_probs=flat,
        Y_obs=np.asarray(y0, dtype=np.float32),
        Y0=np.asarray(y0, dtype=np.float32),
        Y1=np.asarray(y1, dtype=np.float32),
        MU0=np.asarray(mu0, dtype=np.float32),
        MU1=np.asarray(mu1, dtype=np.float32),
        gamma_hetero=gamma,
    )


def _replicated_record(
    index: int,
    n_rows: int,
    k: int,
    sd_signal: float,
    sd_noise: float,
    rng: np.random.Generator,
):
    """A record built from an explicit replicate structure.

    ``Q(x_i) ~ N(0, sd_signal^2)`` is the true CATE; each of the ``k``
    replicates draws ``tau = Q + N(0, sd_noise^2)``. ``MU1 - MU0`` is the mean
    over all ``k``, and the retained replicate is one of them -- exactly the
    generator's contract, so the estimators must recover ``sd_signal`` and
    ``sd_noise``.
    """
    q = rng.normal(0.0, sd_signal, size=n_rows)
    tau = q[:, None] + rng.normal(0.0, sd_noise, size=(n_rows, k))
    retained = rng.integers(k, size=n_rows)

    # Encode the effect in the Y1 / MU1 arm; the zero arm keeps the algebra
    # visible (ITE = Y1 - Y0 = tau of the retained replicate).
    y0 = np.zeros(n_rows)
    y1 = tau[np.arange(n_rows), retained]
    return _record(index, y0, y1, mu0=y0, mu1=tau.mean(axis=1)), q


# --------------------------------------------------------------------------
# Leave-one-out identity
# --------------------------------------------------------------------------


def test_leave_one_out_is_the_mean_of_the_other_replicates():
    rng = np.random.default_rng(0)
    k = 8
    tau = rng.normal(size=(64, k))
    retained = rng.integers(k, size=64)

    ite = tau[np.arange(64), retained]
    cate = tau.mean(axis=1)
    q_loo = leave_one_out_cate(ite, cate, k)

    for i in range(64):
        others = np.delete(tau[i], retained[i])
        assert q_loo[i] == pytest.approx(others.mean(), rel=1e-10)


def test_leave_one_out_rejects_k_of_one():
    with pytest.raises(ValueError, match="K >= 2"):
        leave_one_out_cate(np.zeros(3), np.zeros(3), 1)


# --------------------------------------------------------------------------
# The estimators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sd_signal, sd_noise", [(1.0, 1.0), (1.0, 0.25), (0.5, 2.0)])
def test_estimators_recover_a_known_split(sd_signal, sd_noise):
    """Signal variance from the covariance of two independent measurements,
    replicate noise from their paired difference."""
    rng = np.random.default_rng(1)
    k = 20
    record, _ = _replicated_record(0, 20000, k, sd_signal, sd_noise, rng)

    row = target_frame([record], k).iloc[0]
    assert row["var_q"] == pytest.approx(sd_signal**2, rel=0.08)
    assert row["sigma2"] == pytest.approx(sd_noise**2, rel=0.05)
    assert row["nsr"] == pytest.approx((sd_noise / sd_signal) ** 2, rel=0.1)
    # S(1) and the regression slope agree up to sampling error.
    assert row["slope"] == pytest.approx(row["S_1"], rel=0.05)


def test_implied_variance_identities_hold():
    """Var(ITE) = Var(Q) + E[sigma^2] and Var(MU1-MU0) = Var(Q) + E[sigma^2]/K
    are never used in the estimation. The first is near-algebraic given how the
    estimators are built; the second genuinely tests that the target's spread
    is the one the replicate model predicts."""
    rng = np.random.default_rng(2)
    k = 25
    record, _ = _replicated_record(0, 20000, k, 1.0, 1.5, rng)

    row = target_frame([record], k).iloc[0]
    assert row["var_ite_residual"] == pytest.approx(0.0, abs=0.03)
    assert row["var_cate_residual"] == pytest.approx(0.0, abs=0.03)


def test_signal_share_matches_the_target_against_truth():
    """S(K) is meant to be the R^2 of the K-replicate target against the true
    CATE. Check that reading directly, using the q known by construction."""
    rng = np.random.default_rng(3)
    k = 10
    record, q = _replicated_record(0, 20000, k, 1.0, 2.0, rng)

    row = target_frame([record], k).iloc[0]
    cate = np.asarray(record.cate, dtype=np.float64)
    observed_r2 = np.corrcoef(cate, q)[0, 1] ** 2
    assert signal_share(row["nsr"], k) == pytest.approx(observed_r2, rel=0.06)


def test_signal_share_is_monotone_in_k():
    shares = [signal_share(2.0, k) for k in K_GRID]
    assert shares == sorted(shares)
    assert signal_share(2.0, 1) == pytest.approx(1 / 3)


def test_panel_k_values_are_quotable_from_the_table():
    """Every K drawn in panel (b) must also be reported in the table and
    summary, so a number read off the figure can be cited in the text."""
    assert set(SHARE_PANEL_K) <= set(K_GRID)
    assert SHARE_PANEL_K == tuple(sorted(SHARE_PANEL_K))


# --------------------------------------------------------------------------
# Degenerate cases
# --------------------------------------------------------------------------


def test_k_one_cache_is_rejected_with_a_pointer_to_the_preset():
    rng = np.random.default_rng(4)
    y0 = rng.normal(size=16)
    record = _record(0, y0, y0 + 1.0, y0, y0 + 1.0)
    with pytest.raises(ValueError, match="mc_targets"):
        target_frame([record], 1)


def test_low_gamma_datasets_are_excluded():
    rng = np.random.default_rng(5)
    k = 10
    usable, _ = _replicated_record(0, 512, k, 1.0, 1.0, rng)
    degenerate, _ = _replicated_record(1, 512, k, 1.0, 1.0, rng)
    degenerate = _record(
        1, degenerate.Y0, degenerate.Y1, degenerate.MU0, degenerate.MU1, gamma=0.001
    )

    frame = target_frame([usable, degenerate], k)
    assert bool(frame.loc[0, "usable"])
    assert not bool(frame.loc[1, "usable"])


def test_constant_effect_gives_no_positive_signal():
    """A dataset whose true CATE does not vary has no signal to measure; the
    estimator must mark it unusable rather than emit a huge NSR."""
    k = 10
    n = 512
    y0 = np.zeros(n)
    record = _record(0, y0, y0 + 2.0, mu0=y0, mu1=y0 + 2.0)
    frame = target_frame([record], k)
    assert not bool(frame.loc[0, "usable"])
    assert np.isnan(frame.loc[0, "nsr"])


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cache():
    spec = PriorSpec(
        name="tiny-mc",
        n_datasets=3,
        seq_len=128,
        mc_cate=8,
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
    return gen.generate(spec, n_jobs=1, progress=False)


def test_analysis_runs_on_a_real_cache(tiny_cache, tmp_path):
    artifacts = McTargetsAnalysis().run(tiny_cache, tmp_path)

    assert [p.name for p in artifacts.figures] == [
        "cate_vs_ite.pdf",
        "cate_vs_ite.png",
    ]
    assert [p.name for p in artifacts.tables] == [
        "mc_signal_share.tex",
        "mc_signal_share.csv",
    ]
    assert all(p.exists() for p in artifacts.figures + artifacts.tables)
    assert artifacts.summary["K"] == 8
    assert artifacts.summary["n_datasets"] == len(tiny_cache)


def test_mc_targets_preset_carries_replicates():
    """The one property the analysis actually requires. Size is a budget knob
    the author is expected to scale, so it is deliberately not asserted."""
    assert PRESETS["mc_targets"].mc_cate >= 2
