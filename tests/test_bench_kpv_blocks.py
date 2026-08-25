"""KPV's per-arm solve must reproduce the undecomposed system exactly.

``KPVEstimator._fit_stages`` decomposes *both* stages by arm, which is valid
only because ``k_A`` is an indicator and multiplies every gram in the pipeline:
K_ZAX on both splits, and Sigma. That is a structural fact about the estimator,
not a tolerance to be tuned, so these tests compare against the dense system
written out longhand and demand agreement to solver precision.

The reference below is deliberately a transcription of the formulas in the
module docstring — Gamma, Sigma, alpha = Gamma * v — rather than a call into
``kpv.py``: if the two shared code the test would pass no matter what either
did. Same reasoning, and the same shape, as
:mod:`tests.test_bench_pmmr_blocks`.

**On the tolerance.** The two routes are the same arithmetic in a different
order, so the floor is the conditioning of the Stage-2 system rather than
machine epsilon. Measured spread over the cases below is 1e-15 to 1e-12
relative, so ``_RTOL`` sits well above what was observed and a failure here
means a changed formula, not drifting numerics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve

from benchmarks.methods.kernels import BinaryKernel, ColumnWiseGaussianKernel
from benchmarks.methods.kpv import (
    KPVConfig,
    KPVEstimator,
    _RidgeSpectrum,
    _loocv_stage1,
    _loocv_stage1_block,
    _loocv_stage2,
    _loocv_stage2_block,
    _loocv_stage2_spectral_block,
)

#: Relative agreement demanded of alpha; see the module docstring.
_RTOL = 1e-8


def make_data(n=200, seed=0, d_x=2, d_w=2, d_z=2, p_treat=0.5):
    """A confounded proximal draw. ``d_x=0`` exercises the no-covariate path."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_x))
    Z = U + 0.5 * rng.normal(size=(n, d_z))
    W = U + 0.5 * rng.normal(size=(n, d_w))
    A = (rng.random((n, 1)) < p_treat).astype(float)
    Y = A[:, 0] + 2.0 * U[:, 0] + 0.3 * rng.normal(size=n)
    return X, A, Y, W, Z


def dense_fit(X, A, Y, W, Z, cfg):
    """The undecomposed pipeline, transcribed from the module docstring.

    Reuses the complete sample in both stages, exactly as ``fit`` does.
    """
    n = len(Y)
    i1 = i2 = np.arange(n)
    n1 = n2 = n
    has_x = X.shape[1] > 0

    k_A = BinaryKernel().fit(A[i1])
    k_Z = ColumnWiseGaussianKernel().fit(Z[i1], scale=cfg.kernel_scale)
    k_W = ColumnWiseGaussianKernel().fit(W[i1], scale=cfg.kernel_scale)
    k_X = (
        ColumnWiseGaussianKernel().fit(X[i1], scale=cfg.kernel_scale) if has_x else None
    )
    kx = (lambda a, b: k_X(a, b)) if has_x else (lambda a, b: 1.0)

    K11 = k_A(A[i1], A[i1]) * k_Z(Z[i1], Z[i1]) * kx(X[i1], X[i1])
    K12 = k_A(A[i1], A[i2]) * k_Z(Z[i1], Z[i2]) * kx(X[i1], X[i2])
    K_W11 = k_W(W[i1], W[i1])

    lam1 = cfg.lam1
    if lam1 is None:
        grid = np.logspace(
            np.log10(cfg.lam1_min), np.log10(cfg.lam1_max), cfg.n_lam1_search
        )
        lam1 = min(grid, key=lambda l: _loocv_stage1(K11, K_W11, l))

    Gamma = solve(K11 + lam1 * n1 * np.eye(n1), K12, assume_a="pos")
    Sigma = (Gamma.T @ K_W11 @ Gamma) * k_A(A[i2], A[i2]) * kx(X[i2], X[i2])

    lam2 = cfg.lam2
    if lam2 is None:
        grid = np.logspace(
            np.log10(cfg.lam2_min), np.log10(cfg.lam2_max), cfg.n_lam2_search
        )
        lam2 = min(grid, key=lambda l: _loocv_stage2(Sigma, Y[i2], l))

    v = solve(Sigma + lam2 * n2 * np.eye(n2), Y[i2], assume_a="pos")
    return {
        "alpha": Gamma * v[np.newaxis, :],
        "lam1": lam1,
        "lam2": lam2,
        "score": _loocv_stage2(Sigma, Y[i2], lam2),
    }


@pytest.mark.parametrize(
    "d_x, p_treat, lams",
    [
        (2, 0.5, (None, None)),  # both stages tuned by LOOCV
        (0, 0.5, (None, None)),  # no covariates
        (2, 0.2, (None, None)),  # badly imbalanced arms
        (3, 0.5, (1e-3, 1e-2)),  # fixed lambdas, no tuning
        (0, 0.8, (1e-4, 1e-1)),
    ],
)
def test_per_arm_fit_matches_the_dense_pipeline(d_x, p_treat, lams):
    X, A, Y, W, Z = make_data(d_x=d_x, p_treat=p_treat, seed=d_x + int(10 * p_treat))
    cfg = KPVConfig(lam1=lams[0], lam2=lams[1])

    est = KPVEstimator(cfg).fit(X, A, Y, W=W, Z=Z)
    ref = dense_fit(X, A, Y, W, Z, cfg)

    # The tuned lambdas must be the *same grid points*, not merely close: the
    # per-arm LOOCV sums are what select them.
    assert est.lam1_ == pytest.approx(ref["lam1"], rel=1e-12)
    assert est.lam2_ == pytest.approx(ref["lam2"], rel=1e-12)

    scale = max(float(np.abs(ref["alpha"]).max()), 1e-300)
    assert np.abs(est._alpha - ref["alpha"]).max() <= _RTOL * scale


def test_alpha_is_exactly_zero_across_arms():
    """Structural, so exact: no coefficient links a D1 row to a D2 row of the
    other arm. This is the fact the decomposition rests on."""
    X, A, Y, W, Z = make_data(d_x=2)
    est = KPVEstimator(KPVConfig(lam1=1e-3, lam2=1e-2)).fit(X, A, Y, W=W, Z=Z)

    arm2 = est._A2.ravel()
    arm1 = A.ravel()

    for a in (0.0, 1.0):
        rows = np.flatnonzero(arm1 == a)
        cols = np.flatnonzero(arm2 != a)
        if rows.size and cols.size:
            assert np.all(est._alpha[np.ix_(rows, cols)] == 0.0)


def test_both_stages_reuse_the_full_context():
    X, A, Y, W, Z = make_data(n=120, seed=11)
    est = KPVEstimator(KPVConfig(lam1=1e-3, lam2=1e-2)).fit(X, A, Y, W=W, Z=Z)

    assert est._W1.shape[0] == len(Y)
    assert est._A2.shape[0] == len(Y)
    assert est._alpha.shape == (len(Y), len(Y))


def test_loocv_scores_add_over_arms():
    """Both criteria are sums over the blocks — stage 2 in squares."""
    rng = np.random.default_rng(3)
    n = 60
    A = (rng.random(n) < 0.5).astype(float)
    Z = rng.normal(size=(n, 2))
    W = rng.normal(size=(n, 2))
    Y = rng.normal(size=n)

    k_Z = ColumnWiseGaussianKernel().fit(Z)
    k_W = ColumnWiseGaussianKernel().fit(W)
    K = BinaryKernel().fit(A[:, None])(A[:, None], A[:, None]) * k_Z(Z, Z)
    K_W = k_W(W, W)
    lam = 1e-2

    per_arm1 = 0.0
    per_arm2 = 0.0
    for a in (0.0, 1.0):
        i = np.flatnonzero(A == a)
        per_arm1 += _loocv_stage1_block(k_Z(Z[i], Z[i]), K_W[np.ix_(i, i)], lam, n)
        per_arm2 += _loocv_stage2_block(K[np.ix_(i, i)], Y[i], lam, n)

    assert per_arm1 == pytest.approx(_loocv_stage1(K, K_W, lam), rel=1e-10)
    assert np.sqrt(per_arm2) == pytest.approx(_loocv_stage2(K, Y, lam), rel=1e-10)


@pytest.mark.parametrize("rank", [3, 24])
def test_stage2_spectral_sweep_matches_dense_solves(rank):
    """One spectrum reproduces every dense-solve score, including singular K."""
    rng = np.random.default_rng(41 + rank)
    m = 24
    features = rng.normal(size=(m, rank))
    Sigma = features @ features.T
    Y = rng.normal(size=m)
    n_full = 57
    spectrum = _RidgeSpectrum.from_psd(Sigma)

    for lam in np.logspace(-5, 1, 10):
        expected = _loocv_stage2_block(Sigma, Y, lam, n_full)
        actual = _loocv_stage2_spectral_block(spectrum, Y, lam, n_full)
        assert actual == pytest.approx(expected, rel=2e-11, abs=1e-12)

        expected_v = solve(Sigma + lam * n_full * np.eye(m), Y, assume_a="pos")
        # The deliberately rank-deficient case becomes ill-conditioned at the
        # smallest lambda; Cholesky and eigen solves then differ at the
        # condition-number-amplified roundoff floor despite solving the same
        # system. Full-rank systems agree more tightly.
        solve_rtol = 5e-10 if rank < m else 2e-11
        np.testing.assert_allclose(
            spectrum.solve(Y, lam, n_full),
            expected_v,
            rtol=solve_rtol,
            atol=1e-12,
        )
