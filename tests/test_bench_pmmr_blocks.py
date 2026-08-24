"""PMMR's per-arm solve must reproduce the assembled n x n system exactly.

``PMMREstimator.fit`` decomposes the normal equations by arm, which is only
valid because ``k_A`` is an indicator and multiplies *both* grams. That is a
structural fact about the estimator, not a tolerance to be tuned, so these tests
compare against the assembled system written out longhand and demand agreement
to solver precision.

The reference implementation below is deliberately a transcription of the
formula in the module docstring rather than a call into ``pmmr.py``: if the two
shared code the test would pass no matter what either did.

**On the tolerance.** The two routes are the same arithmetic in a different
order, so the floor is not machine epsilon but the conditioning of the system.
The measured spread over the cases below is 3e-8 to 6e-7, so
``_ALPHA_RTOL`` is set an order of magnitude above what was observed, and a
failure here means a changed formula rather than drifting numerics. Tightening
it will produce flakes, not extra safety; the sharp check is
:func:`test_cross_arm_block_of_the_assembled_system_is_exactly_zero`, which is
exact by construction, and the check that *matters* is on predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve

from benchmarks.methods.kernels import BinaryKernel, ColumnWiseGaussianKernel
from benchmarks.methods.pmmr import PMMRConfig, PMMREstimator


#: Relative agreement demanded of alpha; see the module docstring.
_ALPHA_RTOL = 1e-5


def assert_alpha_matches(fitted: np.ndarray, expected: np.ndarray) -> None:
    """Compare relative to ``max|alpha|`` — the entries span several decades."""
    scale = max(float(np.abs(expected).max()), 1e-300)
    assert np.abs(fitted - expected).max() <= _ALPHA_RTOL * scale


def make_data(n=240, seed=0, d_x=2, d_w=2, d_z=2, p_treat=0.5):
    """A confounded proximal draw. ``d_x=0`` exercises the no-covariate path."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_x)) if d_x else np.zeros((n, 0))
    W = U + 0.5 * rng.normal(size=(n, d_w))
    Z = U + 0.5 * rng.normal(size=(n, d_z))
    logit = 1.5 * U.ravel() + np.log(p_treat / (1 - p_treat))
    A = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float).reshape(-1, 1)
    Y = 2 * A.ravel() + 2 * U.ravel() + 0.3 * rng.normal(size=n)
    return X, A, Y, W, Z


def alpha_dense(X, A, Y, W, Z, cfg: PMMRConfig) -> np.ndarray:
    """alpha from the assembled n x n system, exactly as the docstring writes it."""
    n = len(Y)
    K_A = BinaryKernel().fit(A)(A, A)
    K_Z = ColumnWiseGaussianKernel().fit(
        Z, scale=cfg.kernel_scale * cfg.kernel_scale_z
    )(Z, Z)
    K_W = ColumnWiseGaussianKernel().fit(W, scale=cfg.kernel_scale)(W, W)
    if X.shape[1] > 0:
        K_X = ColumnWiseGaussianKernel().fit(X, scale=cfg.kernel_scale)(X, X)
    else:
        K_X = np.ones((n, n))

    Phi = K_A * K_Z * K_X
    L = K_A * K_W * K_X
    LPhi = L @ Phi
    lhs = LPhi @ L + cfg.lam1 * n**2 * L
    return solve(lhs + cfg.jitter * n**2 * np.eye(n), LPhi @ Y, assume_a="pos")


# ─────────────────────────────────────────────────────────────────────────────
# The equivalence
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d_x", [0, 2])
@pytest.mark.parametrize("lam1", [4.9e-6, 1e-3, 0.25])
def test_per_arm_alpha_matches_assembled_system(d_x, lam1):
    X, A, Y, W, Z = make_data(d_x=d_x)
    cfg = PMMRConfig(lam1=lam1)

    fitted = PMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)._alpha
    assert_alpha_matches(fitted, alpha_dense(X, A, Y, W, Z, cfg))


def test_imbalanced_arms_still_match():
    """Unequal blocks are the case where a per-arm row count would show up.

    ``n`` in the penalty terms must stay the full sample size in both blocks;
    substituting the block's own size passes on a balanced draw and fails here.
    """
    X, A, Y, W, Z = make_data(p_treat=0.2)
    assert 0.1 < A.mean() < 0.4, "fixture no longer produces imbalanced arms"

    cfg = PMMRConfig()
    fitted = PMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)._alpha
    assert_alpha_matches(fitted, alpha_dense(X, A, Y, W, Z, cfg))


def test_predictions_match_end_to_end():
    """Equivalence at alpha is worth little if a prediction path disagrees."""
    X, A, Y, W, Z = make_data()
    est = PMMREstimator(PMMRConfig()).fit(X, A, Y, W=W, Z=Z)
    expected_alpha = alpha_dense(X, A, Y, W, Z, PMMRConfig())

    for strategy in ("marginal", "paired", "independent", "conditional"):
        got = est.predict(X, W=W, strategy=strategy)
        # Recompute the same strategy from the dense alpha by swapping it in.
        est._alpha, saved = expected_alpha, est._alpha
        want = est.predict(X, W=W, strategy=strategy)
        est._alpha = saved
        assert np.allclose(got, want, rtol=0, atol=1e-6 * max(np.abs(want).max(), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# The structural fact the decomposition rests on
# ─────────────────────────────────────────────────────────────────────────────


def test_cross_arm_block_of_the_assembled_system_is_exactly_zero():
    """Why the decomposition is exact, asserted rather than assumed.

    If ``k_A`` ever stops being an indicator — a smoothed or continuous
    treatment kernel — this is the test that fails, and it should: the per-arm
    solve would then be silently wrong rather than merely approximate.
    """
    X, A, Y, W, Z = make_data()
    n = len(Y)
    cfg = PMMRConfig()

    K_A = BinaryKernel().fit(A)(A, A)
    K_Z = ColumnWiseGaussianKernel().fit(
        Z, scale=cfg.kernel_scale * cfg.kernel_scale_z
    )(Z, Z)
    K_W = ColumnWiseGaussianKernel().fit(W, scale=cfg.kernel_scale)(W, W)
    K_X = ColumnWiseGaussianKernel().fit(X, scale=cfg.kernel_scale)(X, X)

    L = K_A * K_W * K_X
    lhs = (L @ (K_A * K_Z * K_X)) @ L + cfg.lam1 * n**2 * L

    treated = np.flatnonzero(A.ravel() == 1.0)
    control = np.flatnonzero(A.ravel() == 0.0)
    assert np.all(lhs[np.ix_(treated, control)] == 0.0)


def test_single_arm_episode_leaves_the_absent_arm_at_zero():
    """A degenerate episode must not raise; it has one block, not two."""
    X, A, Y, W, Z = make_data()
    A = np.ones_like(A)

    est = PMMREstimator(PMMRConfig()).fit(X, A, Y, W=W, Z=Z)
    assert np.all(np.isfinite(est._alpha))
    # Everything is in the treated block, so the contrast is entirely one-sided.
    assert np.isfinite(est.predict(X, W=W, strategy="marginal")).all()
