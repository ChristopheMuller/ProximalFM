"""NMMR's moment kernel: the degeneracy, and the opt-in repair.

The failure this pins is silent — a near-diagonal K turns the V-statistic into
ordinary least squares on (A, W, X) and the U-statistic into a vanishing
remainder, and both look like healthy convergence in the loss curve. See
``nmmr_length_scale.txt``.
"""

import numpy as np
import pytest
import torch

from benchmarks.methods.nmmr import (
    NMMRConfig,
    NMMREstimator,
    _delta_kernel_matrix,
    _median_bandwidth_sq,
    _rbf_kernel_matrix,
)


def _episode(n=256, d_X=8, d_Z=5, d_W=3, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, d_X))
    Z = U + rng.normal(size=(n, d_Z))
    W = U + rng.normal(size=(n, d_W))
    A = (rng.random((n, 1)) < 0.5).astype(np.float64)
    Y = (A[:, 0] + U[:, 0] + X[:, 0] + rng.normal(size=n) * 0.1).astype(np.float64)
    return X, A, Y, W, Z


def _fit(**cfg_kwargs):
    X, A, Y, W, Z = _episode()
    cfg = NMMRConfig(n_epochs=1, hidden_dim=8, n_layers=1, device="cpu", **cfg_kwargs)
    est = NMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)
    return est, (X, A, Y, W, Z)


def _offdiag_median(K):
    n = K.shape[0]
    return float(K[~torch.eye(n, dtype=torch.bool)].median())


def test_reference_kernel_is_degenerate_at_our_dimensions():
    """The published kernel: dimension-blind, and effectively the identity here."""
    est, _ = _fit(adaptive_kernel=False)

    assert est.diagnostics_["kernel_adaptive"] == 0.0
    assert est.diagnostics_["kernel_dim"] == 14
    assert est.diagnostics_["kernel_offdiag_median"] < 1e-3


def test_adaptive_kernel_keeps_moment_signal_at_high_dimension():
    """The floor that would catch a silent regression to a near-identity K.

    The joint median heuristic puts each block's median pair at exp(-1), so
    over the same-arm pairs the diagnostic measures (see
    ``_record_kernel_diagnostics``) the median lands around exp(-2) — orders of
    magnitude above the reference kernel's 1e-05 at the same dimension.
    """
    est, _ = _fit(adaptive_kernel=True)

    assert est.diagnostics_["kernel_adaptive"] == 1.0
    assert est.diagnostics_["kernel_offdiag_median"] > 1e-2


@pytest.mark.parametrize("d_extra", [0, 20, 100])
def test_adaptive_kernel_is_dimension_independent(d_extra):
    """Adding inert columns must not collapse the kernel; under the reference
    bandwidth it does."""
    n = 128
    rng = np.random.default_rng(1)
    block = rng.normal(size=(n, 1 + d_extra))
    t = torch.tensor(block, dtype=torch.float32)

    ls2 = _median_bandwidth_sq(t)
    K = torch.exp(-torch.cdist(t, t).pow(2) / (2.0 * ls2))

    # Median pair sits at exp(-1) whatever the width.
    assert _offdiag_median(K) == pytest.approx(np.exp(-1.0), abs=0.02)


def test_delta_kernel_matches_within_arm_only():
    A = torch.tensor([[0.0], [1.0], [1.0]])
    K = _delta_kernel_matrix(A)
    assert torch.equal(K, torch.tensor([[1.0, 0, 0], [0, 1, 1], [0, 1, 1]]))


def test_bandwidths_are_fitted_once_and_stored():
    """Per-minibatch bandwidths would make the training objective non-stationary."""
    est, (X, A, Y, W, Z) = _fit(adaptive_kernel=True)

    ls2_Z, ls2_X = est._ls2_Z, est._ls2_X
    assert ls2_Z > 0 and ls2_X > 0

    # Refitting the same episode reproduces them; scoring does not mutate them.
    est.score(X, A, Y, W=W, Z=Z)
    assert (est._ls2_Z, est._ls2_X) == (ls2_Z, ls2_X)


def test_kernel_scale_widens_the_derived_bandwidth():
    wide, _ = _fit(adaptive_kernel=True, kernel_scale=2.0)
    base, _ = _fit(adaptive_kernel=True, kernel_scale=1.0)

    assert wide._ls2_Z == pytest.approx(4.0 * base._ls2_Z)
    assert wide.diagnostics_["kernel_offdiag_median"] > (
        base.diagnostics_["kernel_offdiag_median"]
    )


def test_score_defaults_to_u_stat_under_the_adaptive_kernel():
    """V's diagonal is MSE/n, which varies by candidate and would inject an MSE
    preference into the ranking; U removes it exactly."""
    est, (X, A, Y, W, Z) = _fit(adaptive_kernel=True)

    assert est.score(X, A, Y, W=W, Z=Z) == est.score(X, A, Y, W=W, Z=Z, loss="u_stat")

    ref, (X, A, Y, W, Z) = _fit(adaptive_kernel=False)
    assert ref.score(X, A, Y, W=W, Z=Z) == ref.score(X, A, Y, W=W, Z=Z, loss="v_stat")


def test_score_ignores_the_candidate_kernel_scale():
    """Scoring must vary by episode but not by candidate."""
    X, A, Y, W, Z = _episode()

    scores = []
    for scale in (0.5, 2.0):
        cfg = NMMRConfig(
            n_epochs=0,
            hidden_dim=8,
            n_layers=1,
            device="cpu",
            adaptive_kernel=True,
            kernel_scale=scale,
        )
        est = NMMREstimator(cfg).fit(X, A, Y, W=W, Z=Z)
        scores.append(est.score(X, A, Y, W=W, Z=Z))

    assert scores[0] == pytest.approx(scores[1])


def test_default_path_still_builds_the_published_kernel():
    """adaptive_kernel is opt-in: by default the trained objective is byte-for-byte
    the joint RBF over the standardised concatenation, as before."""
    est, (X, A, Y, W, Z) = _fit(adaptive_kernel=False, length_scale=1.0)

    A_t, Z_t, X_t = (
        torch.tensor(v, dtype=torch.float32) for v in (A, Z, np.asarray(X))
    )
    expected = _rbf_kernel_matrix(
        torch.cat([A_t, Z_t / est._Z_scale, X_t / est._X_scale], dim=1),
        length_scale=1.0,
    )

    assert torch.equal(est._moment_kernel(A_t, Z_t, X_t), expected)
