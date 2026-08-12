"""Mastouri et al.'s synthetic design.

The design's value is entirely in an asymmetry -- ``U_1`` is recoverable from
``U_2`` but not the reverse, and each proxy is deliberately informative about
the *other* one's weak dimension -- so the tests here check that asymmetry
survives implementation, not just that the code runs. The sharpest are the
exact ones: ``Y`` is noiseless given the latent, so ``Y0`` must equal its
formula to floating point, which simultaneously pins that ``W`` never enters the
outcome.

``sigma^2 = 3`` and ``beta^2 = 0.05`` are variances; reading them as standard
deviations makes both proxies ~3x more informative and quietly turns this into
an easier benchmark. ``test_proxy_noise_is_the_published_variance`` is what
catches that.
"""

from __future__ import annotations

import math

import pytest
import torch

from lit_eval.base import EpisodeSpec
from lit_eval.datasets.synthetic_mastouri import SyntheticMastouri, response
from lit_eval.registry import build_dataset

T = 20000
TRAIN = 16000

U_, X_, W_, Z_, A_, P_, YOBS_, Y0_, Y1_, MU0_, MU1_, S2_ = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    13,
    14,
    15,
)


def episode(
    seed: int = 2, mc_cate: int = 1, seq_len: int = T, train: int = TRAIN, **opts
):
    ds = SyntheticMastouri(**opts)
    spec = EpisodeSpec(
        index=0, seed=seed, seq_len=seq_len, train_size=train, mc_cate=mc_cate
    )
    tensors, info = ds.generate(spec)
    return ds, tensors, info


@pytest.fixture(scope="module")
def ep():
    return episode()


def _corr(a, b):
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


# ---------------------------------------------------------------------------
# The latent, and the asymmetry the design is built on
# ---------------------------------------------------------------------------


def test_latent_coupling_is_the_published_indicator_shift(ep):
    """``U_1 ~ U[0,1] - 1[0 <= U_2 <= 1]``: the shift is exact, not stochastic.

    Given ``U_2`` the range of ``U_1`` is known; given ``U_1 in [0,1]``, ``U_2``
    is equally likely in ``[-1,0]`` or ``[1,2]``. That one-way informativeness
    is the design.
    """
    _, t, _ = ep
    U1, U2 = t[U_][:, 0], t[U_][:, 1]
    assert -1.0 <= float(U2.min()) and float(U2.max()) <= 2.0
    assert abs(float(U2.mean()) - 0.5) < 0.05

    inside = (U2 >= 0) & (U2 <= 1)
    assert float(U1[inside].max()) <= 0.0 and float(U1[inside].min()) >= -1.0
    assert float(U1[~inside].min()) >= 0.0 and float(U1[~inside].max()) <= 1.0

    # The reverse direction really is ambiguous: given U_1 >= 0, U_2 splits
    # about evenly between the two intervals it can occupy.
    low = U2[(U1 >= 0) & (U2 < 0)].numel()
    high = U2[(U1 >= 0) & (U2 > 1)].numel()
    assert 0.4 < low / (low + high) < 0.6


def test_proxy_noise_is_the_published_variance(ep):
    """sigma^2 = 3 means sd = sqrt(3), and the uniform noise is [-1, 1]."""
    _, t, _ = ep
    U1, U2 = t[U_][:, 0], t[U_][:, 1]
    W, Z = t[W_], t[Z_]

    assert abs(float((W[:, 1] - U2).std()) - math.sqrt(3.0)) < 0.05
    assert abs(float((Z[:, 0] - U1).std()) - math.sqrt(3.0)) < 0.05

    for resid in (W[:, 0] - U1, Z[:, 1] - U2):
        assert abs(float(resid.min()) + 1.0) < 0.02
        assert abs(float(resid.max()) - 1.0) < 0.02
        # Uniform[-1,1] has sd 1/sqrt(3), not 1.
        assert abs(float(resid.std()) - 1 / math.sqrt(3.0)) < 0.02


def test_each_proxy_is_informative_about_the_other_ones_weak_dimension(ep):
    """Neither proxy set alone is complete for U; together they are.

    ``W`` sees ``U_1`` well and ``U_2`` badly; ``Z`` is the mirror image. A
    method adjusting on ``W`` alone therefore cannot recover ``U_2``, which is
    the dimension that drives ``Y``.
    """
    _, t, _ = ep
    U1, U2 = t[U_][:, 0], t[U_][:, 1]
    W, Z = t[W_], t[Z_]
    assert _corr(W[:, 0], U1) > 0.6 > _corr(W[:, 1], U2)
    assert _corr(Z[:, 1], U2) > 0.6 > _corr(Z[:, 0], U1)


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


def test_outcome_is_exactly_its_formula_and_ignores_the_proxies(ep):
    """``Y^a = U_2 cos(2(a + 0.3 U_1 + 0.2))``, noiseless.

    Exact equality also pins that ``W`` never enters the outcome -- it is a pure
    child of ``U`` in this design, which is what makes it a leaf w.r.t. the
    target and lets it be generated on the observed rows alone.
    """
    ds, t, _ = ep
    U1, U2 = t[U_][:, :1], t[U_][:, 1:2]
    a_lo, a_hi = ds.options["action_low"], ds.options["action_high"]
    torch.testing.assert_close(t[Y0_], response(a_lo, U1, U2), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(t[Y1_], response(a_hi, U1, U2), rtol=1e-6, atol=1e-6)


def test_default_arms_are_a_confounded_contrast(ep):
    """The cosine response makes the naive bias change sign along the curve.

    At the default ``(0.5, 1.5)`` an unadjusted estimate overstates the effect
    by ~70%; at ``(0.0, 1.0)`` it is nearly unbiased. Picking the contrast by
    accident would flatter every method, so the default is pinned here.
    """
    _, t, _ = ep
    true = float((t[Y1_] - t[Y0_]).mean())
    A = t[A_].reshape(-1).bool()
    y = t[YOBS_].reshape(-1)
    naive = float(y[A].mean() - y[~A].mean())
    assert abs(true - (-0.475)) < 0.05, true
    assert abs(naive - (-0.819)) < 0.05, naive
    assert abs(naive - true) > 0.2

    # The nearly-unconfounded pair, for contrast -- and as a guard on anyone
    # "fixing" the default.
    _, unconf, _ = episode(action_low=0.0, action_high=1.0)
    true_u = float((unconf[Y1_] - unconf[Y0_]).mean())
    Au = unconf[A_].reshape(-1).bool()
    yu = unconf[YOBS_].reshape(-1)
    assert abs(float(yu[Au].mean() - yu[~Au].mean()) - true_u) < 0.1


# ---------------------------------------------------------------------------
# Treatment, and the shared adaptations
# ---------------------------------------------------------------------------


def test_assignment_is_driven_by_the_latent_but_keeps_overlap(ep):
    """``A = U_2 + N(0, 0.05)``, so selection is essentially on ``U_2`` --
    the dimension that also drives ``Y``. That is the confounding."""
    _, t, _ = ep
    p = t[P_].reshape(-1)
    assert _corr(p, t[U_][:, 1]) > 0.95
    assert float(p.min()) > 0.02 and float(p.max()) < 0.98
    assert 0.4 < float(t[A_].mean()) < 0.6


def test_shared_arm_options_are_wired_in():
    _, shifted, info = episode(assignment_shift=1.5)
    assert float(shifted[A_].mean()) > 0.6
    assert info["sampled_hp"]["assignment_shift"] == 1.5
    with pytest.raises(ValueError, match="positivity"):
        episode(assignment_temp=0.0)
    with pytest.raises(ValueError, match="zero-width block"):
        episode(n_covariates=0)


def test_replicates_estimate_a_constant_tau_star(ep):
    """No covariates => constant CATE, so the row spread is pure MC noise."""
    _, t, _ = episode(mc_cate=256, seq_len=1024, train=768)
    tau = (t[MU1_] - t[MU0_])[768:].reshape(-1)
    mc_sd = float(t[S2_][768:].mean().sqrt())
    assert 0.7 < float(tau.std()) / mc_sd < 1.4
    # And it lands on the design's true ATE.
    assert abs(float(tau.mean()) - (-0.475)) < 0.1


def test_x_effect_scale_makes_tau_star_vary():
    _, t, _ = episode(mc_cate=256, seq_len=1024, train=768, x_effect_scale=1.0)
    tau = (t[MU1_] - t[MU0_])[768:].reshape(-1)
    mc_sd = float(t[S2_][768:].mean().sqrt())
    assert float(tau.std()) > 5 * mc_sd


def test_registered_under_the_requested_name():
    ds = build_dataset("synthetic_Mastouri")
    assert isinstance(ds, SyntheticMastouri)
    assert "Mastouri" in ds.citation
    assert (ds.max_dims.d_U, ds.max_dims.d_W, ds.max_dims.d_Z) == (2, 2, 2)
