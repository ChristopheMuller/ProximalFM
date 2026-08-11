"""The Demand benchmark adapter.

Two kinds of check. The first kind pins **fidelity to the published DGP** by
reconstructing each structural equation from the emitted tensors: with the noise
scale known, ``W - 7g(U) - 45`` must be standard normal, and ``Y1 - Y0`` on the
observed rows has an exact closed form because ``eps_5`` and ``-5g(U)`` cancel
in the difference. Those are the tests that would catch a transcription slip in
a formula.

The second kind pins the **two adaptations** -- binarizing a continuous price,
and having no observed covariates -- since both are places where a defensible
choice and a silently wrong one look identical in the output.
"""

from __future__ import annotations

import math

import pytest
import torch

from lit_eval.base import EpisodeSpec
from lit_eval.datasets.demand import Demand, g, potential_outcome
from lit_eval.registry import build_dataset

T = 512
TRAIN = 384
K = 400

# Index into EPISODE_FIELDS.
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


def episode(seed: int = 3, mc_cate: int = K, **options):
    ds = Demand(**options)
    spec = EpisodeSpec(index=0, seed=seed, seq_len=T, train_size=TRAIN, mc_cate=mc_cate)
    tensors, info = ds.generate(spec)
    return ds, tensors, info


@pytest.fixture(scope="module")
def ep():
    return episode()


# ---------------------------------------------------------------------------
# Fidelity to the published structural equations
# ---------------------------------------------------------------------------


def test_g_matches_the_published_formula():
    u = torch.tensor([0.0, 5.0, 10.0])
    expected = torch.tensor(
        [
            2 * ((0 - 5) ** 4 / 600 + math.exp(-4 * 25) + 0.0 - 2),
            2 * ((0.0) + 1.0 + 0.5 - 2),
            2 * ((5.0) ** 4 / 600 + math.exp(-100) + 1.0 - 2),
        ]
    )
    torch.testing.assert_close(g(u), expected, rtol=1e-6, atol=1e-6)


def test_demand_is_uniform_on_zero_ten(ep):
    _, t, _ = ep
    U = t[U_].reshape(-1)
    assert float(U.min()) >= 0.0 and float(U.max()) <= 10.0
    # Uniform(0,10): mean 5, sd 10/sqrt(12) = 2.887.
    assert abs(float(U.mean()) - 5.0) < 0.4
    assert abs(float(U.std()) - 10 / math.sqrt(12)) < 0.3


def test_views_and_cost_reproduce_their_equations(ep):
    """``W = 7g(U) + 45 + eps_3`` and ``Z = [2sin, 2cos] + eps``, sd(eps) = 1."""
    _, t, _ = ep
    U = t[U_].reshape(-1)
    resid_w = t[W_].reshape(-1) - (7.0 * g(U) + 45.0)
    assert abs(float(resid_w.std()) - 1.0) < 0.15
    assert abs(float(resid_w.mean())) < 0.15

    ang = 2 * torch.pi * U / 10.0
    resid_z1 = t[Z_][:, 0] - 2.0 * torch.sin(ang)
    resid_z2 = t[Z_][:, 1] - 2.0 * torch.cos(ang)
    for r in (resid_z1, resid_z2):
        assert abs(float(r.std()) - 1.0) < 0.15
        assert abs(float(r.mean())) < 0.15


def test_potential_outcome_difference_is_exact_on_the_observed_rows(ep):
    """``eps_5`` and ``-5g(U)`` are shared across arms, so their difference is
    a deterministic function of ``W`` alone -- an exact check on the link."""
    ds, t, _ = ep
    W = t[W_]
    p_lo, p_hi = ds.options["price_low"], ds.options["price_high"]
    zeros = torch.zeros_like(W)
    expected = potential_outcome(p_hi, W, zeros) - potential_outcome(p_lo, W, zeros)
    torch.testing.assert_close(t[Y1_] - t[Y0_], expected, rtol=1e-5, atol=1e-4)


def test_the_link_saturates_where_the_published_min_says_it_does(ep):
    ds, t, _ = ep
    W = t[W_]
    p_lo = ds.options["price_low"]
    y = potential_outcome(p_lo, W, torch.zeros_like(W))
    assert float(y.max()) <= 5.0 * p_lo + 1e-4


# ---------------------------------------------------------------------------
# Adaptation 1: binarizing a continuous price
# ---------------------------------------------------------------------------


def test_assignment_keeps_positivity_and_stays_confounded(ep):
    """Overlap must survive the binarization, and confounding must not.

    The naive difference in means is the whole point of the benchmark: if it
    landed on the true effect there would be nothing for a proximal method to
    do. At the default temperature it sits near -10 against a true -19.
    """
    _, t, _ = ep
    p = t[P_].reshape(-1)
    assert float(p.min()) > 0.01 and float(p.max()) < 0.99
    treated = float(t[A_].mean())
    assert 0.25 < treated < 0.75

    A = t[A_].reshape(-1).bool()
    y = t[YOBS_].reshape(-1)
    naive = float(y[A].mean() - y[~A].mean())
    true_ate = float((t[MU1_] - t[MU0_])[TRAIN:].mean())
    assert true_ate < -15.0, true_ate
    assert naive - true_ate > 5.0, (naive, true_ate)


def test_a_hard_threshold_is_refused():
    with pytest.raises(ValueError, match="positivity"):
        episode(assignment_temp=0.0)


def test_lower_temperature_sharpens_assignment(ep):
    _, sharp, _ = episode(assignment_temp=0.25)
    _, t, _ = ep
    assert float(sharp[P_].std()) > float(t[P_].std())


def test_assignment_shift_moves_only_the_arm_balance(ep):
    """A location shift in logit space: imbalance without changing anything else.

    ``temp`` sets selection *strength*, ``shift`` sets arm *imbalance*, and the
    two are orthogonal by construction. The shift is applied after every RNG
    draw in the episode, so the potential outcomes must come out byte-identical
    -- that is what makes an imbalance sweep a one-variable experiment.
    """
    _, base, _ = ep
    fracs = []
    for shift in (-2.0, -1.0, 0.0, 1.0, 2.0):
        _, t, info = episode(assignment_shift=shift)
        fracs.append(float(t[P_].mean()))
        # Only assignment moved: the potential outcomes are the same draws.
        torch.testing.assert_close(t[Y0_], base[Y0_], rtol=0, atol=0)
        torch.testing.assert_close(t[Y1_], base[Y1_], rtol=0, atol=0)
        assert info["sampled_hp"]["assignment_shift"] == shift

    assert fracs == sorted(fracs), fracs
    assert fracs[0] < 0.2 and fracs[-1] > 0.8
    assert 0.4 < fracs[2] < 0.6  # shift = 0 is the balanced default

    # Confounding strength is a property of the logit's slope, not its level:
    # U and Z keep their published coefficients under any shift.
    _, shifted, _ = episode(assignment_shift=2.0)
    assert float(shifted[P_].std()) > 0.05


# ---------------------------------------------------------------------------
# Adaptation 2: no observed covariates
# ---------------------------------------------------------------------------


def test_tau_star_is_constant_so_its_row_spread_is_pure_mc_noise(ep):
    """Faithful Demand is an ATE benchmark; this is what that looks like.

    ``sd_rows(tau*) ~= sqrt(mean S2)`` is the signature of a constant target
    estimated by Monte Carlo -- and the reason ``prior_eval.verify``'s group 4
    (which reads ``s / sd(tau*)``) is meaningless on a faithful build.
    """
    _, t, _ = ep
    tau = (t[MU1_] - t[MU0_])[TRAIN:].reshape(-1)
    mc_sd = float(t[S2_][TRAIN:].mean().sqrt())
    assert 0.7 < float(tau.std()) / mc_sd < 1.4


def test_x_effect_scale_makes_tau_star_actually_vary(ep):
    """The opt-in deviation, and the check that it is off by default."""
    _, base, _ = ep
    _, het, _ = episode(x_effect_scale=4.0)

    def spread(t):
        tau = (t[MU1_] - t[MU0_])[TRAIN:].reshape(-1)
        return float(tau.std()), float(t[S2_][TRAIN:].mean().sqrt())

    base_sd, base_mc = spread(base)
    het_sd, het_mc = spread(het)
    assert base_sd < 1.5 * base_mc  # noise only
    assert het_sd > 5 * het_mc  # genuine heterogeneity

    # The low-price arm keeps its published law: only Y1 was touched.
    torch.testing.assert_close(base[Y0_], het[Y0_], rtol=1e-5, atol=1e-4)


def test_zero_covariates_is_refused_with_a_readable_error():
    """The format has no zero-width block; without this the failure surfaces as
    an unreadable ``view(-1, 0)`` reshape error inside the sparse writer."""
    with pytest.raises(ValueError, match="zero-width block"):
        episode(n_covariates=0, mc_cate=1)


def test_covariate_width_does_not_change_the_published_law():
    """X is inert: only the RNG stream offset moves with ``n_covariates``.

    Byte-equality is the wrong check (the X draw shifts the stream), so this
    compares the *law* -- which is what "faithful to the published DGP" means
    here.
    """

    def moments(nc):
        ds = Demand(n_covariates=nc)
        cols = {k: [] for k in ("U", "W", "Y0", "Y1")}
        for i in range(12):
            t, _ = ds.generate(
                EpisodeSpec(
                    index=i, seed=500 + i, seq_len=512, train_size=384, mc_cate=1
                )
            )
            for k, idx in (("U", U_), ("W", W_), ("Y0", Y0_), ("Y1", Y1_)):
                cols[k].append(t[idx].reshape(-1))
        return {
            k: (float(torch.cat(v).mean()), float(torch.cat(v).std()))
            for k, v in cols.items()
        }

    one, three = moments(1), moments(3)
    for k in one:
        mean_a, sd_a = one[k]
        mean_b, sd_b = three[k]
        assert abs(mean_a - mean_b) < 0.1 * sd_a, (k, mean_a, mean_b)
        assert abs(sd_a - sd_b) < 0.1 * sd_a, (k, sd_a, sd_b)


def test_covariates_are_inert_by_default(ep):
    """X is noise standing in for "no covariates" -- it must enter nothing."""
    _, t, _ = ep
    X = t[X_].reshape(-1)
    for other in (t[U_].reshape(-1), t[W_].reshape(-1), t[YOBS_].reshape(-1)):
        corr = torch.corrcoef(torch.stack([X, other]))[0, 1]
        assert abs(float(corr)) < 0.15


def test_registered_with_its_provenance():
    ds = build_dataset("demand")
    assert isinstance(ds, Demand)
    assert "Xu" in ds.citation
    assert ds.max_dims.d_Z == 2 and ds.max_dims.d_U == 1
