"""Closed-form checks for ``proximalfm.train._dist_metrics``.

Every assertion here is against an analytic value, not a smoke test: these are
the only pieces of the distributional-eval stack with exact answers, so a
regression should be a number that moved, not a shape that broke.
"""

import math

import pytest
import torch

from proximalfm.train._dist_metrics import (
    coverage,
    crps,
    oracle_discretized_crps,
    out_of_grid_fraction,
    pit,
    pit_ks,
    predictive_cdf,
    predictive_mean_var,
    predictive_quantile,
    variance_identity,
)
from proximalfm.train._losses import gaussian_bin_mass


def make_grid(vmin=-8.0, vmax=8.0, nbins=800):
    return torch.linspace(vmin, vmax, nbins + 1)


def discretized_gaussian(bin_edges, center, sigma):
    """Bin-integrated N(center, sigma^2), renormalized over the grid."""
    center = torch.as_tensor(center, dtype=torch.float32)
    p = gaussian_bin_mass(bin_edges, center, sigma)
    return p / p.sum(dim=-1, keepdim=True)


def analytic_crps_gaussian(y, mu, sigma):
    """CRPS of N(mu, sigma^2) at y: sigma * [z(2Phi(z) - 1) + 2phi(z) - 1/sqrt(pi)]."""
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


# --------------------------------------------------------------------------
# Representation: CDF, moments, quantiles
# --------------------------------------------------------------------------


def test_cdf_endpoints_and_piecewise_linearity():
    edges = torch.tensor([0.0, 1.0, 2.0])
    probs = torch.tensor([0.25, 0.75])

    # Exactly 0 / 1 outside, and linear inside each bin.
    assert predictive_cdf(probs, edges, torch.tensor(-1.0)).item() == 0.0
    assert predictive_cdf(probs, edges, torch.tensor(3.0)).item() == 1.0
    assert predictive_cdf(probs, edges, torch.tensor(0.5)).item() == pytest.approx(
        0.125
    )
    assert predictive_cdf(probs, edges, torch.tensor(1.0)).item() == pytest.approx(0.25)
    assert predictive_cdf(probs, edges, torch.tensor(1.5)).item() == pytest.approx(
        0.625
    )


def test_single_bin_variance_is_the_grid_floor():
    """All mass in one bin => sd = w/sqrt(12), not 0. Module docstring, point 2."""
    w = 0.4
    edges = torch.tensor([0.0, w, 2 * w, 3 * w])
    probs = torch.tensor([0.0, 1.0, 0.0])

    mean, var = predictive_mean_var(probs, edges)

    assert mean.item() == pytest.approx(1.5 * w)
    assert var.sqrt().item() == pytest.approx(w / math.sqrt(12.0), rel=1e-5)


def test_mean_var_matches_gaussian_moments():
    edges = make_grid()
    probs = discretized_gaussian(edges, 0.7, 1.3)

    mean, var = predictive_mean_var(probs, edges)

    assert mean.item() == pytest.approx(0.7, abs=1e-3)
    assert var.sqrt().item() == pytest.approx(1.3, rel=2e-3)


def test_quantile_inverts_the_cdf():
    edges = make_grid()
    probs = discretized_gaussian(edges, -0.3, 0.9)
    levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])

    q = predictive_quantile(probs, edges, levels)

    # Round trip, and agreement with the analytic Gaussian quantiles.
    assert torch.allclose(predictive_cdf(probs, edges, q), levels, atol=1e-4)
    expected = -0.3 + 0.9 * torch.tensor([-1.6449, -0.6745, 0.0, 0.6745, 1.6449])
    assert torch.allclose(q, expected, atol=5e-3)


def test_batched_shapes():
    edges = make_grid(nbins=64)
    probs = torch.rand(3, 5, 64)
    probs = probs / probs.sum(-1, keepdim=True)
    y = torch.randn(3, 5)

    assert crps(probs, edges, y).shape == (3, 5)
    assert pit(probs, edges, y).shape == (3, 5)
    assert predictive_quantile(probs, edges, torch.tensor([0.1, 0.9])).shape == (
        3,
        5,
        2,
    )
    assert predictive_quantile(probs, edges, 0.5).shape == (3, 5)


# --------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------


def test_crps_of_point_mass_is_absolute_error():
    """A one-bin forecast is charged |error|, up to the bin's own w/6 floor."""
    w = 1e-3
    edges = torch.tensor([-0.5 * w, 0.5 * w])  # single bin centred at 0
    probs = torch.tensor([1.0])
    y = torch.tensor([2.0, -3.0, 0.5])

    assert torch.allclose(crps(probs, edges, y), y.abs(), atol=w)


def test_crps_matches_analytic_gaussian():
    edges = make_grid()
    probs = discretized_gaussian(edges, 0.0, 1.0)

    for y in (0.0, 0.5, -1.7, 3.0):
        got = crps(probs, edges, torch.tensor(y)).item()
        assert got == pytest.approx(analytic_crps_gaussian(y, 0.0, 1.0), rel=2e-3)


def test_crps_charges_distance_past_the_grid():
    """Out-of-grid targets are charged the distance to the edge, not clipped."""
    edges = torch.tensor([-1.0, 0.0, 1.0])
    probs = torch.tensor([0.5, 0.5])

    inside = crps(probs, edges, torch.tensor(1.0)).item()
    for extra in (0.5, 5.0, 50.0):
        far = crps(probs, edges, torch.tensor(1.0 + extra)).item()
        assert far == pytest.approx(inside + extra, rel=1e-5)


def test_crps_is_minimized_by_the_truth():
    """Propriety, empirically: a wrong centre or a wrong width both cost."""
    edges = make_grid()
    y = torch.randn(4000, generator=torch.Generator().manual_seed(0))

    def mean_crps(sigma, shift=0.0):
        probs = discretized_gaussian(edges, shift, sigma).expand(y.numel(), -1)
        return crps(probs, edges, y).mean().item()

    best = mean_crps(1.0)
    assert best < mean_crps(0.5)
    assert best < mean_crps(2.0)
    assert best < mean_crps(1.0, shift=0.5)


def test_oracle_crps_is_the_bin_floor():
    w = 0.6
    edges = torch.tensor([0.0, w, 2 * w])

    # Bin centre is the best case (w/12); an edge is the worst (w/3).
    assert oracle_discretized_crps(
        torch.tensor(0.5 * w), edges
    ).item() == pytest.approx(w / 12.0, rel=1e-5)
    assert oracle_discretized_crps(
        torch.tensor(1.0 * w), edges
    ).item() == pytest.approx(w / 3.0, rel=1e-5)


def test_crps_over_oracle_never_below_one():
    edges = make_grid(nbins=40)  # deliberately coarse
    y = torch.randn(500, generator=torch.Generator().manual_seed(1))
    probs = discretized_gaussian(edges, 0.0, 1.0).expand(y.numel(), -1)

    ratio = crps(probs, edges, y).mean() / oracle_discretized_crps(y, edges).mean()
    assert ratio.item() > 1.0


# --------------------------------------------------------------------------
# Calibration: PIT, KS, coverage, variance identity
# --------------------------------------------------------------------------


def test_pit_ks_band_shrinks_with_n():
    u = torch.rand(10_000, generator=torch.Generator().manual_seed(2))
    stat_small, band_small = pit_ks(u[:100])
    stat_big, band_big = pit_ks(u)

    assert band_small.item() > band_big.item()
    assert band_big.item() == pytest.approx(
        math.sqrt(-0.5 * math.log(0.025) / 10_000), rel=1e-9
    )
    # Genuinely uniform input passes at both sample sizes.
    assert stat_small.item() < band_small.item()
    assert stat_big.item() < band_big.item()


def test_pit_ks_batched_over_episodes():
    u = torch.rand(7, 500, generator=torch.Generator().manual_seed(3))
    stat, band = pit_ks(u)
    assert stat.shape == (7,) and band.shape == (7,)
    assert bool((stat < band).all())


def test_correct_forecast_is_calibrated():
    """A correct Gaussian forecast: KS inside band, sd/rmse ~ 1, cov90 ~ 0.90."""
    edges = make_grid()
    n = 20_000
    y = torch.randn(n, generator=torch.Generator().manual_seed(4))
    probs = discretized_gaussian(edges, 0.0, 1.0).expand(n, -1)

    stat, band = pit_ks(pit(probs, edges, y))
    assert stat.item() < band.item()

    ident = variance_identity(probs, edges, y)
    assert ident["sd_over_rmse"] == pytest.approx(1.0, abs=0.03)

    cov = coverage(probs, edges, y)
    assert cov["cov90"] == pytest.approx(0.90, abs=0.01)
    assert cov["cov50"] == pytest.approx(0.50, abs=0.02)
    # Width is a property of the forecast alone, so it is exact.
    assert cov["width90"] == pytest.approx(2 * 1.6449, abs=0.01)
    assert cov["width50"] == pytest.approx(2 * 0.6745, abs=0.01)

    assert out_of_grid_fraction(edges, y) == 0.0


def test_too_narrow_forecast_fails_both_tests():
    """Over-confidence must show up in the PIT *and* in sd/rmse.

    Either alone is ambiguous -- a U-shaped PIT can also come from a biased
    centre, and sd/rmse is a single global ratio -- so the pair is the test.
    """
    edges = make_grid()
    n = 20_000
    y = torch.randn(n, generator=torch.Generator().manual_seed(5))
    probs = discretized_gaussian(edges, 0.0, 0.5).expand(n, -1)

    u = pit(probs, edges, y)
    stat, band = pit_ks(u)
    assert stat.item() > band.item()

    # U-shaped: far more mass in the extreme 10% than the nominal 10%.
    tail_frac = ((u < 0.05) | (u > 0.95)).double().mean().item()
    assert tail_frac > 0.25

    ident = variance_identity(probs, edges, y)
    assert ident["sd_over_rmse"] == pytest.approx(0.5, abs=0.03)
    assert coverage(probs, edges, y)["cov90"] < 0.75


def test_too_wide_forecast_is_dome_shaped():
    edges = make_grid()
    n = 20_000
    y = torch.randn(n, generator=torch.Generator().manual_seed(6))
    probs = discretized_gaussian(edges, 0.0, 2.0).expand(n, -1)

    u = pit(probs, edges, y)
    stat, band = pit_ks(u)
    assert stat.item() > band.item()

    tail_frac = ((u < 0.05) | (u > 0.95)).double().mean().item()
    assert tail_frac < 0.05

    assert variance_identity(probs, edges, y)["sd_over_rmse"] == pytest.approx(
        2.0, abs=0.1
    )
    assert coverage(probs, edges, y)["cov90"] > 0.98


def test_variance_identity_reduces_to_var_over_mse_without_noise():
    edges = make_grid()
    n = 4000
    y = torch.randn(n, generator=torch.Generator().manual_seed(7))
    probs = discretized_gaussian(edges, 0.0, 1.0).expand(n, -1)

    ident = variance_identity(probs, edges, y, s2=None)

    assert ident["mean_s2"] == 0.0
    assert ident["mse_corrected"] == pytest.approx(ident["mse_obs"])
    assert ident["var_ratio"] == pytest.approx(ident["mean_var"] / ident["mse_obs"])
    assert ident["sd_over_rmse"] == pytest.approx(math.sqrt(ident["var_ratio"]))
    # s2 = 0 explicitly must agree with s2 = None.
    zeros = variance_identity(probs, edges, y, s2=torch.zeros(n))
    assert zeros["var_ratio"] == pytest.approx(ident["var_ratio"])


def test_variance_identity_subtracts_target_noise():
    """A correct posterior scored against a noisy target reads over-confident
    until s^2 is subtracted -- and the sign of that correction matters."""
    edges = make_grid()
    n = 20_000
    gen = torch.Generator().manual_seed(8)
    tau = torch.randn(n, generator=gen)  # the estimand; posterior sd 1 is correct
    s = 0.5
    y = tau + s * torch.randn(n, generator=gen)
    probs = discretized_gaussian(edges, 0.0, 1.0).expand(n, -1)

    naive = variance_identity(probs, edges, y, s2=None)
    corrected = variance_identity(probs, edges, y, s2=torch.full((n,), s**2))

    # MSE_obs ~ 1 + s^2 = 1.25, so the naive ratio understates width.
    assert naive["sd_over_rmse"] == pytest.approx(1.0 / math.sqrt(1.25), abs=0.03)
    assert corrected["sd_over_rmse"] == pytest.approx(1.0, abs=0.03)
    assert corrected["s_over_sd"] == pytest.approx(s, abs=0.02)


def test_variance_identity_tolerates_missing_s2_rows():
    edges = make_grid(nbins=64)
    n = 128
    y = torch.randn(n, generator=torch.Generator().manual_seed(9))
    probs = discretized_gaussian(edges, 0.0, 1.0).expand(n, -1)

    s2 = torch.full((n,), 0.1)
    s2[:10] = float("nan")
    s2[10:20] = -1.0

    ident = variance_identity(probs, edges, y, s2=s2)
    assert math.isfinite(ident["var_ratio"])
    assert ident["mean_s2"] == pytest.approx(0.1 * (n - 20) / n)


# --------------------------------------------------------------------------
# Out-of-grid accounting
# --------------------------------------------------------------------------


def test_out_of_grid_fraction_and_pit_pinning():
    edges = torch.tensor([-1.0, 0.0, 1.0])
    probs = torch.tensor([0.5, 0.5]).expand(4, -1)
    y = torch.tensor([-2.0, 0.0, 0.5, 3.0])

    assert out_of_grid_fraction(edges, y) == pytest.approx(0.5)
    u = pit(probs, edges, y)
    assert u[0].item() == 0.0 and u[3].item() == 1.0


def test_shape_contract_errors():
    edges = make_grid(nbins=8)
    with pytest.raises(ValueError, match="bins"):
        crps(torch.ones(4) / 4, edges, torch.tensor(0.0))
    with pytest.raises(ValueError, match="1-D"):
        crps(torch.ones(8) / 8, edges.expand(2, -1), torch.tensor(0.0))
    with pytest.raises(ValueError, match="at least 2"):
        pit_ks(torch.tensor([0.5]))
