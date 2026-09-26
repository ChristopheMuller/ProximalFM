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
    out_of_grid_fraction,
    pit,
    pit_ks,
    predictive_cdf,
    predictive_mean_var,
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


def test_crps_matches_analytic_gaussian():
    edges = make_grid()
    probs = discretized_gaussian(edges, 0.0, 1.0)

    for y in (0.0, 0.5, -1.7, 3.0):
        got = crps(probs, edges, torch.tensor(y)).item()
        assert got == pytest.approx(analytic_crps_gaussian(y, 0.0, 1.0), rel=2e-3)


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


def test_out_of_grid_fraction_and_pit_pinning():
    edges = torch.tensor([-1.0, 0.0, 1.0])
    probs = torch.tensor([0.5, 0.5]).expand(4, -1)
    y = torch.tensor([-2.0, 0.0, 0.5, 3.0])

    assert out_of_grid_fraction(edges, y) == pytest.approx(0.5)
    u = pit(probs, edges, y)
    assert u[0].item() == 0.0 and u[3].item() == 1.0
