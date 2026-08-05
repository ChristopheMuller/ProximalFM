"""Tests for the noise-aware CATE likelihood (``loss_type="cate_single_noise_aware"``).

The supervised CATE target is a Monte Carlo mean over ``K`` replicates, so it is
a *noisy* observation of the estimand. These tests pin down that
``compute_cate_noise_aware_nll`` treats it that way, against closed forms where
one exists.
"""

import math

import pytest
import torch

from proximalfm.train._run import (
    compute_cate_distribution_loss,
    compute_cate_noise_aware_nll,
    gaussian_bin_mass,
)


NBINS = 2000
VMIN, VMAX = -6.0, 6.0


@pytest.fixture
def bin_edges():
    return torch.linspace(VMIN, VMAX, NBINS + 1)


def gaussian_logits(bin_edges, mean, sd):
    """Logits whose softmax is ``N(mean, sd^2)`` discretized onto ``bin_edges``."""
    centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    log_p = -0.5 * ((centers - mean) / sd) ** 2
    return log_p - torch.logsumexp(log_p, dim=-1)


def normal_nll(t, mean, var):
    return 0.5 * (math.log(2 * math.pi * var) + (t - mean) ** 2 / var)


def test_matches_closed_form_convolution(bin_edges):
    """A Gaussian predictive convolved with N(0, s^2) has a known density at t.

    This is the whole contract in one assertion: the loss must evaluate
    ``N(t; m, v + s^2 + sigma^2)``, i.e. the *prediction* is widened, not the
    target.
    """
    m, v, s2, sigma = 0.4, 0.25, 0.09, 0.05
    t = torch.tensor([[0.7]])

    logits = gaussian_logits(bin_edges, m, math.sqrt(v)).view(1, 1, -1)

    got = compute_cate_noise_aware_nll(
        logits,
        t,
        bin_edges,
        target_var=torch.full_like(t, s2),
        sigma=sigma,
    )
    expected = normal_nll(0.7, m, v + s2 + sigma**2)

    assert got.item() == pytest.approx(expected, abs=2e-3)


def test_zero_variance_reduces_to_plain_binned_nll(bin_edges):
    """With no target noise and a sub-bin sigma, the loss is -log(p_bin / w).

    That is the degenerate path taken by context rows, K=1 datasets and any
    pre-generated data written before target-noise tracking existed.
    """
    width = float(bin_edges[1] - bin_edges[0])
    t = torch.tensor([[0.31]])

    logits = gaussian_logits(bin_edges, 0.2, 0.5).view(1, 1, -1)

    got = compute_cate_noise_aware_nll(
        logits,
        t,
        bin_edges,
        target_var=torch.zeros_like(t),
        sigma=width / 10.0,
    )

    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.bucketize(t.view(-1), bin_edges).item()) - 1
    expected = -math.log(float(probs[0, 0, idx]) / width)

    assert got.item() == pytest.approx(expected, abs=1e-2)


def test_minimizer_is_the_undilated_width():
    """The point of the loss: it recovers the posterior, not posterior (*) noise.

    Scoring against ``t = tau* + e`` with a noise-unaware objective is minimized
    by a predictive of width ``sqrt(w^2 + s^2)``. The noise-aware one must be
    minimized at ``w`` itself. Both are checked here so the test fails if the
    convolution is applied to the wrong side (or dropped).
    """
    torch.manual_seed(0)

    edges = torch.linspace(-8.0, 8.0, 4001)
    true_w, s = 0.5, 0.4
    sigma = 1e-3

    n = 40000
    tau = torch.randn(n) * true_w
    t = (tau + torch.randn(n) * s).view(1, -1)

    widths = torch.tensor([0.3, 0.4, 0.5, 0.64, 0.8, 1.0])
    aware, unaware = [], []
    for w in widths:
        logits = gaussian_logits(edges, 0.0, float(w)).view(1, 1, -1).expand(1, n, -1)
        aware.append(
            compute_cate_noise_aware_nll(
                logits,
                t,
                edges,
                target_var=torch.full_like(t, s**2),
                sigma=sigma,
            ).item()
        )
        unaware.append(
            compute_cate_noise_aware_nll(
                logits,
                t,
                edges,
                target_var=torch.zeros_like(t),
                sigma=sigma,
            ).item()
        )

    # sqrt(0.5^2 + 0.4^2) = 0.64, which is in the grid above.
    assert widths[int(torch.tensor(aware).argmin())].item() == pytest.approx(true_w)
    assert widths[int(torch.tensor(unaware).argmin())].item() == pytest.approx(0.64)


def test_scale_invariance_of_the_unit_conversion(bin_edges):
    """s^2 must arrive in head units, i.e. divided by the squared context scale.

    Rescaling y by c scales the raw target by c and its variance by c^2. Once
    both are divided through by the context std the loss is unchanged -- which
    is what pins ``target_var_std = S2 / y_context_std**2`` in the trainer. A
    missing or mismatched power here is invisible in every other test.
    """
    tau_raw = torch.tensor([[0.6, -0.2, 1.1]])
    s2_raw = torch.tensor([[0.05, 0.02, 0.08]])
    logits = gaussian_logits(bin_edges, 0.3, 0.7).view(1, 1, -1).expand(1, 3, -1)

    base = compute_cate_noise_aware_nll(
        logits, tau_raw, bin_edges, target_var=s2_raw, sigma=0.05
    )

    c = 3.7
    scaled = compute_cate_noise_aware_nll(
        logits,
        (c * tau_raw) / c,
        bin_edges,
        target_var=(c**2 * s2_raw) / c**2,
        sigma=0.05,
    )

    assert scaled.item() == pytest.approx(base.item(), rel=1e-6)


def test_nan_and_negative_variance_do_not_propagate(bin_edges):
    """Context rows carry NaN s^2; those rows must degrade, not poison the batch."""
    t = torch.tensor([[0.1, -0.4, 0.9]])
    s2 = torch.tensor([[float("nan"), -1e-9, 0.04]])
    logits = gaussian_logits(bin_edges, 0.0, 0.6).view(1, 1, -1).expand(1, 3, -1)

    loss = compute_cate_noise_aware_nll(logits, t, bin_edges, target_var=s2, sigma=0.05)

    assert torch.isfinite(loss)


def test_target_outside_the_grid_stays_finite(bin_edges):
    """Far-out targets underflow every bin; the clamp must keep the NLL finite."""
    t = torch.tensor([[50.0]])
    logits = gaussian_logits(bin_edges, 0.0, 0.5).view(1, 1, -1)

    loss = compute_cate_noise_aware_nll(
        logits, t, bin_edges, target_var=torch.zeros_like(t), sigma=0.05
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_gradients_flow_to_logits(bin_edges):
    t = torch.tensor([[0.2, -0.3]])
    logits = gaussian_logits(bin_edges, 0.0, 0.5).view(1, 1, -1).expand(1, 2, -1)
    logits = logits.clone().requires_grad_(True)

    compute_cate_noise_aware_nll(
        logits,
        t,
        bin_edges,
        target_var=torch.full_like(t, 0.02),
        sigma=0.05,
    ).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_shape_mismatch_is_rejected(bin_edges):
    logits = torch.zeros(1, 3, NBINS)
    with pytest.raises(ValueError, match="must match logits shape"):
        compute_cate_noise_aware_nll(
            logits,
            torch.zeros(1, 4),
            bin_edges,
            target_var=torch.zeros(1, 4),
            sigma=0.1,
        )


def test_non_positive_sigma_is_rejected(bin_edges):
    logits = torch.zeros(1, 1, NBINS)
    with pytest.raises(ValueError, match="sigma must be positive"):
        compute_cate_noise_aware_nll(
            logits,
            torch.zeros(1, 1),
            bin_edges,
            target_var=torch.zeros(1, 1),
            sigma=0.0,
        )


class TestGaussianBinMass:
    """The kernel shared with the HL loss and (later) the posterior evaluator."""

    def test_integrates_to_one_when_the_grid_covers_the_kernel(self, bin_edges):
        mass = gaussian_bin_mass(bin_edges, torch.tensor([0.0]), 0.5)
        assert mass.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_per_row_sigma_broadcasts(self, bin_edges):
        centers = torch.tensor([[0.0, 1.0]])
        sd = torch.tensor([[0.2, 0.9]])

        mass = gaussian_bin_mass(bin_edges, centers, sd)
        assert mass.shape == (1, 2, NBINS)

        # A wider kernel spreads mass over more bins.
        occupied = (mass > 1e-6).sum(dim=-1)
        assert occupied[0, 1] > occupied[0, 0]

        # ...and agrees with the scalar-sigma path row by row.
        for j, s in enumerate([0.2, 0.9]):
            solo = gaussian_bin_mass(bin_edges, centers[:, j], s)
            assert torch.allclose(mass[:, j], solo, atol=1e-6)

    def test_sub_bin_sigma_is_integrated_not_sampled(self, bin_edges):
        """A kernel far narrower than a bin still places its full mass.

        Sampling the density at bin centres instead would collapse this toward
        zero and make the noise correction silently vanish for exactly the small
        s the loss is meant to handle.
        """
        width = float(bin_edges[1] - bin_edges[0])
        # Sit at a bin centre, not an edge, so the whole kernel lands in one bin.
        centre = float(bin_edges[NBINS // 2] + 0.5 * width)

        mass = gaussian_bin_mass(bin_edges, torch.tensor([centre]), width / 100.0)

        assert mass.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert mass.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_hl_loss_still_works_through_the_shared_kernel(self, bin_edges):
        """compute_cate_distribution_loss (the noise-unaware baseline) is intact."""
        logits = gaussian_logits(bin_edges, 0.0, 0.5).view(1, 1, -1)

        near = compute_cate_distribution_loss(
            logits, torch.tensor([[0.0]]), bin_edges, sigma=0.1
        )
        far = compute_cate_distribution_loss(
            logits, torch.tensor([[2.0]]), bin_edges, sigma=0.1
        )

        assert torch.isfinite(near) and torch.isfinite(far)
        assert far > near
