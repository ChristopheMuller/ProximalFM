"""Loss functions for the training heads in ``proximalfm.train._run``.

Pure tensor math, no ``Trainer``/config dependency, so each function is
directly unit-testable (see ``tests/test_noise_aware_loss.py``).
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F

_LOG_FLOOR = 1e-30


def compute_crps_loss(
    logits: torch.Tensor,
    y_target: torch.Tensor,
    bin_edges: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ranked Probability Score (discrete CRPS).

    Hyperparameter-free, and unlike cross-entropy it respects the distance
    between bins. Computed as the MSE between the predicted CDF and the ideal
    step-function CDF of the target.
    """
    if probs is None:
        probs = F.softmax(logits, dim=-1)
    pred_cdf = torch.cumsum(probs, dim=-1)
    target_cdf = (bin_edges[1:].view(1, 1, -1) >= y_target.unsqueeze(-1)).float()

    return torch.mean((pred_cdf - target_cdf) ** 2, dim=-1).mean()


def gaussian_bin_mass(
    bin_edges: torch.Tensor,
    center: torch.Tensor,
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Mass that ``N(center, sigma^2)`` puts in each histogram bin.

    Returns an *unnormalized* ``(..., nbins)`` tensor: it sums to one only when
    ``[bin_edges[0], bin_edges[-1]]`` covers the Gaussian. Callers decide what to
    do with the missing tail -- ``compute_hl_gaussian_cross_entropy_loss``
    renormalizes over the available support, ``compute_cate_noise_aware_nll``
    does not (see its docstring).

    The mass is *integrated* over each bin rather than sampling the density at
    the bin centre. That matters whenever ``sigma`` drops below one bin width,
    where a sampled kernel collapses toward a point mass and any correction
    built on it silently vanishes.

    Parameters
    ----------
    bin_edges : Tensor
        Shape ``(nbins + 1,)``, ascending.

    center : Tensor
        Shape ``(...)``. The Gaussian mean, on the same scale as ``bin_edges``.

    sigma : float or Tensor
        Standard deviation. A tensor must broadcast against ``center``, which is
        what lets a *per-row* noise scale be used; the scalar case covers the
        fixed schedule.

    Returns
    -------
    Tensor
        Shape ``center.shape + (nbins,)``, non-negative.
    """
    center_fp32 = center.float()
    bin_edges_fp32 = bin_edges.to(device=center.device, dtype=torch.float32)

    edge_shape = (1,) * center_fp32.ndim + (-1,)
    edges = bin_edges_fp32.view(edge_shape)
    mu = center_fp32.unsqueeze(-1)

    if isinstance(sigma, torch.Tensor):
        if (sigma <= 0).any():
            raise ValueError("sigma must be positive everywhere")
        scale = sigma.float().unsqueeze(-1)
    else:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        scale = float(sigma)

    normalizer = math.sqrt(2.0) * scale

    cdf_all = 0.5 * (1.0 + torch.erf((edges - mu) / normalizer))

    return (cdf_all[..., 1:] - cdf_all[..., :-1]).clamp_min(0.0)


def compute_hl_gaussian_cross_entropy_loss(
    logits: torch.Tensor,
    y_target: torch.Tensor,
    bin_edges: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
    log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Histogram Loss with Gaussian soft targets.

    For each scalar target y_i, constructs a discrete target distribution p_i
    by integrating N(y_i, sigma^2) over the histogram bins, then computes

        HL = - sum_l p_l log q_l

    where q_l = softmax(logits)_l.

    Args:
        logits:
            Tensor of shape (..., nbins).
        y_target:
            Tensor of shape (...).
        bin_edges:
            Tensor of shape (nbins + 1,).
        sigma:
            Standard deviation of the Gaussian target on the standardized target scale.
        eps:
            Numerical stability constant.
        log_probs:
            Optional precomputed log_softmax probabilities.

    Returns:
        Scalar tensor.
    """
    if y_target.shape != logits.shape[:-1]:
        raise ValueError(
            f"y_target shape {y_target.shape} must match logits shape "
            f"{logits.shape[:-1]} except for the last dimension."
        )

    logits_fp32 = logits.float()

    p = gaussian_bin_mass(bin_edges, y_target, sigma)

    p = p / (p.sum(dim=-1, keepdim=True) + eps)

    if log_probs is None:
        log_probs = F.log_softmax(logits_fp32, dim=-1)

    loss = -torch.sum(p * log_probs, dim=-1)

    return loss.mean()


def compute_cate_distribution_loss(
    logits: torch.Tensor,
    cate_target: torch.Tensor,
    bin_edges: torch.Tensor,
    sigma: float,
    log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distributional term for the single-histogram CATE head ("cate_single").

    Kept as the noise-*unaware* baseline. See
    :func:`compute_cate_noise_aware_nll` for the version that accounts for the
    Monte Carlo error in the target.
    """
    return compute_hl_gaussian_cross_entropy_loss(
        logits, cate_target, bin_edges, sigma=sigma, log_probs=log_probs
    )


def compute_cate_noise_aware_nll(
    logits: torch.Tensor,
    cate_target: torch.Tensor,
    bin_edges: torch.Tensor,
    target_var: torch.Tensor,
    sigma: float,
    log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noise-aware NLL for the CATE head: ``-log[(P * N(0, s^2 + sigma^2))(t)]``.

    The supervised target ``t = MU1 - MU0`` is a Monte Carlo mean over ``K``
    replicates, so ``t = tau* + e`` with ``e ~ N(0, s^2)`` and ``s^2`` the
    per-row ``target_var``. Fitting a histogram directly to ``t`` therefore
    converges to ``posterior(tau) * N(0, s^2)`` -- over-dispersed by
    construction. Convolving the *prediction* with the noise kernel before
    evaluating the likelihood moves the minimizer back onto ``posterior(tau)``.

    Treating ``P`` as the histogram density (``p_i / w`` on bin ``i``),

        (P * N)(t) = sum_i p_i * [Phi((t - e_i)/s) - Phi((t - e_{i+1})/s)] / w

    which is a normalized density in ``t`` over the whole real line, so **no
    edge/leakage policy is required here** -- unlike a convolution that
    re-projects ``P`` back onto the same bin grid, which does push mass past the
    grid ends and needs an explicit clamp. An evaluator sharing
    :func:`gaussian_bin_mass` must not carry this assumption across.

    ``sigma`` is *not* set to ``s``. Doing so would smooth the target by the very
    noise being removed and double the over-dispersion. It stays what it always
    was -- a smoothing regularizer and a hard floor on the learnable width,
    annealed toward one bin width -- and enters as ``s^2 + sigma^2`` inside a
    single kernel. This also gives the degenerate path for free: where ``s^2`` is
    zero or NaN (context rows, ``K = 1``, data predating target-noise tracking)
    the kernel width falls back to ``sigma`` and the loss reduces to the plain
    smoothed NLL. Nothing propagates NaN.

    Parameters
    ----------
    logits : Tensor
        Shape ``(..., nbins)``, unnormalized log-weights over ``bin_edges``.

    cate_target : Tensor
        Shape ``(...)``. The observed noisy target, on the head's grid scale.

    bin_edges : Tensor
        Shape ``(nbins + 1,)``, uniform.

    target_var : Tensor
        Shape broadcastable to ``cate_target``. Per-row ``s^2`` **in the head's
        units**, i.e. already divided by the squared context scale used to
        standardize the target.

    sigma : float
        Smoothing scale from the training schedule.

    log_probs : Tensor, optional
        Precomputed log_softmax probabilities.

    Returns
    -------
    Tensor
        Scalar mean NLL.
    """
    if cate_target.shape != logits.shape[:-1]:
        raise ValueError(
            f"cate_target shape {cate_target.shape} must match logits shape "
            f"{logits.shape[:-1]} except for the last dimension."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    s2 = torch.nan_to_num(target_var.float(), nan=0.0, posinf=0.0).clamp_min(0.0)
    s2 = s2.expand_as(cate_target)
    kernel_sd = torch.sqrt(s2 + float(sigma) ** 2)

    mass = gaussian_bin_mass(bin_edges, cate_target, kernel_sd)

    bin_width = (bin_edges[1] - bin_edges[0]).to(mass.device, torch.float32)

    log_kernel = torch.log(mass.clamp_min(_LOG_FLOOR)) - torch.log(bin_width)
    if log_probs is None:
        log_probs = F.log_softmax(logits.float(), dim=-1)

    nll = -torch.logsumexp(log_probs + log_kernel, dim=-1)

    return nll.mean()


def compute_gaussian_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood for a Gaussian distribution.
    Assumes isotropic gaussian (used here for ATE modeling).
    """
    nll = 0.5 * torch.exp(-log_var) * (target - mean) ** 2 + 0.5 * log_var
    return nll.mean()
