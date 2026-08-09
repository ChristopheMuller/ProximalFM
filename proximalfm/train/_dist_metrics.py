"""Distributional (calibration/sharpness) metrics for the histogram CATE head.

Pure tensor math: torch in, torch out, batched over leading dimensions, no
``Trainer``, no config, no I/O, no model. Every function here has a closed form
and is checked against it in ``tests/test_dist_metrics.py``.

This module is the reference for the ``*/dist/*`` half of the metric namespace
(``prior_dist/crps``, ``prior_dist/cov90_pooled``, ...). Its point-estimate counterpart
is not a module: point metrics (``*/point/*``) are three lines of arithmetic each
and live fused into the streaming accumulator in
:class:`proximalfm.train._eval.CausalEvaluator`. There is nothing to factor out of
them, which is exactly why the distributional ones get their own file -- they are
the half with non-obvious closed forms.

---------------------------------------------------------------------------
The representation: a histogram is a piecewise-uniform *density*
---------------------------------------------------------------------------

Every function reads ``(probs, bin_edges)`` as the density

.. math::
    f(t) = \\frac{p_i}{w_i} \\quad\\text{for } t \\in [e_i, e_{i+1}),
    \\qquad w_i = e_{i+1} - e_i ,

with ``bin_edges`` :math:`= (e_0, \\dots, e_B)` strictly ascending and
``probs`` :math:`= (p_0, \\dots, p_{B-1})` summing to one. The CDF is therefore
*continuous and piecewise linear*:

.. math::
    F(t) = \\sum_i p_i \\, \\mathrm{clip}\\!\\left(\\frac{t - e_i}{w_i}, 0, 1\\right).

The alternative reading -- point masses at bin centres -- is not used anywhere
in this module, and the difference is not cosmetic. Three consequences, each of
which is a bug someone would otherwise introduce:

1. **CRPS is exact and PIT is continuous.** Because :math:`F` is continuous,
   :func:`crps` has a per-bin closed form (no quadrature) and :func:`pit` needs
   no randomisation to be uniform under a correct forecast. Under the
   point-mass reading the PIT is discrete and its KS statistic has a floor set
   by the bin count, so "miscalibrated" and "coarse grid" become
   indistinguishable.

2. **Predicted variance picks up the within-bin term** :math:`w^2/12`, which is
   *the grid's floor on predicted sd*: a head that puts all its mass in one bin
   reports :math:`\\mathrm{sd} = w/\\sqrt{12}`, not 0. Use the discrete moment
   :math:`\\sum_i p_i (m_i - \\mu)^2` instead and a sharp posterior looks
   sharper than the grid can express, corrupting ``sd_over_rmse`` in exactly the
   regime the fine-tune is heading into. (Note that
   ``Trainer._target_noise_metrics``' ``noise/mean_sd_pred`` *is* the discrete
   moment -- it is a training-side diagnostic of the loss, not a calibration
   metric, and the two series differ by that term. Do not compare them.)

3. **Targets off the grid are charged the distance, not clipped.** A target
   beyond :math:`e_B` adds :math:`y - e_B` to its CRPS and gets
   :math:`\\mathrm{PIT} = 1`. Clipping instead would make an unrepresentable
   target look merely extreme. The share of such rows is
   :func:`out_of_grid_fraction`, reported as ``out_of_grid_frac``; a non-trivial
   value means the grid (``--cate_vmin``/``--cate_vmax``) is too tight and every
   other number here is suspect.

---------------------------------------------------------------------------
What the metrics mean, and what to want from them
---------------------------------------------------------------------------

Let :math:`P` be the predicted distribution for one row and :math:`y` its
target. All quantities are in the *standardized* outcome scale (the caller's
job; see the frozen set's ``frozen_meta.pt`` for why the scale must be fixed
across episodes rather than per-episode).

``crps``
    :math:`\\mathrm{CRPS}(P, y) = \\int_{\\mathbb{R}} \\big(F(t) -
    \\mathbb{1}\\{t \\ge y\\}\\big)^2 \\, dt`. Proper: minimized only by the true
    predictive law. Lower is better, but it mixes calibration and sharpness, so
    it ranks models and never diagnoses one. Reduces to :math:`|{\\hat y} - y|`
    for a point forecast, so it is on the scale of an absolute error.

``crps_over_oracle``
    ``crps`` divided by :func:`oracle_discretized_crps`, the CRPS of a forecast
    that puts *all* mass in the bin containing the target. As this ratio
    approaches 1 the remaining CRPS is the **bin grid**, not the model, and the
    remedy is more ``--nbins``, not more steps. Without it a converged run and a
    grid-limited run look identical.

``pit`` / ``pit_ks``
    :math:`u = F(y)`. If :math:`P` is the law of :math:`y` then
    :math:`u \\sim \\mathrm{Uniform}(0,1)`; :func:`pit_ks` returns the
    Kolmogorov-Smirnov distance to that uniform *and* the critical band, so
    "inside the band" is decidable rather than eyeballed. Shape reads the
    failure: U-shaped (mass at 0 and 1) = too narrow, dome-shaped = too wide,
    tilted = biased.

``cov{L}`` / ``width{L}``
    Empirical coverage of the central :math:`L` interval
    :math:`[Q((1-L)/2), Q((1+L)/2)]`, and its mean width. **Always together**:
    coverage alone is trivially achieved by a wide interval, width alone by a
    confidently wrong one.

``sd_over_rmse``
    :math:`\\sqrt{\\mathbb{E}[\\mathrm{Var}\\, P] \\big/ (\\mathrm{MSE} - s^2)}`,
    the square root of the variance identity below. Target **1.0**; below 1 is
    over-confidence. Never report a width, or a width trend, without it -- a
    beautiful shrinkage curve at ``sd_over_rmse = 0.6`` is a model that is
    confidently wrong.

The variance identity (:func:`variance_identity`) is the reason that last one is
worth a metric of its own. If :math:`P` is calibrated for :math:`\\tau^\\star`
and the target is the Monte-Carlo estimate :math:`y = \\tau^\\star + e` with
:math:`\\mathrm{Var}(e) = s^2` independent of :math:`P`, then

.. math::
    \\mathbb{E}\\big[(\\mu_P - y)^2\\big]
      = \\underbrace{\\mathbb{E}\\big[(\\mu_P - \\tau^\\star)^2\\big]}_{\\text{model error}}
        + s^2,
    \\qquad
    \\mathbb{E}\\big[\\mathrm{Var}\\, P\\big] = \\mathbb{E}\\big[(\\mu_P - \\tau^\\star)^2\\big],

so :math:`\\mathbb{E}[\\mathrm{Var}\\, P] \\big/ (\\mathrm{MSE}_{\\text{obs}} -
s^2) = 1` exactly. Subtracting :math:`s^2` is one line and its sign matters:
omit it and the denominator is inflated, making the model read as *more*
over-confident than it is.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "predictive_cdf",
    "predictive_mean_var",
    "predictive_quantile",
    "crps",
    "oracle_discretized_crps",
    "pit",
    "pit_ks",
    "coverage",
    "variance_identity",
    "out_of_grid_fraction",
]

_EPS = 1e-12


def _f64(t: torch.Tensor) -> torch.Tensor:
    """Detach to CPU float64.

    Every reduction in this module accumulates in float64 to keep the pooled
    sums exact, but MPS has no float64 at all -- ``.double()`` on an MPS tensor
    raises. The reductions here are all per-episode-sized and end in a Python
    float anyway, so the transfer is free in practice; this mirrors what
    ``_eval.CausalEvaluator`` already does (``.cpu()`` on every per-episode
    value before ``.double()``).
    """
    return t.detach().cpu().double()


def _check(probs: torch.Tensor, bin_edges: torch.Tensor) -> None:
    """Shape contract shared by every function taking ``(probs, bin_edges)``.

    ``bin_edges`` is required to be 1-D: the model registers a single shared
    ``cate_bin_edges`` buffer, and allowing a per-row grid would buy nothing
    while making the gathers in :func:`predictive_quantile` considerably harder
    to read. Values must be strictly ascending; that is *not* checked, because
    the check is a device sync on a hot path.
    """
    if bin_edges.ndim != 1:
        raise ValueError(f"bin_edges must be 1-D, got shape {tuple(bin_edges.shape)}")
    if probs.shape[-1] + 1 != bin_edges.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[-1]} bins but bin_edges has "
            f"{bin_edges.shape[0]} edges (expected {probs.shape[-1] + 1})"
        )


def _prepare(
    probs: torch.Tensor, bin_edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(p, e_lo, e_hi, w)`` in fp32, with ``p`` renormalized.

    Renormalization is defensive: softmax output already sums to one, but a
    grid-reprojecting operation (e.g. the noise convolution of step 7) can lose
    mass off the ends, and silently scoring an unnormalized "distribution" is
    worse than absorbing the clamp here.
    """
    _check(probs, bin_edges)
    p = probs.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    edges = bin_edges.to(device=p.device, dtype=torch.float32)
    e_lo, e_hi = edges[:-1], edges[1:]
    return p, e_lo, e_hi, e_hi - e_lo


def predictive_cdf(
    probs: torch.Tensor, bin_edges: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """:math:`F(y)` for the piecewise-uniform density, clamped to ``[0, 1]``.

    .. math::
        F(y) = \\sum_i p_i \\,
               \\mathrm{clip}\\!\\left(\\frac{y - e_i}{w_i}, 0, 1\\right)

    Parameters
    ----------
    probs : Tensor
        Shape ``(..., B)``, non-negative.
    bin_edges : Tensor
        Shape ``(B + 1,)``, strictly ascending.
    y : Tensor
        Shape ``(...)``, broadcastable against ``probs.shape[:-1]``.

    Returns
    -------
    Tensor
        Shape ``(...)``. Exactly 0 below ``bin_edges[0]`` and exactly 1 above
        ``bin_edges[-1]``.
    """
    p, e_lo, _, w = _prepare(probs, bin_edges)
    frac = ((y.float().unsqueeze(-1) - e_lo) / w).clamp(0.0, 1.0)
    return (p * frac).sum(dim=-1)


def predictive_mean_var(
    probs: torch.Tensor, bin_edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of the piecewise-uniform density.

    .. math::
        \\mu = \\sum_i p_i m_i, \\qquad
        \\mathbb{E}[Y^2] = \\sum_i p_i \\left(m_i^2 + \\frac{w_i^2}{12}\\right),
        \\qquad m_i = \\tfrac{1}{2}(e_i + e_{i+1}).

    The :math:`w_i^2/12` term is the within-bin variance of a uniform, i.e. the
    grid's floor on predicted sd (module docstring, point 2). It is why a head
    with all mass in one bin reports ``sd = w / sqrt(12)`` rather than 0.

    Returns
    -------
    tuple of Tensor
        ``(mean, var)``, both shape ``probs.shape[:-1]``. ``var`` is clamped at
        0 against float cancellation.
    """
    p, e_lo, e_hi, w = _prepare(probs, bin_edges)
    centers = 0.5 * (e_lo + e_hi)
    mean = (p * centers).sum(dim=-1)
    second = (p * (centers**2 + w**2 / 12.0)).sum(dim=-1)
    return mean, (second - mean**2).clamp_min(0.0)


def predictive_quantile(
    probs: torch.Tensor, bin_edges: torch.Tensor, q: torch.Tensor | float
) -> torch.Tensor:
    """Inverse CDF :math:`F^{-1}(q)`, exact for the piecewise-linear :math:`F`.

    Within the bin :math:`i` where the cumulative mass first reaches ``q``,

    .. math::
        F^{-1}(q) = e_i + w_i \\, \\frac{q - C_i}{p_i},
        \\qquad C_i = \\sum_{j<i} p_j ,

    falling back to :math:`e_i` for an empty bin. No interpolation error and no
    randomisation: the quantiles used by :func:`coverage` are the exact
    quantiles of the object being scored.

    Parameters
    ----------
    q : Tensor or float
        A scalar, or a 1-D tensor of ``K`` levels in ``[0, 1]``.

    Returns
    -------
    Tensor
        Shape ``probs.shape[:-1]`` for a scalar ``q``, else
        ``probs.shape[:-1] + (K,)``.
    """
    p, e_lo, _, w = _prepare(probs, bin_edges)

    scalar = not torch.is_tensor(q) or q.ndim == 0
    levels = torch.as_tensor(q, device=p.device, dtype=torch.float32).reshape(-1)

    cdf = p.cumsum(dim=-1)
    below = cdf - p

    target = levels.expand(*p.shape[:-1], levels.numel()).contiguous()
    idx = torch.searchsorted(cdf.contiguous(), target).clamp_max(p.shape[-1] - 1)

    p_i = p.gather(-1, idx)
    c_i = below.gather(-1, idx)
    lo_i, w_i = e_lo[idx], w[idx]

    out = torch.where(
        p_i > _EPS, lo_i + w_i * (target - c_i) / p_i.clamp_min(_EPS), lo_i
    )
    return out.squeeze(-1) if scalar else out


def crps(probs: torch.Tensor, bin_edges: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Continuous ranked probability score, in exact closed form.

    .. math::
        \\mathrm{CRPS}(P, y) = \\int_{\\mathbb{R}}
            \\big(F(t) - \\mathbb{1}\\{t \\ge y\\}\\big)^2 dt .

    Because :math:`F` is linear on each bin, the integral splits into per-bin
    terms with :math:`u = (t - e_i)/w_i` and :math:`\\int_0^1 (a + b u)^2 du =
    a^2 + ab + b^2/3`:

    * ``y`` at or above :math:`e_{i+1}` (indicator 0): :math:`w_i\\,(C_i^2 + C_i
      p_i + p_i^2/3)`.
    * ``y`` at or below :math:`e_i` (indicator 1): the same with
      :math:`C_i \\to C_i - 1`.
    * ``y`` inside the bin, at :math:`v = (y - e_i)/w_i`: the two pieces
      split at :math:`v`, giving
      :math:`w_i [\\,a^2 v + a b v^2 + b^2 v^3/3 + a'^2(1-v) + a' b (1-v^2)
      + b^2 (1-v^3)/3\\,]` with :math:`a = C_i`, :math:`a' = C_i - 1`,
      :math:`b = p_i`.

    Outside the grid the integrand is 1 over the whole gap, so a target beyond
    the ends is charged :math:`(e_0 - y)_+ + (y - e_B)_+` on top -- the distance,
    not a clip (module docstring, point 3).

    Returns
    -------
    Tensor
        Shape ``probs.shape[:-1]``, non-negative, in the units of ``y``.
    """
    p, e_lo, e_hi, w = _prepare(probs, bin_edges)
    yb = y.float().unsqueeze(-1)

    cdf = p.cumsum(dim=-1)
    a = cdf - p
    a_shift = a - 1.0
    b = p

    full_right = w * (a**2 + a * b + b**2 / 3.0)
    full_left = w * (a_shift**2 + a_shift * b + b**2 / 3.0)

    v = ((yb - e_lo) / w).clamp(0.0, 1.0)
    v2, v3 = v * v, v * v * v
    split = w * (
        a**2 * v
        + a * b * v2
        + b**2 * v3 / 3.0
        + a_shift**2 * (1.0 - v)
        + a_shift * b * (1.0 - v2)
        + b**2 * (1.0 - v3) / 3.0
    )

    inside = (yb > e_lo) & (yb < e_hi)
    below = yb <= e_lo
    per_bin = torch.where(inside, split, torch.where(below, full_left, full_right))

    tails = (e_lo[0] - yb.squeeze(-1)).clamp_min(0.0) + (
        yb.squeeze(-1) - e_hi[-1]
    ).clamp_min(0.0)
    return per_bin.sum(dim=-1) + tails


def oracle_discretized_crps(y: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """CRPS of the sharpest grid-representable forecast: all mass in ``y``'s bin.

    This is the **bin-resolution ceiling**. A forecaster that knew :math:`y`
    exactly still cannot express a point mass on this grid; the best it can do
    is commit to the single bin containing :math:`y`, which by the closed form
    above costs

    .. math::
        w_i \\left(\\frac{v^3}{3} + \\frac{(1-v)^3}{3}\\right)
        \\in \\left[\\frac{w_i}{12}, \\frac{w_i}{3}\\right],
        \\qquad v = \\frac{y - e_i}{w_i},

    minimized when :math:`y` sits at the bin centre. Out-of-grid targets get the
    nearest edge bin, so the distance penalty carries through.

    Committing to one bin is the natural floor rather than a proven infimum over
    all grid distributions -- spreading mass over neighbouring bins only puts
    :math:`F > 0` where the step function is still 0. Read ``crps_over_oracle``
    as "how much of my CRPS is grid, not model", not as a normalized score.

    Returns
    -------
    Tensor
        Shape ``y.shape``, strictly positive.
    """
    edges = bin_edges.to(device=y.device, dtype=torch.float32)
    nbins = edges.numel() - 1
    idx = (torch.searchsorted(edges, y.float().contiguous()) - 1).clamp(0, nbins - 1)
    one_hot = torch.zeros(
        *y.shape, nbins, device=y.device, dtype=torch.float32
    ).scatter_(-1, idx.unsqueeze(-1), 1.0)
    return crps(one_hot, edges, y)


def pit(probs: torch.Tensor, bin_edges: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Probability integral transform :math:`u = F(y)`.

    Uniform on ``[0, 1]`` iff ``probs`` is the law of ``y``. Continuous by
    construction here -- no randomised PIT needed, because :math:`F` is
    continuous (module docstring, point 1). Out-of-grid targets land exactly on
    0 or 1 and are *kept*: they are genuine tail failures, and
    :func:`out_of_grid_fraction` says how many there are.

    Returns
    -------
    Tensor
        Shape ``probs.shape[:-1]``, in ``[0, 1]``.
    """
    return predictive_cdf(probs, bin_edges, y)


def pit_ks(u: torch.Tensor, alpha: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """KS distance of PIT values from ``Uniform(0, 1)``, with its critical band.

    For sorted :math:`u_{(1)} \\le \\dots \\le u_{(n)}`,

    .. math::
        D_n = \\max_i \\max\\!\\left(\\frac{i}{n} - u_{(i)},\\;
              u_{(i)} - \\frac{i-1}{n}\\right),

    compared against the asymptotic Kolmogorov critical value
    :math:`c_\\alpha = \\sqrt{-\\tfrac{1}{2}\\ln(\\alpha/2)} / \\sqrt{n}`
    (:math:`\\approx 1.358/\\sqrt{n}` at :math:`\\alpha = 0.05`). Returning the
    band alongside the statistic is the point: :math:`D_n` shrinks with
    :math:`n`, so a bare value cannot be compared across evaluation sets, while
    ``stat < band`` is a decision.

    The band is a *guide*, not an exact test, on two counts: it is asymptotic,
    and PIT values from the same episode are dependent, so the effective sample
    size is below :math:`n`. Treat a marginal excursion as marginal.

    Parameters
    ----------
    u : Tensor
        Shape ``(..., n)``; the KS is taken over the last dimension, so
        per-episode statistics come from a batched call. Needs
        :math:`n \\ge 2`.

    Returns
    -------
    tuple of Tensor
        ``(stat, band)``, both shape ``u.shape[:-1]``.
    """
    if u.shape[-1] < 2:
        raise ValueError("pit_ks needs at least 2 PIT values")

    n = u.shape[-1]
    values = _f64(u).sort(dim=-1).values
    upper = torch.arange(1, n + 1, device=values.device, dtype=torch.float64) / n
    lower = upper - 1.0 / n

    stat = torch.maximum(upper - values, values - lower).amax(dim=-1)
    band = torch.full_like(stat, math.sqrt(-0.5 * math.log(alpha / 2.0) / n))
    return stat, band


def coverage(
    probs: torch.Tensor,
    bin_edges: torch.Tensor,
    y: torch.Tensor,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[str, float]:
    """Empirical coverage **and** mean width of central credible intervals.

    For each level :math:`L`, the interval is
    :math:`[F^{-1}((1-L)/2),\\, F^{-1}((1+L)/2)]` and the two reported numbers
    are :math:`\\Pr(y \\in \\text{interval})` and
    :math:`\\mathbb{E}[\\text{hi} - \\text{lo}]`.

    They are returned together, and should be *read* together, because either
    alone is gameable: coverage is trivially achieved by widening, width by
    being confidently wrong. Coverage below the nominal level with a small width
    is over-confidence; coverage above it with a large width is a wasted
    posterior. Neither identifies *where* the failure sits -- that is what the
    PIT shape is for.

    Returns
    -------
    dict
        ``{"cov50": ..., "width50": ..., "cov90": ..., "width90": ...}``, floats
        reduced over *all* elements. Accumulate as a weighted sum on element
        count.
    """
    out: dict[str, float] = {}
    for level in levels:
        q = torch.tensor(
            [(1.0 - level) / 2.0, (1.0 + level) / 2.0],
            device=probs.device,
            dtype=torch.float32,
        )
        bounds = predictive_quantile(probs, bin_edges, q)
        lo, hi = bounds[..., 0], bounds[..., 1]
        inside = (y.float() >= lo) & (y.float() <= hi)

        tag = f"{round(level * 100)}"
        out[f"cov{tag}"] = _f64(inside).mean().item()
        out[f"width{tag}"] = _f64(hi - lo).mean().item()
    return out


def variance_identity(
    probs: torch.Tensor,
    bin_edges: torch.Tensor,
    y: torch.Tensor,
    s2: torch.Tensor | None = None,
) -> dict[str, float]:
    """The calibration identity :math:`\\mathbb{E}[\\mathrm{Var}\\,P] /
    (\\mathrm{MSE}_{\\text{obs}} - s^2)`, and its parts.

    With a target :math:`y = \\tau^\\star + e`, :math:`\\mathrm{Var}(e) = s^2`
    independent of the prediction, a calibrated :math:`P` satisfies

    .. math::
        \\mathbb{E}\\big[\\mathrm{Var}\\, P\\big]
          = \\mathbb{E}\\big[(\\mu_P - \\tau^\\star)^2\\big]
          = \\mathbb{E}\\big[(\\mu_P - y)^2\\big] - s^2 ,

    so ``var_ratio`` is 1 for a calibrated posterior, ``< 1`` when it is
    over-confident and ``> 1`` when it is too wide. ``sd_over_rmse`` is its
    square root, which is the form to report: it is in sd units, so 0.8 reads
    directly as "intervals 20 % too narrow".

    The :math:`s^2` subtraction is unconditional here, and its sign matters:
    dropping it inflates the denominator and makes the model read as more
    over-confident than it is. ``s2`` **must be in the same (standardized) units
    as** ``y`` -- the raw-vs-head scale mixup is the single most likely bug on
    this path, which is why the frozen set precomputes ``s2_std`` rather than
    leaving the conversion to callers.

    This is a global identity, not a per-row one: :math:`\\mathrm{Var}\\,P` is
    not an estimate of that row's squared error. Do not read a per-row version
    of this ratio.

    Parameters
    ----------
    s2 : Tensor, optional
        Per-row target sampling variance, broadcastable against ``y``.
        ``None`` means an exact target (:math:`s^2 = 0`), in which case the
        ratio is the plain :math:`\\mathbb{E}[\\mathrm{Var}]/\\mathrm{MSE}`.
        Non-finite entries are treated as 0.

    Returns
    -------
    dict
        ``mean_var``, ``mean_sd``, ``mse_obs``, ``mean_s2``, ``mse_corrected``,
        ``var_ratio``, ``sd_over_rmse``, ``s_over_sd``. All floats reduced over
        every element.

    Notes
    -----
    ``s_over_sd`` = :math:`\\sqrt{\\mathbb{E}[s^2]} /
    \\sqrt{\\mathbb{E}[\\mathrm{Var}\\,P]}` is the ratio that decides whether the
    target noise is worth correcting for at all: the width inflation it induces
    is :math:`\\sqrt{1 + (s/\\sigma)^2} - 1`, i.e. under 2 % below 0.2. It is
    reported unconditionally because it *drifts up* as the posterior sharpens.
    """
    mean, var = predictive_mean_var(probs, bin_edges)

    sq_err = (mean - y.float()) ** 2
    if s2 is None:
        s2_t = torch.zeros_like(sq_err)
    else:
        s2_t = torch.nan_to_num(s2.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(
            0.0
        )
        s2_t = s2_t.expand_as(sq_err)

    mean_var = _f64(var).mean().item()
    mean_sd = _f64(var.clamp_min(0.0).sqrt()).mean().item()
    mse_obs = _f64(sq_err).mean().item()
    mean_s2 = _f64(s2_t).mean().item()

    mse_corrected = max(mse_obs - mean_s2, 0.0)
    var_ratio = mean_var / mse_corrected if mse_corrected > _EPS else float("nan")

    return {
        "mean_var": mean_var,
        "mean_sd": mean_sd,
        "mse_obs": mse_obs,
        "mean_s2": mean_s2,
        "mse_corrected": mse_corrected,
        "var_ratio": var_ratio,
        "sd_over_rmse": math.sqrt(var_ratio)
        if var_ratio == var_ratio
        else float("nan"),
        "s_over_sd": (
            math.sqrt(mean_s2 / mean_var) if mean_var > _EPS else float("nan")
        ),
    }


def out_of_grid_fraction(bin_edges: torch.Tensor, y: torch.Tensor) -> float:
    """Share of targets outside ``[bin_edges[0], bin_edges[-1]]``.

    The health check for every other metric in this module. Those targets are
    scored honestly (charged the distance, PIT pinned to 0 or 1), but no
    forecast on this grid can cover them, so a non-trivial fraction means the
    grid is too tight -- widen ``--cate_vmin``/``--cate_vmax`` rather than
    reading the calibration numbers.
    """
    edges = bin_edges.to(device=y.device, dtype=torch.float32)
    outside = (y.float() < edges[0]) | (y.float() > edges[-1])
    return _f64(outside).mean().item()
