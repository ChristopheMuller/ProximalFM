"""ROADMAP A6 -- the ``n``-sweep: does the posterior actually contract?

    uv run python -m posterior_eval.sweep \\
        --dir data/posterior_eval/frozen_prod_v1 --ckpt path/to/model.ckpt

Scores the same frozen set at nested context prefixes. The set needs no
regeneration: prefix ``n`` is the first ``n`` context rows of an episode that
already exists, and the sidecar's frozen comparison scale (``y_context_std`` at
``n_max``) is what makes widths at different ``n`` commensurable at all --
per-episode standardisation moves with ``n``, so scoring in each prefix's own
scale would manufacture exactly the trend this module is measuring.

Two questions, and they are different
=====================================

**1. Does the width fall at the right rate?** Fitted as a power law,

.. math:: \\log \\mathrm{sd}(n) = \\alpha + \\beta \\log n .

The logs stay. Contraction is a power law (``sd ~ n^{-1/2}`` for a parametric
posterior), so log-log is the coordinate system in which the prediction is a
straight line and ``beta`` is the whole claim; on the raw scale there is no
linear structure to fit and the estimate would be dominated by the small-``n``
end. It also makes the fit **scale-free**: the frozen standardisation is an
arbitrary per-episode constant, and a constant moves ``alpha`` only. That is
what lets betas from episodes with wildly different ``tau`` scales be compared,
which their widths cannot be (metrics.md §5).

The fit is therefore reported **per episode** (median and IQR of the betas) as
the headline, with the pooled fit beside it. Pooled ``mean_sd`` is a row-weighted
mean over episodes whose ``tau`` scales differ by orders of magnitude, so its
slope is whatever the largest-scale episodes do. Note the geometric-mean curve
``mean_sd_geo`` has, by construction, exactly the slope that is the mean of the
per-episode betas -- the two headline numbers agree because they are the same
computation, and a gap between either and the row-weighted pooled fit is a
scale-mix artifact, not a finding.

``beta`` is also fitted on the RMSE. **The pair is the point.** A width that
contracts faster than the error is a posterior growing over-confident with ``n``;
equivalently, ``sd_over_rmse`` trending across the sweep. The roadmap's done-when
is exactly this: ``mean_sd`` falls monotonically while ``sd_over_rmse`` stays ~1.

**Two width curves are reported, and they are not the same claim.** The model
standardises ``y`` by the context rows it was handed, so its grid arrives in
raw-``y`` units and ``episode_predictive`` divides by the sidecar's frozen
``y_context_std``; the leftover factor is ``grid_scale = y_std_predict /
y_std_frozen``, exactly 1 at ``n_ctx == n_max`` and *systematically below it* at
short prefixes, because a sample sd from few rows of a heavy-tailed ``y``
under-estimates. That shrinks the frozen-unit width at small ``n`` on its own,
partly cancelling -- on a small set, sometimes reversing -- the contraction being
measured.

- ``mean_sd`` / ``mean_sd_geo`` (**frozen units**) is the headline. It is the
  width the model actually claims about ``tau`` on the fixed comparison scale,
  it is the one commensurable with the RMSE, and its ratio to the RMSE *is*
  ``sd_over_rmse``. Calibration statements live here.
- ``mean_sd_head`` (**the model's own units**, i.e. divided by that episode's
  ``grid_scale``) isolates how much the head narrows given more context, free of
  its own standardisation. Learning statements live here.

``beta(sd) - beta(sd_head)`` is therefore not noise; it is the contribution of
the shrinking context-sd estimate, and it is why ``grid_scale``'s median and p95
are printed per ``n`` rather than kept internal.

Two ends of the curve are not power law and must not be fitted through:

- **Upper end, the floor.** Predicted sd cannot go below
  ``sqrt(sigma^2 + w^2/12)`` -- the schedule's smoothing and the bin grid, adding
  in quadrature (C2). Points within ``FLOOR_MARGIN`` of ``sd_floor`` are flagged
  and excluded from the default fit window, because a curve flattening onto the
  floor attenuates ``beta`` toward 0 and reads as "contraction stopped".
  **Pass ``--hl_sigma``**: a checkpoint does not carry the schedule, so without
  it ``sd_floor`` is the grid term alone -- typically ~1e-3, two orders of
  magnitude under the real floor -- and a schedule-pinned flat width curve is
  indistinguishable from a converged one. That is the single likeliest
  misreading of this whole module.
- **Lower end, the prior.** As ``n -> 0`` the width does not diverge, it
  *plateaus* at the prior sd of ``tau``. Which is question 2.

**2. Does the small-``n`` limit land on the prior?** At ``n_ctx`` small the model
has almost no episode-specific information, so its predictive should approach the
CATE prior implied by the frozen set. Two statistics, and **only the second one
is about ``n``**:

``prior_ks`` / ``prior_w1``
    Distance from the pooled model mixture (the average predictive CDF over
    every scored row) to the empirical CDF of ``tau_star_std``. **This is a
    marginal-calibration check, not a prior-limit check.** Averaging a
    calibrated posterior over the data-generating distribution returns the prior
    marginal *at every ``n``* -- that is an identity, not a property of short
    prefixes -- so these should be small everywhere, and a curve that sits on
    the empirical CDF at ``n = n_max`` is passing, not failing. It is close kin
    to the pooled PIT: necessary, cheap, and blind to exactly the thing question
    2 asks. Read it as "is anything grossly miscalibrated in the marginal".

``within_frac`` (and ``between_sd``)
    The discriminator. Decompose the mixture variance,
    ``Var(mix) = E[Var per row] + Var(row means)``: the first term is what the
    model still does not know, the second is what it has learned to tell apart.
    ``within_frac`` is the first over the total and runs **1 at the prior limit
    to 0 as context accumulates**. ``within_frac ~ 1`` with ``mixture_sd ~
    sd(tau*)`` at the shortest prefix is the prior limit reached; ``within_frac``
    that does not move across the sweep is a model whose width ignores its
    context, whatever the KS column says.

Two caveats that make small ``n`` its own regime, both reported rather than
hidden:

- The same self-standardisation as above: at small ``n`` the limit being
  approached is the prior over ``tau / hat s``, not over ``tau / sd(y)``. The
  distortion is real and belongs to the model, so it is reported, not corrected
  away -- but it is why a small ``prior_ks`` at short prefixes is evidence and
  not proof.
- Below ``MIN_N`` rows the context sd is not estimable at all (``predict``
  clamps at 1e-5 and the grid explodes), so the sweep refuses to go there. The
  "limit" is ``n = MIN_N``, not ``n = 0``.

Offline only, single process, no wandb producer -- same deliberate key-budget
choice as ``score_strata`` (a sweep is ``len(n_list)`` forward passes over the
whole set, which is not a training-cadence cost).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from posterior_eval.build_prior_frozen_set import (
    frozen_content_digest,
    load_frozen_meta,
)
from posterior_eval.report import load_model
from proximalfm.train._dist_eval import (
    PosteriorEvaluator,
    UnsupportedLossTypeError,
    episode_predictive,
    iter_frozen_episodes,
)
from proximalfm.train._dist_metrics import (
    _f64,
    predictive_cdf,
    predictive_mean_var,
)

#: Smallest prefix worth scoring. Below two rows the context sd is not a
#: quantity; at two it is a very noisy one, which the ``grid_scale`` columns
#: expose rather than hide.
MIN_N = 2

#: Prefixes added below the sidecar's ``n_grid`` (which starts at 32), because
#: the prior limit is the whole of question 2 and 32 rows is not "no data".
SMALL_N = (2, 4, 8, 16)

#: Reference grid for the pooled-mixture CDF comparison. 129 points over the
#: pooled ``tau`` range: this feeds a KS/W1 distance and a plot, not a score.
GRID_POINTS = 129

#: Grid points evaluated at once. The CDF is ``(rows, points, bins)`` before
#: reduction, so this caps a transient that would otherwise be ~100 MB at
#: ``n_q=128, nbins=1667``.
GRID_CHUNK = 16

#: A width within this factor of the sd floor is floor-bound, not converged, and
#: is dropped from the default fit window.
FLOOR_MARGIN = 3.0


def default_n_list(meta: Dict[str, Any]) -> List[int]:
    """The sidecar's ``n_grid``, extended down and always closed at ``n_max``."""
    n_max = int(meta["global"]["n_max"])
    grid = set(int(n) for n in meta["global"].get("n_grid", ()))
    grid |= set(SMALL_N)
    grid.add(n_max)
    return sorted(n for n in grid if MIN_N <= n <= n_max)


def parse_n_list(text: str, meta: Dict[str, Any]) -> List[int]:
    n_max = int(meta["global"]["n_max"])
    out = sorted({int(part) for part in text.replace(" ", "").split(",") if part})
    bad = [n for n in out if not MIN_N <= n <= n_max]
    if bad:
        raise ValueError(f"n_ctx values outside [{MIN_N}, {n_max}]: {bad}")
    return out


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------


class _NAccumulator:
    """Everything accumulated at one prefix length.

    Deliberately not :func:`~proximalfm.train._dist_eval.score_episode`: the
    mixture CDF of question 2 needs the raw ``probs``, which that function
    reduces away. The frozen-space scale logic -- the part where a bug would be
    invisible -- still comes from ``episode_predictive`` and is not duplicated.
    """

    def __init__(self, n: int, grid: torch.Tensor):
        self.n = n
        self.grid = grid
        self.dist = PosteriorEvaluator()
        self.rows = 0
        self.mse_sum = 0.0
        self.s2_sum = 0.0
        self.scale: List[float] = []
        self.mean_sd: List[float] = []  # per episode, frozen units
        self.mean_sd_head: List[float] = []  # per episode, the model's own units
        self.rmse: List[float] = []  # per episode
        self.cdf_sum = torch.zeros(grid.numel(), dtype=torch.float64)
        # Mixture moments, exact (no grid truncation): Var of a mixture is the
        # mean of the component variances plus the variance of their means.
        self.mu_sum = 0.0
        self.mu2_sum = 0.0
        self.var_sum = 0.0

    def update(self, probs: torch.Tensor, edges: torch.Tensor, tau, s2, scale: float):
        self.dist.update(probs=probs, bin_edges=edges, y=tau, s2=s2)

        mean, var = predictive_mean_var(probs, edges)
        sq_err = _f64((mean - tau) ** 2)
        n_rows = int(tau.numel())

        self.rows += n_rows
        self.mse_sum += float(sq_err.sum())
        self.s2_sum += float(_f64(s2).sum())
        self.scale.append(scale)
        sd = float(_f64(var.clamp_min(0)).sqrt().mean())
        self.mean_sd.append(sd)
        # Exact, not approximate: the whole grid is ``head_edges * scale``, so
        # dividing undoes the de-standardization row for row.
        self.mean_sd_head.append(sd / scale if scale > 0 else float("nan"))
        self.rmse.append(float(sq_err.mean().sqrt()))

        mean64, var64 = _f64(mean), _f64(var)
        self.mu_sum += float(mean64.sum())
        self.mu2_sum += float((mean64**2).sum())
        self.var_sum += float(var64.sum())

        for lo in range(0, self.grid.numel(), GRID_CHUNK):
            pts = self.grid[lo : lo + GRID_CHUNK].to(probs.device)
            cdf = predictive_cdf(
                probs.unsqueeze(1), edges, pts.expand(n_rows, pts.numel())
            )
            self.cdf_sum[lo : lo + pts.numel()] += _f64(cdf).sum(dim=0)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    t = torch.tensor(list(values), dtype=torch.float64)
    return float(t.quantile(torch.tensor(q, dtype=torch.float64)))


def _geomean(values: Sequence[float]) -> float:
    positive = [v for v in values if v > 0 and v == v]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def sweep_n(
    model,
    out: Path,
    *,
    n_list: Optional[Sequence[int]] = None,
    device: str = "cpu",
    max_episodes: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    hl_sigma: Optional[float] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Score a frozen set at every prefix in ``n_list``; one load, ``|n_list|`` forwards.

    Episode-outer, ``n``-inner: the shards are read once, which is the only part
    that can be shared -- the forward pass genuinely differs per prefix.

    ``hl_sigma`` is the training schedule's smoothing width **in head units**
    (``get_hl_sigma()`` / ``loss/hl_sigma`` at the checkpoint's step). Pass it.
    A checkpoint does not carry it, and without it ``sd_floor`` is only the grid
    term ``w/sqrt(12)`` -- typically two orders of magnitude below the real
    floor, which makes a schedule-pinned flat curve look like a model that
    stopped learning (C2). This is the single most consequential optional
    argument here.
    """
    out = Path(out)
    meta = load_frozen_meta(out) if meta is None else meta
    n_max = int(meta["global"]["n_max"])
    records = meta["episodes"]
    ns = list(default_n_list(meta) if n_list is None else n_list)
    if not ns:
        raise ValueError("n_list is empty")

    total = len(records) if max_episodes is None else min(len(records), max_episodes)

    # The empirical CATE prior of this set, and the grid it is compared on.
    tau_pool = torch.cat(
        [r["tau_star_std"].reshape(-1).double() for r in records[:total]]
    )
    span = float(tau_pool.quantile(torch.tensor(0.999, dtype=torch.float64)))
    span = max(
        span, -float(tau_pool.quantile(torch.tensor(0.001, dtype=torch.float64)))
    )
    span = span * 1.2 if span > 0 else 1.0
    grid = torch.linspace(-span, span, GRID_POINTS, dtype=torch.float64)
    target_cdf = (tau_pool.reshape(-1, 1) <= grid.reshape(1, -1)).double().mean(dim=0)

    accs = {n: _NAccumulator(n, grid) for n in ns}

    episodes = tqdm(
        enumerate(iter_frozen_episodes(out / "val", max_episodes)),
        total=total,
        desc="episodes",
        unit="ep",
        disable=not progress,
    )
    with torch.inference_mode():
        for i, batch in episodes:
            record = records[i]
            tau = record["tau_star_std"].reshape(-1).float()
            s2 = record["s2_std"].reshape(-1).float()
            for n in ns:
                probs, edges = episode_predictive(
                    model, batch, record, n_ctx=n, n_max=n_max, device=device
                )
                head = model.cate_bin_edges.to(edges.device).float()
                scale = float(
                    (edges[-1] - edges[0]) / (head[-1] - head[0]).clamp_min(1e-12)
                )
                accs[n].update(
                    probs,
                    edges,
                    tau.to(probs.device),
                    s2.to(probs.device),
                    scale,
                )

    head_edges = model.cate_bin_edges.float()
    head_width = float((head_edges[1:] - head_edges[:-1]).mean())

    by_n: Dict[int, Dict[str, Any]] = {}
    for n, acc in accs.items():
        if not acc.rows:
            continue
        metrics = acc.dist.compute(prefix="", include_count=True)
        mse = acc.mse_sum / acc.rows
        mean_s2 = acc.s2_sum / acc.rows
        model_cdf = acc.cdf_sum / acc.rows
        mu = acc.mu_sum / acc.rows
        within = acc.var_sum / acc.rows
        between = max(acc.mu2_sum / acc.rows - mu**2, 0.0)
        mix_var = within + between
        scale_p50 = _quantile(acc.scale, 0.5)

        metrics.update(
            {
                "n_ctx": float(n),
                "mean_sd_geo": _geomean(acc.mean_sd),
                "mean_sd_head_geo": _geomean(acc.mean_sd_head),
                # Target-noise corrected, unlike ``prior_point/cate/rmse_pooled``:
                # this one is the denominator of ``sd_over_rmse``, so the two
                # curves being fitted have to be the same two quantities whose
                # ratio the calibration check reads.
                "rmse_pooled": math.sqrt(max(mse - mean_s2, 0.0)),
                "rmse_geo": _geomean(acc.rmse),
                "grid_scale/p50": scale_p50,
                "grid_scale/p95": _quantile(acc.scale, 0.95),
                # The grid term always; the schedule's sigma only if the caller
                # supplied it (a checkpoint does not carry it). They add in
                # quadrature, so without --hl_sigma this is a lower bound, and
                # usually a very loose one.
                "sd_floor": math.hypot(head_width / math.sqrt(12.0), hl_sigma or 0.0)
                * scale_p50,
                "sd_floor_is_grid_only": float(hl_sigma is None),
                # The n-discriminator. Var(mixture) = E[Var per row] + Var(means
                # per row); the first term is what the model still does not
                # know, the second is what it has learned to tell apart. The
                # ratio runs 1 -> 0 as context accumulates, and *that* is the
                # prior limit -- unlike prior_ks, which a calibrated posterior
                # passes at every n (see the module docstring).
                "within_frac": within / mix_var if mix_var > 0 else float("nan"),
                "between_sd": math.sqrt(between),
                "prior_ks": float((model_cdf - target_cdf).abs().max()),
                "prior_w1": float(
                    torch.trapezoid((model_cdf - target_cdf).abs(), grid)
                ),
                "mixture_sd": math.sqrt(max(mix_var, 0.0)),
                "mixture_mean": mu,
                "_cdf": model_cdf.tolist(),
            }
        )
        by_n[n] = metrics

    ns_present = sorted(by_n)
    result: Dict[str, Any] = {
        "n_list": ns_present,
        "n_max": n_max,
        "episodes": total,
        "by_n": by_n,
        "per_episode": {
            "mean_sd": {n: accs[n].mean_sd for n in ns_present},
            "mean_sd_head": {n: accs[n].mean_sd_head for n in ns_present},
            "rmse": {n: accs[n].rmse for n in ns_present},
        },
        "prior": {
            "grid": grid.tolist(),
            "cdf": target_cdf.tolist(),
            "tau_sd": float(tau_pool.std(unbiased=False)),
            "tau_mean": float(tau_pool.mean()),
            "tau_rows": int(tau_pool.numel()),
        },
    }
    result["fit"] = contraction_summary(result)
    return result


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def fit_power_law(ns: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    """OLS of ``log(values)`` on ``log(ns)``.

    Returns ``beta`` (the contraction exponent), ``alpha``, ``r2``, ``se`` (the
    standard error of ``beta``, ``nan`` with fewer than 3 points) and
    ``n_points``. Pairs with a non-positive or non-finite value are dropped --
    a zero width is a degenerate episode, not a data point at ``-inf``.
    """
    pairs = [
        (math.log(n), math.log(v))
        for n, v in zip(ns, values)
        if n > 0 and v == v and v > 0
    ]
    if len(pairs) < 2:
        return {
            "beta": float("nan"),
            "alpha": float("nan"),
            "r2": float("nan"),
            "se": float("nan"),
            "n_points": float(len(pairs)),
        }
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    k = len(pairs)
    mx, my = sum(xs) / k, sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx if sxx > 0 else float("nan")
    alpha = my - beta * mx
    resid = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    ss_res = sum(r**2 for r in resid)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    se = (
        math.sqrt(ss_res / (k - 2) / sxx)
        if k > 2 and sxx > 0 and ss_res >= 0
        else float("nan")
    )
    return {
        "beta": beta,
        "alpha": alpha,
        "r2": r2,
        "se": se,
        "n_points": float(k),
    }


def fit_window(result: Dict[str, Any], floor_margin: float = FLOOR_MARGIN) -> List[int]:
    """The prefixes the power law is fitted over.

    Drops the floor-bound upper end (``mean_sd`` within ``floor_margin`` of the
    reported ``sd_floor``): a curve flattening onto the floor attenuates ``beta``
    toward 0, and reading that as "contraction stopped" is the single most
    likely misreading of this table (C2). The small-``n`` plateau is *not*
    dropped automatically -- where it starts is the answer to question 2, not an
    input to it -- so if ``prior_ks`` is already at its floor over several
    prefixes, refit with an explicit ``--fit_min``.
    """
    keep = []
    for n in result["n_list"]:
        row = result["by_n"][n]
        floor = row.get("sd_floor", 0.0)
        sd = row.get("mean_sd_geo", float("nan"))
        if sd == sd and floor > 0 and sd < floor_margin * floor:
            continue
        keep.append(n)
    return keep


def contraction_summary(
    result: Dict[str, Any],
    *,
    window: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Per-episode and pooled power-law fits, plus the calibration cross-check."""
    ns = list(fit_window(result) if window is None else window)
    by_n = result["by_n"]
    per_ep = result["per_episode"]

    fits: Dict[str, Any] = {
        "window": ns,
        "dropped": [n for n in result["n_list"] if n not in ns],
    }
    if len(ns) < 2:
        return fits

    fits["sd_geo"] = fit_power_law(ns, [by_n[n]["mean_sd_geo"] for n in ns])
    fits["sd_pooled"] = fit_power_law(ns, [by_n[n]["mean_sd"] for n in ns])
    fits["sd_head_geo"] = fit_power_law(ns, [by_n[n]["mean_sd_head_geo"] for n in ns])
    fits["rmse_geo"] = fit_power_law(ns, [by_n[n]["rmse_geo"] for n in ns])
    fits["rmse_pooled"] = fit_power_law(ns, [by_n[n]["rmse_pooled"] for n in ns])

    for tag, series in (
        ("sd", "mean_sd"),
        ("sd_head", "mean_sd_head"),
        ("rmse", "rmse"),
    ):
        betas = [
            fit_power_law(ns, [per_ep[series][n][e] for n in ns])["beta"]
            for e in range(len(per_ep[series][ns[0]]))
        ]
        finite = [b for b in betas if b == b]
        fits[f"per_episode_{tag}"] = {
            "p25": _quantile(finite, 0.25),
            "p50": _quantile(finite, 0.5),
            "p75": _quantile(finite, 0.75),
            "n": float(len(finite)),
        }
    fits["monotone_sd"] = all(
        by_n[a]["mean_sd_geo"] >= by_n[b]["mean_sd_geo"]
        for a, b in zip(result["n_list"], result["n_list"][1:])
    )
    return fits


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(value: Optional[float], width: int = 9, prec: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    if value != value:
        return "nan".rjust(width)
    return f"{value:>{width}.{prec}f}"


def print_report(result: Dict[str, Any]) -> None:
    by_n = result["by_n"]
    print("\n" + "=" * 92)
    print("n-sweep: posterior contraction (ROADMAP A6)")
    print("=" * 92)
    print(
        f"\n{result['episodes']} episodes, n_max={result['n_max']}, "
        f"prior sd(tau*) = {result['prior']['tau_sd']:.4f} "
        f"over {result['prior']['tau_rows']} rows"
    )

    cols = [
        ("mean_sd_geo", "  sd_geo"),
        ("mean_sd_head_geo", "  sd_head"),
        ("rmse_geo", " rmse_geo"),
        ("sd_over_rmse", "  sd/rmse"),
        ("crps", "     crps"),
        ("cov90_pooled", "    cov90"),
        ("pit_ks_pooled", "    pitKS"),
        ("sd_floor", "    floor"),
    ]
    print("\n--- contraction ---")
    header = f"  {'n_ctx':>6} " + " ".join(h.rjust(9) for _, h in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in result["n_list"]:
        row = by_n[n]
        cells = " ".join(_fmt(row.get(k)) for k, _ in cols)
        flag = "  <- floor-bound" if n in result["fit"].get("dropped", []) else ""
        print(f"  {n:>6} {cells}{flag}")

    print("\n--- prior limit ---")
    print(
        "    within_frac is the n-discriminator (1 = prior, 0 = fully informed);"
        "\n    prior_KS/W1 is marginal calibration and should be small at EVERY n."
    )
    header = (
        f"  {'n_ctx':>6} {'within_frac':>11} {'between_sd':>10} {'mix_sd':>9} "
        f"{'prior_KS':>9} {'prior_W1':>9} {'scale_p50':>9} {'scale_p95':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in result["n_list"]:
        row = by_n[n]
        print(
            f"  {n:>6} {_fmt(row.get('within_frac'), 11)} "
            f"{_fmt(row.get('between_sd'), 10)} {_fmt(row.get('mixture_sd'))} "
            f"{_fmt(row.get('prior_ks'))} {_fmt(row.get('prior_w1'))} "
            f"{_fmt(row.get('grid_scale/p50'))} {_fmt(row.get('grid_scale/p95'))}"
        )
    print(
        f"  {'target':>6} {1.0:>11.4f} {0.0:>10.4f} "
        f"{result['prior']['tau_sd']:>9.4f}    (at the prior limit)"
    )

    _print_verdict(result)


def _print_verdict(result: Dict[str, Any]) -> None:
    fit = result["fit"]
    print("\n--- Reading ---")
    if fit.get("dropped"):
        print(
            f"  fit window {fit['window']}; dropped {fit['dropped']} as within "
            f"{FLOOR_MARGIN}x the sd floor (the schedule/grid floor flattens the "
            "curve; that is C2, not convergence)."
        )
    per_sd = fit.get("per_episode_sd")
    if per_sd:
        print(
            f"  beta(sd)   per-episode median {per_sd['p50']:+.3f} "
            f"[p25 {per_sd['p25']:+.3f}, p75 {per_sd['p75']:+.3f}]   "
            f"pooled-geo {fit['sd_geo']['beta']:+.3f} "
            f"(r2 {fit['sd_geo']['r2']:.3f})"
        )
    per_head = fit.get("per_episode_sd_head")
    if per_head:
        print(
            f"  beta(sd_head) per-episode median {per_head['p50']:+.3f} "
            f"[p25 {per_head['p25']:+.3f}, p75 {per_head['p75']:+.3f}]   "
            f"pooled-geo {fit['sd_head_geo']['beta']:+.3f}"
        )
    if per_sd and per_head:
        drift = per_sd["p50"] - per_head["p50"]
        if abs(drift) > 0.05:
            print(
                f"  beta(sd) - beta(sd_head) = {drift:+.3f}: that much of the "
                "frozen-unit curve is the\n    model's own context-sd estimate "
                "moving with n, not the head narrowing. Both are real; the\n"
                "    first is what the posterior claims, the second is what it "
                "learned."
            )
    per_rmse = fit.get("per_episode_rmse")
    if per_rmse:
        print(
            f"  beta(rmse) per-episode median {per_rmse['p50']:+.3f} "
            f"[p25 {per_rmse['p25']:+.3f}, p75 {per_rmse['p75']:+.3f}]   "
            f"pooled-geo {fit['rmse_geo']['beta']:+.3f} "
            f"(r2 {fit['rmse_geo']['r2']:.3f})"
        )
    if per_sd and per_rmse:
        gap = per_sd["p50"] - per_rmse["p50"]
        if abs(gap) < 0.05:
            print(
                f"  beta(sd) - beta(rmse) = {gap:+.3f}: width and error contract "
                "at the same rate, so calibration holds across the sweep."
            )
        elif gap < 0:
            print(
                f"  ! beta(sd) - beta(rmse) = {gap:+.3f} < 0: the width contracts "
                "FASTER than the error.\n    The posterior is growing "
                "over-confident as context grows -- check sd_over_rmse per n."
            )
        else:
            print(
                f"  ! beta(sd) - beta(rmse) = {gap:+.3f} > 0: the error contracts "
                "faster than the width;\n    the posterior is not cashing in the "
                "information it is using."
            )

    if not fit.get("monotone_sd", True):
        print("  ! mean_sd is not monotone in n_ctx -- the roadmap's done-when fails.")

    ns = result["n_list"]
    if len(ns) >= 2:
        lo, hi = result["by_n"][ns[0]], result["by_n"][ns[-1]]
        tau_sd = result["prior"]["tau_sd"]
        ratio = lo["mixture_sd"] / tau_sd if tau_sd > 0 else float("nan")
        print(
            f"\n  prior limit at n={ns[0]}: within_frac {lo['within_frac']:.3f} "
            f"(1.0 is the prior), mixture sd {lo['mixture_sd']:.4f} vs prior sd "
            f"{tau_sd:.4f} (ratio {ratio:.3f})"
        )
        moved = lo["within_frac"] - hi["within_frac"]
        if moved < 0.05:
            print(
                f"  ! within_frac moves only {moved:+.3f} from n={ns[0]} to "
                f"n={ns[-1]}: the split between what the\n    model does not know "
                "and what it has resolved barely responds to context. A flat "
                "width\n    curve plus a flat within_frac is a marginal, not a "
                "posterior."
            )
        else:
            print(
                f"  within_frac falls {lo['within_frac']:.3f} -> "
                f"{hi['within_frac']:.3f} across the sweep: the model is "
                "converting context into resolved variation."
            )
        print(
            f"  prior_KS ranges {min(result['by_n'][n]['prior_ks'] for n in ns):.4f}"
            f"-{max(result['by_n'][n]['prior_ks'] for n in ns):.4f} across n. "
            "Small everywhere is correct\n    and expected: it is marginal "
            "calibration, not evidence about the prior limit."
        )

    if any(result["by_n"][n].get("sd_floor_is_grid_only") for n in ns):
        print(
            "\n  ! sd_floor is the GRID term only -- --hl_sigma was not passed, so the "
            "schedule's\n    smoothing floor is missing and is usually the binding "
            "one by two orders of\n    magnitude. A flat sd curve cannot be "
            "distinguished from a schedule-pinned one\n    until you re-run with "
            "--hl_sigma set to loss/hl_sigma at this checkpoint's step."
        )
    print()


def plot_sweep(result: Dict[str, Any], path: Path) -> None:
    """Two panels: the contraction curve, and the prior-limit CDF overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = result["n_list"]
    by_n = result["by_n"]
    fit = result["fit"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    sd = [by_n[n]["mean_sd_geo"] for n in ns]
    rmse = [by_n[n]["rmse_geo"] for n in ns]
    ax0.plot(ns, sd, "o-", label="posterior sd (frozen units)")
    ax0.plot(
        ns,
        [by_n[n]["mean_sd_head_geo"] for n in ns],
        "^-",
        alpha=0.6,
        label="posterior sd (head units)",
    )
    ax0.plot(ns, rmse, "s-", label="RMSE (geo. mean)")
    floor_label = (
        "grid floor only (pass --hl_sigma)"
        if by_n[ns[0]].get("sd_floor_is_grid_only")
        else "sd floor (sigma + grid)"
    )
    ax0.plot(
        ns, [by_n[n]["sd_floor"] for n in ns], ":", color="grey", label=floor_label
    )
    ax0.axhline(
        result["prior"]["tau_sd"], ls="--", color="k", lw=0.8, label="prior sd(tau*)"
    )
    if "sd_geo" in fit and fit["sd_geo"]["beta"] == fit["sd_geo"]["beta"]:
        w = fit["window"]
        line = [
            math.exp(fit["sd_geo"]["alpha"]) * n ** fit["sd_geo"]["beta"] for n in w
        ]
        ax0.plot(
            w,
            line,
            "-",
            color="C0",
            alpha=0.4,
            lw=3,
            label=f"fit beta={fit['sd_geo']['beta']:+.3f}",
        )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("n_ctx")
    ax0.set_ylabel("frozen units")
    ax0.set_title("posterior contraction")
    ax0.legend(fontsize=7)

    grid = result["prior"]["grid"]
    ax1.plot(grid, result["prior"]["cdf"], "k-", lw=2, label="empirical tau* (prior)")
    for n in ns:
        cdf = by_n[n].get("_cdf")
        if cdf:
            ax1.plot(grid, cdf, lw=1, alpha=0.8, label=f"model mixture, n={n}")
    ax1.set_xlabel("tau (frozen units)")
    ax1.set_ylabel("CDF")
    ax1.set_title("prior limit")
    ax1.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_safe(result: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(result, default=float))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m posterior_eval.sweep",
        description="Score a checkpoint at nested context prefixes of a frozen set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", type=Path, required=True, help="frozen set ROOT")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument(
        "--n_list",
        type=str,
        default=None,
        help="comma-separated prefixes; defaults to the sidecar's n_grid "
        f"extended down to {MIN_N}",
    )
    p.add_argument(
        "--fit_min",
        type=int,
        default=None,
        help="drop prefixes below this from the power-law fit (use when the "
        "small-n plateau is visible in the table)",
    )
    p.add_argument(
        "--hl_sigma",
        type=float,
        default=None,
        help="the schedule's smoothing sigma in HEAD units at this checkpoint's "
        "step (wandb loss/hl_sigma). Without it sd_floor is the grid term only "
        "and a schedule-pinned flat width curve is indistinguishable from a "
        "converged one",
    )
    p.add_argument("--json_out", type=Path, default=None)
    p.add_argument("--plot", type=Path, default=None, help="write a two-panel PNG here")
    p.add_argument("--no_progress", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    meta = load_frozen_meta(args.dir)
    ns = None if args.n_list is None else parse_n_list(args.n_list, meta)
    model, step = load_model(args.ckpt, device=args.device)

    try:
        result = sweep_n(
            model,
            args.dir,
            n_list=ns,
            device=args.device,
            max_episodes=args.max_episodes,
            meta=meta,
            hl_sigma=args.hl_sigma,
            progress=not args.no_progress,
        )
    except UnsupportedLossTypeError as exc:
        print(f"\n{exc}\n")
        return 2

    if args.fit_min is not None:
        window = [n for n in fit_window(result) if n >= args.fit_min]
        result["fit"] = contraction_summary(result, window=window)

    print(f"\ncheckpoint: {args.ckpt}" + (f" (step {step})" if step else ""))
    print(f"frozen set: {args.dir}  [content_digest={frozen_content_digest(meta)}]")
    print_report(result)

    if args.plot:
        plot_sweep(result, args.plot)
        print(f"Plot written to {args.plot}")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True)
        )
        print(f"Written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
