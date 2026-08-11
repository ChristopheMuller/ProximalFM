"""Streaming distributional evaluation of the CATE posterior.

Point **and** distributional metrics on a set carrying a low-noise ``tau_star``
and a ``frozen_meta.pt`` sidecar. Two such sets exist, and ``score_frozen_set``
namespaces them by caller-supplied prefix: the frozen high-``K`` prior set built
by ``prior_eval/`` (``prior_point/*`` + ``prior_dist/*``) and the realistic
replicate set built by ``synt_eval/build_proximal_datasets.py --mc_cate K``
(``synt_point/*`` + ``synt_dist/*``, both from the same pass -- point metrics are a
side effect of the per-episode forward pass that produces the distributional
ones, not a second, separately-gated one).

What such a set has and a plain ``mc_cate=1`` set does not is the Monte-Carlo
CEPO target. A ``mc_cate=1`` set carries only sampled potential outcomes, so
``y1 - y0`` there is an ITE draw, and scoring against it -- point or
distributional -- is either minimized by an over-wide posterior or inflated by
the outcome noise. The sidecar's *absence* is the gate: without it there is no
frozen scale and no ``tau_star``, and this module fails loudly rather than
producing a plausible wrong number. See ``proximalfm/train/metrics.md`` §7.

Three things here are decisions, not implementation details:

*The comparison scale is frozen, not per-episode.* ``ProximalFM.predict``
returns raw-``y`` units obtained by de-standardizing with the *current* context
statistics, and those move with ``n_ctx`` and across episodes. Everything below
is scored in the sidecar's frozen space (``y_context_std`` at ``n_max``, the
trainer's formula), against its precomputed ``tau_star_std`` / ``s2_std``.
Rescaling the *grid* rather than the target is what keeps the histogram and the
target in one space without ever converting ``s2`` by hand -- the raw-vs-head
mixup POSTERIOR_EVAL_NOTES calls the likeliest bug on this path is not
expressible here.

*Forward-only, single pass -- never ``predict_ensemble``.* Averaging feature
permutations yields a mixture: strictly wider and differently calibrated. It is
a legitimate object to evaluate, but a different one, and mixing the two makes
every width number meaningless.

*Per-episode KS beside the pooled one.* Pooling hides cancellation:
over-confident on half the episodes and under-confident on the other half is
uniform overall. ``pit_ks/p90`` is the number that catches it.

The class mirrors :class:`proximalfm.train._eval.CausalEvaluator`'s
``reset``/``update``/``compute`` shape, so the trainer wiring is mechanical.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from proximalfm.train._dist_metrics import (
    _f64,
    coverage,
    crps,
    oracle_discretized_crps,
    out_of_grid_fraction,
    pit,
    pit_ks,
    predictive_mean_var,
    variance_identity,
)
from proximalfm.train._eval import CausalEvaluator

__all__ = [
    "PosteriorEvaluator",
    "UnsupportedLossTypeError",
    "episode_predictive",
    "require_single_cate_head",
    "score_episode",
    "score_frozen_set",
]

# ``ProximalFM.predict``'s std clamp, duplicated here because it is a literal in
# ``predict`` with no name to import. It only bites on a degenerate context
# (sd < 1e-5); the frozen scale uses the trainer's 1e-3 floor instead, and the
# ratio of the two is exactly what :func:`episode_predictive` folds into the
# grid, so a divergence changes the grid rather than passing unnoticed.
PREDICT_Y_STD_FLOOR = 1e-5

_EPS = 1e-12


class UnsupportedLossTypeError(ValueError):
    """The checkpoint has no single distribution over ``tau`` to score."""


class PosteriorEvaluator:
    """Streaming accumulator for the ``*/dist/*`` metrics.

    One :meth:`update` per episode. Everything is kept as a weighted sum over
    rows except the PIT values, which are retained because the KS statistic
    needs the sample, and the per-episode KS statistics and coverages, which are
    the point of reporting a distribution of calibration rather than one pooled
    number.

    Coverage is reported both ways and they answer different questions.
    ``cov90_pooled`` is the row-weighted fraction over every row in the pass;
    ``cov90/p25``, ``/p50``, ``/p75`` are quantiles of the *per-episode*
    coverages. Pooling cancels: over-covering on half the episodes and
    under-covering on the other half reads as a perfect ``cov90_pooled = 0.90``
    while ``cov90/p25`` and ``cov90/p75`` straddle it. This is the same failure
    ``pit_ks/p90`` exists to catch on the PIT side.

    The per-episode spread has a **binomial floor**: even a perfectly calibrated
    model scores ``Binomial(n, L)/n`` per episode, so the p25--p75 gap is
    ``~1.35 * sqrt(L(1-L)/n)`` before any miscalibration -- 0.06 for ``cov50``
    and 0.036 for ``cov90`` at ``n = 128``. Read the IQR against that, never
    against zero.

    Parameters
    ----------
    levels : tuple of float
        Central credible levels for coverage/width. Both numbers are reported
        for each level, always together.
    alpha : float
        Significance level for the KS band.
    max_pit : int
        Cap on retained PIT values, applied by uniform subsampling *at compute
        time* rather than during accumulation, so the pooled sample stays an
        unbiased draw from all episodes rather than a prefix of them. A frozen
        set is bounded (``n_datasets * n_q`` rows), so this is a memory guard,
        not a routine path.
    seed : int
        Seeds the subsample, so the pooled KS is reproducible.
    """

    def __init__(
        self,
        levels: Tuple[float, ...] = (0.5, 0.9),
        alpha: float = 0.05,
        max_pit: int = 200_000,
        seed: int = 0,
    ):
        self.levels = tuple(levels)
        self.alpha = alpha
        self.max_pit = max_pit
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._sums: Dict[str, float] = {}
        self._rows = 0
        self._episodes = 0
        self._pit: List[torch.Tensor] = []
        self._episode_ks: List[float] = []
        self._episode_ks_band: List[float] = []
        self._episode_cov: Dict[str, List[float]] = {}

    @property
    def empty(self) -> bool:
        return self._episodes == 0

    def _add(self, key: str, value: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value)

    @torch.no_grad()
    def update(
        self,
        *,
        probs: torch.Tensor,
        bin_edges: torch.Tensor,
        y: torch.Tensor,
        s2: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one episode.

        Parameters
        ----------
        probs : Tensor
            Shape ``(n, B)``: the predictive histogram of ``tau`` for each of
            the episode's ``n`` query rows, on ``bin_edges``.
        bin_edges : Tensor
            Shape ``(B + 1,)``, **in the same units as** ``y``. One episode at a
            time is what allows this to stay 1-D while each episode carries its
            own de-standardization (see :func:`episode_predictive`).
        y : Tensor
            Shape ``(n,)``: the target, ``tau_star`` in the frozen space.
        s2 : Tensor, optional
            Shape ``(n,)``: per-row sampling variance of ``y``, in the same
            units. ``None`` means an exact target.
        """
        probs = probs.detach()
        y = y.detach().float().reshape(-1)
        if probs.shape[0] != y.shape[0]:
            raise ValueError(f"probs has {probs.shape[0]} rows but y has {y.shape[0]}")
        s2_t = None if s2 is None else s2.detach().float().reshape(-1)

        n = int(y.numel())
        if n == 0:
            return

        self._add("crps", _f64(crps(probs, bin_edges, y)).sum().item())
        self._add(
            "crps_oracle",
            _f64(oracle_discretized_crps(y, bin_edges)).sum().item(),
        )

        cov = coverage(probs, bin_edges, y, self.levels)
        for key, value in cov.items():
            self._add(key, value * n)

        # Retained as a sample, not a sum: quantiles over episodes are not
        # recoverable from a row-weighted mean, which is the whole reason the
        # per-episode keys exist alongside the pooled one. Gated on n >= 2 for
        # the same reason as the per-episode KS below -- a coverage over one row
        # is 0 or 1 and would drag the quantiles toward the extremes.
        if n >= 2:
            for level in self.levels:
                tag = f"{round(level * 100)}"
                self._episode_cov.setdefault(f"cov{tag}", []).append(cov[f"cov{tag}"])

        # variance_identity returns means over the episode; scale them back to
        # sums so the pooled ratios below are over rows, not over episodes.
        vi = variance_identity(probs, bin_edges, y, s2_t)
        for key in ("mean_var", "mean_sd", "mse_obs", "mean_s2"):
            self._add(key, vi[key] * n)

        self._add("out_of_grid", out_of_grid_fraction(bin_edges, y) * n)

        u = pit(probs, bin_edges, y).flatten().float().cpu()
        self._pit.append(u)
        if u.numel() >= 2:
            stat, band = pit_ks(u, alpha=self.alpha)
            self._episode_ks.append(float(stat))
            self._episode_ks_band.append(float(band))

        self._rows += n
        self._episodes += 1

    def _sum_keys(self) -> List[str]:
        """The accumulator's keys, in a fixed order.

        Laid out from ``self.levels`` rather than from ``self._sums``, so an
        evaluator that accumulated nothing still reports the same key set.
        """
        keys = [
            "crps",
            "crps_oracle",
            "mean_var",
            "mean_sd",
            "mse_obs",
            "mean_s2",
            "out_of_grid",
        ]
        for level in self.levels:
            tag = f"{round(level * 100)}"
            keys += [f"cov{tag}", f"width{tag}"]
        return keys

    def _pooled_pit(self, cap: Optional[int] = None) -> torch.Tensor:
        cap = self.max_pit if cap is None else cap
        if not self._pit:
            return torch.empty(0)
        pool = torch.cat(self._pit)
        if pool.numel() <= cap:
            return pool
        gen = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(pool.numel(), generator=gen)[:cap]
        return pool[idx]

    def compute(
        self,
        *,
        prefix: str = "prior_dist/",
        include_count: bool = True,
    ) -> Dict[str, float]:
        """Summarize; returns ``{}`` when nothing was accumulated."""
        if self._episodes == 0 or self._rows == 0:
            return {}

        keys = self._sum_keys()
        rows = float(self._rows)
        episodes = float(self._episodes)
        sums = [float(self._sums.get(k, 0.0)) for k in keys]

        pooled = self._pooled_pit()
        episode_ks = list(self._episode_ks)
        episode_ks_band = list(self._episode_ks_band)
        episode_cov = {k: list(v) for k, v in self._episode_cov.items()}

        if rows <= 0 or episodes <= 0:
            return {}

        mean = {k: v / rows for k, v in zip(keys, sums)}
        metrics: Dict[str, float] = {}

        metrics["crps"] = mean["crps"]
        oracle = mean["crps_oracle"]
        metrics["crps_oracle"] = oracle
        metrics["crps_over_oracle"] = (
            mean["crps"] / oracle if oracle > _EPS else float("nan")
        )

        for level in self.levels:
            tag = f"{round(level * 100)}"
            metrics[f"cov{tag}_pooled"] = mean[f"cov{tag}"]
            metrics[f"width{tag}"] = mean[f"width{tag}"]

            per_episode = episode_cov.get(f"cov{tag}")
            if per_episode:
                sample = torch.tensor(per_episode, dtype=torch.float64)
                for q, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
                    metrics[f"cov{tag}/{name}"] = float(
                        sample.quantile(torch.tensor(q, dtype=torch.float64))
                    )

        # The variance identity, pooled over rows. Same algebra as
        # ``_dist_metrics.variance_identity``, applied to the accumulated means:
        # the s^2 subtraction is unconditional, and dropping it would inflate the
        # denominator and make the model read as more over-confident than it is.
        mean_var = mean["mean_var"]
        mse_obs = mean["mse_obs"]
        mean_s2 = mean["mean_s2"]
        mse_corrected = max(mse_obs - mean_s2, 0.0)
        var_ratio = mean_var / mse_corrected if mse_corrected > _EPS else float("nan")
        metrics["mean_sd"] = mean["mean_sd"]
        metrics["mean_var"] = mean_var
        metrics["mse_obs"] = mse_obs
        metrics["mean_s2"] = mean_s2
        metrics["var_ratio"] = var_ratio
        metrics["sd_over_rmse"] = (
            math.sqrt(var_ratio) if var_ratio == var_ratio else float("nan")
        )
        metrics["s_over_sd"] = (
            math.sqrt(mean_s2 / mean_var) if mean_var > _EPS else float("nan")
        )

        metrics["out_of_grid_frac"] = mean["out_of_grid"]

        if pooled.numel() >= 2:
            stat, band = pit_ks(pooled, alpha=self.alpha)
            metrics["pit_ks_pooled"] = float(stat)
            metrics["pit_ks_pooled_band"] = float(band)

        if episode_ks:
            eps_ks = torch.tensor(episode_ks, dtype=torch.float64)
            metrics["pit_ks/p50"] = float(eps_ks.median())
            metrics["pit_ks/p90"] = float(
                eps_ks.quantile(torch.tensor(0.9, dtype=torch.float64))
            )
            band = torch.tensor(episode_ks_band, dtype=torch.float64)
            metrics["pit_ks/band"] = float(band.median())

        if include_count:
            metrics["eval_episodes"] = episodes
            metrics["eval_rows"] = rows

        return {f"{prefix}{k}": v for k, v in metrics.items()}


# --------------------------------------------------------------------------
# Driving a model over the frozen set
# --------------------------------------------------------------------------


def require_single_cate_head(model) -> None:
    """Raise :class:`UnsupportedLossTypeError` unless ``model`` predicts one
    histogram over ``tau`` on a single shared grid.

    Called both eagerly (``Trainer.__init__``, so a ``bins_po``/``hl`` config
    with ``--prior_eval_dir`` set fails at construction rather than hundreds of
    steps into training) and per-episode inside :func:`episode_predictive`.
    """
    from proximalfm._model.proximalfm import CATE_SINGLE_LOSS_TYPES

    loss_type = getattr(model, "loss_type", None)
    if loss_type in CATE_SINGLE_LOSS_TYPES:
        return
    if loss_type in ("bins_po", "hl"):
        raise UnsupportedLossTypeError(
            f"loss_type={loss_type!r} predicts two per-arm histograms, not a "
            "distribution over tau. The posterior of a difference is not "
            "identified by its two marginals -- any dependence between the arms "
            "is unconstrained -- so there is nothing here to score. Point "
            "metrics (prior_point/*) remain available; distributional ones need "
            "a cate_single* checkpoint."
        )
    raise UnsupportedLossTypeError(
        f"loss_type={loss_type!r} is not supported: the distributional metrics "
        "need one histogram over tau on a single shared grid, and "
        "'cate_decomposition' anchors its grid on a per-row predicted ATE."
    )


@torch.no_grad()
def episode_predictive(
    model,
    batch: Sequence[torch.Tensor],
    record: Dict[str, Any],
    *,
    n_ctx: int,
    n_max: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predictive histogram for one frozen episode, in the frozen space.

    Uses the first ``n_ctx`` context rows plus every query row (the layout of
    the frozen set: rows ``[0, n_max)`` are context, ``[n_max, T)`` are the
    replicated queries), exactly as ``prior_eval/report.py`` does.

    ``predict`` de-standardizes with the context statistics of the rows it was
    handed, so its grid is ``cate_bin_edges * y_std_predict`` in raw-``y`` units.
    Dividing by the sidecar's frozen ``y_context_std`` puts that grid in the same
    space as ``tau_star_std``/``s2_std``. Note this rescales the *grid*, never
    the target: ``s2`` is never converted here, which is why the raw-vs-head
    scale mixup cannot happen on this path.

    Returns
    -------
    tuple of Tensor
        ``(probs, bin_edges)`` with shapes ``(n_q, B)`` and ``(B + 1,)``.
    """
    require_single_cate_head(model)

    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        Y_obs,
        Y0,
        Y1,
        MU0,
        MU1,
        d_U,
        d_X,
        d_W,
        d_Z,
        *_rest,
    ) = batch

    def rows(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device)
        return torch.cat([t[:, :n_ctx], t[:, n_max:]], dim=1)

    def trim(t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Drop the dense feature padding beyond this episode's true width.

        ``LoadPriorDataset`` densifies a batch file to the widest ``d`` it
        contains -- the *file's* max, not the episode's -- so a frozen prior
        episode arrives padded to ``(20, 10, 10)`` whatever its own dims are.
        The model masks those columns out, but only after embedding them: they
        cost a full ``ColEmbedding`` + ``RowInteraction`` token each. Scoring is
        one episode per call, so unlike the training path (where a batch must be
        padded to its own max) the padding is pure waste, and dropping it is
        exact -- the columns removed are the ones the mask discards.

        Measured on ``frozen_prod_v1``: 370 -> 294 ms/episode on MPS, with
        predicted probabilities agreeing to 3e-8.
        """
        if d.numel() != 1:
            return t
        width = int(d.reshape(-1)[0])
        return t if t.shape[-1] <= width else t[..., :width]

    y_train = Y_obs.to(device)[:, :n_ctx].reshape(1, n_ctx)

    out = model.predict(
        X=rows(trim(X, d_X)),
        Z=rows(trim(Z, d_Z)),
        W=rows(trim(W, d_W)),
        A=rows(A).reshape(1, -1),
        y_train=y_train,
        target="cate",
        output_type="distribution",
        d_X=d_X.to(device),
        d_Z=d_Z.to(device),
        d_W=d_W.to(device),
    )
    probs = out["weights"].reshape(-1, out["weights"].shape[-1])

    y_std_predict = (
        y_train.reshape(-1).float().std(unbiased=False).clamp(min=PREDICT_Y_STD_FLOOR)
    )
    y_std_frozen = record["y_context_std"].to(probs.device).float()
    edges = model.cate_bin_edges.to(probs.device).float() * (
        y_std_predict / y_std_frozen
    )
    return probs, edges


@torch.no_grad()
def score_episode(
    model,
    batch: Sequence[torch.Tensor],
    record: Dict[str, Any],
    *,
    n_ctx: int,
    n_max: int,
    device: str = "cpu",
    dist_evaluator: Optional[PosteriorEvaluator] = None,
    point_evaluator: Optional[CausalEvaluator] = None,
    dist_evaluators: Sequence[Tuple[PosteriorEvaluator, Optional[torch.Tensor]]] = (),
    point_evaluators: Sequence[Tuple[CausalEvaluator, Optional[torch.Tensor]]] = (),
) -> Dict[str, float]:
    """Score one episode into the given evaluators.

    Returns the per-episode pooled squared error and row count, which the
    driver accumulates into ``cate/rmse_pooled``. That key exists for one
    reason: ``CausalEvaluator.cate/rmse`` is a *median over episodes*, so it is
    not the quantity ``prior_eval/report.py`` prints, and the cheapest place
    to catch a scale bug is a number that must match it exactly.

    Also returns ``grid_scale``, the factor :func:`episode_predictive` applied
    to the head's own grid to land in the frozen space. It is read off the
    scored grid rather than recomputed from ``y_train``, so it is by
    construction the factor actually used. Callers that hold a quantity in head
    units -- the trainer's ``sigma`` schedule is the one that matters -- need it
    to put that quantity on the same axis as ``prior_dist/mean_sd``.

    ``dist_evaluators``/``point_evaluators`` fan the *same* forward pass out to
    additional accumulators, each with an optional boolean row mask: the
    stratified evaluation in ``prior_eval.score_strata`` scores every cut without predicting
    more than once, because :class:`PosteriorEvaluator` accumulates row-weighted
    sums and is therefore exactly additive over disjoint row subsets. A ``None``
    mask means every row. Both default to empty, so the training path is
    unchanged. Masks selecting no rows are skipped -- an evaluator must not see
    an empty update, or its per-episode KS list falls out of step with its
    episode count.
    """
    probs, edges = episode_predictive(
        model, batch, record, n_ctx=n_ctx, n_max=n_max, device=device
    )

    tau_star = record["tau_star_std"].to(probs.device).float().reshape(-1)
    s2 = record["s2_std"].to(probs.device).float().reshape(-1)

    if dist_evaluator is not None:
        dist_evaluator.update(probs=probs, bin_edges=edges, y=tau_star, s2=s2)

    for evaluator, mask in dist_evaluators:
        if mask is None:
            evaluator.update(probs=probs, bin_edges=edges, y=tau_star, s2=s2)
            continue
        rows = mask.to(probs.device).reshape(-1)
        if not bool(rows.any()):
            continue
        evaluator.update(
            probs=probs[rows], bin_edges=edges, y=tau_star[rows], s2=s2[rows]
        )

    mean, _ = predictive_mean_var(probs, edges)
    sq_err = (mean - tau_star) ** 2

    if point_evaluator is not None:
        point_evaluator.update(
            pred_cate=mean.reshape(1, -1),
            true_cate=tau_star.reshape(1, -1),
        )

    for evaluator, mask in point_evaluators:
        rows = None if mask is None else mask.to(mean.device).reshape(-1)
        if rows is not None and not bool(rows.any()):
            continue
        sel = slice(None) if rows is None else rows
        evaluator.update(
            pred_cate=mean[sel].reshape(1, -1),
            true_cate=tau_star[sel].reshape(1, -1),
        )

    head_edges = model.cate_bin_edges.to(edges.device).float()
    grid_scale = float(
        (edges[-1] - edges[0]) / (head_edges[-1] - head_edges[0]).clamp_min(_EPS)
    )

    return {
        "mse": float(_f64(sq_err).mean()),
        "rows": int(tau_star.numel()),
        "grid_scale": grid_scale,
    }


def iter_frozen_episodes(
    val_dir: Path, max_episodes: Optional[int] = None
) -> Iterable[Sequence[torch.Tensor]]:
    """Yield frozen episodes one at a time, in the order they were built."""
    from proximalfm.prior._genload import LoadPriorDataset

    val_dir = Path(val_dir)
    n_files = len([p for p in val_dir.iterdir() if p.name.startswith("batch_")])
    loader = LoadPriorDataset(
        data_dir=str(val_dir), batch_size=1, max_batches=n_files, device="cpu"
    )
    for i, batch in enumerate(loader):
        if max_episodes is not None and i >= max_episodes:
            return
        yield batch


def score_frozen_set(
    model,
    out: Path,
    *,
    n_ctx: Optional[int] = None,
    device: str = "cpu",
    max_episodes: Optional[int] = None,
    levels: Tuple[float, ...] = (0.5, 0.9),
    alpha: float = 0.05,
    meta: Optional[Dict[str, Any]] = None,
    prefix_point: Optional[str] = "prior_point/",
    prefix_dist: str = "prior_dist/",
) -> Dict[str, float]:
    """Score a whole frozen set; returns the merged point + distributional metric dict.

    ``out`` is the frozen set's *root* (the directory holding both ``val/`` and
    ``frozen_meta.pt``), because the sidecar is not optional: without it there
    is no frozen comparison scale and no ``tau_star``.

    ``meta`` lets a caller that already loaded the sidecar (``Trainer``, which
    loads it once at construction because it is the expensive part) pass it
    straight through instead of paying ``load_frozen_meta`` again on every
    pass. ``None`` (the CLI's case) loads it here.

    ``prefix_point``/``prefix_dist`` namespace the two returned surfaces. The
    defaults are the prior set's, so every existing caller is unchanged. The
    trainer's *realistic* replicate pass passes
    ``prefix_point="synt_point/"``, ``prefix_dist="synt_dist/"`` -- both metric
    kinds from this one call, since point metrics are a side effect of the same
    per-episode forward pass that produces the distributional ones
    (``score_episode``'s ``CausalEvaluator`` fan-out), not a second pass.
    ``prefix_point=None`` skips the point surface entirely, for a caller that
    only wants the distributional half.
    """
    from prior_eval.build_prior_frozen_set import load_frozen_meta

    out = Path(out)
    if meta is None:
        meta = load_frozen_meta(out)
    n_max = int(meta["global"]["n_max"])
    n_ctx = n_max if n_ctx is None else int(n_ctx)
    if not 1 <= n_ctx <= n_max:
        raise ValueError(f"n_ctx={n_ctx} must be in [1, n_max={n_max}]")
    records = meta["episodes"]

    dist_eval = PosteriorEvaluator(levels=levels, alpha=alpha)
    point_eval = CausalEvaluator()

    mse_sum, scale_sum, rows = 0.0, 0.0, 0
    with torch.inference_mode():
        for i, batch in enumerate(iter_frozen_episodes(out / "val", max_episodes)):
            stats = score_episode(
                model,
                batch,
                records[i],
                n_ctx=n_ctx,
                n_max=n_max,
                device=device,
                dist_evaluator=dist_eval,
                point_evaluator=point_eval,
            )
            mse_sum += stats["mse"] * stats["rows"]
            # Row-weighted, matching how ``mean_sd`` averages -- the two are
            # multiplied together downstream, so they must pool the same way.
            scale_sum += stats["grid_scale"] * stats["rows"]
            rows += stats["rows"]

    metrics: Dict[str, float] = {}
    if prefix_point is not None:
        metrics.update(point_eval.compute(prefix=prefix_point, include_count=True))
    metrics.update(dist_eval.compute(prefix=prefix_dist, include_count=True))
    if rows:
        if prefix_point is not None:
            metrics[f"{prefix_point}cate/rmse_pooled"] = math.sqrt(mse_sum / rows)
        # Head units -> frozen space. Exactly 1.0 at n_ctx == n_max (the two
        # scales are then the same rows under the same formula), so this only
        # departs from 1 on an n-sweep prefix. Published because the trainer
        # needs it to express its sigma schedule in mean_sd's units, and a
        # caller that silently assumed 1.0 would be wrong precisely where the
        # sweep is interesting.
        metrics[f"{prefix_dist}grid_scale"] = scale_sum / rows
    metrics[f"{prefix_dist}n_ctx"] = float(n_ctx)
    return metrics
