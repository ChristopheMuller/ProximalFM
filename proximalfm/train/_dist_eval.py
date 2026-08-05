"""Streaming distributional evaluation of the CATE posterior (EVAL_PLAN step 5).

This is the ``prior/*`` surface: point **and** distributional metrics on the
frozen high-``K`` set built by ``posterior_eval/``. It is the only surface where
a distribution can be scored honestly, because it is the only one carrying a
low-noise ``tau_star`` -- see EVAL_PLAN.md's table and ``_dist_metrics``'
docstring for why ``--val_dir`` is deliberately excluded.

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
uniform overall. ``pit_ks_episode_p90`` is the number that catches it.

The class mirrors :class:`proximalfm.train._eval.CausalEvaluator`'s
``reset``/``update``/``compute`` shape so the trainer wiring (step 6) is
mechanical, ``sync_ddp`` included: :func:`score_frozen_set` shards the episodes
round-robin across ranks and every rank then enters the same collectives, so a
rank whose shard came out empty still participates rather than hanging the
group.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from proximalfm.train._dist_metrics import (
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
    needs the sample, and the per-episode KS statistics, which are the point of
    reporting a distribution of calibration rather than one pooled number.

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
        not a routine path. Under DDP the cap is applied *per rank* as
        ``max_pit // world_size`` before the gather -- gathering first and
        subsampling after would defeat the memory bound that motivates it.
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

        self._add("crps", crps(probs, bin_edges, y).double().sum().item())
        self._add(
            "crps_oracle",
            oracle_discretized_crps(y, bin_edges).double().sum().item(),
        )

        for key, value in coverage(probs, bin_edges, y, self.levels).items():
            self._add(key, value * n)

        # variance_identity returns means over the episode; scale them back to
        # sums so the pooled ratios below are over rows, not over episodes.
        vi = variance_identity(probs, bin_edges, y, s2_t)
        for key in ("mean_var", "mean_sd", "mse_obs", "mean_s2"):
            self._add(key, vi[key] * n)

        self._add("out_of_grid", out_of_grid_fraction(bin_edges, y) * n)

        u = pit(probs, bin_edges, y).flatten()
        self._pit.append(u.float().cpu())
        if u.numel() >= 2:
            stat, band = pit_ks(u, alpha=self.alpha)
            self._episode_ks.append(float(stat))
            self._episode_ks_band.append(float(band))

        self._rows += n
        self._episodes += 1

    def _sum_keys(self) -> List[str]:
        """The accumulator's keys, in a rank-independent order.

        Not ``sorted(self._sums)``: a rank whose shard came out empty has an
        empty ``_sums``, so the all-reduce payload has to be laid out from
        ``self.levels`` (identical on every rank by construction) rather than
        from what this rank happened to see.
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
        prefix: str = "prior/dist/",
        include_count: bool = True,
        sync_ddp: bool = False,
        device: str | torch.device | None = None,
    ) -> Dict[str, float]:
        """Summarize; returns ``{}`` when nothing was accumulated.

        With ``sync_ddp=True`` this reduces across the process group and **must
        be entered by every rank**, including one that accumulated nothing --
        the early return below is therefore conditioned on the group being
        inactive, exactly as :meth:`CausalEvaluator.compute` does it.
        """
        ddp_active = sync_ddp and dist.is_available() and dist.is_initialized()

        if not ddp_active and (self._episodes == 0 or self._rows == 0):
            return {}

        keys = self._sum_keys()
        rows = float(self._rows)
        episodes = float(self._episodes)
        sums = [float(self._sums.get(k, 0.0)) for k in keys]

        world_size = dist.get_world_size() if ddp_active else 1
        # Subsample locally *before* the gather: the cap is a memory bound, and
        # gathering the full pools first would spend exactly the memory it
        # exists to save.
        pooled = self._pooled_pit(max(1, self.max_pit // world_size))
        episode_ks = list(self._episode_ks)
        episode_ks_band = list(self._episode_ks_band)

        if ddp_active:
            if device is None:
                raise ValueError("device must be provided when sync_ddp=True")

            payload = torch.tensor(
                [rows, episodes, *sums], dtype=torch.float64, device=device
            )
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            rows = payload[0].item()
            episodes = payload[1].item()
            sums = payload[2:].tolist()

            # The three things that do not all-reduce. The PIT sample is needed
            # as a *sample* (KS is not a sum), and the per-episode statistics
            # are quantiles over episodes -- a few hundred floats either way.
            gathered: List[dict | None] = [None] * world_size
            dist.all_gather_object(
                gathered,
                {"pit": pooled.cpu(), "ks": episode_ks, "band": episode_ks_band},
            )
            pits = [
                g["pit"] for g in gathered if g is not None and g["pit"].numel() > 0
            ]
            pooled = torch.cat(pits) if pits else torch.empty(0)
            episode_ks = [x for g in gathered if g is not None for x in g["ks"]]
            episode_ks_band = [x for g in gathered if g is not None for x in g["band"]]

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
            metrics[f"cov{tag}"] = mean[f"cov{tag}"]
            metrics[f"width{tag}"] = mean[f"width{tag}"]

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
            metrics["pit_ks"] = float(stat)
            metrics["pit_ks_band"] = float(band)

        if episode_ks:
            eps_ks = torch.tensor(episode_ks, dtype=torch.float64)
            metrics["pit_ks_episode_median"] = float(eps_ks.median())
            metrics["pit_ks_episode_p90"] = float(
                eps_ks.quantile(torch.tensor(0.9, dtype=torch.float64))
            )
            band = torch.tensor(episode_ks_band, dtype=torch.float64)
            metrics["pit_ks_episode_band"] = float(band.median())

        if include_count:
            metrics["eval/episodes"] = episodes
            metrics["eval/rows"] = rows

        return {f"{prefix}{k}": v for k, v in metrics.items()}


# --------------------------------------------------------------------------
# Driving a model over the frozen set
# --------------------------------------------------------------------------


def require_single_cate_head(model) -> None:
    """Raise :class:`UnsupportedLossTypeError` unless ``model`` predicts one
    histogram over ``tau`` on a single shared grid.

    Called both eagerly (``Trainer.__init__``, so a ``bins_po``/``hl`` config
    with ``--dist_eval_dir`` set fails at construction rather than hundreds of
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
            "metrics (prior/point/*) remain available; distributional ones need "
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
    replicated queries), exactly as ``posterior_eval/report.py`` does.

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

    y_train = Y_obs.to(device)[:, :n_ctx].reshape(1, n_ctx)

    out = model.predict(
        X=rows(X),
        Z=rows(Z),
        W=rows(W),
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
) -> Dict[str, float]:
    """Score one episode into the given evaluators.

    Returns the per-episode pooled squared error and row count, which the
    driver accumulates into ``cate/rmse_pooled``. That key exists for one
    reason: ``CausalEvaluator.cate/rmse`` is a *median over episodes*, so it is
    not the quantity ``posterior_eval/report.py`` prints, and the cheapest place
    to catch a scale bug is a number that must match it exactly.
    """
    probs, edges = episode_predictive(
        model, batch, record, n_ctx=n_ctx, n_max=n_max, device=device
    )

    tau_star = record["tau_star_std"].to(probs.device).float().reshape(-1)
    s2 = record["s2_std"].to(probs.device).float().reshape(-1)

    if dist_evaluator is not None:
        dist_evaluator.update(probs=probs, bin_edges=edges, y=tau_star, s2=s2)

    mean, _ = predictive_mean_var(probs, edges)
    sq_err = (mean - tau_star) ** 2

    if point_evaluator is not None:
        point_evaluator.update(
            pred_cate=mean.reshape(1, -1),
            true_cate=tau_star.reshape(1, -1),
        )

    return {"mse": float(sq_err.double().mean()), "rows": int(tau_star.numel())}


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
    rank: int = 0,
    world_size: int = 1,
    sync_ddp: bool = False,
) -> Dict[str, float]:
    """Score a whole frozen set; returns the merged ``prior/*`` metric dict.

    ``out`` is the frozen set's *root* (the directory holding both ``val/`` and
    ``frozen_meta.pt``), because the sidecar is not optional: without it there
    is no frozen comparison scale and no ``tau_star``.

    ``meta`` lets a caller that already loaded the sidecar (``Trainer``, which
    loads it once at construction because it is the expensive part) pass it
    straight through instead of paying ``load_frozen_meta`` again on every
    pass. ``None`` (the CLI's case) loads it here.

    ``rank``/``world_size`` shard the episodes round-robin (episode ``i`` is
    scored by rank ``i % world_size``), and ``sync_ddp`` reduces the three
    accumulators back into the same numbers a single process would produce.
    The sharding is applied to the *global* episode index, after
    ``max_episodes``, so which episodes get scored does not depend on the world
    size -- only who scores them. A rank may legitimately end up with an empty
    shard (``max_episodes < world_size``); it must still reach the reductions
    below, so nothing here returns early on an empty local pass.
    """
    from posterior_eval.build_prior_frozen_set import load_frozen_meta

    out = Path(out)
    if meta is None:
        meta = load_frozen_meta(out)
    n_max = int(meta["global"]["n_max"])
    n_ctx = n_max if n_ctx is None else int(n_ctx)
    if not 1 <= n_ctx <= n_max:
        raise ValueError(f"n_ctx={n_ctx} must be in [1, n_max={n_max}]")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} must be in [0, world_size={world_size})")
    records = meta["episodes"]

    dist_eval = PosteriorEvaluator(levels=levels, alpha=alpha)
    point_eval = CausalEvaluator()

    mse_sum, rows = 0.0, 0
    with torch.inference_mode():
        for i, batch in enumerate(iter_frozen_episodes(out / "val", max_episodes)):
            if i % world_size != rank:
                continue
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
            rows += stats["rows"]

    # ``cate/rmse_pooled`` is accumulated here in the driver, outside either
    # evaluator, so it needs its own reduce -- the easiest of the three to
    # forget, and it would silently report only rank 0's shard.
    if sync_ddp and dist.is_available() and dist.is_initialized():
        pooled = torch.tensor(
            [mse_sum, float(rows)], dtype=torch.float64, device=device
        )
        dist.all_reduce(pooled, op=dist.ReduceOp.SUM)
        mse_sum, rows = pooled[0].item(), pooled[1].item()

    metrics = dict(
        point_eval.compute(
            prefix="prior/point/",
            include_count=True,
            sync_ddp=sync_ddp,
            device=device,
        )
    )
    metrics.update(
        dist_eval.compute(
            prefix="prior/dist/",
            include_count=True,
            sync_ddp=sync_ddp,
            device=device,
        )
    )
    if rows:
        metrics["prior/point/cate/rmse_pooled"] = math.sqrt(mse_sum / rows)
    metrics["prior/n_ctx"] = float(n_ctx)
    return metrics
