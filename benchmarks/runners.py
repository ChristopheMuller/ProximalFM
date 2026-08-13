"""runners.py — how a method's episodes get executed.

This is the inversion point. The run script no longer owns a pool and push episodes
through it; it hands a method its resolved `ComputeConfig` and calls `run_method`, and
the runner decides whether to fan the episodes out through the shared `WorkerManager`
or to keep them in this process with a device held resident.

Both shapes reduce to the same loop — build the estimator, fit on the context rows,
predict the query rows — so the only real difference is *where* `_run_episode` runs.
`WorkerManager.map` collapses that difference, which is why there is no longer a
separate sequential implementation to drift out of sync with the parallel one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from benchmarks.compute import INLINE, ComputeConfig, apply_compute
from benchmarks.methods import build_method

__all__ = ["EpisodeTask", "EpisodeResult", "method_output_names", "run_method"]


def method_output_names(method_cfg: dict) -> list[str]:
    """Return the list of prediction-set names a single method config yields.

    A plain config produces one output named after ``cfg['name']``. A config with an
    ``outputs`` list (multi-output methods, e.g. NMMR emitting several CATE variants
    from one trained model) produces one output per entry.
    """
    outputs = method_cfg.get("outputs")
    if outputs:
        return [o["name"] for o in outputs]
    return [method_cfg["name"]]


@dataclass
class EpisodeTask:
    """One episode's worth of work, as sent to a worker.

    Everything here crosses a `spawn` process boundary, so it is plain numpy and plain
    dicts — no torch tensors, no open handles. The `compute` travels *with* the task
    rather than being set once per pool, because a session-wide pool serves methods
    with different torch-thread budgets.
    """

    b: int
    method_cfg: dict
    compute: ComputeConfig
    U_ctx: np.ndarray
    X_ctx: np.ndarray
    A_ctx: np.ndarray
    Y_ctx: np.ndarray
    W_ctx: np.ndarray
    Z_ctx: np.ndarray
    U_q: np.ndarray
    X_q: np.ndarray
    W_q: np.ndarray
    Z_q: np.ndarray


@dataclass
class EpisodeResult:
    b: int
    preds: Optional[dict[str, np.ndarray]]
    elapsed: float
    error: Optional[str]


def _run_episode(task: EpisodeTask) -> EpisodeResult:
    """Fit and predict one episode. Module-level so a spawn pool can pickle it.

    Failures are returned, not raised: episodes are independent, so one bad draw
    leaves NaN in its row and is reported as a failure rate rather than taking the
    whole run down.
    """
    import time

    try:
        # The one place threads are set. A pooled worker may have just run a method
        # with a different budget, so this is per task rather than per pool.
        apply_compute(task.compute)

        t0 = time.perf_counter()
        _, method = build_method(task.method_cfg, task.compute)

        X_ctx, X_q = task.X_ctx, task.X_q
        if task.method_cfg.get("is_oracle", False):
            X_ctx = np.concatenate([X_ctx, task.U_ctx], axis=-1)
            X_q = np.concatenate([X_q, task.U_q], axis=-1)

        method.fit(X_ctx, task.A_ctx, task.Y_ctx, W=task.W_ctx, Z=task.Z_ctx)

        # One fit → one or more named prediction sets. The default estimator returns
        # a single output keyed by None, which we map to the config name.
        variants = method.predict_variants(X_q, W=task.W_q, Z=task.Z_q)
        name = task.method_cfg["name"]
        preds = {
            (name if k is None else k): np.asarray(v).astype(np.float32)
            for k, v in variants.items()
        }
        return EpisodeResult(task.b, preds, time.perf_counter() - t0, None)
    except Exception as exc:
        return EpisodeResult(task.b, None, 0.0, str(exc))


def _build_tasks(
    method_cfg: dict, compute: ComputeConfig, batches: list[dict]
) -> list[EpisodeTask]:
    return [
        EpisodeTask(
            b=b["b"],
            method_cfg=method_cfg,
            compute=compute,
            U_ctx=b["U_ctx"],
            X_ctx=b["X_ctx"],
            A_ctx=b["A_ctx"],
            Y_ctx=b["Y_ctx"],
            W_ctx=b["W_ctx"],
            Z_ctx=b["Z_ctx"],
            U_q=b["U_q"],
            X_q=b["X_q"],
            W_q=b["W_q"],
            Z_q=b["Z_q"],
        )
        for b in batches
    ]


def run_method(
    method_cfg: dict,
    batches: list[dict],
    n_test: int,
    compute: ComputeConfig,
    manager: Any,
    *,
    on_progress: Any = None,
    on_error: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Run one method over every episode and return its predictions and timings.

    Args:
        method_cfg: one entry from the config's `methods:` list.
        batches: the per-episode numpy dicts from `run.build_batches`.
        n_test: query rows per episode, i.e. the width of the prediction arrays.
        compute: this method's resolved budget — see `compute.resolve_method_compute`.
        manager: the session's `WorkerManager`. Used only when `compute.execution`
            says to fan out; an `inline` method never touches a pool.
        on_progress: called once per finished episode (for the progress bar).
        on_error: called with `(b, message)` for each failed episode.

    Returns:
        `(preds, times, n_errors)`. `preds` and `times` are dicts keyed by output
        name: `preds[name]` is `(n_batch, n_test)` with NaN wherever an episode
        failed, `times[name]` is `(n_batch,)` seconds. All of a multi-output method's
        variants share one fit, so they share the recorded time.
    """
    n_batch = len(batches)
    output_names = method_output_names(method_cfg)
    preds = {
        name: np.full((n_batch, n_test), np.nan, dtype=np.float32)
        for name in output_names
    }
    times = {name: np.zeros(n_batch, dtype=np.float32) for name in output_names}

    tasks = _build_tasks(method_cfg, compute, batches)
    n_errors = 0

    for result in manager.map(_run_episode, tasks, compute):
        if result.preds is not None:
            for name, arr in result.preds.items():
                preds[name][result.b] = arr
                times[name][result.b] = result.elapsed
        else:
            n_errors += 1
            if on_error is not None:
                on_error(result.b, result.error)
        if on_progress is not None:
            on_progress()

    return preds, times, n_errors


def is_inline(compute: ComputeConfig) -> bool:
    return compute.execution == INLINE or compute.n_workers <= 1
