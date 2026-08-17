"""
tuning.py — Generic hyperparameter-tuning scaffolding for baselines.

Design
──────
The *mechanics* of tuning are identical across methods — split the context
into train/val, enumerate or sample candidate hyperparameters, train each,
score it, keep the best — so they live here, once. What differs between
methods is the *selection metric* (NMMR: held-out kernel moment loss;
meta-learners: outcome MSE; …) and *how a candidate is trained*.

A method therefore only supplies a callback::

    fit_score(hparams: dict) -> float      # lower is better

and this module owns everything else. This keeps the tuner method-agnostic
while letting each estimator own the statistics that make it unique.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TuningConfig:
    """Parsed form of a method config's ``tune:`` block.

    Two search spaces, because they cost differently:

    - ``grid`` is *sampled*: ``n_trials`` draws from its Cartesian product (the
      whole product when it fits). This is the budgeted axis.
    - ``sweep`` is *enumerated*: every combination is tried, for every draw from
      ``grid``. So the evaluated candidate count is ``n_trials * |sweep|`` and
      ``n_trials`` counts the sampled axes only.

    ``sweep`` exists because some parameters are nearly free to vary once a
    candidate has been prepared — PMMR's ``lam1``/``lam2`` enter only the
    assembled left-hand side, so re-solving costs one Cholesky against a full
    refit (see :class:`benchmarks.methods.pmmr._PreparedSystem`). Putting such a
    parameter in ``sweep`` rather than ``grid`` buys exhaustive coverage of the
    axis that matters most for roughly the cost of one point on it.

    It is a *declaration of intent*, not an optimisation the tuner performs on
    its own: whether a swept parameter is actually cheap depends on the method,
    so a method that exploits ``sweep`` must validate which of its parameters
    may appear there. A method that ignores ``sweep`` still gets correct
    results, just no speedup.

    Omitting ``sweep`` reproduces the single-grid behaviour exactly.
    """

    enabled: bool = False
    val_frac: float = 0.2  # fraction of the context held out for scoring
    n_trials: int = 20  # random-search budget over `grid`; full grid if it fits
    grid: dict = field(default_factory=dict)  # param_name -> list of values
    sweep: dict = field(default_factory=dict)  # param_name -> list, exhaustive
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "TuningConfig":
        if not d:
            return cls(enabled=False)
        grid = dict(d.get("grid", {}) or {})
        sweep = dict(d.get("sweep", {}) or {})

        both = sorted(set(grid) & set(sweep))
        if both:
            raise ValueError(
                f"tune: {both} appear in both `grid` and `sweep`; a parameter "
                f"is either sampled or enumerated, not both. Keep the values in "
                f"one block."
            )

        return cls(
            enabled=bool(d.get("enabled", True)),
            val_frac=float(d.get("val_frac", 0.2)),
            n_trials=int(d.get("n_trials", 20)),
            grid=grid,
            sweep=sweep,
            seed=int(d.get("seed", 0)),
        )

    @property
    def search_space(self) -> dict:
        """Both blocks as one ``param -> values`` mapping, for diagnostics.

        Safe to merge because ``from_dict`` refuses overlapping keys.
        """
        return {**self.grid, **self.sweep}


# ─────────────────────────────────────────────────────────────────────────────
# Mechanics
# ─────────────────────────────────────────────────────────────────────────────


def sample_candidates(grid: dict, n_trials: int, seed: int) -> list[dict]:
    """Return candidate hyperparameter dicts.

    If the full Cartesian product of the grid fits within ``n_trials`` it is
    returned exhaustively; otherwise a reproducible random subsample (without
    replacement) of the product is drawn.
    """
    if not grid:
        return [{}]

    keys = list(grid.keys())
    value_lists = [
        list(v) if isinstance(v, (list, tuple)) else [v] for v in grid.values()
    ]
    combos = list(itertools.product(*value_lists))

    if n_trials is not None and 0 < n_trials < len(combos):
        combos = random.Random(seed).sample(combos, n_trials)

    return [dict(zip(keys, combo)) for combo in combos]


def train_val_split(
    n: int, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) row indices for an ``n``-row context."""
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def search(
    candidates: list[dict],
    fit_score: Callable[[dict], float],
) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Evaluate every candidate and return the best (lowest-scoring) one.

    Args:
        candidates: hyperparameter dicts to try.
        fit_score:  callback that trains a candidate and returns its validation
                    score (lower is better). NaN/inf scores are skipped.

    Returns:
        (best_hparams, best_score, all_results).
    """
    return search_nested(
        candidates,
        [{}],
        prepare=lambda hp: hp,
        score=lambda hp, _sweep: fit_score(hp),
    )


def search_nested(
    outer: list[dict],
    inner: list[dict],
    prepare: Callable[[dict], object | None],
    score: Callable[[object, dict], float],
) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Two-level search: prepare once per ``outer`` candidate, score every ``inner``.

    For methods where part of a fit does not depend on part of the search space
    (see :class:`TuningConfig`), this is the shape that collects the saving —
    and it collects it *structurally*, by building the shared object once in the
    outer loop, rather than by memoising on hyperparameter values. That matters
    for more than tidiness: a memo keyed on hyperparameters alone carries no
    identity for the *data*, so one that outlived a call would happily serve
    another episode's grams inside a worker pool. Here there is nothing to
    invalidate, and the reuse is total rather than opportunistic.

    Args:
        outer: candidates whose values require a fresh ``prepare``.
        inner: candidates re-evaluated against every prepared object. Pass
            ``[{}]`` for none.
        prepare: builds the shared object for one outer candidate. Return
            ``None`` to declare the candidate unfittable; all of its inner
            candidates then score ``inf`` and are skipped, as with a non-finite
            score from ``fit_score``.
        score: ``(prepared, inner_hparams) -> float``, lower is better.

    Returns:
        ``(best_hparams, best_score, all_results)``, identical in contract to
        :func:`search`. Every reported hparams dict is the merged
        ``{**outer, **inner}``, so a caller cannot tell the search was nested —
        which is what lets :func:`search_diagnostics` stay unaware of the split.
    """
    best_hparams: dict = {**outer[0], **inner[0]} if outer and inner else {}
    best_score = float("inf")
    results: list[tuple[dict, float]] = []

    for base in outer:
        prepared = prepare(base)
        for sweep_hp in inner:
            hp = {**base, **sweep_hp}
            s = float("inf") if prepared is None else float(score(prepared, sweep_hp))
            results.append((hp, s))
            if np.isfinite(s) and s < best_score:
                best_score, best_hparams = s, hp

    return best_hparams, best_score, results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def _grid_key(value) -> str:
    """A stable, printable spelling of a grid value for a diagnostic key."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:g}"


def search_diagnostics(
    grid: dict,
    candidates: list[dict],
    best: dict,
    best_score: float,
    results: list[tuple[dict, float]],
) -> dict[str, float]:
    """Scalar record of what a search actually did, for ``CATEEstimator.diagnostics``.

    Emitted keys:

    - ``tune_n_candidates`` — how many candidates were *evaluated*, which is not
      the grid size whenever ``n_trials`` is smaller than the Cartesian product.
      This is the one that catches a search that has silently collapsed to a
      single random draw.
    - ``tune_best_score`` — the winning held-out score.
    - ``tune_score_range`` — spread between the best and worst finite score. Near
      zero means the criterion did not separate the candidates, so the "chosen"
      value is arbitrary and reading it as a tuned optimum is a mistake.
    - ``tune_n_failed`` — candidates whose score was NaN/inf and were skipped.
    - ``tune_<param>_frac@<value>`` — 1.0 for the selected value of each grid
      parameter, 0.0 for a value that was **offered to the search and lost**,
      and *absent* for a configured value no candidate carried.

    That last distinction matters as soon as ``n_trials`` is below the grid
    size, because then the candidates are a random subset and most configured
    values are never tried. Reporting those as 0.0 would read as "lost", which
    is the opposite of the truth — it is the *grid* that failed to reach them,
    not the value that failed. Omitting the key instead reuses the convention
    the runner already has: ``run_method`` accumulates diagnostics by key and
    pads an episode that did not emit one with NaN, and the summary mean is
    NaN-aware, so a never-offered value surfaces as ``None`` rather than as a
    confident zero. Read alongside ``tune_n_candidates``, which says how much of
    the grid was reachable at all.

    Within one episode the emitted fractions for a parameter still sum to 1, so
    :meth:`PLearner.diagnostics`'s concern — averaging fractions computed over
    different populations — is handled rather than dodged: with a fixed
    ``tune.seed`` the candidate set is identical in every episode and the key
    set therefore does not vary, and if the seed is ever varied per episode each
    surviving key is the conditional rate "of the episodes where this value was
    a candidate, how often it won", which is the number you would want.
    """
    finite = [s for _, s in results if np.isfinite(s)]

    diag: dict[str, float] = {
        "tune_n_candidates": float(len(candidates)),
        "tune_n_failed": float(len(results) - len(finite)),
    }
    if finite:
        diag["tune_best_score"] = float(best_score)
        diag["tune_score_range"] = float(max(finite) - min(finite))

    for param, values in grid.items():
        values = list(values) if isinstance(values, (list, tuple)) else [values]
        offered = {c.get(param) for c in candidates}
        chosen = best.get(param)
        for v in values:
            if v in offered:
                diag[f"tune_{param}_frac@{_grid_key(v)}"] = float(v == chosen)

    return diag
