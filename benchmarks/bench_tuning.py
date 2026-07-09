"""
bench_tuning.py — Generic hyperparameter-tuning scaffolding for baselines.

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
    """Parsed form of a method config's ``tune:`` block."""

    enabled: bool = False
    val_frac: float = 0.2  # fraction of the context held out for scoring
    n_trials: int = 20  # random-search budget; full grid if it fits
    grid: dict = field(default_factory=dict)  # param_name -> list of values
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "TuningConfig":
        if not d:
            return cls(enabled=False)
        return cls(
            enabled=bool(d.get("enabled", True)),
            val_frac=float(d.get("val_frac", 0.2)),
            n_trials=int(d.get("n_trials", 20)),
            grid=dict(d.get("grid", {}) or {}),
            seed=int(d.get("seed", 0)),
        )


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
    best_hparams: dict = candidates[0] if candidates else {}
    best_score = float("inf")
    results: list[tuple[dict, float]] = []

    for hp in candidates:
        score = float(fit_score(hp))
        results.append((hp, score))
        if np.isfinite(score) and score < best_score:
            best_score, best_hparams = score, hp

    return best_hparams, best_score, results
