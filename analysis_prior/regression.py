"""Cross-fitted out-of-fold regression, with a prediction cache.

The shared dependency of every audit that asks "how well can block(s) B
predict target t?" (mechanism output composition in B.2.3; the exclusion /
confounding / relevance rungs of B.3). Three design rules:

- **Every row predicted exactly once, out of fold.** ``oof_predictions`` fits
  on K-1 folds and predicts the held-out fold, so the returned vector is the
  same length as ``y`` and never contaminated by in-fold fitting.
- **Folds are a function of ``(n_rows, n_splits, seed)`` only.** Incremental
  comparisons (R² with block B minus R² without it) difference out the fold
  split rather than resampling it, so the contrast is never polluted by fold
  luck. ``fold_indices`` is exposed and pinned by tests.
- **Results are cached on disk, predictions included.** Foundation-model
  inference dominates the cost of an audit; re-rendering a figure must not
  re-run it. Cache entries are content-addressed by a caller-supplied tag plus
  the regression parameters, and store the OOF predictions alongside the R²
  so later audits can reuse them (e.g. residual analyses).

Models: ``"linear"`` (fast, for tests), ``"rf"`` (neutral baseline),
``"tabicl"`` (in-context foundation model; a single shared instance, since
construction loads a checkpoint -- ``fit`` only stores the context, so reuse
across folds and datasets is safe).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

#: Default location for cached regressions. Sits next to the dataset cache.
DEFAULT_REGRESSION_CACHE_DIR = Path(__file__).parent / "cache" / "regression"

# A single shared TabICL instance: construction loads a checkpoint (seconds),
# while fit() merely stores the in-context data. Lazy so that importing this
# module (e.g. for RF-only tests) never touches the checkpoint.
_TABICL_INSTANCE = {}


def _tabicl():
    if "default" not in _TABICL_INSTANCE:
        from tabicl import TabICLRegressor

        _TABICL_INSTANCE["default"] = TabICLRegressor()
    return _TABICL_INSTANCE["default"]


def _model_factory(model: str, seed: int) -> Callable:
    """Return a zero-argument callable producing a ready-to-fit estimator."""
    if model == "linear":
        return LinearRegression
    if model == "rf":
        return lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )
    if model == "tabicl":
        return _tabicl
    raise ValueError(f"Unknown model {model!r}. Choices: 'linear', 'rf', 'tabicl'.")


def fold_indices(
    n_rows: int, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The (train, test) index pairs -- a pure function of the arguments.

    Every predictor set evaluated on the same ``(n_rows, n_splits, seed)``
    sees the identical split, which is what makes ΔR² contrasts fold-stable.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.empty((n_rows, 1)))]


def oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions of ``y`` from ``X``; same length as ``y``."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X {X.shape} and y {y.shape} are inconsistent")
    if X.shape[1] == 0:
        # No predictors: the OOF best constant is the train-fold mean.
        pred = np.empty_like(y)
        for train, test in fold_indices(len(y), n_splits, seed):
            pred[test] = y[train].mean()
        return pred

    factory = _model_factory(model, seed)
    pred = np.empty_like(y)
    for train, test in fold_indices(len(y), n_splits, seed):
        estimator = factory()
        estimator.fit(X[train], y[train])
        pred[test] = np.asarray(estimator.predict(X[test]), dtype=np.float32)
    return pred


def _cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.npz"


def cached_oof(
    tag: str,
    X: np.ndarray,
    y: np.ndarray,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
    cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
) -> Dict[str, np.ndarray]:
    """OOF predictions + R², cached on disk under ``tag`` + parameters.

    ``tag`` must identify the *data* (e.g. dataset cache key, record index,
    target name, predictor-set name); the regression parameters are appended
    here, so one tag can be evaluated under several models without collision.
    Returns ``{"pred": ndarray, "r2": float}``.
    """
    key = f"{tag}|model={model}|splits={n_splits}|seed={seed}"
    path = _cache_path(cache_dir, key)

    if path.exists():
        payload = np.load(path, allow_pickle=False)
        if str(payload["key"]) == key:  # hash-collision guard
            return {"pred": payload["pred"], "r2": float(payload["r2"])}

    pred = oof_predictions(X, y, model=model, n_splits=n_splits, seed=seed)
    r2 = float(r2_score(y, pred))

    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" unless the name already ends in it, so the temp
    # name must too or the file lands one suffix over from where we rename it.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, key=np.str_(key), pred=pred, r2=np.float64(r2))
    tmp.replace(path)
    return {"pred": pred, "r2": r2}


def leave_one_out_r2(
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    tag_prefix: str,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
    cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
) -> Dict[str, float]:
    """R² of the full predictor set and of each leave-one-block-out set.

    ``blocks`` maps block name to a ``(T, d_b)`` matrix (``d_b`` may be 0).
    Returns ``{"full": R²(all blocks)} | {name: R²(all except name)}``; the
    caller derives unique contributions as ``full - without[name]``. All sets
    share the same folds by construction.
    """
    names = list(blocks)

    def r2_of(subset: Iterable[str], label: str) -> float:
        X = np.hstack([np.asarray(blocks[n], dtype=np.float32) for n in subset])
        return cached_oof(
            f"{tag_prefix}|predictors={label}",
            X,
            y,
            model=model,
            n_splits=n_splits,
            seed=seed,
            cache_dir=cache_dir,
        )["r2"]

    out = {"full": r2_of(names, "+".join(names))}
    for name in names:
        rest = [n for n in names if n != name]
        out[name] = r2_of(rest, "+".join(rest))
    return out
