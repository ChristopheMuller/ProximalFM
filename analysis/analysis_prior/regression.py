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

Beyond ``R^2``, :func:`t_learner_mu` provides the one thing an ``R^2`` cannot:
an actual *counterfactual* prediction, which is what the confounding-strength
ladder (B.3.3) adjusts with.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
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


@lru_cache(maxsize=64)
def fold_indices(
    n_rows: int, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The (train, test) index pairs -- a pure function of the arguments.

    Every predictor set evaluated on the same ``(n_rows, n_splits, seed)``
    sees the identical split, which is what makes ΔR² contrasts fold-stable.

    Memoised: an audit calls this once per scored regression (tens of thousands
    of times over a full run) with only a handful of distinct argument triples,
    and rebuilding the ``KFold`` permutation each time was ~15% of the wall
    clock on a fully cached re-render. The returned arrays are shared between
    callers, so **treat them as read-only** -- every caller here only indexes
    with them.
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


def clip_to_train_range(
    pred: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> np.ndarray:
    """Clip each fold's OOF predictions to that fold's train-target range.

    A cross-fitted prediction far outside the training range is pure
    extrapolation error; left unbounded, a single fold's blow-up (a regressor
    predicting, say, 1e8 on one held-out row) dominates ``SS_res`` and drives
    ``R²`` to absurd magnitudes. Clipping to ``[min, max]`` of the *training*
    targets is a leakage-free guard -- the bound uses only training data -- that
    keeps ``R²`` in a sensible range without altering well-behaved predictions.
    Applied at scoring time, so it also rescues predictions already cached raw.
    """
    y = np.asarray(y, dtype=np.float64)
    clipped = np.asarray(pred, dtype=np.float64).copy()
    for train, test in fold_indices(len(y), n_splits, seed):
        clipped[test] = np.clip(clipped[test], y[train].min(), y[train].max())
    return clipped


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
    Returns ``{"pred": ndarray, "r2": float}`` where ``pred`` is clipped to the
    per-fold train range (see :func:`clip_to_train_range`) and ``r2`` is scored
    on that clipped prediction. The **raw** prediction is what gets stored, so
    clipping is a pure post-process: it applies to cache hits too, and the
    stored file stays model-faithful for other consumers.
    """
    key = f"{tag}|model={model}|splits={n_splits}|seed={seed}"
    path = _cache_path(cache_dir, key)

    raw = None
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        if str(payload["key"]) == key:  # hash-collision guard
            raw = payload["pred"]

    if raw is None:
        raw = oof_predictions(X, y, model=model, n_splits=n_splits, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" unless the name already ends in it, so the
        # temp name must too or the file lands one suffix over from the rename.
        # The pid keeps it unique per writer: two processes computing the same
        # entry (running two audits at once is normal -- they deliberately share
        # predictor sets) would otherwise write the *same* temp file and race,
        # one of them then failing its rename on a file the other already moved.
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        # Store the raw prediction; r2 is recomputed (clipped) on every read, so
        # the stored r2 is informational only.
        np.savez(
            tmp,
            key=np.str_(key),
            pred=raw,
            r2=np.float64(r2_score(y, clip_to_train_range(raw, y, n_splits, seed))),
        )
        tmp.replace(path)

    clipped = clip_to_train_range(raw, y, n_splits, seed)
    return {"pred": clipped, "r2": float(r2_score(y, clipped))}


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


# ---------------------------------------------------------------------------
# T-learner counterfactual predictions
# ---------------------------------------------------------------------------


def t_learner_mu(
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-fitted T-learner CEPOs ``(mu0_hat, mu1_hat)``, one per row.

    One regression per arm: the arm-``a`` model is fitted on ``{X, Y}`` of the
    rows with ``A == a`` and used to fill ``mu_a`` for *every* row. The two
    halves of that vector are produced differently on purpose:

    - **Opposite-arm rows** (``A != a``) are a genuine counterfactual: they were
      never in the arm-``a`` training set, so a single fit on the whole arm
      predicts them out-of-sample already.
    - **Same-arm rows** (``A == a``) are their own training data, so they are
      predicted *out of fold* (k-fold within that arm only). Without this the
      factual half would carry an in-sample-inflated fit and the two halves of
      ``mu_a`` would not be comparable -- which would bias
      ``mean(mu1 - mu0)`` by an amount that depends on the treated fraction.

    Predictions are clipped to the range of the targets that trained them (see
    :func:`clip_to_train_range` for why), so ``ATE_hat = mean(mu1 - mu0)`` cannot
    be driven by a single extrapolation blow-up.

    An arm with no rows yields ``NaN`` for that arm; an arm too small to split
    falls back to its own mean. Both are degenerate-overlap cases, and returning
    ``NaN``/constant is more honest than a fit on a handful of rows.
    """
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A).ravel()
    Y = np.asarray(Y, dtype=np.float32).ravel()
    if X.ndim != 2 or len(X) != len(Y) or len(A) != len(Y):
        raise ValueError(f"X {X.shape}, A {A.shape}, Y {Y.shape} are inconsistent")

    mu = []
    for arm in (0, 1):
        own = np.flatnonzero(A == arm)
        other = np.flatnonzero(A != arm)
        out = np.full(len(Y), np.nan, dtype=np.float64)
        if own.size == 0:
            mu.append(out)
            continue

        y_own = Y[own]
        if own.size < 2 * n_splits or X.shape[1] == 0:
            # Too few rows to cross-fit (or nothing to fit on): the best
            # available answer for both halves is the arm mean.
            out[:] = float(y_own.mean())
            mu.append(out)
            continue

        pred_own = oof_predictions(
            X[own], y_own, model=model, n_splits=n_splits, seed=seed
        )
        out[own] = clip_to_train_range(pred_own, y_own, n_splits, seed)

        if other.size:
            estimator = _model_factory(model, seed)()
            estimator.fit(X[own], y_own)
            out[other] = np.clip(
                np.asarray(estimator.predict(X[other]), dtype=np.float64),
                y_own.min(),
                y_own.max(),
            )
        mu.append(out)
    return mu[0], mu[1]


def cached_t_learner(
    tag: str,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
    cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
) -> Dict[str, np.ndarray]:
    """:func:`t_learner_mu` with the on-disk cache of :func:`cached_oof`.

    ``tag`` must identify the data *and* the predictor set (e.g. dataset cache
    key, record index, ``predictors=X+U``); the regression parameters are
    appended here. Both arms live in one entry -- they are always computed
    together and always read together, so splitting them would double the file
    count for no reuse. Returns ``{"mu0", "mu1", "ate"}``.
    """
    key = f"tlearner|{tag}|model={model}|splits={n_splits}|seed={seed}"
    path = _cache_path(cache_dir, key)

    mu0 = mu1 = None
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        if str(payload["key"]) == key:  # hash-collision guard
            mu0, mu1 = payload["mu0"], payload["mu1"]

    if mu0 is None:
        mu0, mu1 = t_learner_mu(X, A, Y, model=model, n_splits=n_splits, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" unless the name already ends in it, so the
        # temp name must too or the file lands one suffix over from the rename.
        # The pid keeps it unique per writer: two processes computing the same
        # entry (running two audits at once is normal -- they deliberately share
        # predictor sets) would otherwise write the *same* temp file and race,
        # one of them then failing its rename on a file the other already moved.
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, key=np.str_(key), mu0=mu0, mu1=mu1)
        tmp.replace(path)

    return {"mu0": mu0, "mu1": mu1, "ate": float(np.nanmean(mu1 - mu0))}
