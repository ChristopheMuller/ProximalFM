"""Disk cache for an audit's assembled per-dataset frame.

:mod:`analysis_prior.regression` already caches every individual regression, so
nothing is *refitted* on a re-render. That is not the same as being free: an
audit still opens one file and rescores one ``R^2`` per (dataset, target,
predictor set), which for the relevance audit is ~5 entries per ``U`` column and
lands at ~28 ms per dataset -- half a minute of pure bookkeeping to redraw a
figure over the full 1024-dataset prior.

This module caches the *assembled* frame instead, so a re-render reads one file.
The regression cache stays the source of truth for the expensive part; this is
a second, cheaper layer in front of it.

Two design points:

- **Coverage is per dataset index, like the dataset cache.** An entry records
  which indices it holds; asking for a subset takes a prefix, asking for more
  builds only the missing ones and appends. So ``--n-datasets 200`` after a full
  run costs one file read, and growing 600 to 1024 pays only for the 424.
- **Invalidation is automatic, from the builder's source.** The cache key folds
  in a hash of the source of the functions that produce the frame, so editing a
  measurement invalidates its cached frames without anyone having to remember a
  version constant. Reformatting invalidates too -- a cache miss is cheap and a
  stale frame is not.

A corrupt or unreadable entry is treated as a miss rather than an error: this
is a regenerable convenience cache and must never be the reason an audit fails.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

#: Default location for cached frames. Sits next to the other caches.
DEFAULT_FRAME_CACHE_DIR = Path(__file__).parent / "cache" / "frames"


def _source_digest(functions: Sequence[Callable]) -> str:
    """A hash of the given functions' source, for automatic invalidation.

    Falls back to the qualified name when the source is unavailable (a C
    builtin, or a REPL-defined function): the cache then behaves as it would
    have with a manual version constant, which is the pre-existing situation
    rather than a regression.
    """
    parts = []
    for function in functions:
        try:
            parts.append(inspect.getsource(function))
        except (OSError, TypeError):
            parts.append(getattr(function, "__qualname__", repr(function)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def cached_frame(
    tag: str,
    indices: Sequence[int],
    build: Callable[[Sequence[int]], pd.DataFrame],
    depends_on: Sequence[Callable] = (),
    cache_dir: Path = DEFAULT_FRAME_CACHE_DIR,
    index_column: str = "index",
) -> pd.DataFrame:
    """Return the frame for ``indices``, building only what is not cached.

    Parameters
    ----------
    tag : str
        Identifies the *data and parameters* the frame was built under (dataset
        cache key, model, folds, seed, row cap). Must not encode how many
        datasets were asked for -- that is what ``indices`` is for.

    indices : sequence of int
        The dataset indices wanted. Rows are returned for exactly these, in
        this order.

    build : callable
        ``build(missing_indices) -> DataFrame`` with an ``index_column``. Called
        only when something is missing, and only for the missing indices.

    depends_on : sequence of callable
        Functions whose source determines the frame's contents. Editing any of
        them invalidates the cache (see the module docstring).

    Notes
    -----
    An index that ``build`` legitimately returns no rows for (e.g. a dataset the
    audit skips) would be re-attempted on every call, since coverage is measured
    from the rows present. Callers must therefore filter such indices out
    *before* calling -- which they have to do anyway to report how many were
    skipped.
    """
    wanted = list(indices)
    key = f"{tag}|source={_source_digest(depends_on)}"
    path = Path(cache_dir) / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.pkl"

    cached: pd.DataFrame | None = None
    if path.exists():
        try:
            payload = pd.read_pickle(path)
            # The key is stored alongside the frame as a hash-collision guard,
            # exactly as in the regression cache.
            if payload.get("key") == key:
                cached = payload["frame"]
        except Exception:  # noqa: BLE001 -- a bad cache file is a miss, not a crash
            cached = None

    have = set() if cached is None else set(cached[index_column].unique())
    missing = [i for i in wanted if i not in have]

    if missing or cached is None:
        # ``cached is None`` with nothing missing means an empty request against
        # a cold cache: still call the builder, so the returned frame carries
        # the right columns rather than being a shapeless empty.
        built = build(missing)
        cached = built if cached is None else pd.concat([cached, built])

    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a pid-unique temp name so two audits running at once
        # cannot interleave a partial file, matching the regression cache.
        tmp = path.with_suffix(f".tmp{os.getpid()}.pkl")
        pd.to_pickle({"key": key, "frame": cached}, tmp)
        tmp.replace(path)

    selection = cached[cached[index_column].isin(wanted)]
    return selection.sort_values(index_column).reset_index(drop=True)
