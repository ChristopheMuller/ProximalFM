"""Adapters from a frozen evaluation set to ``analysis_prior``'s record type.

`analysis_prior` already measures everything ROADMAP A2 wants to cut on --
proxy relevance, delivered confounding share, overlap, heterogeneity -- but it
speaks :class:`analysis_prior.generate.PriorDatasetRecord`, built from its own
cache. The frozen set holds the same information in a different container: the
per-row arrays in the ``val/`` shards, the per-episode scalars in
``frozen_meta.pt``. These two adapters bridge that gap so the cuts reuse those
measurements rather than growing a second, silently drifting copy.

Two levels, because most cuts do not need the data at all:

:class:`SidecarRecord`
    Sidecar-only. Carries ``index``, ``sampled_hp``, ``mechanisms``,
    ``gamma_hetero`` and a ``dims`` mapping -- exactly the attributes
    ``mechanism_frame``, ``delivered_shares`` and ``proxy_relevance._knobs``
    touch. Building 1000 of these reads one file and no tensors.

:func:`frozen_to_records`
    The full thing, for the cuts that need rows (overlap, covariate
    extrapolation) and for the opt-in cross-fitted proxy-relevance regression.
    Loads the ``val/`` shards.

The duck-typing is deliberate and load-bearing: ``PriorDatasetRecord.dims`` is
a *property* derived from array shapes, so a sidecar-only record cannot be an
instance of it. Every consumer we target reads ``dims`` as a mapping and never
inspects the type, so a structural match is the honest way to say "this is the
subset of the record those functions actually use".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from analysis_prior.generate import PriorDatasetRecord
from posterior_eval.build_prior_frozen_set import load_frozen_meta

#: Feature blocks stored padded to ``max_dim_*`` and trimmed with ``d_*``.
_BLOCKS = ("U", "X", "W", "Z")


@dataclass
class SidecarRecord:
    """The subset of :class:`PriorDatasetRecord` derivable without the shards.

    Structurally compatible with the ``analysis_prior`` helpers that read only
    hyperparameters and diagnostics -- ``mechanism_inputs.mechanism_frame``,
    ``confounding.delivered_shares``, ``proxy_relevance._knobs`` and
    ``proxy_relevance._slack``. Anything touching rows needs
    :func:`frozen_to_records` instead.
    """

    index: int
    seed: int
    sampled_hp: Dict[str, Any]
    mechanisms: Dict[str, Any]
    gamma_hetero: Optional[float] = None
    a_logit_mean: Optional[float] = None
    a_logit_std: Optional[float] = None
    _dims: Dict[str, int] = field(default_factory=dict)

    @property
    def dims(self) -> Dict[str, int]:
        """Delivered (post-constant-filter) widths, matching the real record.

        The sidecar's ``d_U``/``d_X``/``d_W``/``d_Z``, **not** ``sampled_hp``'s
        ``dim_*``: a constant column is dropped before the data is written, and
        ``_slack`` is documented as holding for what was actually delivered.
        """
        return dict(self._dims)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value)
    return float(value)


def sidecar_records(out: Path, meta: Optional[Dict[str, Any]] = None) -> List:
    """Every episode of the frozen set as a :class:`SidecarRecord`.

    Cheap: one ``torch.load`` of ``frozen_meta.pt`` and no shard I/O.
    """
    out = Path(out)
    meta = load_frozen_meta(out) if meta is None else meta
    records = []
    for rec in meta["episodes"]:
        records.append(
            SidecarRecord(
                index=int(rec["index"]),
                seed=int(rec["seed"]),
                sampled_hp=dict(rec.get("sampled_hp") or {}),
                mechanisms=dict(rec.get("mechanisms") or {}),
                gamma_hetero=_as_float(rec.get("gamma_hetero")),
                a_logit_mean=_as_float(rec.get("a_logit_mean")),
                a_logit_std=_as_float(rec.get("a_logit_std")),
                _dims={f"d_{b}": int(rec[f"d_{b}"]) for b in _BLOCKS},
            )
        )
    return records


def _trim(block: torch.Tensor, width: int, rows: slice) -> np.ndarray:
    """Padded ``(T, max_dim)`` block -> active ``(n, width)`` float64 array."""
    return block[rows, :width].double().numpy()


def frozen_to_records(
    out: Path,
    *,
    indices: Optional[Sequence[int]] = None,
    rows: str = "context",
    meta: Optional[Dict[str, Any]] = None,
) -> List[PriorDatasetRecord]:
    """Load frozen episodes as full :class:`PriorDatasetRecord` objects.

    Parameters
    ----------
    out
        The frozen set's **root** (holding ``val/`` and ``frozen_meta.pt``).
    indices
        Episode indices to load; ``None`` loads all. Iteration is still
        sequential over the shards -- the filter skips conversion, not reading.
    rows
        ``"context"`` (default) keeps rows ``[0, n_max)``, ``"query"`` keeps
        ``[n_max, T)``, ``"all"`` keeps everything.

        **``"context"`` is the right default for every diagnostic here.** The
        query rows carry Monte Carlo replicate structure and NaN ``MU0``/``MU1``
        does not appear there, but the context rows are the observational
        sample a method would actually see, which is what overlap and proxy
        relevance are statements about.
    meta
        Pre-loaded sidecar, to avoid re-reading it.

    Notes
    -----
    ``MU0``/``MU1`` are **NaN on the context prefix** by the frozen set's
    construction (``mc_targets_on_context=False``), so ``record.cate`` is NaN
    under ``rows="context"``. Anything wanting the CATE target must read
    ``tau_star``/``tau_star_std`` from the sidecar instead -- that is the whole
    reason the sidecar exists.
    """
    from proximalfm.train._dist_eval import iter_frozen_episodes

    out = Path(out)
    meta = load_frozen_meta(out) if meta is None else meta
    sidecar = meta["episodes"]
    n_max = int(meta["global"]["n_max"])
    seq_len = int(meta["global"]["seq_len"])

    if rows == "context":
        row_slice = slice(0, n_max)
    elif rows == "query":
        row_slice = slice(n_max, seq_len)
    elif rows == "all":
        row_slice = slice(0, seq_len)
    else:
        raise ValueError(f"rows must be context/query/all, got {rows!r}")

    wanted = None if indices is None else set(int(i) for i in indices)
    out_records: List[PriorDatasetRecord] = []

    for i, batch in enumerate(iter_frozen_episodes(out / "val")):
        if wanted is not None and i not in wanted:
            continue
        rec = sidecar[i]
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
        widths = {
            "U": int(d_U[0]),
            "X": int(d_X[0]),
            "W": int(d_W[0]),
            "Z": int(d_Z[0]),
        }
        blocks = {"U": U[0], "X": X[0], "W": W[0], "Z": Z[0]}
        vectors = {
            "A": A[0],
            "A_probs": A_probs[0],
            "Y_obs": Y_obs[0],
            "Y0": Y0[0],
            "Y1": Y1[0],
            "MU0": MU0[0],
            "MU1": MU1[0],
        }
        out_records.append(
            PriorDatasetRecord(
                index=int(rec["index"]),
                seed=int(rec["seed"]),
                **{k: _trim(v, widths[k], row_slice) for k, v in blocks.items()},
                **{
                    k: v.reshape(-1)[row_slice].double().numpy()
                    for k, v in vectors.items()
                },
                sampled_hp=dict(rec.get("sampled_hp") or {}),
                mechanisms=dict(rec.get("mechanisms") or {}),
                gamma_hetero=_as_float(rec.get("gamma_hetero")),
                a_logit_mean=_as_float(rec.get("a_logit_mean")),
                a_logit_std=_as_float(rec.get("a_logit_std")),
            )
        )
        if wanted is not None and len(out_records) == len(wanted):
            break

    return out_records


def iter_frozen_arrays(
    out: Path, meta: Optional[Dict[str, Any]] = None
) -> Iterable[Dict[str, Any]]:
    """Stream per-episode arrays without materialising 1000 records at once.

    Yields ``{"index", "X_context", "X_query", "A_probs_context", "A_context"}``
    -- the minimum the row-touching cuts need. The full-record path above holds
    every block for every episode simultaneously, which is fine for a
    regression build but wasteful for a two-column summary.
    """
    from proximalfm.train._dist_eval import iter_frozen_episodes

    out = Path(out)
    meta = load_frozen_meta(out) if meta is None else meta
    n_max = int(meta["global"]["n_max"])

    for i, batch in enumerate(iter_frozen_episodes(out / "val")):
        X, A, A_probs = batch[1], batch[4], batch[5]
        d_X = int(batch[12][0])
        x = X[0][:, :d_X].double().numpy()
        yield {
            "index": i,
            "X_context": x[:n_max],
            "X_query": x[n_max:],
            "A_context": A[0].reshape(-1)[:n_max].double().numpy(),
            "A_probs_context": A_probs[0].reshape(-1)[:n_max].double().numpy(),
        }
