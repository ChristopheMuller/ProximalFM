"""Load raw (un-standardized) observational tuples from a ``val/`` set.

Deliberately *not* routed through ``benchmarks.data``: that module standardizes
every column with context-only statistics, which is exactly right for scoring a
method and irrelevant here — the discretization below is quantile-based, hence
invariant to any monotone rescaling, and keeping the raw values makes the bin
edges readable against the generator's own units.

Only the observational tuple ``(X, W, Z, A)`` is loaded.  ``Y`` plays no role in
the operator whose spectrum we are after, and ``U`` is kept because it is the
one thing that lets a diagnostic ask "how much of the W-Z dependence is the
confounder".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import torch

from proximalfm.prior._genload import sparse2dense

Rows = Literal["context", "query", "all"]


@dataclass(frozen=True)
class SpectralEpisode:
    """One dataset: the rows a method would actually see, raw.

    ``X``/``W``/``Z``/``U`` are ``[n, d]`` (``d`` may be 0 — the small set has
    ``d_X = 0``), ``A`` is ``[n]`` of 0/1.
    """

    index: int
    X: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    U: np.ndarray
    A: np.ndarray

    @property
    def n(self) -> int:
        return self.A.shape[0]

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"SpectralEpisode(index={self.index}, n={self.n}, "
            f"d_X={self.X.shape[1]}, d_W={self.W.shape[1]}, "
            f"d_Z={self.Z.shape[1]}, d_U={self.U.shape[1]}, "
            f"p_treated={self.A.mean():.3f})"
        )


def _densify(payload: dict, field: str, dims_key: str, T: int) -> torch.Tensor:
    """Sparse 1D column store -> dense ``[b, T, d]`` at this file's max width."""
    dims = payload[dims_key]
    b = dims.shape[0]
    max_d = int(dims.max().item())
    if max_d == 0:
        return torch.zeros(b, T, 0)
    flat = sparse2dense(payload[field], dims.repeat_interleave(T), max_len=max_d)
    return flat.view(b, T, max_d)


def iter_episodes(
    val_dir: str | Path,
    *,
    rows: Rows = "context",
    limit: Optional[int] = None,
) -> Iterator[SpectralEpisode]:
    """Yield episodes from a ``val/`` directory, in file order.

    ``rows="context"`` (default) keeps the first ``train_size`` rows.  Those are
    the plain observational draws; the trailing ``n_q`` query rows exist to carry
    Monte-Carlo CEPO targets and, in builds with ``mc_targets_on_context=False``
    and on-disk replicates, are not one i.i.d. row each.  Nothing here needs
    them, so the safe subset is the default.
    """
    val_dir = Path(val_dir)
    files = sorted(val_dir.glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files found in {val_dir}")

    emitted = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["A"].dim() != 2:
            raise ValueError(f"{path}: expected non-nested val files.")
        T = int(payload["seq_lens"][0].item())
        if not (payload["seq_lens"] == T).all():
            raise ValueError(f"{path}: episodes must share a single seq_len.")

        dense = {
            name: _densify(payload, name, f"d_{name}", T)
            for name in ("X", "W", "Z", "U")
        }
        A = payload["A"]

        for i in range(A.shape[0]):
            train_size = int(payload["train_sizes"][i].item())
            if rows == "context":
                sl = slice(0, train_size)
            elif rows == "query":
                sl = slice(train_size, T)
            else:
                sl = slice(0, T)

            def take(name: str) -> np.ndarray:
                d = int(payload[f"d_{name}"][i].item())
                return dense[name][i, sl, :d].numpy().astype(np.float64)

            yield SpectralEpisode(
                index=emitted,
                X=take("X"),
                W=take("W"),
                Z=take("Z"),
                U=take("U"),
                A=A[i, sl].numpy().astype(np.int64),
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def load_episodes(
    val_dir: str | Path,
    *,
    rows: Rows = "context",
    limit: Optional[int] = None,
) -> list[SpectralEpisode]:
    """Eager :func:`iter_episodes`."""
    return list(iter_episodes(val_dir, rows=rows, limit=limit))
