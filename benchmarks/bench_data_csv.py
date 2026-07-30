"""
bench_data_csv.py — Build the standardized benchmark data dict from a single
real-world CSV (e.g. a Causal Chambers export), given a user-supplied column
to variable-block mapping.

Unlike ``load_benchmark_from_val_dir`` (many synthetic episodes sharing one
``seq_len``/``train_size``), this loader always produces exactly one episode
(``N_BATCH == 1``): a single dataset, evaluated on its own train/test split.
Downstream (``bench_methods.py``, ``bench_metrics.py``) needs no changes for
this — every estimator and metric already operates per-episode.

The mapping mirrors the one used in
``benchmarks/causal_chambers/Proximal Causal Inference on Causal Chambers.Rmd``
(``load_cc_data()``): ``U`` is the unobserved confounder (oracle-only, may be
omitted), ``X`` the covariates, ``Z`` the treatment proxy, ``W`` the outcome
proxy, ``A`` the treatment, ``Y`` the observed outcome, and ``Y0``/``Y1`` the
ground-truth potential outcomes (obtained on the chamber by physically forcing
the treatment to each extreme) that make ``true_ITEs`` — and therefore the
whole existing metrics stack — meaningful for real, non-synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.bench_data import build_benchmark_dict

_REQUIRED_KEYS = ("X", "W", "Z", "A", "Y", "Y0", "Y1")


def _as_list(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def load_benchmark_from_csv(
    csv_path: str | Path,
    columns: dict,
    *,
    prop_train: float = 0.75,
    shuffle: bool = True,
    seed: int = 0,
    normalize_treatment: bool = True,
) -> dict:
    """Load a single chamber-style CSV into the standardized benchmark dict.

    ``columns`` maps variable blocks to CSV column names: ``X``/``W``/``Z``
    (lists, possibly multi-dimensional), ``A``/``Y``/``Y0``/``Y1`` (single
    column names), and an optional ``U`` (list; omitted => zero-width, used
    only by oracle methods when present).
    """
    missing = [k for k in _REQUIRED_KEYS if k not in columns]
    if missing:
        raise ValueError(
            f"columns mapping is missing required key(s) {missing}; "
            f"a single-CSV benchmark needs X/W/Z/A/Y plus ground-truth "
            f"Y0/Y1 (forced-counterfactual potential outcomes)."
        )

    csv_path = Path(csv_path)
    # Raw chamber exports carry a leading unnamed index column; canonical files
    # written by prepare_chamber_csv.py do not. Detect rather than assume — a
    # hardcoded index_col=0 silently swallows the first real column of any CSV
    # saved with index=False.
    df = pd.read_csv(csv_path)
    if len(df.columns) and str(df.columns[0]).startswith("Unnamed:"):
        df = df.drop(columns=df.columns[0])

    def cols(key: str) -> list[str]:
        names = _as_list(columns[key]) if key in columns else []
        unknown = [c for c in names if c not in df.columns]
        if unknown:
            raise ValueError(
                f"columns['{key}'] references unknown CSV column(s) {unknown}"
            )
        return names

    x_cols, w_cols, z_cols, u_cols = (cols(k) for k in ("X", "W", "Z", "U"))
    a_col, y_col, y0_col, y1_col = (columns[k] for k in ("A", "Y", "Y0", "Y1"))

    T = len(df)
    X = (
        df[x_cols].to_numpy(dtype=np.float32)
        if x_cols
        else np.zeros((T, 0), dtype=np.float32)
    )
    W = (
        df[w_cols].to_numpy(dtype=np.float32)
        if w_cols
        else np.zeros((T, 0), dtype=np.float32)
    )
    Z = (
        df[z_cols].to_numpy(dtype=np.float32)
        if z_cols
        else np.zeros((T, 0), dtype=np.float32)
    )
    U = (
        df[u_cols].to_numpy(dtype=np.float32)
        if u_cols
        else np.zeros((T, 0), dtype=np.float32)
    )
    A = df[a_col].to_numpy(dtype=np.float32)
    Y_obs = df[y_col].to_numpy(dtype=np.float32)
    Y0 = df[y0_col].to_numpy(dtype=np.float32)
    Y1 = df[y1_col].to_numpy(dtype=np.float32)

    if normalize_treatment:
        a_min, a_max = A.min(), A.max()
        if a_max > a_min:
            A = (A - a_min) / (a_max - a_min)

    if shuffle:
        perm = np.random.default_rng(seed).permutation(T)
        X, W, Z, U, A, Y_obs, Y0, Y1 = (
            arr[perm] for arr in (X, W, Z, U, A, Y_obs, Y0, Y1)
        )

    train_size = round(prop_train * T)

    def to_batched(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    return build_benchmark_dict(
        U=to_batched(U),
        X=to_batched(X),
        W=to_batched(W),
        Z=to_batched(Z),
        A=to_batched(A),
        Y_obs=to_batched(Y_obs),
        Y0=to_batched(Y0),
        Y1=to_batched(Y1),
        d_X=torch.tensor([X.shape[-1]]),
        d_W=torch.tensor([W.shape[-1]]),
        d_Z=torch.tensor([Z.shape[-1]]),
        train_size=train_size,
    )
