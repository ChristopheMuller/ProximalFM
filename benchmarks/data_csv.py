"""
data_csv.py — Build the standardized benchmark data dict from one or more
real-world CSVs (e.g. Causal Chambers exports), given a user-supplied column
to variable-block mapping.

Each CSV is one episode with its own context-only standardization and its own
train/test split. ``load_benchmark_from_csvs`` stacks those episodes, so the
ordinary benchmark reporting aggregates over files rather than concatenating
their rows and treating several physical runs as one dataset.

The mapping mirrors the one used in
``benchmarks/causal_chambers/Proximal Causal Inference on Causal Chambers.Rmd``
(``load_cc_data()``): ``U`` is the unobserved confounder (oracle-only, may be
omitted), ``X`` the covariates, ``Z`` the treatment-confounding proxy, ``W`` the
outcome-confounding proxy, ``A`` the treatment, ``Y`` the observed outcome, and
``Y0``/``Y1`` the
ground-truth potential outcomes (obtained on the chamber by physically forcing
the treatment to each extreme) that make ``true_ITEs`` — and therefore the
whole existing metrics stack — meaningful for real, non-synthetic data.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.data import build_benchmark_dict

_REQUIRED_KEYS = ("X", "W", "Z", "A", "Y", "Y0", "Y1")


def _as_list(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _resolve_feature_columns(
    configured, available: list[str], *, key: str
) -> list[str]:
    """Resolve exact names and shell-style patterns in one feature block.

    Matches follow CSV header order within each pattern. Overlapping patterns
    select a column only once, which makes spellings such as
    ``["demographic_*", "X_*"]`` safe to compose.
    """
    resolved: list[str] = []
    unmatched: list[str] = []
    for pattern in _as_list(configured):
        matches = [name for name in available if fnmatchcase(name, pattern)]
        if not matches:
            unmatched.append(pattern)
            continue
        resolved.extend(name for name in matches if name not in resolved)

    if unmatched:
        raise ValueError(
            f"columns['{key}'] references unknown CSV column(s) or unmatched "
            f"pattern(s) {unmatched}"
        )
    return resolved


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
    (lists or shell-style patterns such as ``"X_*"``),
    ``A``/``Y``/``Y0``/``Y1`` (single exact column names), and an optional
    ``U`` (list/pattern; omitted => zero-width, used only by oracle methods).
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
        if key not in columns:
            return []
        return _resolve_feature_columns(
            columns[key], [str(name) for name in df.columns], key=key
        )

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


def load_benchmark_from_csvs(
    csv_paths: list[str | Path],
    columns: dict,
    *,
    prop_train: float = 0.75,
    shuffle: bool = True,
    seed: int = 0,
    normalize_treatment: bool = True,
) -> dict:
    """Load several CSVs, treating each file as one benchmark episode.

    Files must have the same number of rows and mapped feature dimensions: the
    dense benchmark cache has one shared ``N_TRAIN``/``N_TEST`` and shape.
    Standardization happens before stacking, so statistics never cross files.
    """
    paths = [Path(path) for path in csv_paths]
    if not paths:
        raise ValueError("csv_paths must contain at least one CSV file.")

    episodes = [
        load_benchmark_from_csv(
            path,
            columns,
            prop_train=prop_train,
            shuffle=shuffle,
            seed=seed,
            normalize_treatment=normalize_treatment,
        )
        for path in paths
    ]
    expected = {key: episodes[0][key] for key in ("N_TRAIN", "N_TEST")}
    for path, episode in zip(paths[1:], episodes[1:]):
        mismatched = {
            key: (expected[key], episode[key])
            for key in expected
            if episode[key] != expected[key]
        }
        if mismatched:
            raise ValueError(
                f"{path}: all CSV files must produce the same train/test sizes; "
                f"got {mismatched}."
            )
        for key in ("U", "X", "W", "Z"):
            if episode[key].shape[2:] != episodes[0][key].shape[2:]:
                raise ValueError(
                    f"{path}: mapped {key} dimension differs across CSV files "
                    f"({episode[key].shape[-1]} != {episodes[0][key].shape[-1]})."
                )

    tensor_keys = ("U", "X", "Z", "W", "A", "Y", "d_X", "d_Z", "d_W")
    combined = {
        key: torch.cat([episode[key] for episode in episodes], dim=0)
        for key in tensor_keys
    }
    combined["true_ITEs"] = np.concatenate(
        [episode["true_ITEs"] for episode in episodes], axis=0
    )
    combined.update(
        {
            "true_CATEs": None,
            "cate_s2": None,
            "cate_K_eff": None,
            "N_TRAIN": expected["N_TRAIN"],
            "N_TEST": expected["N_TEST"],
            "N_BATCH": len(episodes),
            "episode_files": [str(path) for path in paths],
        }
    )
    return combined


def resolve_csv_paths(data_cfg: dict) -> list[Path]:
    """Resolve the mutually exclusive CSV input spellings in ``data:``."""
    supplied = [
        key
        for key in ("csv_path", "csv_paths", "csv_dir")
        if data_cfg.get(key) is not None
    ]
    if len(supplied) != 1:
        raise ValueError(
            "data.source: csv requires exactly one of csv_path, csv_paths, "
            "or csv_dir."
        )

    key = supplied[0]
    if key == "csv_path":
        paths = [Path(data_cfg[key])]
    elif key == "csv_paths":
        configured = data_cfg[key]
        if isinstance(configured, (str, Path)):
            raise ValueError(
                "data.csv_paths must be a list; use csv_path for one file."
            )
        paths = [Path(path) for path in configured]
    else:
        paths = sorted(Path(data_cfg[key]).glob(data_cfg.get("csv_pattern", "*.csv")))

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found for data.{key}={data_cfg.get(key)!r}."
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV input file(s) do not exist: {missing}")
    return paths
