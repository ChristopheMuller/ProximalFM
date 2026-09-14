"""Generic, deterministic CSV ingestion for semi-synthetic evaluation sets.

Rows are partitioned before any data-dependent preprocessing.  A fixed
calibration partition fits numeric imputation, clipping and standardization and
categorical encodings; episodes are sampled only from the disjoint evaluation
partition.  The benchmark independently standardizes each generated episode
from its context prefix.

The automatic X/U split is intentionally schema-driven rather than tied to any
named dataset.  Explicit column lists are supported for scientifically chosen
splits, but dropping a new CSV into ``datasets/`` is enough to make it usable.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import torch

Tensor = torch.Tensor
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(frozen=True)
class PreparedCSV:
    """Calibration/evaluation pools and their deterministic X/U partition."""

    path: Path
    digest: str
    calibration_values: Tensor
    evaluation_values: Tensor
    columns: tuple[str, ...]
    x_indices: tuple[int, ...]
    u_indices: tuple[int, ...]
    x_columns: tuple[str, ...]
    u_columns: tuple[str, ...]
    calibration_row_indices: tuple[int, ...]
    evaluation_row_indices: tuple[int, ...]
    partition_seed: int
    preprocessing: dict[str, dict]

    @property
    def X_cal(self) -> Tensor:
        return self.calibration_values[:, self.x_indices]

    @property
    def U_cal(self) -> Tensor:
        return self.calibration_values[:, self.u_indices]

    @property
    def X_eval(self) -> Tensor:
        return self.evaluation_values[:, self.x_indices]

    @property
    def U_eval(self) -> Tensor:
        return self.evaluation_values[:, self.u_indices]

    @property
    def values(self) -> Tensor:
        """Evaluation values retained as a compatibility alias."""
        return self.evaluation_values

    @property
    def X(self) -> Tensor:
        """Evaluation X retained as a compatibility alias."""
        return self.X_eval

    @property
    def U(self) -> Tensor:
        """Evaluation U retained as a compatibility alias."""
        return self.U_eval


def discover_csvs(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Path]:
    """Return every CSV in ``data_dir`` in stable filename order."""
    return sorted(Path(data_dir).glob("*.csv"), key=lambda p: p.name.casefold())


def resolve_csv(name: str | Path, data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Resolve a path, filename, or unique stem under ``data_dir``."""
    requested = Path(name)
    if requested.is_file():
        return requested.resolve()

    root = Path(data_dir)
    direct = root / requested.name
    if direct.is_file():
        return direct.resolve()
    if requested.suffix.lower() != ".csv":
        direct = root / f"{requested.name}.csv"
        if direct.is_file():
            return direct.resolve()

    matches = [p for p in discover_csvs(root) if p.stem == requested.stem]
    if len(matches) == 1:
        return matches[0].resolve()
    available = ", ".join(p.name for p in discover_csvs(root)) or "<none>"
    raise FileNotFoundError(f"CSV {name!r} not found; available in {root}: {available}")


def _parse_columns(value: Optional[str | Sequence[str]]) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _stable_order(columns: Iterable[str], split_seed: int) -> list[str]:
    def key(column: str) -> tuple[str, str]:
        token = f"{int(split_seed)}\0{column}".encode("utf-8")
        return hashlib.sha256(token).hexdigest(), column

    return sorted(columns, key=key)


def _partition_seed(split_seed: int, source_name: str) -> int:
    token = f"{int(split_seed)}\0{source_name}\0row_partition".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") & (2**63 - 1)


def _partition_rows(
    n_rows: int, n_calibration: int, split_seed: int, source_name: str
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if n_calibration < 1:
        raise ValueError("n_calibration must be >= 1")
    if n_rows <= n_calibration:
        raise ValueError(
            f"{source_name}: has {n_rows} rows but n_calibration={n_calibration} "
            "leaves no evaluation rows"
        )
    seed = _partition_seed(split_seed, source_name)
    order = torch.randperm(n_rows, generator=torch.Generator().manual_seed(seed))
    calibration = tuple(sorted(int(i) for i in order[:n_calibration]))
    evaluation = tuple(sorted(int(i) for i in order[n_calibration:]))
    return calibration, evaluation, seed


def _standardize(
    calibration: Tensor, evaluation: Tensor
) -> tuple[Tensor, Tensor, float, float]:
    mean = calibration.mean()
    std = calibration.std(unbiased=False)
    if not torch.isfinite(calibration).all() or not torch.isfinite(std) or std <= 1e-12:
        raise ValueError("calibration column is non-finite or constant")
    return (
        ((calibration - mean) / std).float(),
        ((evaluation - mean) / std).float(),
        float(mean),
        float(std),
    )


def _encode_column(
    series: pd.Series,
    calibration_rows: tuple[int, ...],
    evaluation_rows: tuple[int, ...],
) -> Optional[tuple[Tensor, Tensor, dict]]:
    """Fit one column on calibration rows and transform both partitions."""
    calibration_series = series.iloc[list(calibration_rows)]
    numeric_cal = pd.to_numeric(calibration_series, errors="coerce")
    non_missing = int(calibration_series.notna().sum())
    numeric_share = float(numeric_cal.notna().sum()) / max(non_missing, 1)
    if pd.api.types.is_numeric_dtype(series) or numeric_share >= 0.95:
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        finite = numeric.replace([float("inf"), float("-inf")], pd.NA)
        cal_finite = finite.iloc[list(calibration_rows)]
        fill = cal_finite.median()
        if pd.isna(fill):
            return None
        filled = finite.fillna(fill)
        cal_filled = filled.iloc[list(calibration_rows)]
        clip_low = float(cal_filled.quantile(0.005))
        clip_high = float(cal_filled.quantile(0.995))
        clipped = filled.clip(lower=clip_low, upper=clip_high)
        calibration = torch.as_tensor(
            clipped.iloc[list(calibration_rows)].to_numpy(dtype="float64")
        )
        evaluation = torch.as_tensor(
            clipped.iloc[list(evaluation_rows)].to_numpy(dtype="float64")
        )
        try:
            calibration, evaluation, mean, std = _standardize(calibration, evaluation)
        except ValueError:
            return None
        metadata = {
            "kind": "numeric",
            "imputation_median": float(fill),
            "clip_low": clip_low,
            "clip_high": clip_high,
            "mean": mean,
            "std": std,
        }
    else:
        text = series.astype("string").fillna("<MISSING>")
        cal_text = text.iloc[list(calibration_rows)]
        categories = sorted(cal_text.unique().tolist())
        mapping = {category: i for i, category in enumerate(categories)}
        encoded = torch.tensor(
            [mapping.get(value, -1) for value in text], dtype=torch.float64
        )
        calibration = encoded[list(calibration_rows)]
        evaluation = encoded[list(evaluation_rows)]
        try:
            calibration, evaluation, mean, std = _standardize(calibration, evaluation)
        except ValueError:
            return None
        metadata = {
            "kind": "categorical",
            "categories": categories,
            "unknown_code": -1,
            "mean": mean,
            "std": std,
        }
    if not torch.isfinite(evaluation).all():
        return None
    return calibration, evaluation, metadata


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a CSV after checking its unmangled header for duplicate names."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(65_536)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error as error:
            raise ValueError(
                f"{path.name}: could not detect a CSV delimiter"
            ) from error
        header = next(csv.reader(f, dialect), None)
    if not header:
        raise ValueError(f"{path.name}: CSV has no header")
    names = [str(value).strip() for value in header]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate column names: {duplicates}")
    return pd.read_csv(path, sep=dialect.delimiter, encoding="utf-8-sig")


def load_csv(
    name: str | Path,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    x_columns: Optional[str | Sequence[str]] = None,
    u_columns: Optional[str | Sequence[str]] = None,
    drop_columns: Optional[str | Sequence[str]] = None,
    n_features_x: int = 7,
    n_features_u: int = 5,
    n_calibration: int = 1000,
    split_seed: int = 0,
) -> PreparedCSV:
    """Load any tabular CSV and deterministically split its features into X/U.

    Exactly ``n_features_x`` and ``n_features_u`` usable columns are required.
    When explicit column lists are omitted, that many columns are selected by a
    stable hash order. Explicit lists must be supplied together and their
    lengths must match the requested dimensions exactly.
    """
    path = resolve_csv(name, data_dir)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = _read_frame(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    calibration_rows, evaluation_rows, partition_seed = _partition_rows(
        len(frame), n_calibration, split_seed, path.name
    )

    drops = set(_parse_columns(drop_columns) or [])
    missing_drops = drops - set(frame.columns)
    if missing_drops:
        raise ValueError(
            f"{path.name}: unknown --drop_columns: {sorted(missing_drops)}"
        )

    encoded: dict[str, tuple[Tensor, Tensor, dict]] = {}
    for column in frame.columns:
        if column not in drops:
            values = _encode_column(frame[column], calibration_rows, evaluation_rows)
            if values is not None:
                encoded[column] = values
    if n_features_x < 1 or n_features_u < 1:
        raise ValueError("n_features_x and n_features_u must both be >= 1")
    n_required = int(n_features_x) + int(n_features_u)
    if len(encoded) < n_required:
        raise ValueError(
            f"{path.name}: requested n_features_x={n_features_x} and "
            f"n_features_u={n_features_u} ({n_required} total), but only "
            f"{len(encoded)} non-constant usable columns are available"
        )

    explicit_x = _parse_columns(x_columns)
    explicit_u = _parse_columns(u_columns)
    if (explicit_x is None) != (explicit_u is None):
        raise ValueError("--x_columns and --u_columns must be supplied together")

    if explicit_x is not None and explicit_u is not None:
        if len(explicit_x) != len(set(explicit_x)) or len(explicit_u) != len(
            set(explicit_u)
        ):
            raise ValueError("X/U column lists must not contain duplicates")
        overlap = set(explicit_x) & set(explicit_u)
        unknown = (set(explicit_x) | set(explicit_u)) - set(encoded)
        if overlap:
            raise ValueError(f"X/U column lists overlap: {sorted(overlap)}")
        if unknown:
            raise ValueError(f"unusable or unknown X/U columns: {sorted(unknown)}")
        if not explicit_x or not explicit_u:
            raise ValueError("X and U must each contain at least one column")
        if len(explicit_x) != n_features_x or len(explicit_u) != n_features_u:
            raise ValueError(
                f"explicit lists must contain exactly n_features_x={n_features_x} "
                f"and n_features_u={n_features_u} columns; got "
                f"{len(explicit_x)} and {len(explicit_u)}"
            )
        x_names, u_names = explicit_x, explicit_u
        selected = list(dict.fromkeys([*x_names, *u_names]))
    else:
        selected = _stable_order(encoded, split_seed)[:n_required]
        ordered = _stable_order(encoded, split_seed)
        u_names = ordered[::2][:n_features_u]
        x_names = ordered[1::2][:n_features_x]
        if len(u_names) != n_features_u or len(x_names) != n_features_x:
            raise ValueError(
                f"{path.name}: insufficient columns for stable alternating X/U selection"
            )
        selected = [*u_names, *x_names]

    calibration_values = torch.stack([encoded[c][0] for c in selected], dim=1)
    evaluation_values = torch.stack([encoded[c][1] for c in selected], dim=1)
    positions = {column: i for i, column in enumerate(selected)}
    return PreparedCSV(
        path=path,
        digest=digest,
        calibration_values=calibration_values,
        evaluation_values=evaluation_values,
        columns=tuple(selected),
        x_indices=tuple(positions[c] for c in x_names),
        u_indices=tuple(positions[c] for c in u_names),
        x_columns=tuple(x_names),
        u_columns=tuple(u_names),
        calibration_row_indices=calibration_rows,
        evaluation_row_indices=evaluation_rows,
        partition_seed=partition_seed,
        preprocessing={column: encoded[column][2] for column in selected},
    )


def dimension_view(source: PreparedCSV, dimension: int) -> PreparedCSV:
    """Take role-specific prefixes without repeating source preprocessing."""
    if not 1 <= dimension <= min(len(source.x_columns), len(source.u_columns)):
        raise ValueError("dimension exceeds prepared source feature capacity")
    x_names = source.x_columns[:dimension]
    u_names = source.u_columns[:dimension]
    names = (*u_names, *x_names)
    positions = [source.columns.index(name) for name in names]
    return replace(
        source,
        calibration_values=source.calibration_values[:, positions],
        evaluation_values=source.evaluation_values[:, positions],
        columns=names,
        u_indices=tuple(range(dimension)),
        x_indices=tuple(range(dimension, 2 * dimension)),
        x_columns=x_names,
        u_columns=u_names,
        preprocessing={name: source.preprocessing[name] for name in names},
    )
