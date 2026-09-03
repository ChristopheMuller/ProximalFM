"""Generic, deterministic CSV ingestion for semi-synthetic evaluation sets.

Every usable source column is converted to one finite floating-point column.
Numeric columns are median-imputed; categorical columns use stable sorted
category codes.  The resulting columns are standardized against the complete
CSV, which is the fixed empirical population from which episodes are sampled.
The benchmark independently standardizes each episode from its context prefix.

The automatic X/U split is intentionally schema-driven rather than tied to any
named dataset.  Explicit column lists are supported for scientifically chosen
splits, but dropping a new CSV into ``datasets/`` is enough to make it usable.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import torch

Tensor = torch.Tensor
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(frozen=True)
class PreparedCSV:
    """Finite standardized rows and their deterministic X/U partition."""

    path: Path
    digest: str
    values: Tensor
    columns: tuple[str, ...]
    x_indices: tuple[int, ...]
    u_indices: tuple[int, ...]
    x_columns: tuple[str, ...]
    u_columns: tuple[str, ...]

    @property
    def X(self) -> Tensor:
        return self.values[:, self.x_indices]

    @property
    def U(self) -> Tensor:
        return self.values[:, self.u_indices]


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


def _encode_column(series: pd.Series) -> Optional[Tensor]:
    """Encode one raw column to one float column, or drop it if constant."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    numeric_share = float(numeric.notna().sum()) / max(non_missing, 1)
    if pd.api.types.is_numeric_dtype(series) or numeric_share >= 0.95:
        values = numeric.astype(float)
        finite = values.replace([float("inf"), float("-inf")], pd.NA)
        fill = finite.median()
        if pd.isna(fill):
            return None
        array = torch.as_tensor(finite.fillna(fill).to_numpy(dtype="float64"))
    else:
        text = series.astype("string").fillna("<MISSING>")
        categories = sorted(text.unique().tolist())
        mapping = {category: i for i, category in enumerate(categories)}
        array = torch.tensor([mapping[value] for value in text], dtype=torch.float64)

    std = array.std(unbiased=False)
    if not torch.isfinite(array).all() or not torch.isfinite(std) or std <= 1e-12:
        return None
    return ((array - array.mean()) / std).float()


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
    max_features: int = 12,
    u_fraction: float = 0.4,
    split_seed: int = 0,
) -> PreparedCSV:
    """Load any tabular CSV and deterministically split its features into X/U.

    When ``x_columns`` and ``u_columns`` are omitted, at most ``max_features``
    usable columns are selected by a stable hash order, then ``u_fraction`` of
    them become U.  Explicit X and U lists must be supplied together and are
    never truncated.
    """
    path = resolve_csv(name, data_dir)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = _read_frame(path)
    frame.columns = [str(c).strip() for c in frame.columns]

    drops = set(_parse_columns(drop_columns) or [])
    missing_drops = drops - set(frame.columns)
    if missing_drops:
        raise ValueError(
            f"{path.name}: unknown --drop_columns: {sorted(missing_drops)}"
        )

    encoded: dict[str, Tensor] = {}
    for column in frame.columns:
        if column not in drops:
            values = _encode_column(frame[column])
            if values is not None:
                encoded[column] = values
    if len(encoded) < 2:
        raise ValueError(
            f"{path.name}: need at least two non-constant usable columns to split X/U; "
            f"found {len(encoded)}"
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
        x_names, u_names = explicit_x, explicit_u
        selected = list(dict.fromkeys([*x_names, *u_names]))
    else:
        if max_features < 2:
            raise ValueError("max_features must be >= 2")
        if not 0.0 < u_fraction < 1.0:
            raise ValueError("u_fraction must lie strictly between 0 and 1")
        selected = _stable_order(encoded, split_seed)[: int(max_features)]
        n_u = max(1, min(len(selected) - 1, round(len(selected) * u_fraction)))
        u_names = selected[:n_u]
        x_names = selected[n_u:]

    values = torch.stack([encoded[c] for c in selected], dim=1)
    positions = {column: i for i, column in enumerate(selected)}
    return PreparedCSV(
        path=path,
        digest=digest,
        values=values,
        columns=tuple(selected),
        x_indices=tuple(positions[c] for c in x_names),
        u_indices=tuple(positions[c] for c in u_names),
        x_columns=tuple(x_names),
        u_columns=tuple(u_names),
    )
