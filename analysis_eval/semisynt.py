"""Validated loading and summaries for the paired semisynthetic matrix."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LEVELS = ("low", "high")
FAMILIES = ("linear", "nonlinear")
CELL_ORDER = tuple(
    (family, confounding, proxy)
    for family in FAMILIES
    for confounding in LEVELS
    for proxy in LEVELS
)
REQUIRED_COLUMNS = {
    "method",
    "episode",
    "n_ctx",
    "target",
    "estimand",
    "is_reference",
    "failed",
    "npehe",
}


@dataclass(frozen=True)
class MatrixCell:
    case: str
    family: str
    confounding: str
    proxy: str
    results_path: Path


def _case_factors(case: str) -> tuple[str, str, str]:
    """Parse the strict case names emitted by the format-v4 recipe."""
    parts = case.split("_")
    if len(parts) != 5 or parts[1] != "conf" or parts[3] != "proxy":
        raise ValueError(
            f"case {case!r} does not match "
            "<linear|nonlinear>_conf_<low|high>_proxy_<low|high>"
        )
    family, confounding, proxy = parts[0], parts[2], parts[4]
    if family not in FAMILIES or confounding not in LEVELS or proxy not in LEVELS:
        raise ValueError(f"case {case!r} has unknown factorial levels")
    return family, confounding, proxy


def _read_cases(manifest_path: Path) -> list[dict]:
    """Read the case records from one matrix manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{manifest_path}: 'cases' must be a list")
    if not all(isinstance(item, dict) for item in cases):
        raise ValueError(f"{manifest_path}: every item in 'cases' must be an object")
    return cases


def _dimension_from_root(path: Path) -> int | None:
    """Return the dimension encoded by a ``d-N`` directory name, if any."""
    match = re.fullmatch(r"d-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _matrix_cases(
    matrix_root: Path, *, dimension: int | None
) -> tuple[Path, list[dict], int | None]:
    """Resolve legacy and per-dimension matrix layouts to one cell root."""
    matrix_root = Path(matrix_root)
    manifest_path = matrix_root / "manifest.json"

    # A dimension subdirectory deliberately has no manifest of its own.  Let
    # callers keep using it directly, while reading its declaration from the
    # build-root manifest.
    if not manifest_path.exists():
        root_dimension = _dimension_from_root(matrix_root)
        parent_manifest = matrix_root.parent / "manifest.json"
        if root_dimension is not None and parent_manifest.exists():
            if dimension is not None and dimension != root_dimension:
                raise ValueError(
                    f"--dim {dimension} conflicts with dimension root {matrix_root}"
                )
            return _matrix_cases(matrix_root.parent, dimension=root_dimension)
        raise FileNotFoundError(f"matrix manifest not found: {manifest_path}")

    cases = _read_cases(manifest_path)
    has_dimension = ["dimension" in item for item in cases]
    if any(has_dimension) and not all(has_dimension):
        raise ValueError(
            f"{manifest_path}: matrix cases inconsistently declare dimensions"
        )
    if not any(has_dimension):
        if dimension is not None:
            raise ValueError(
                f"{manifest_path}: --dim is only valid for a per-dimension matrix"
            )
        return matrix_root, cases, None

    values = [item["dimension"] for item in cases]
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError(f"{manifest_path}: case dimensions must be positive integers")
    dimensions = set(values)
    selected_dimension = 5 if dimension is None else dimension
    if selected_dimension not in dimensions:
        available = ", ".join(str(value) for value in sorted(dimensions))
        raise ValueError(
            f"dimension {selected_dimension} is not in {manifest_path}; "
            f"available dimensions: {available}"
        )
    cell_root = matrix_root / f"d-{selected_dimension}"
    if not cell_root.is_dir():
        raise FileNotFoundError(
            f"dimension directory missing for dimension {selected_dimension}: {cell_root}"
        )
    return (
        cell_root,
        [item for item in cases if item["dimension"] == selected_dimension],
        selected_dimension,
    )


def is_matrix_root(matrix_root: Path) -> bool:
    """Whether ``matrix_root`` names a legacy or per-dimension case matrix."""
    try:
        _matrix_cases(matrix_root, dimension=None)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return True


def discover_cells(
    matrix_root: Path,
    *,
    selected_case: str | None = None,
    dimension: int | None = None,
) -> list[MatrixCell]:
    """Discover the full matrix or one selected case from its manifest.

    The default path validates the complete eight-case matrix. A selected case
    only needs its own benchmark results, which allows partially completed
    benchmark roots to be used for single-case plots.
    """
    matrix_root, cases, _ = _matrix_cases(matrix_root, dimension=dimension)
    if selected_case is not None:
        _case_factors(selected_case)

    cells = []
    seen = set()
    for item in cases:
        case = item.get("case")
        if not isinstance(case, str):
            raise ValueError("matrix case names must be strings")
        family, confounding, proxy = _case_factors(case)
        factors = (family, confounding, proxy)
        if factors in seen:
            raise ValueError(f"duplicate factorial cell {factors}")
        seen.add(factors)
        if selected_case is not None and case != selected_case:
            continue
        results_path = matrix_root / case / "bench" / "results" / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(
                f"benchmark results missing for {case}: {results_path}"
            )
        cells.append(MatrixCell(case, family, confounding, proxy, results_path))

    if selected_case is None:
        missing = set(CELL_ORDER) - seen
        extra = seen - set(CELL_ORDER)
        if missing or extra:
            raise ValueError(
                f"expected the complete 2x2x2 matrix; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
    elif not cells:
        raise ValueError(
            f"case {selected_case!r} is not declared in the matrix manifest"
        )
    order = {factors: i for i, factors in enumerate(CELL_ORDER)}
    return sorted(
        cells,
        key=lambda cell: order[(cell.family, cell.confounding, cell.proxy)],
    )


def load_cate_results(
    matrix_root: Path,
    *,
    selected_case: str | None = None,
    dimension: int | None = None,
) -> pd.DataFrame:
    """Load successful competitor CATE(X) rows from all or one matrix cell."""
    frames = []
    for cell in discover_cells(
        matrix_root, selected_case=selected_case, dimension=dimension
    ):
        frame = pd.read_csv(cell.results_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{cell.results_path}: missing columns {sorted(missing)}")
        frame = frame[
            frame["target"].eq("cate")
            & frame["estimand"].eq("cate(x)")
            & ~frame["is_reference"].astype(bool)
            & ~frame["failed"].astype(bool)
        ].copy()
        frame["npehe"] = pd.to_numeric(frame["npehe"], errors="coerce")
        frame = frame[np.isfinite(frame["npehe"])]
        frame["case"] = cell.case
        frame["family"] = cell.family
        frame["confounding"] = cell.confounding
        frame["proxy"] = cell.proxy
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    keys = ["case", "method", "episode", "n_ctx"]
    duplicates = result.duplicated(keys, keep=False)
    if duplicates.any():
        sample = result.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(f"duplicate benchmark rows, e.g. {sample}")
    if result.empty:
        raise ValueError("the matrix contains no finite competitor CATE(X) scores")
    return result


def summarize_learning_curves(
    frame: pd.DataFrame, methods: Iterable[str] | None = None
) -> pd.DataFrame:
    """Median and interquartile nPEHE by cell, method, and context size."""
    if methods is not None:
        methods = list(methods)
        absent = set(methods) - set(frame["method"])
        if absent:
            raise ValueError(f"requested methods absent from results: {sorted(absent)}")
        frame = frame[frame["method"].isin(methods)]
    group = ["case", "family", "confounding", "proxy", "method", "n_ctx"]
    summary = (
        frame.groupby(group, observed=True)["npehe"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            std="std",
            n="count",
        )
        .reset_index()
        .sort_values(group)
        .reset_index(drop=True)
    )
    summary["ste"] = summary["std"] / summary["n"].pow(0.5)
    return summary


def complete_learning_curve_methods(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split methods into complete and incomplete factorial/context coverage.

    Completeness concerns visible curve points, not the number of episodes at
    each point. The renderer separately validates that every selected point is
    present; this helper lets the default CLI exclude partially run methods
    while reporting them instead of failing the entire finished matrix.
    """
    cells = set(frame["case"].unique())
    contexts = set(frame["n_ctx"].unique())
    expected = {(case, n_ctx) for case in cells for n_ctx in contexts}
    complete, incomplete = [], []
    for method, group in frame.groupby("method", sort=False):
        observed = set(group[["case", "n_ctx"]].itertuples(index=False, name=None))
        (complete if observed == expected else incomplete).append(method)
    return complete, incomplete


def summarize_speed_rank(
    frame: pd.DataFrame,
    methods: Iterable[str],
    *,
    context_size: int | None = None,
    by_confounding: bool = True,
) -> pd.DataFrame:
    """Paired nPEHE ranks and mean episode latency at one context size.

    Ranks are computed within each ``(case, episode)`` block and only then
    summarized. By default, low- and high-confounding cases are summarized
    independently across their four linearity/proxy cells. Set
    ``by_confounding=False`` to recover one pooled eight-cell summary. Timing is
    averaged over the exact same complete blocks used for the ranks.

    ``time`` is the measured end-to-end fit-and-predict latency.
    ``time_post_tune`` is the corresponding latency after removing explicit
    hyperparameter-selection work. It is optional so historical results remain
    plottable; missing post-selection measurements stay missing.
    """
    required = {
        "time",
        "npehe",
        "case",
        "episode",
        "method",
        "n_ctx",
        "confounding",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"speed/rank summary requires columns {sorted(missing)}")

    methods = list(methods)
    if not methods:
        raise ValueError("speed/rank summary requires at least one method")
    absent = set(methods) - set(frame["method"])
    if absent:
        raise ValueError(f"requested methods absent from results: {sorted(absent)}")

    available_contexts = sorted(int(value) for value in frame["n_ctx"].unique())
    if context_size is None:
        context_size = available_contexts[-1]
    if context_size not in available_contexts:
        raise ValueError(
            f"context size {context_size} is absent; available={available_contexts}"
        )

    work = frame[frame["n_ctx"].eq(context_size) & frame["method"].isin(methods)].copy()
    rows = []
    groups = (
        work.groupby("confounding", sort=False)
        if by_confounding
        else [("pooled", work)]
    )
    for confounding, confounding_frame in groups:
        all_cases = set(confounding_frame["case"].unique())
        incomplete = {}
        for method in methods:
            observed_cases = set(
                confounding_frame.loc[confounding_frame["method"].eq(method), "case"]
            )
            if observed_cases != all_cases:
                incomplete[method] = sorted(all_cases - observed_cases)
        if incomplete:
            raise ValueError(
                f"methods lack {confounding}-confounding cases at the selected "
                "context size: "
                + "; ".join(
                    f"{method}: {cases}" for method, cases in incomplete.items()
                )
            )

        block_columns = ["case", "episode"]
        scores = confounding_frame.pivot(
            index=block_columns, columns="method", values="npehe"
        )
        scores = scores.reindex(columns=methods).dropna(axis=0, how="any")
        if scores.empty:
            raise ValueError(
                "no complete (case, episode) blocks remain for "
                f"{confounding}-confounding ranks"
            )
        ranks = scores.rank(axis=1, method="average", ascending=True)
        complete_blocks = scores.index
        timed = (
            confounding_frame.set_index(block_columns)
            .loc[complete_blocks]
            .reset_index()
        )

        for method in methods:
            method_ranks = ranks[method]
            method_times = pd.to_numeric(
                timed.loc[timed["method"].eq(method), "time"], errors="coerce"
            )
            method_times = method_times[np.isfinite(method_times) & method_times.gt(0)]
            if method_times.empty:
                raise ValueError(f"method {method!r} has no finite positive timings")

            post_time = float("nan")
            if "time_post_tune" in timed:
                post = pd.to_numeric(
                    timed.loc[timed["method"].eq(method), "time_post_tune"],
                    errors="coerce",
                )
                post = post[np.isfinite(post) & post.gt(0)]
                if not post.empty:
                    post_time = float(post.mean())

            rows.append(
                {
                    "confounding": confounding,
                    "method": method,
                    "n_ctx": context_size,
                    "rank_median": float(method_ranks.median()),
                    "rank_q25": float(method_ranks.quantile(0.25)),
                    "rank_q75": float(method_ranks.quantile(0.75)),
                    "time_mean": float(method_times.mean()),
                    "time_post_tune_mean": post_time,
                    "n_blocks": len(complete_blocks),
                }
            )
    return pd.DataFrame(rows)


def summarize_error_decomposition(
    frame: pd.DataFrame,
    methods: Iterable[str],
    *,
    aggregation: str = "median",
) -> pd.DataFrame:
    """Aggregate the two additive components of squared CATE nPEHE.

    Episode-wise, ``npehe² = ate_bias_norm² + (1 - shape_r2)``. With ``mean``
    the aggregated coordinates preserve this identity exactly. ``median`` is
    the robust default for visualisation: its coordinates are the separate
    component medians, so their sum defines a robust decomposed score but need
    not equal median episode nPEHE².
    """
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")
    required = {"ate_bias_norm", "shape_r2", "npehe"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"decomposition requires columns {sorted(missing)}")
    methods = list(methods)
    absent = set(methods) - set(frame["method"])
    if absent:
        raise ValueError(f"requested methods absent from results: {sorted(absent)}")

    work = frame[frame["method"].isin(methods)].copy()
    work["level_error"] = work["ate_bias_norm"].pow(2)
    work["shape_error"] = 1.0 - work["shape_r2"]
    work["npehe_sq"] = work["npehe"].pow(2)
    group = ["case", "family", "confounding", "proxy", "method", "n_ctx"]
    summary = (
        work.groupby(group, observed=True)
        .agg(
            level_error=("level_error", aggregation),
            shape_error=("shape_error", aggregation),
            npehe_sq=("npehe_sq", aggregation),
            n=("episode", "count"),
        )
        .reset_index()
        .sort_values(group)
        .reset_index(drop=True)
    )
    summary["decomposed_npehe"] = np.sqrt(
        np.maximum(summary["level_error"] + summary["shape_error"], 0.0)
    )
    summary["identity_gap"] = (
        summary["level_error"] + summary["shape_error"] - summary["npehe_sq"]
    )
    if aggregation == "mean" and not np.allclose(
        summary["identity_gap"], 0.0, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("mean error decomposition does not preserve nPEHE identity")
    return summary
