"""Plot per-dataset Causal Chambers method ranks at a fixed context size.

Run from the repository root::

    uv run python -m analysis_eval.causal_chambers_rank

The figure reads the benchmark runner's long-form ``results.csv``.  It first
computes one RMSE per method and condition at the selected context size, ranks
methods within each condition (1 is best), and renders the rank matrix with
columns ordered by mechanism family, confounding strength, and proxy
informativeness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .design import (
    COLORS,
    FULL_PAGE_WIDTH_IN,
    apply_style,
    method_style,
    ordered_methods,
)
from .plotting import plt, save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "causal_chambers" / "results"
DEFAULT_RESULTS_CSV = _RESULTS_ROOT / "results.csv"
DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "causal_chambers_rank_grid"
DEFAULT_CONTEXT_SIZE = 1000
REQUIRED_COLUMNS = frozenset({"condition", "permutation", "n_ctx", "method", "error"})
FAMILY_ORDER = ("linear", "non_linear")
CONFOUNDING_ORDER = ("strong", "weak")
PROXY_ORDER = ("low", "mid", "high")


@dataclass(frozen=True)
class DatasetCondition:
    """Parsed factors for one Causal Chambers dataset condition."""

    name: str
    family: str
    confounding: str
    proxy: str


@dataclass(frozen=True)
class RankGrid:
    """Rank matrix and its ordered condition metadata."""

    ranks: pd.DataFrame
    mean_rank: pd.Series
    conditions: tuple[DatasetCondition, ...]
    context_size: int


_CONDITION_RE = re.compile(
    r"(?P<confounding>strong|weak)_confounding_"
    r"(?P<family>linear|non_linear)_proxy_"
    r"(?P<proxy>low|mid|high)(?:_|$)"
)


def _parse_condition(name: str) -> DatasetCondition:
    match = _CONDITION_RE.search(name)
    if match is None:
        raise ValueError(
            f"could not parse Causal Chambers condition {name!r}; expected "
            "..._<strong|weak>_confounding_<linear|non_linear>_proxy_"
            "<low|mid|high>..."
        )
    return DatasetCondition(name=name, **match.groupdict())


def _condition_order(condition: DatasetCondition) -> tuple[int, int, int]:
    return (
        FAMILY_ORDER.index(condition.family),
        CONFOUNDING_ORDER.index(condition.confounding),
        PROXY_ORDER.index(condition.proxy),
    )


def _validate_results(
    frame: pd.DataFrame, *, source: Path | None = None
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        location = f"{source}: " if source is not None else ""
        raise ValueError(f"{location}results are missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError("Causal Chambers results are empty")

    result = frame[["condition", "permutation", "n_ctx", "method", "error"]].copy()
    for column in ("condition", "method", "permutation"):
        if result[column].isna().any():
            raise ValueError(f"Causal Chambers results contain missing {column} values")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"Causal Chambers results contain blank {column} values")
    for column in ("n_ctx", "error"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["n_ctx", "error"]].isna().any().any():
        raise ValueError("Causal Chambers results contain non-numeric values")

    contexts = result["n_ctx"].to_numpy(dtype=float)
    errors = result["error"].to_numpy(dtype=float)
    if not np.isfinite(contexts).all() or not np.isfinite(errors).all():
        raise ValueError("Causal Chambers results contain non-finite values")
    if (contexts <= 0).any() or not np.equal(contexts, np.floor(contexts)).all():
        raise ValueError("context sizes must be positive integers")
    result["n_ctx"] = result["n_ctx"].astype(int)

    duplicates = result.duplicated(
        ["condition", "permutation", "n_ctx", "method"], keep=False
    )
    if duplicates.any():
        sample = (
            result.loc[duplicates, ["condition", "permutation", "n_ctx", "method"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(
            f"duplicate condition/permutation/context/method rows, e.g. {sample}"
        )
    return result


def load_results(results_csv: Path) -> pd.DataFrame:
    """Load the validated long-form Causal Chambers benchmark results."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Causal Chambers results not found: {results_csv}")
    return _validate_results(pd.read_csv(results_csv), source=results_csv)


def build_rank_grid(
    results: pd.DataFrame,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    methods: Sequence[str] | None = None,
) -> RankGrid:
    """Build a per-condition method-rank matrix from long-form results."""
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    results = _validate_results(results)
    selected = results[results["n_ctx"].eq(context_size)].copy()
    if selected.empty:
        available = sorted(int(value) for value in results["n_ctx"].unique())
        raise ValueError(
            f"context size {context_size} is absent from results; available={available}"
        )

    parsed = {}
    parsed_factors = set()
    for name in selected["condition"].unique():
        condition = _parse_condition(name)
        key = (condition.family, condition.confounding, condition.proxy)
        if key in parsed_factors:
            raise ValueError(f"duplicate condition factors in results: {key}")
        parsed[name] = condition
        parsed_factors.add(key)
    expected = {
        (family, confounding, proxy)
        for family in FAMILY_ORDER
        for confounding in CONFOUNDING_ORDER
        for proxy in PROXY_ORDER
    }
    observed = set(parsed.values())
    observed_factors = {
        (condition.family, condition.confounding, condition.proxy)
        for condition in observed
    }
    if observed_factors != expected:
        raise ValueError(
            "Causal Chambers results must contain the complete 2x2x3 condition "
            f"grid; missing={sorted(expected - observed_factors)}, "
            f"extra={sorted(observed_factors - expected)}"
        )
    conditions = tuple(sorted(parsed.values(), key=_condition_order))

    available = set(selected["method"])
    chosen = list(available) if methods is None else list(methods)
    absent = set(chosen) - available
    if absent:
        raise ValueError(f"requested methods absent from results: {sorted(absent)}")
    if not chosen:
        raise ValueError("rank grid has no methods to plot")
    ordered = ordered_methods(chosen)

    permutations = tuple(selected["permutation"].drop_duplicates())
    expected_rows = {
        (condition.name, permutation, method)
        for condition in conditions
        for permutation in permutations
        for method in ordered
    }
    observed_rows = set(
        selected[["condition", "permutation", "method"]].itertuples(
            index=False, name=None
        )
    )
    missing_rows = expected_rows - observed_rows
    if missing_rows:
        raise ValueError(
            "rank grid requires every selected method on every condition and "
            f"permutation; missing e.g. {sorted(missing_rows)[:5]}"
        )

    # With one permutation this is exactly |error|. With multiple permutations,
    # compute the per-condition RMSE first, then rank the methods on that score.
    scores = (
        selected[selected["method"].isin(ordered)]
        .groupby(["condition", "method"], sort=False)["error"]
        .agg(lambda values: float(np.sqrt(np.mean(np.square(values.to_numpy())))))
        .rename("rmse")
        .unstack("method")
        .reindex(index=[condition.name for condition in conditions], columns=ordered)
    )
    if scores.isna().any().any():
        raise ValueError("rank grid contains missing per-condition RMSE values")

    ranks = scores.rank(axis=1, method="average", ascending=True).T
    mean_rank = ranks.mean(axis=1)
    order_index = {method: index for index, method in enumerate(ordered)}
    row_order = sorted(
        ordered, key=lambda method: (mean_rank[method], order_index[method])
    )
    ranks = ranks.reindex(row_order)
    mean_rank = mean_rank.reindex(row_order)
    ranks.index.name = "method"
    ranks.columns.name = "condition"
    mean_rank.name = "mean_rank"
    return RankGrid(ranks, mean_rank, conditions, context_size)


def load_rank_grid(
    results_csv: Path,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    methods: Sequence[str] | None = None,
) -> RankGrid:
    """Load long-form results and construct the rank grid."""
    return build_rank_grid(
        load_results(results_csv), context_size=context_size, methods=methods
    )


def _rank_label(value: float) -> str:
    return f"{value:.0f}" if np.isclose(value, round(value)) else f"{value:.1f}"


def make_rank_grid_figure(grid: RankGrid):
    """Create the hierarchical Causal Chambers rank-grid figure."""
    ranks = grid.ranks
    conditions = grid.conditions
    n_methods, n_conditions = ranks.shape
    if n_methods == 0 or n_conditions == 0:
        raise ValueError("rank grid is empty")

    apply_style()
    figure_height = max(5.8, 2.1 + 0.30 * n_methods)
    fig = plt.figure(figsize=(FULL_PAGE_WIDTH_IN, figure_height))
    layout = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.95, max(3.0, 0.30 * n_methods)),
        hspace=0.02,
    )
    header = fig.add_subplot(layout[0])
    ax = fig.add_subplot(layout[1])

    values = ranks.to_numpy(dtype=float)
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap="RdYlGn_r",
        vmin=1,
        vmax=n_methods,
    )
    ax.set_xticks(np.arange(n_conditions))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    row_labels = [
        f"{method_style(method).label}  (avg {grid.mean_rank[method]:.2f})"
        for method in ranks.index
    ]
    ax.set_yticks(np.arange(n_methods), labels=row_labels)
    ax.set_ylabel("Method (ordered by mean rank)")
    ax.set_xlabel(f"Rank at context size {grid.context_size:,} (1 = best)")
    ax.set_xlim(-0.5, n_conditions - 0.5)
    ax.set_ylim(n_methods - 0.5, -0.5)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n_conditions, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_methods, 1), minor=True)
    ax.grid(which="minor", color=COLORS["grey"], alpha=0.35, linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row, method in enumerate(ranks.index):
        for column, condition in enumerate(conditions):
            value = float(ranks.loc[method, condition.name])
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                column,
                row,
                _rank_label(value),
                ha="center",
                va="center",
                color=COLORS["black"] if luminance > 0.58 else "white",
                fontsize=8,
                fontweight="bold",
            )

    header.set_xlim(-0.5, n_conditions - 0.5)
    header.set_ylim(0, 3)
    header.axis("off")
    for y in (0, 1, 2, 3):
        header.hlines(
            y,
            -0.5,
            n_conditions - 0.5,
            color=COLORS["light_grey"],
            linewidth=0.5,
        )
    for boundary in range(1, n_conditions):
        if (
            conditions[boundary - 1].family != conditions[boundary].family
            or conditions[boundary - 1].confounding != conditions[boundary].confounding
        ):
            header.vlines(
                boundary - 0.5,
                0,
                3,
                color=COLORS["grey"],
                linewidth=0.8,
            )
    for column, condition in enumerate(conditions):
        header.text(
            column,
            0.5,
            condition.proxy.capitalize(),
            ha="center",
            va="center",
        )

    for family in FAMILY_ORDER:
        indices = [
            i for i, condition in enumerate(conditions) if condition.family == family
        ]
        header.text(
            float(np.mean(indices)),
            2.5,
            "Non-linear" if family == "non_linear" else "Linear",
            ha="center",
            va="center",
            fontweight="bold",
        )
    for confounding in CONFOUNDING_ORDER:
        # There is one contiguous group per family; put a label in each group.
        for family in FAMILY_ORDER:
            group = [
                i
                for i, condition in enumerate(conditions)
                if condition.family == family and condition.confounding == confounding
            ]
            header.text(
                float(np.mean(group)),
                1.5,
                f"{confounding.capitalize()} confounding",
                ha="center",
                va="center",
            )

    fig.suptitle(
        f"Causal Chambers — method rank by dataset at {grid.context_size:,} datapoints"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.04)
    colorbar.set_label("Rank (1 = best)")
    colorbar.set_ticks(range(1, n_methods + 1))
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Causal Chambers results.csv (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help=f"Context size to rank (default: {DEFAULT_CONTEXT_SIZE})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--name", default=OUTPUT_NAME, help="Output filename stem")
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method subset; names must match results.csv exactly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grid = load_rank_grid(
        args.results_csv, context_size=args.context_size, methods=args.methods
    )
    fig = make_rank_grid_figure(grid)
    paths = save_figure(fig, args.out_dir, args.name)
    print(
        f"Loaded {len(grid.ranks)} methods and {len(grid.conditions)} datasets "
        f"at n_ctx={grid.context_size}."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
