"""Plot Causal Chambers ATE RMSE as the context size grows.

Run from the repository root::

    uv run python -m analysis_eval.causal_chambers \
        --summary-csv benchmarks/causal_chambers/results/ate/summary.csv

The benchmark runner writes ``summary.csv`` by pooling the condition-level
ATE errors for each method and context size.  This module only reads that
summary; it does not rerun estimators or inspect the per-fit cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from .design import FULL_PAGE_WIDTH_IN, apply_style, method_style, ordered_methods
from .plotting import plt, save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "causal_chambers" / "results"
_CANONICAL_SUMMARY = _RESULTS_ROOT / "ate" / "summary.csv"
_EXISTING_SUMMARY = _RESULTS_ROOT / "ate_p2sls" / "summary.csv"
DEFAULT_SUMMARY_CSV = (
    _EXISTING_SUMMARY if _EXISTING_SUMMARY.exists() else _CANONICAL_SUMMARY
)
DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "causal_chambers_rmse"
REQUIRED_COLUMNS = frozenset({"method", "n_ctx", "rmse"})


def _validate_summary(
    frame: pd.DataFrame, *, source: Path | None = None
) -> pd.DataFrame:
    """Validate and normalize the columns needed by the RMSE figure."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        location = f"{source}: " if source is not None else ""
        raise ValueError(f"{location}summary is missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError("Causal Chambers summary is empty")

    result = frame[["method", "n_ctx", "rmse"]].copy()
    if result["method"].isna().any():
        raise ValueError("Causal Chambers summary contains missing method names")
    result["method"] = result["method"].astype(str)
    if result["method"].str.strip().eq("").any():
        raise ValueError("Causal Chambers summary contains blank method names")

    for column in ("n_ctx", "rmse"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["n_ctx", "rmse"]].isna().any().any():
        raise ValueError("Causal Chambers summary contains non-numeric values")

    contexts = result["n_ctx"].to_numpy(dtype=float)
    rmse = result["rmse"].to_numpy(dtype=float)
    if not np.isfinite(contexts).all() or not np.isfinite(rmse).all():
        raise ValueError("Causal Chambers summary contains non-finite values")
    if (contexts <= 0).any() or not np.equal(contexts, np.floor(contexts)).all():
        raise ValueError("context sizes must be positive integers")
    if (rmse < 0).any():
        raise ValueError("RMSE values must be non-negative")

    result["n_ctx"] = result["n_ctx"].astype(int)
    duplicates = result.duplicated(["method", "n_ctx"], keep=False)
    if duplicates.any():
        sample = result.loc[duplicates, ["method", "n_ctx"]].head().to_dict("records")
        raise ValueError(f"duplicate method/context rows in summary, e.g. {sample}")
    if result["n_ctx"].nunique() < 2:
        raise ValueError("RMSE context curves require at least two context sizes")
    return result.sort_values(["method", "n_ctx"]).reset_index(drop=True)


def load_rmse_summary(summary_csv: Path) -> pd.DataFrame:
    """Load the validated method/context/RMSE summary from ``summary_csv``."""
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Causal Chambers summary not found: {summary_csv}")
    return _validate_summary(pd.read_csv(summary_csv), source=summary_csv)


def _selected_methods(
    summary: pd.DataFrame, methods: Sequence[str] | None
) -> list[str]:
    available = set(summary["method"])
    selected = list(available) if methods is None else list(methods)
    absent = set(selected) - available
    if absent:
        raise ValueError(f"requested methods absent from summary: {sorted(absent)}")
    if not selected:
        raise ValueError("RMSE figure has no methods to plot")

    ordered = ordered_methods(selected)
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())
    expected = {(method, context) for method in ordered for context in contexts}
    observed = set(summary[["method", "n_ctx"]].itertuples(index=False, name=None))
    missing = expected - observed
    if missing:
        raise ValueError(
            "RMSE curves must contain every selected method at every context size; "
            f"missing e.g. {sorted(missing)[:5]}"
        )
    return ordered


def make_causal_chambers_figure(
    summary: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
):
    """Create the Causal Chambers ATE RMSE learning-curve figure."""
    summary = _validate_summary(summary)
    methods = _selected_methods(summary, methods)
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())

    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_PAGE_WIDTH_IN, 4.25))
    for method in methods:
        curve = summary[summary["method"].eq(method)].sort_values("n_ctx")
        style = method_style(method)
        ax.plot(
            curve["n_ctx"].to_numpy(dtype=float),
            curve["rmse"].to_numpy(dtype=float),
            label=style.label,
            **style.line_kwargs(),
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    # y-axis -> log
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Context size")
    ax.set_ylabel("ATE RMSE")
    ax.set_title("Causal Chambers")
    fig.legend(
        loc="outside lower center",
        ncol=min(4, len(methods)),
        columnspacing=1.1,
        handlelength=2.6,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Causal Chambers summary.csv (default: {DEFAULT_SUMMARY_CSV})",
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
        help="Optional method subset; names must match summary.csv exactly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = load_rmse_summary(args.summary_csv)
    methods = _selected_methods(summary, args.methods)
    fig = make_causal_chambers_figure(summary, methods=methods)
    paths = save_figure(fig, args.out_dir, args.name)
    print(
        f"Loaded {len(methods)} methods and {summary['n_ctx'].nunique()} "
        "context sizes."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
