"""Plot Causal Chambers ATE absolute-error learning curves by condition.

The physical benchmark has a 2 x 2 x 3 factorial design: linear/non-linear
control functions, weak/strong confounding, and low/mid/high nominal proxy
informativeness. This module reads its long-form results and renders one
learning curve per physical condition. It never fits estimators.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .design import (
    FULL_PAGE_WIDTH_IN,
    SELECTION_CAUSAL_CHAMERS,
    apply_style,
    method_style,
    ordered_methods,
)
from .plotting import plt, save_figure
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_CSV = (
    REPO_ROOT / "benchmarks" / "causal_chambers" / "results" / "results.csv"
)
DEFAULT_OUT_DIR = Path(__file__).with_name("results") / "causal_chambers"
OUTPUT_NAME = "causal_chambers_absolute_error"
REQUIRED_COLUMNS = frozenset(
    {"condition", "permutation", "n_ctx", "method", "abs_error"}
)
FAMILY_ORDER = ("linear", "non_linear")
CONFOUNDING_ORDER = ("strong", "weak")
PROXY_ORDER = ("low", "mid", "high")
_CONDITION_RE = re.compile(
    r"(?P<confounding>strong|weak)_confounding_"
    r"(?P<family>linear|non_linear)_proxy_"
    r"(?P<proxy>low|mid|high)(?:_|$)"
)


@dataclass(frozen=True)
class DatasetCondition:
    """One cell of the Causal Chambers physical factorial design."""

    name: str
    family: str
    confounding: str
    proxy: str


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

    result = frame[["condition", "permutation", "n_ctx", "method", "abs_error"]].copy()
    for column in ("condition", "method", "permutation"):
        if result[column].isna().any():
            raise ValueError(f"Causal Chambers results contain missing {column} values")
        result[column] = result[column].astype(str)
        if result[column].str.strip().eq("").any():
            raise ValueError(f"Causal Chambers results contain blank {column} values")
    for column in ("n_ctx", "abs_error"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["n_ctx", "abs_error"]].isna().any().any():
        raise ValueError("Causal Chambers results contain non-numeric values")

    contexts = result["n_ctx"].to_numpy(dtype=float)
    errors = result["abs_error"].to_numpy(dtype=float)
    if not np.isfinite(contexts).all() or not np.isfinite(errors).all():
        raise ValueError("Causal Chambers results contain non-finite values")
    if (contexts <= 0).any() or not np.equal(contexts, np.floor(contexts)).all():
        raise ValueError("context sizes must be positive integers")
    if (errors < 0).any():
        raise ValueError("absolute errors must be non-negative")
    if (errors == 0).any():
        raise ValueError(
            "log-scale absolute-error plots require strictly positive errors"
        )
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
            "duplicate condition/permutation/context/method rows, e.g. " f"{sample}"
        )
    return result


def load_results(results_csv: Path) -> pd.DataFrame:
    """Load the validated long-form Causal Chambers ATE results."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        raise FileNotFoundError(f"Causal Chambers results not found: {results_csv}")
    return _validate_results(pd.read_csv(results_csv), source=results_csv)


def _selected_methods(
    results: pd.DataFrame, methods: Sequence[str] | None
) -> list[str]:
    available = set(results["method"])
    selected = list(SELECTION_CAUSAL_CHAMERS if methods is None else methods)
    absent = set(selected) - available
    if absent:
        raise ValueError(f"requested methods absent from results: {sorted(absent)}")
    ordered = ordered_methods(selected)
    if not ordered:
        raise ValueError("Causal Chambers figure has no methods to plot")
    return ordered


def _conditions(results: pd.DataFrame) -> tuple[DatasetCondition, ...]:
    parsed = tuple(_parse_condition(name) for name in results["condition"].unique())
    factors = {(item.family, item.confounding, item.proxy) for item in parsed}
    expected = {
        (family, confounding, proxy)
        for family in FAMILY_ORDER
        for confounding in CONFOUNDING_ORDER
        for proxy in PROXY_ORDER
    }
    if factors != expected or len(parsed) != len(factors):
        raise ValueError(
            "Causal Chambers results must contain the complete 2x2x3 condition "
            f"grid; missing={sorted(expected - factors)}, extra={sorted(factors - expected)}"
        )
    return tuple(sorted(parsed, key=_condition_order))


def _display_label(method: str) -> str:
    """Use concise labels because the entire figure is an ATE evaluation."""
    labels = {
        "P-Kernel-ATE": "P-learner",
        "KPV-ATE": "KPV",
        "PMMR-ATE": "PMMR",
        "NMMR-V-fixed-ATE": "NMMR (V, fixed)",
        "CausalPFN-XWZ": "CausalPFN",
        "ProximalFM-ATE": "ProximalFM",
    }
    return labels.get(method, method_style(method).label)


def _curve_errors(results: pd.DataFrame, condition: str, method: str) -> pd.DataFrame:
    """Aggregate repeated acquisition orders by geometric mean absolute error."""
    curve = results[results["condition"].eq(condition) & results["method"].eq(method)]
    return (
        curve.groupby("n_ctx", as_index=False)["abs_error"]
        .agg(lambda values: float(np.exp(np.log(values.to_numpy()).mean())))
        .sort_values("n_ctx")
    )


def make_causal_chambers_figure(
    results: pd.DataFrame,
    *,
    methods: Sequence[str] | None = None,
):
    """Create a shared-scale 12-panel Causal Chambers learning-curve figure."""
    results = _validate_results(results)
    methods = _selected_methods(results, methods)
    conditions = _conditions(results)
    contexts = sorted(int(value) for value in results["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("learning curves require at least two context sizes")

    expected = {
        (condition.name, method, context)
        for condition in conditions
        for method in methods
        for context in contexts
    }
    observed = set(
        results[results["method"].isin(methods)][
            ["condition", "method", "n_ctx"]
        ].itertuples(index=False, name=None)
    )
    missing = expected - observed
    if missing:
        raise ValueError(
            "learning curves require every selected method at every condition "
            f"and context size; missing e.g. {sorted(missing)[:5]}"
        )

    apply_style()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(FULL_PAGE_WIDTH_IN, 8.5),
        sharex=True,
        sharey=True,
    )
    for axis, condition in zip(axes.flat, conditions):
        for method in methods:
            curve = _curve_errors(results, condition.name, method)
            axis.plot(
                curve["n_ctx"].to_numpy(dtype=float),
                curve["abs_error"].to_numpy(dtype=float),
                label=_display_label(method),
                **method_style(method).line_kwargs(),
            )
        axis.set_title(
            f"{condition.family.replace('_', '-').capitalize()}; "
            f"{condition.confounding} confounding; {condition.proxy} proxy info",
            loc="left",
            fontsize=7.5,
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(contexts)
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.grid(axis="y", which="major", alpha=0.22, linewidth=0.45)

    for axis in axes[-1]:
        axis.set_xlabel("Context size")
    fig.supylabel("Absolute ATE error (sensor counts)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        columnspacing=1.25,
        handlelength=2.6,
    )
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
    results = load_results(args.results_csv)
    methods = _selected_methods(results, args.methods)
    figure = make_causal_chambers_figure(results, methods=methods)
    paths = save_figure(figure, args.out_dir, args.name)
    print(
        f"Loaded {len(methods)} methods, {results['condition'].nunique()} conditions, "
        f"and {results['n_ctx'].nunique()} context sizes."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
