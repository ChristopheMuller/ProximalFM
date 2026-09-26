"""Two-panel learning-curve and error-decomposition figure.

Run from the repository root::

    uv run python -m analysis_eval.learning_curves_error_decomposition \
      --case nonlinear_conf_high_proxy_low

Standalone evaluation roots without factorial cases are also supported::

    uv run python -m analysis_eval.learning_curves_error_decomposition \
      --matrix-root data/synt_eval/synt_val_8192_v1 \
      --name synt_data \
      --y-max 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .design import (
    FULL_PAGE_WIDTH_IN,
    apply_style,
    method_style,
    ordered_methods,
)
from .error_decomposition import draw_error_decomposition_panel
from .learning_curves import (
    CONFIDENCE_INTERVAL_CHOICES,
    DEFAULT_MATRIX_ROOT,
    DEFAULT_Y_MAX,
    draw_learning_curve_panel,
)
from .plotting import plt, save_figure
from .semisynt import (
    REQUIRED_COLUMNS,
    complete_learning_curve_methods,
    is_matrix_root,
    load_cate_results,
    summarize_error_decomposition,
    summarize_learning_curves,
)

DEFAULT_OUT_DIR = Path(__file__).with_name("results")
OUTPUT_NAME = "semisynt_learning_curves_error_decomposition"
FIGSIZE = (FULL_PAGE_WIDTH_IN, 4.5)
DEFAULT_METHODS = (
    "T-OLS",
    "S-TabICL-WZ-kNN-16",
    "P-Kernel-OLS",
    "P-Kernel-TabICL",
    "NMMR-V-fixed-Wcond",
    "CausalPFN",
    "CausalPFN-WZ-kNN-16",
    "ProximalFM",
    "ProximalFM-FT",
)

STANDALONE_FACTORS = ("synthetic", "mixed", "mixed")


def _standalone_results_path(matrix_root: Path) -> Path:
    """Find benchmark results in a root that does not contain cases."""
    matrix_root = Path(matrix_root)
    candidates = (
        matrix_root / "bench" / "results" / "results.csv",
        matrix_root / "results" / "results.csv",
        matrix_root / "results.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"standalone benchmark results not found under {matrix_root}; "
        f"expected one of: {expected}"
    )


def _load_standalone_results(matrix_root: Path) -> pd.DataFrame:
    """Load CATE rows from a single benchmark root without case metadata."""
    results_path = _standalone_results_path(matrix_root)
    frame = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{results_path}: missing columns {sorted(missing)}")

    frame = frame[
        frame["target"].eq("cate")
        & frame["estimand"].eq("cate(x)")
        & ~frame["is_reference"].astype(bool)
        & ~frame["failed"].astype(bool)
    ].copy()
    frame["npehe"] = pd.to_numeric(frame["npehe"], errors="coerce")
    frame = frame[np.isfinite(frame["npehe"])]
    if frame.empty:
        raise ValueError(
            f"{results_path}: no finite competitor CATE(X) scores in standalone set"
        )

    # The summary functions intentionally use the matrix schema. A standalone
    # set gets one synthetic case and one mixed factorial cell so the same
    # aggregation and plotting code can be reused without inventing per-row
    # cases that are not present in the data.
    frame["case"] = Path(matrix_root).name
    frame["family"], frame["confounding"], frame["proxy"] = STANDALONE_FACTORS
    return frame


def _load_results(
    matrix_root: Path, *, selected_case: str | None, dimension: int | None = None
) -> pd.DataFrame:
    """Load either a factorial case matrix or a standalone benchmark root."""
    matrix_root = Path(matrix_root)
    if is_matrix_root(matrix_root):
        return load_cate_results(
            matrix_root, selected_case=selected_case, dimension=dimension
        )

    if selected_case is not None:
        raise ValueError(
            "--case is only valid for a case matrix; omit it for a standalone "
            "benchmark root"
        )
    return _load_standalone_results(matrix_root)


def _selected_cell(summary, case: str | None):
    if case is None:
        cases = summary["case"].drop_duplicates().tolist()
        if len(cases) != 1:
            raise ValueError(
                "a combined figure without --case requires a standalone root "
                f"with exactly one case; found {len(cases)}"
            )
        case = cases[0]
    cell = summary[summary["case"].eq(case)]
    if cell.empty:
        available = sorted(str(value) for value in summary["case"].unique())
        raise ValueError(
            f"unknown case {case!r}; available cases are: {', '.join(available)}"
        )
    factors = cell[["family", "confounding", "proxy"]].drop_duplicates()
    if len(factors) != 1:
        raise ValueError(f"case {case!r} maps to multiple factorial cells")
    return cell, next(factors.itertuples(index=False, name=None))


def make_combined_figure(
    learning_summary,
    decomposition_summary,
    *,
    methods: Sequence[str],
    case: str | None = None,
    y_max: float = DEFAULT_Y_MAX,
    confidence_interval: str = "false",
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
):
    """Render the selected case as learning-curve and decomposition panels."""
    if y_max <= 0:
        raise ValueError("y_max must be positive")
    if confidence_interval not in CONFIDENCE_INTERVAL_CHOICES:
        raise ValueError(
            f"confidence_interval must be one of {CONFIDENCE_INTERVAL_CHOICES}"
        )
    methods = ordered_methods(methods)
    learning_cell, factors = _selected_cell(learning_summary, case)
    decomposition_cell, decomposition_factors = _selected_cell(
        decomposition_summary, case
    )
    if factors != decomposition_factors:
        raise ValueError(
            "learning and decomposition summaries describe different cells"
        )
    family, confounding, proxy = factors
    contexts = sorted(int(value) for value in learning_cell["n_ctx"].unique())
    decomposition_contexts = sorted(
        int(value) for value in decomposition_cell["n_ctx"].unique()
    )
    if contexts != decomposition_contexts:
        raise ValueError(
            "learning and decomposition summaries use different context sizes"
        )

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_learning_curve_panel(
        axes[0],
        learning_cell,
        methods=methods,
        contexts=contexts,
        family=family,
        confounding=confounding,
        proxy=proxy,
        y_max=y_max,
        confidence_interval=confidence_interval,
    )
    draw_error_decomposition_panel(
        axes[1],
        decomposition_cell,
        methods=methods,
        contexts=contexts,
        family=family,
        confounding=confounding,
        proxy=proxy,
        xlim=xlim,
        ylim=ylim,
    )
    for ax, tag in zip(axes, "ab"):
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(f"({tag}) {title}", loc="left")

    handles = [
        Line2D(
            [],
            [],
            label=method_style(method).label,
            **method_style(method).line_kwargs(),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(5, len(handles)),
        columnspacing=1.0,
        handlelength=2.2,
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Dimension in a per-dimension matrix (default: 5)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    parser.add_argument(
        "--case",
        "--config",
        dest="case",
        metavar="NAME",
        help=(
            "Matrix case to render in both panels; omit for a standalone "
            "benchmark root without cases"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method subset; defaults to this figure's DEFAULT_METHODS",
    )
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX)
    parser.add_argument(
        "--confidence-interval",
        type=str.lower,
        choices=CONFIDENCE_INTERVAL_CHOICES,
        default="false",
    )
    parser.add_argument(
        "--aggregation",
        choices=("median", "mean"),
        default="median",
    )
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = _load_results(args.matrix_root, selected_case=args.case, dimension=args.dim)
    complete, incomplete = complete_learning_curve_methods(frame)
    if args.methods is not None:
        selected = args.methods
    else:
        selected = [method for method in DEFAULT_METHODS if method in complete]
        if not selected:
            raise ValueError(
                "none of this figure's DEFAULT_METHODS have complete context coverage"
            )
    learning_summary = summarize_learning_curves(frame, selected)
    methods = ordered_methods(set(learning_summary["method"]))
    decomposition_summary = summarize_error_decomposition(
        frame, methods, aggregation=args.aggregation
    )
    fig = make_combined_figure(
        learning_summary,
        decomposition_summary,
        methods=methods,
        case=args.case,
        y_max=args.y_max,
        confidence_interval=args.confidence_interval,
        xlim=args.xlim,
        ylim=args.ylim,
    )
    paths = save_figure(fig, args.out_dir, args.name)
    rendered_case = args.case
    if rendered_case is None and frame["case"].nunique() == 1:
        rendered_case = str(frame["case"].iloc[0])
    print(
        f"Rendered {len(methods)} methods for {rendered_case} over "
        f"{frame['n_ctx'].nunique()} context sizes ({args.aggregation})."
    )
    if args.methods is None and incomplete:
        print(
            "Excluded methods without complete context coverage: "
            + ", ".join(sorted(incomplete))
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
