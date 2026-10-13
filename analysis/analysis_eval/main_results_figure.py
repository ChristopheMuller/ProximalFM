"""Compact main-text figure for speed, calibration, and large contexts.

Run from the repository root::

    .venv/bin/python -m analysis.analysis_eval.main_results_figure

This is cache-only: no estimator or checkpoint is run. Panel (c) reads the
extended-context summary used by the appendix table, since the current matrix
benchmark cache only contains contexts through 2048.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .plotting import plt, save_figure  # sets MPLCONFIGDIR before matplotlib import
from matplotlib.ticker import ScalarFormatter
from .design import COLORS, FULL_PAGE_WIDTH_IN, apply_style, method_style
from .semisynt import load_cate_results
from .speed_context import DEFAULT_BENCH_DIR, load_speed_results, summarize_speed
from .speed_rank import (
    DEFAULT_MATRIX_ROOT,
    _rank_panels,
    _select_rank_methods,
    summarize_sequential_speed_rank,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR_COVERAGE = (
    ROOT
    / "analysis/analysis_eval/results/posterior_calibration"
    / "prior_posterior_calibration_v1__whole_1-best_model/coverage.csv"
)
DEFAULT_SEMISYNT_COVERAGE = (
    ROOT
    / "analysis/analysis_eval/results/posterior_calibration"
    / "nonlinear_conf_high_proxy_low__whole_1-best_model/coverage.csv"
)
_SELECTION_CSV = ROOT / "docs/latex/assets/results/semisynt_selection_npehe.csv"
_LEGACY_EXTENDED_CSV = (
    ROOT / "docs/latex/assets/results/semisynt_exhaustive_npehe_extended.csv"
)
DEFAULT_EXTENDED_CSV = (
    _SELECTION_CSV if _SELECTION_CSV.exists() else _LEGACY_EXTENDED_CSV
)
DEFAULT_OUT_DIR = ROOT / "docs/latex/assets/main"
OUTPUT_NAME = "practical_results"
CONTEXT_SIZE = 2048
EXTENDED_METHODS = ("P-Kernel-TabICL", "ProximalFM", "ProximalFM-FT")
SPEED_HIGHLIGHTS = (
    "P-Kernel-TabICL",
    "CausalPFN-WZ-kNN-16",
    "ProximalFM-CPU",
    "ProximalFM-MPS",
)


def _speed_summary(matrix_root: Path, speed_dir: Path, dim: int | None) -> pd.DataFrame:
    scores = load_cate_results(matrix_root, dimension=dim)
    speed = summarize_speed(load_speed_results(speed_dir))
    panels = _rank_panels(scores, scope="overall", case=None, cases=None)
    methods = _select_rank_methods(
        scores, speed, context_size=CONTEXT_SIZE, panels=panels, requested=None
    )
    summary = summarize_sequential_speed_rank(
        scores, speed, methods, context_size=CONTEXT_SIZE, panels=panels
    )
    missing = set(SPEED_HIGHLIGHTS) - set(summary["method"])
    if missing:
        raise ValueError(f"highlighted speed methods missing: {sorted(missing)}")
    return summary


def _coverage(path: Path, level: float = 0.9) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"n_ctx", "nominal", "empirical"}
    if required - set(data):
        raise ValueError(
            f"{path} lacks coverage columns {sorted(required - set(data))}"
        )
    selected = data[np.isclose(data["nominal"], level)].sort_values("n_ctx")
    if selected.empty or selected["n_ctx"].duplicated().any():
        raise ValueError(f"{path} needs one coverage row per context at {level:g}")
    return selected


def _extended(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"case", "method", "n_ctx", "mean_npehe", "n_sources"}
    if required - set(data):
        raise ValueError(
            f"{path} lacks extended-result columns {sorted(required - set(data))}"
        )
    data = data[data["case"].eq("nonlinear_conf_high_proxy_low")]
    data = data[data["method"].isin(EXTENDED_METHODS)]
    contexts = {128, 256, 512, 1024, 2048, 4096, 8192}
    for method in EXTENDED_METHODS:
        rows = data[data["method"].eq(method)]
        if set(rows["n_ctx"]) != contexts or rows["n_ctx"].duplicated().any():
            raise ValueError(f"{path}: incomplete extended context curve for {method}")
        if not rows["n_sources"].eq(12).all():
            raise ValueError(f"{path}: expected 12 sources for {method}")
    return data


def make_figure(
    speed: pd.DataFrame,
    prior: pd.DataFrame,
    semisynt: pd.DataFrame,
    extended: pd.DataFrame,
):
    """Render three deliberately sparse panels at full paper width."""
    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_PAGE_WIDTH_IN, 3.05), layout="constrained"
    )
    ax_speed, ax_cal, ax_large = axes

    # Rank is computed on the full common method set; muted points preserve
    # its context without requiring a 30-item legend in the main text.
    background = speed[~speed["method"].isin(SPEED_HIGHLIGHTS)]
    ax_speed.scatter(
        background["time_median"],
        background["rank_median"],
        s=12,
        color=COLORS["light_grey"],
        alpha=0.65,
        zorder=1,
    )
    for row in speed[speed["method"].isin(SPEED_HIGHLIGHTS)].itertuples():
        method = row.method.replace("-CPU", "").replace("-MPS", "")
        style = method_style(method)
        color, marker = style.color, style.marker
        ax_speed.scatter(
            row.time_median,
            row.rank_median,
            s=34,
            marker=marker,
            facecolor=("white" if row.method == "ProximalFM-CPU" else color),
            edgecolor=color,
            linewidth=1.1,
            zorder=3,
        )
    plotted = speed.set_index("method")
    cpu, mps = plotted.loc["ProximalFM-CPU"], plotted.loc["ProximalFM-MPS"]
    ax_speed.plot(
        [mps.time_median, cpu.time_median],
        [mps.rank_median, cpu.rank_median],
        color=COLORS["vermillion"],
        lw=1.1,
        zorder=2,
    )
    ax_speed.annotate(
        "ProximalFM\nMPS",
        (mps.time_median, mps.rank_median),
        xytext=(0, -9),
        textcoords="offset points",
        ha="center",
        va="top",
        color=COLORS["vermillion"],
        fontsize=6.7,
    )
    ax_speed.annotate(
        "CPU",
        (cpu.time_median, cpu.rank_median),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=COLORS["vermillion"],
        fontsize=6.7,
    )
    tabicl = plotted.loc["P-Kernel-TabICL"]
    ax_speed.annotate(
        "P-learner\n(TabICL)",
        (tabicl.time_median, tabicl.rank_median),
        xytext=(6, -7),
        textcoords="offset points",
        va="top",
        color=COLORS["green"],
        fontsize=6.4,
    )
    causal = plotted.loc["CausalPFN-WZ-kNN-16"]
    ax_speed.annotate(
        "CausalPFN\n+ kNN marg.",
        (causal.time_median, causal.rank_median),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=COLORS["blue"],
        fontsize=6.4,
    )
    ax_speed.set_xscale("log")
    ax_speed.set_xlabel("End-to-end runtime (s)")
    ax_speed.set_ylabel("Median CATE nPEHE rank ↓")
    ax_speed.set_ylim(33.5, 0.5)
    ax_speed.set_yticks([1, 5, 10, 20, 30])
    ax_speed.grid(axis="y", alpha=0.16)
    ax_speed.set_title("(a) Accuracy–runtime frontier")

    for data, label, color, marker in (
        (prior, "Pre-training prior", COLORS["blue"], "o"),
        (semisynt, "Semi-synthetic", COLORS["vermillion"], "s"),
    ):
        ax_cal.plot(
            data["n_ctx"],
            data["empirical"],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.6,
        )
    ax_cal.axhline(0.9, color=COLORS["grey"], linestyle=":", linewidth=1)
    ax_cal.axvline(
        CONTEXT_SIZE, color=COLORS["light_grey"], linestyle="--", linewidth=0.8
    )
    ax_cal.set_xscale("log", base=2)
    ax_cal.set_xticks([128, 512, 2048, 8192])
    ax_cal.xaxis.set_major_formatter(ScalarFormatter())
    ax_cal.minorticks_off()
    ax_cal.set_ylim(0, 1.02)
    ax_cal.set_yticks([0, 0.3, 0.6, 0.9])
    ax_cal.set_xlabel("Context size")
    ax_cal.set_ylabel("Empirical 90% coverage")
    ax_cal.legend(loc="lower left", fontsize=6.6, frameon=False)
    ax_cal.set_title("(b) Posterior calibration")

    for method in EXTENDED_METHODS:
        rows = extended[extended["method"].eq(method)].sort_values("n_ctx")
        style = method_style(method)
        ax_large.plot(
            rows["n_ctx"],
            rows["mean_npehe"],
            label=style.label,
            **style.line_kwargs(),
        )
    ax_large.axhline(1.0, color=COLORS["grey"], linestyle=":", linewidth=0.9)
    ax_large.axvline(
        CONTEXT_SIZE, color=COLORS["light_grey"], linestyle="--", linewidth=0.8
    )
    ax_large.set_xscale("log", base=2)
    ax_large.set_xticks([128, 512, 2048, 8192])
    ax_large.xaxis.set_major_formatter(ScalarFormatter())
    ax_large.minorticks_off()
    ax_large.set_ylim(0.55, 1.6)
    ax_large.set_xlabel("Context size")
    ax_large.set_ylabel("Mean CATE nPEHE ↓")
    ax_large.legend(loc="upper right", fontsize=6.2, frameon=False)
    ax_large.set_title("(c) Beyond training range")
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--speed-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--prior-coverage", type=Path, default=DEFAULT_PRIOR_COVERAGE)
    parser.add_argument(
        "--semisynt-coverage", type=Path, default=DEFAULT_SEMISYNT_COVERAGE
    )
    parser.add_argument("--extended-csv", type=Path, default=DEFAULT_EXTENDED_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    args = parser.parse_args(argv)
    figure = make_figure(
        _speed_summary(args.matrix_root, args.speed_dir, args.dim),
        _coverage(args.prior_coverage),
        _coverage(args.semisynt_coverage),
        _extended(args.extended_csv),
    )
    for path in save_figure(figure, args.out_dir, args.name):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
