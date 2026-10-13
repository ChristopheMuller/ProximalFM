"""Plot semi-synthetic benchmark latency as context size grows.

Run from the repository root::

    uv run python -m analysis.analysis_eval.speed_context

The benchmark stores one ``*_preds.npz`` cache per method and context size.
This module reads only its timing arrays; it never re-runs an estimator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .design import COLORS, METHOD_STYLES, apply_style, method_style, ordered_methods
from .plotting import plt, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_DIR = (
    REPO_ROOT
    / "data"
    / "semisynt_eval"
    / "semisynt_val_v1"
    / "d-5"
    / "nonlinear_conf_high_proxy_low"
    / "bench_speed_sequential"
)
DEFAULT_OUT_DIR = Path(__file__).with_name("results") / "speed"
OUTPUT_NAME = "semisynt_speed_by_context"

# A fixed, representative selection keeps the figure readable while the full
# benchmark remains visible as a muted reference layer.  The CPU/MPS pair is
# deliberately included: it makes the hardware comparison explicit instead of
# hiding it behind the generic ProximalFM style used elsewhere.
FEATURED_METHODS = (
    "T-OLS",
    "T-OLS-WZ-kNN-16",
    "P-Kernel-OLS",
    "P-Kernel-TabICL",
    "KPV-X-Wcond",
    "PMMR-X-Wcond",
    "NMMR-V-fixed-Wcond",
    "CausalFM",
    "DoPFN",
    "CausalPFN",
    "CausalPFN-WZ-kNN-16",
    "ProximalFM-CPU",
    "ProximalFM-MPS",
)
_CACHE_NAME = re.compile(r"^(?P<method>.+)_nctx-(?P<n_ctx>\d+)_preds\.npz$")


def load_speed_results(bench_dir: Path) -> pd.DataFrame:
    """Return per-episode timings from a benchmark prediction-cache directory."""
    preds_dir = Path(bench_dir) / "preds"
    if not preds_dir.is_dir():
        raise FileNotFoundError(f"prediction-cache directory not found: {preds_dir}")

    rows: list[dict[str, object]] = []
    for path in sorted(preds_dir.glob("*_nctx-*_preds.npz")):
        match = _CACHE_NAME.match(path.name)
        if match is None:
            continue
        with np.load(path, allow_pickle=False) as cache:
            if "times" not in cache:
                raise ValueError(f"timing array missing from {path}")
            times = np.asarray(cache["times"], dtype=float).reshape(-1)
            post_tune = np.asarray(
                cache["times_post_tune"]
                if "times_post_tune" in cache
                else np.full_like(times, np.nan),
                dtype=float,
            ).reshape(-1)
        if post_tune.shape != times.shape:
            raise ValueError(
                f"post-selection timing shape differs from total timing in {path}"
            )
        finite_times = times[np.isfinite(times) & (times > 0)]
        if finite_times.size == 0:
            raise ValueError(f"no finite positive timings in {path}")
        rows.extend(
            {
                "method": match["method"],
                "n_ctx": int(match["n_ctx"]),
                "seconds": float(value),
                "seconds_post_tune": float(post_value),
            }
            for value, post_value in zip(times, post_tune)
            if np.isfinite(value) and value > 0
        )

    if not rows:
        raise FileNotFoundError(
            f"no timing caches matching *_nctx-*_preds.npz in {preds_dir}"
        )
    frame = pd.DataFrame(rows)
    duplicated = frame.groupby(["method", "n_ctx"], observed=True).size()
    if duplicated.empty:
        raise ValueError(f"no timing measurements found in {preds_dir}")
    return frame


def summarize_speed(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize latency robustly over episodes at every context size."""
    required = {"method", "n_ctx", "seconds", "seconds_post_tune"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"speed summary requires columns {sorted(missing)}")
    summary = (
        frame.groupby(["method", "n_ctx"], observed=True)
        .agg(
            median=("seconds", "median"),
            q25=("seconds", lambda values: values.quantile(0.25)),
            q75=("seconds", lambda values: values.quantile(0.75)),
            post_median=("seconds_post_tune", "median"),
            post_q25=("seconds_post_tune", lambda values: values.quantile(0.25)),
            post_q75=("seconds_post_tune", lambda values: values.quantile(0.75)),
            n=("seconds", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(["method", "n_ctx"]).reset_index(drop=True)


def _display_style(method: str) -> tuple[str, str, str, str, float, float, int]:
    """Return a stable style, including explicit labels for device variants."""
    device_styles = {
        "ProximalFM-CPU": (
            "ProximalFM (CPU)",
            COLORS["vermillion"],
            "o",
            "-",
            2.15,
            5.0,
            5,
        ),
        "ProximalFM-MPS": (
            "ProximalFM (MPS)",
            COLORS["vermillion"],
            "o",
            "--",
            2.15,
            5.0,
            6,
        ),
        # These names are emitted only by the speed configuration.  They are
        # deliberately local rather than additions to the paper-wide registry.
        "T-OLS-WZ-kNN-16": (
            "T-learner (ridge) + kNN marg.",
            COLORS["sky_blue"],
            "h",
            "-.",
            1.45,
            4.0,
            2,
        ),
        "T-OLS-WZ-Kernel-16": (
            "T-learner (ridge) + kernel marg.",
            COLORS["sky_blue"],
            "d",
            "-.",
            1.45,
            4.0,
            2,
        ),
        "S-TabICL-WZ-kNN-16": (
            "S-learner (TabICL) + kNN marg.",
            COLORS["purple"],
            "h",
            "-.",
            1.45,
            4.0,
            2,
        ),
        "S-TabICL-WZ-Kernel-16": (
            "S-learner (TabICL) + kernel marg.",
            COLORS["purple"],
            "d",
            "-.",
            1.45,
            4.0,
            2,
        ),
        "CausalPFN-WZ-Kernel-16": (
            "CausalPFN + kernel marg.",
            COLORS["blue"],
            "d",
            "-.",
            1.85,
            4.6,
            4,
        ),
    }
    if method in device_styles:
        return device_styles[method]
    if method.startswith("NMMR-"):
        condition = "cond. W" if method.endswith("-Wcond") else "independent W"
        variant = (
            method.removesuffix("-Wcond").removesuffix("-Wind").removeprefix("NMMR-")
        )
        return (
            f"NMMR ({variant.replace('-', ', ')}; {condition})",
            COLORS["wine"],
            "v" if condition == "cond. W" else "^",
            "-" if condition == "cond. W" else "--",
            1.85,
            4.6,
            3,
        )
    style = method_style(method)
    return (
        style.label,
        style.color,
        style.marker,
        style.linestyle,
        style.linewidth,
        style.markersize,
        style.zorder,
    )


def _ordered_speed_methods(methods: set[str]) -> list[str]:
    """Order registered methods first, then speed-only variants by name."""
    known = methods & set(METHOD_STYLES)
    return ordered_methods(known) + sorted(methods - known)


def make_speed_figure(summary: pd.DataFrame, methods: Sequence[str]):
    """Draw a log-log latency curve with IQR ribbons for selected methods."""
    required = {"method", "n_ctx", "median", "q25", "q75", "n"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"speed figure requires columns {sorted(missing)}")
    if summary.empty:
        raise ValueError("speed figure received an empty summary")

    available = set(summary["method"])
    selected = list(methods)
    absent = sorted(set(selected) - available)
    if absent:
        raise ValueError(f"requested methods absent from timing caches: {absent}")
    contexts = sorted(int(value) for value in summary["n_ctx"].unique())
    if len(contexts) < 2:
        raise ValueError("speed figure needs timings at two or more context sizes")

    apply_style()
    dense_legend = len(selected) > 15
    fig, ax = plt.subplots(figsize=(9.4, 8.4 if dense_legend else 4.8))

    # Preserve the scale of the complete run without forcing a 30-item legend.
    for method, group in summary[~summary["method"].isin(selected)].groupby(
        "method", sort=False
    ):
        group = group.sort_values("n_ctx")
        ax.plot(
            group["n_ctx"],
            group["median"],
            color=COLORS["light_grey"],
            linewidth=0.8,
            alpha=0.38,
            zorder=1,
        )

    handles = []
    for method in selected:
        group = summary[summary["method"].eq(method)].sort_values("n_ctx")
        label, color, marker, linestyle, linewidth, markersize, zorder = _display_style(
            method
        )
        ax.fill_between(
            group["n_ctx"],
            group["q25"],
            group["q75"],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        (line,) = ax.plot(
            group["n_ctx"],
            group["median"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=zorder,
            solid_capstyle="round",
            dash_capstyle="round",
        )
        handles.append(line)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{value:,}" for value in contexts])
    ax.set_xlabel("Context size (rows)")
    ax.set_ylabel("Wall-clock time per episode (seconds)")
    fig.suptitle(
        "Semi-synthetic benchmark speed scales with context size",
        x=0.125,
        ha="left",
        weight="bold",
    )
    ax.set_title(
        "Lines: median across episodes  •  bands: interquartile range  •  grey: remaining benchmark methods",
        loc="left",
        fontsize=7.4,
        color=COLORS["charcoal"],
    )
    if dense_legend:
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=3,
            columnspacing=1.2,
            handlelength=2.4,
        )
    else:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            borderaxespad=0,
            ncol=1,
            handlelength=2.4,
        )
    ax.margins(x=0.06, y=0.12)
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default=OUTPUT_NAME)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Methods to emphasize and include in the legend (default: compact fixed set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Emphasize every method (useful for a complete but denser figure)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.methods is not None and args.all:
        raise ValueError("use either --methods or --all, not both")
    summary = summarize_speed(load_speed_results(args.bench_dir))
    available = set(summary["method"])
    if args.all:
        methods = _ordered_speed_methods(available)
    elif args.methods is not None:
        methods = args.methods
    else:
        methods = [method for method in FEATURED_METHODS if method in available]
        if not methods:
            raise ValueError(
                "none of the default featured methods are present; pass --methods"
            )
    figure = make_speed_figure(summary, methods)
    paths = save_figure(figure, args.out_dir, args.name)
    n_episodes = int(summary["n"].iloc[0])
    print(
        f"Plotted {len(methods)} highlighted methods across "
        f"{summary['n_ctx'].nunique()} context sizes ({n_episodes} episodes per cache)."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
