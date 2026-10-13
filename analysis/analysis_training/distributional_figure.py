"""Render distributional validation curves (``fig:distributional_validation``).

```bash
uv run python -m analysis.analysis_training.distributional_figure
uv run python -m analysis.analysis_training.distributional_figure --run ProximalFM/<id> \
    --outdir <paper>/assets/training
```

The four panels show the distributional validation metrics logged on the fixed
realistic-mechanism suite: CRPS, the standard-deviation calibration ratio,
50- and 90-percent coverage, and central credible-interval widths. The coverage
curves are pooled metrics, with their per-episode p25--p75 variation shown as
shaded bands but excluded from the legend. The final panel overlays the widths
of the central 90-percent and 50-percent intervals.

All metrics are read at their logged validation cadence and lightly summarized
with log-spaced median bins for display. The x-axis is logarithmic to make the
early and late parts of training visible.
The horizontal reference lines mark calibrated values: one for
``sd_over_rmse`` and the nominal levels for 50- and 90-percent coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.analysis_training.design import (
    apply_style,
    binned,
    draw_binned,
    figsize_grid,
    panel_tag,
    plt,
    reference_line,
    save_figure,
    series,
)
from analysis.analysis_training.wandb_history import DEFAULT_RUN, History, fetch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "analysis" / "analysis_training" / "results"
NAME = "distributional_validation"

KEYS = (
    "synt_dist/crps",
    "synt_dist/sd_over_rmse",
    "synt_dist/cov50_pooled",
    "synt_dist/cov50/p25",
    "synt_dist/cov50/p75",
    "synt_dist/cov90_pooled",
    "synt_dist/cov90/p25",
    "synt_dist/cov90/p75",
    "synt_dist/width90",
    "synt_dist/width50",
)


def _metric_frame(
    history: History, key: str, *, band: bool, band_key: str | None = None
):
    """Return sorted, non-missing rows for one logged metric."""
    band_key = key if band_key is None else band_key
    columns = ["_step", key]
    if band:
        columns.extend((f"{band_key}/p25", f"{band_key}/p75"))

    missing = [column for column in columns if column not in history.frame]
    if missing:
        raise KeyError(f"History is missing validation keys: {missing}")

    frame = history.frame[columns].dropna().sort_values("_step")
    if frame.empty:
        raise ValueError(f"No logged validation values found for {key!r}")
    return frame


def _draw_metric(
    ax,
    history: History,
    key: str,
    *,
    ylabel: str,
    band: bool,
    band_key: str | None = None,
    n_bins: int,
) -> None:
    """Draw a log-binned checkpoint trajectory and its optional episode IQR."""
    band_key = key if band_key is None else band_key
    frame = _metric_frame(history, key, band=band, band_key=band_key)
    style = series(key)

    summary = binned(frame["_step"], frame[key], log=True, n_bins=n_bins)
    if band:
        lower = binned(
            frame["_step"],
            frame[f"{band_key}/p25"],
            log=True,
            n_bins=n_bins,
        )
        upper = binned(
            frame["_step"],
            frame[f"{band_key}/p75"],
            log=True,
            n_bins=n_bins,
        )
        if not (
            np.array_equal(summary.x, lower.x) and np.array_equal(summary.x, upper.x)
        ):
            raise ValueError(f"Band bins do not align for {key!r}")
        ax.fill_between(
            summary.x,
            lower.median,
            upper.median,
            color=style.color,
            alpha=0.18,
            lw=0,
            label="_nolegend_",
        )
    draw_binned(ax, summary, style, band=False)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)


def build_figure(history: History, *, n_bins: int = 70) -> "plt.Figure":
    """Assemble the four distributional validation panels."""
    apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize_grid(ncols=2, nrows=2, panel_aspect=0.78),
    )

    _draw_metric(
        axes[0, 0],
        history,
        "synt_dist/crps",
        ylabel="CRPS",
        band=False,
        n_bins=n_bins,
    )
    axes[0, 0].set_title("CRPS")

    _draw_metric(
        axes[0, 1],
        history,
        "synt_dist/sd_over_rmse",
        ylabel="predictive SD / RMSE",
        band=False,
        n_bins=n_bins,
    )
    axes[0, 1].set_title(r"predictive SD / RMSE")
    axes[0, 1].set_yscale("log")
    reference_line(
        axes[0, 1],
        1.0,
        "calibrated",
        x=0.82,
        role="reference",
    )

    for key, band_key in (
        ("synt_dist/cov90_pooled", "synt_dist/cov90"),
        ("synt_dist/cov50_pooled", "synt_dist/cov50"),
    ):
        _draw_metric(
            axes[1, 0],
            history,
            key,
            ylabel="coverage",
            band=True,
            band_key=band_key,
            n_bins=n_bins,
        )
    axes[1, 0].set_title("coverage")
    reference_line(
        axes[1, 0],
        0.9,
        "nominal 90%",
        x=0.82,
        role="reference",
    )
    reference_line(
        axes[1, 0],
        0.5,
        "nominal 50%",
        x=0.82,
        role="reference",
    )
    axes[1, 0].legend(loc="upper left", frameon=False)

    for key in ("synt_dist/width90", "synt_dist/width50"):
        _draw_metric(
            axes[1, 1],
            history,
            key,
            ylabel="credible-interval width",
            band=False,
            n_bins=n_bins,
        )
    axes[1, 1].set_title("credible-interval width")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(loc="upper right", frameon=False)

    max_step = float(history.frame["_step"].max())
    for ax in axes.flat:
        ax.set_xlim(0.8, max_step * 1.2)

    for tag, ax in zip("abcd", axes.flat):
        panel_tag(ax, tag)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=DEFAULT_RUN, help="wandb run to plot.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--force", action="store_true", help="Refetch from wandb.")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=70,
        help="Log-spaced bins for the display summary (default: 70).",
    )
    args = parser.parse_args(argv)

    history = fetch(list(KEYS), run=args.run, force=args.force)
    fig = build_figure(history, n_bins=args.n_bins)
    paths = save_figure(fig, args.outdir, args.name)
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
