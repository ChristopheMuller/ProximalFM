"""Render point-estimate validation curves (``fig:point_validation``).

```bash
uv run python -m analysis_training.point_figure
uv run python -m analysis_training.point_figure --run ProximalFM/<id> \
    --outdir <paper>/assets/training
```

The four panels show the point-estimate validation metrics logged on the fixed
realistic-mechanism suite. ATE RMSE is a buffer-level metric and is therefore
drawn without an interval. The normalized ATE bias, nPEHE, and shape R-squared
are per-episode summaries; their logged p25--p75 values are drawn as shaded
bands. The bands describe variation across validation episodes, not uncertainty
in the estimated training trajectory.

All metrics are read at their logged validation cadence rather than binned: the
validation pass already produces one robust summary at each checkpoint, and
the p25--p75 columns are themselves summaries that should not be re-aggregated.
The x-axis is logarithmic to make the early and late parts of training visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analysis_training.design import (
    apply_style,
    figsize_grid,
    panel_tag,
    plt,
    reference_line,
    save_figure,
    series,
)
from analysis_training.wandb_history import DEFAULT_RUN, History, fetch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "analysis_training" / "results"
NAME = "point_validation"

KEYS = (
    "synt_point/ate/rmse",
    "synt_point/ate/bias_norm",
    "synt_point/ate/bias_norm/p25",
    "synt_point/ate/bias_norm/p75",
    "synt_point/cate/npehe",
    "synt_point/cate/npehe/p25",
    "synt_point/cate/npehe/p75",
    "synt_point/cate/r2_shape",
    "synt_point/cate/r2_shape/p25",
    "synt_point/cate/r2_shape/p75",
)


def _metric_frame(history: History, key: str, *, band: bool):
    """Return sorted, non-missing rows for one logged metric."""
    columns = ["_step", key]
    if band:
        columns.extend((f"{key}/p25", f"{key}/p75"))

    missing = [column for column in columns if column not in history.frame]
    if missing:
        raise KeyError(f"History is missing validation keys: {missing}")

    frame = history.frame[columns].dropna().sort_values("_step")
    if frame.empty:
        raise ValueError(f"No logged validation values found for {key!r}")
    return frame


def _draw_metric(ax, history: History, key: str, *, ylabel: str, band: bool) -> None:
    """Draw one checkpoint trajectory and its optional episode IQR."""
    frame = _metric_frame(history, key, band=band)
    style = series(key)

    if band:
        ax.fill_between(
            frame["_step"].to_numpy(),
            frame[f"{key}/p25"].to_numpy(),
            frame[f"{key}/p75"].to_numpy(),
            color=style.color,
            alpha=0.18,
            lw=0,
            label="episode p25--p75",
        )
    ax.plot(
        frame["_step"].to_numpy(),
        frame[key].to_numpy(),
        color=style.color,
        lw=style.lw,
        label=style.label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)


def build_figure(history: History) -> "plt.Figure":
    """Assemble the four point-estimate validation panels."""
    apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize_grid(ncols=2, nrows=2, panel_aspect=0.78),
    )

    _draw_metric(
        axes[0, 0],
        history,
        "synt_point/ate/rmse",
        ylabel="RMSE",
        band=False,
    )
    axes[0, 0].set_title("ATE RMSE")

    _draw_metric(
        axes[0, 1],
        history,
        "synt_point/ate/bias_norm",
        ylabel="bias / SD(CATE)",
        band=True,
    )
    axes[0, 1].set_title("normalized ATE bias")
    reference_line(
        axes[0, 1],
        0.0,
        "zero bias",
        x=0.78,
        role="reference",
    )

    _draw_metric(
        axes[1, 0],
        history,
        "synt_point/cate/npehe",
        ylabel="nPEHE",
        band=True,
    )
    axes[1, 0].set_title("CATE nPEHE")
    reference_line(
        axes[1, 0],
        1.0,
        "constant-ATE baseline",
        x=0.66,
        role="naive",
    )

    _draw_metric(
        axes[1, 1],
        history,
        "synt_point/cate/r2_shape",
        ylabel=r"shape $R^2$",
        band=True,
    )
    axes[1, 1].set_title(r"CATE shape $R^2$")
    reference_line(
        axes[1, 1],
        0.0,
        "constant-ATE baseline",
        x=0.66,
        role="naive",
    )

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
    args = parser.parse_args(argv)

    history = fetch(list(KEYS), run=args.run, force=args.force)
    fig = build_figure(history)
    paths = save_figure(fig, args.outdir, args.name)
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
