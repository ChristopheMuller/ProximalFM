"""Render the training-curve figure (``fig:training_loss``).

```bash
uv run python -m analysis.analysis_training.loss_figure
uv run python -m analysis.analysis_training.loss_figure --run ProximalFM/<id> --outdir <paper>/assets/training
```

Four panels, each answering a different question about the same run, because a
single loss curve on a single axis cannot answer any of them:

**(a) the objective and its two terms, log step.** ``loss/total`` is
``loss/cate_dist + lambda_mean * loss/cate_mse``, and the point of showing all
three is that the distributional term *is* the objective to within a few percent
-- the mean penalty is a regularizer on the head's centre, not a co-equal half.
Log-spaced steps because everything that happens to the schedule happens in the
first 3 500 of 75 000 steps and a linear axis renders it as a vertical line.

**(b) the same, linear step, past the anneal.** The log axis in (a) compresses
the last 71 500 steps into a fifth of the width. Panel (b) is where a reader can
see that the objective is still descending at 75 000 steps -- the fact the
training budget is an argument about.

**(c) the two schedules**, directly below (a) so the shaded ``sigma`` window and
the log-step axis line up between the two panels. ``sigma`` on a log axis and the
learning rate on a twin axis: the warmup-then-cosine shape, and the anneal from
0.5 to two bin widths. This panel exists because of what it explains in (a) and
(b): under ``noise_aware`` the loss is a cross-entropy against a
target *convolved with a kernel of width sigma*, so shrinking sigma sharpens the
target and **raises the attainable loss**. The level of the distributional term
is therefore not comparable across the anneal, and a curve that moves inside the
shaded window is the schedule, not the model. The shading in (a) marks that
window; nothing else in the appendix would tell a reader to distrust that
segment.

**(d) the weighted mean term in detail.** This is the exact same
``loss/cate_mse_weighted`` curve shown as the orange component in (a) and (b),
now on its own log scale so its progress remains visible without giving it a
second, visually arbitrary scale inside the objective panels.

Style comes from :mod:`analysis.analysis_training.design`, which re-exports Appendix B's
identity so the two appendices' figures are typographically the same figures.
Per-step series are drawn as a binned median with an interquartile band (see
:func:`~analysis.analysis_training.design.binned`): 75 000 raw points are noise, and the
band is the gradient noise a single line would hide.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.analysis_training.design import (
    SCHEDULE_SPAN,
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
from analysis.analysis_training.run_config import TrainingRun, parse_script
from analysis.analysis_training.wandb_history import DEFAULT_RUN, History, fetch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "pretrain_prod.sh"
DEFAULT_OUTDIR = REPO_ROOT / "analysis" / "analysis_training" / "results"
NAME = "training_curves"

KEYS = (
    "loss/total",
    "loss/cate_dist",
    "loss/cate_mse_weighted",
    "loss/cate_mse",
    "loss/hl_sigma",
    "target/cate_std",
    "lr",
)

#: Curves shown in panels (a) and (b), in draw order: components first so the
#: total reads as their envelope rather than as a fourth series.
COMPONENTS = ("loss/cate_dist", "loss/cate_mse_weighted", "loss/total")


def sigma_window(run: TrainingRun) -> tuple[int, int]:
    """The steps over which ``sigma`` moves: constant, then linear decay."""
    config = run.config
    start = int(config.hl_warmup_sigma_steps)
    return start, start + int(config.hl_linear_decay_steps)


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #


def _small(ax) -> float:
    """One point below the axis-label size, for in-axes annotations."""
    return ax.xaxis.label.get_size() - 1


def _components(ax, history: History, *, log: bool, span=None, n_bins: int) -> None:
    frame = history.frame
    for key in COMPONENTS:
        summary = binned(frame["_step"], frame[key], log=log, span=span, n_bins=n_bins)
        draw_binned(ax, summary, series(key), band=key != "loss/total")
    ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
    ax.set_ylabel("loss")


def uniform_loss(run: TrainingRun) -> float:
    """The distributional term attained by a uniform density over the grid.

    A cross-entropy against a density, so the do-nothing value is the
    differential entropy of the uniform on ``[v_min, v_max]``, i.e.
    ``log(v_max - v_min)`` -- ``log 10 = 2.3026`` for the run's plus/minus 5
    grid. The measured first-step value is 2.3037, so the head is initialized
    *flat*: this line is where "has learned nothing" sits, and it is what makes
    panel (a) readable as a scale rather than as arbitrary units.
    """
    return float(np.log(run.config.cate_vmax - run.config.cate_vmin))


def panel_overview(ax, history: History, run: TrainingRun, n_bins: int) -> None:
    """(a) The objective and its terms over the whole run, log step."""
    lo, hi = sigma_window(run)
    ax.axvspan(lo, hi, **SCHEDULE_SPAN)
    _components(ax, history, log=True, span=None, n_bins=n_bins)
    reference_line(ax, uniform_loss(run), "uniform over the grid", x=0.72)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_title("objective and its terms")
    # Placed inside the axes rather than in the caption: the reader needs it at
    # the moment they see the kink, not two inches below.
    ax.annotate(
        r"$\sigma$ anneal",
        xy=(np.sqrt(lo * hi), 0.985),
        xycoords=("data", "axes fraction"),
        ha="center",
        va="top",
        color="0.35",
        fontsize=_small(ax),
    )


def panel_plateau(ax, history: History, run: TrainingRun, n_bins: int) -> None:
    """(b) The same curves past the anneal, linear step."""
    _, hi = sigma_window(run)
    last = float(history.frame["_step"].max())
    _components(ax, history, log=False, span=(hi, last), n_bins=n_bins)
    # Ticks in thousands with the factor in the label, rather than Matplotlib's
    # offset text, which is drawn in the corner where the xlabel already is.
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value / 1000:g}")
    )
    ax.set_xlabel(r"step ($\times 10^3$)")
    ax.set_title(f"past the anneal (step ${hi / 1000:g}$k on)")
    # Headroom for the legend below, rather than letting it sit on the mean
    # term: the panel's own range is narrow, so 45% is a few pixels of white.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.45 * (high - low))
    # The figure's only legend, and it lives here rather than in (a): the same
    # three series appear in both panels, (a) is crossed by its curves at every
    # corner, and (b) has an empty band between the mean term and the objective.
    ax.legend(loc="upper right")


def panel_mean_term_detail(ax, history: History, run: TrainingRun, n_bins: int) -> None:
    """Magnify the weighted mean term already shown in panels (a) and (b)."""
    frame = history.frame
    lo, hi = sigma_window(run)
    ax.axvspan(lo, hi, **SCHEDULE_SPAN)

    key = "loss/cate_mse_weighted"
    summary = binned(frame["_step"], frame[key], log=True, n_bins=n_bins)
    draw_binned(ax, summary, series(key))

    style = series(key)
    final = float(summary.median[-1])
    ax.annotate(
        f"${final:.3f}$",
        xy=(summary.x[-1], final),
        xytext=(-3, 7),
        textcoords="offset points",
        ha="right",
        color=style.color,
        fontsize=_small(ax),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_{\mathrm{mean}}\,\mathcal{L}_{\mathrm{mse}}$")
    ax.set_title("mean term in detail")


def panel_schedules(ax, history: History, run: TrainingRun) -> None:
    """``sigma`` and the learning rate, on twin axes."""
    frame = history.frame
    lo, hi = sigma_window(run)
    ax.axvspan(lo, hi, **SCHEDULE_SPAN)

    sigma = series("loss/hl_sigma")
    # Schedules are deterministic functions of the step, so they are drawn raw:
    # there is no noise to bin away and binning would round the kink at `hi`.
    ax.plot(frame["_step"], frame["loss/hl_sigma"], **sigma.line_kwargs())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\sigma$", color=sigma.color)
    ax.tick_params(axis="y", colors=sigma.color)

    final_sigma = float(frame["loss/hl_sigma"].iloc[-1])
    width = _bin_width(run)
    ax.annotate(
        f"${final_sigma:g} = {final_sigma / width:g}\\,w$",
        xy=(0.86, final_sigma),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        color=sigma.color,
        fontsize=_small(ax),
    )
    # No "sigma anneal" label here: this panel *is* sigma, so the shading needs
    # no gloss. It is labelled in (a), where the shading is the only clue.

    lr = series("lr")
    twin = ax.twinx()
    twin.plot(frame["_step"], frame["lr"], **lr.line_kwargs())
    twin.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value / 1e-4:g}")
    )
    twin.set_ylabel(r"learning rate ($\times 10^{-4}$)", color=lr.color)
    twin.tick_params(axis="y", colors=lr.color)
    twin.grid(False)  # One grid per panel; the left axis owns it.
    twin.set_ylim(bottom=0.0)
    ax.set_title("schedules")


def _bin_width(run: TrainingRun) -> float:
    from proximalfm.train._train_config import hl_bin_width

    return hl_bin_width(run.config)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def build_figure(
    history: History, run: TrainingRun, *, n_bins: int = 90
) -> "plt.Figure":
    """Assemble the four panels."""
    apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize_grid(ncols=2, nrows=2, panel_aspect=0.78),
    )

    # Column layout is load-bearing, not incidental: (c) sits under (a) so their
    # log-step axes and the shaded sigma window line up vertically, which is what
    # lets a reader read the anneal off (c) and find the same window in (a)
    # without translating between two x-axes.
    panel_overview(axes[0, 0], history, run, n_bins)
    panel_plateau(axes[0, 1], history, run, n_bins)
    panel_schedules(axes[1, 0], history, run)
    panel_mean_term_detail(axes[1, 1], history, run, n_bins)

    # Pin the log-step axis on every panel that uses one. Autoscaling would give
    # each its own limits -- a binned panel stops at its last bin's median step,
    # a raw one at the final step -- and the shaded window would then sit at a
    # slightly different x in (a) than in (c), which is the one thing the column
    # layout exists to prevent.
    span = (0.8, float(history.frame["_step"].max()) * 1.2)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.set_xlim(*span)

    for tag, ax in zip("abcd", axes.flat):
        panel_tag(ax, tag)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--run", default=DEFAULT_RUN, help="wandb run to plot.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default=NAME)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=90,
        help="Bins per panel for the median/IQR summary (default: 90).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="Shell variable override for the launch script. Repeatable.",
    )
    parser.add_argument("--force", action="store_true", help="Refetch from wandb.")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            parser.error(f"--set expects VAR=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = value

    run = parse_script(args.script, overrides)
    history = fetch(list(KEYS), run=args.run, force=args.force)
    fig = build_figure(history, run, n_bins=args.n_bins)
    paths = save_figure(fig, args.outdir, args.name)
    print("wrote " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
