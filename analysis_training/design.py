"""Visual identity for the training-run figures.

Appendix C's figures sit next to Appendix B's, so they must be the *same*
figures typographically: same text width, same 9 pt serif, same colourblind-safe
palette, same constrained layout. That identity already exists as
:mod:`analysis_prior.design`, whose own docstring argues against exactly the
duplication a second copy here would be -- so this module **re-exports** it and
adds only what is specific to a training curve.

What is specific is two things.

**A series registry.** ``analysis_prior``'s :data:`~analysis_prior.design.BLOCK_COLORS`
pins a colour per *structural variable* (``U`` is always red). A training curve
has no structural variables; what recurs across Appendix C's figures is a set of
*logged quantities* -- the objective, its two terms, the two schedules -- and the
same rule applies to them: ``loss/total`` must be the same colour in every figure
it appears in. Hence :data:`SERIES`, the same idea one level over.

**Binned summaries.** Per-step metrics over 75 000 steps are far too noisy to
plot raw, and a fixed-width rolling window is the wrong smoother on a log axis:
it over-smooths the early transient and under-smooths the plateau.
:func:`binned` instead reduces to *median and interquartile band per bin*, with
the bins log- or linearly spaced to match the axis. Median rather than mean
because a handful of steps spike two orders of magnitude above the trend (the
objective's first-step value is 17.7 against a final -0.88) and a mean would
follow them; a band rather than a line alone because the spread *is* the
gradient noise, and hiding it would make the plateau look cleaner than it is.
This matches the p25/p75 convention the trainer and the benchmark already use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

# Re-exported so a figure module imports the identity from one place. Listed
# explicitly rather than star-imported: a name that vanishes upstream should
# break here loudly.
from analysis_prior.design import (  # noqa: F401
    FONT_SIZE_PT,
    GOLDEN,
    OKABE_ITO,
    ROLES,
    TEXT_WIDTH_IN,
    apply_style,
    figsize,
    figsize_grid,
)
from analysis_prior.plotting import plt, save_figure  # noqa: F401

# --------------------------------------------------------------------------- #
# series identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Series:
    """How one logged quantity is drawn, wherever it appears."""

    label: str  # Legend text, LaTeX-ready.
    color: str
    linestyle: str = "-"
    lw: float = 1.5

    def line_kwargs(self, **overrides) -> dict:
        return {
            "color": self.color,
            "linestyle": self.linestyle,
            "lw": self.lw,
            **overrides,
        }


#: Colour per logged quantity. The objective is achromatic because it is a *sum*
#: of the two coloured terms -- a reader should see the components as the content
#: and the total as their envelope, which a third hue would fight.
SERIES = {
    "loss/total": Series(r"total", "0.15", lw=1.7),
    "loss/cate_dist": Series(r"distributional term", OKABE_ITO["blue"]),
    "loss/cate_mse_weighted": Series(
        r"mean term ($\lambda_{\mathrm{mean}}\,\mathcal{L}_{\mathrm{mse}}$)",
        OKABE_ITO["orange"],
    ),
    "loss/cate_mse": Series(r"mean term", OKABE_ITO["orange"]),
    "loss/hl_sigma": Series(r"$\sigma$", OKABE_ITO["vermillion"]),
    "lr": Series(r"learning rate", OKABE_ITO["green"]),
    # Not a logged key: the mean head's residual variance as a fraction of the
    # target's, i.e. 1 - R^2 against predicting the context mean.
    "unexplained": Series(r"$1 - R^2$", OKABE_ITO["purple"]),
}


def series(key: str) -> Series:
    try:
        return SERIES[key]
    except KeyError:
        raise KeyError(
            f"No series style for {key!r}. Known: {sorted(SERIES)}"
        ) from None


#: Shading for a scheduled window (the sigma anneal), as kwargs for ``axvspan``.
SCHEDULE_SPAN = dict(color="0.5", alpha=0.10, lw=0, zorder=0)


# --------------------------------------------------------------------------- #
# binned summaries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Binned:
    """A metric reduced to per-bin median and interquartile band."""

    x: np.ndarray
    median: np.ndarray
    p25: np.ndarray
    p75: np.ndarray


def binned(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    n_bins: int = 90,
    log: bool = True,
    span: Tuple[float, float] | None = None,
) -> Binned:
    """Reduce a per-step series to median and p25/p75 per bin.

    ``log`` spaces the bins geometrically, which is what makes a log-x panel
    resolve the first hundred steps and the last ten thousand equally well; the
    plotted ``x`` is each bin's median step rather than its edge, so a sparse bin
    is drawn where its data actually is.
    """
    frame = pd.DataFrame(
        {"step": np.asarray(steps, float), "value": np.asarray(values, float)}
    )
    frame = frame.dropna()
    if span is not None:
        lo, hi = span
        frame = frame[(frame["step"] >= lo) & (frame["step"] <= hi)]
    if frame.empty:
        raise ValueError("no samples in the requested span")

    low, high = frame["step"].min(), frame["step"].max()
    if log:
        edges = np.geomspace(max(low, 1.0), high, n_bins + 1)
    else:
        edges = np.linspace(low, high, n_bins + 1)
    # `right=True` with the final edge folded in, so the last step lands in the
    # last bin instead of a bin of its own.
    index = np.clip(np.digitize(frame["step"], edges[1:-1], right=True), 0, n_bins - 1)

    grouped = frame.groupby(index)["value"]
    x = frame.groupby(index)["step"].median()
    return Binned(
        x=x.to_numpy(),
        median=grouped.median().to_numpy(),
        p25=grouped.quantile(0.25).to_numpy(),
        p75=grouped.quantile(0.75).to_numpy(),
    )


def draw_binned(
    ax,
    summary: Binned,
    style: Series,
    *,
    band: bool = True,
    label: str | None = None,
    **overrides,
):
    """Draw a :class:`Binned` as a median line with its interquartile band."""
    if band:
        ax.fill_between(
            summary.x,
            summary.p25,
            summary.p75,
            color=style.color,
            alpha=0.18,
            lw=0,
        )
    return ax.plot(
        summary.x,
        summary.median,
        label=style.label if label is None else label,
        **style.line_kwargs(**overrides),
    )


def reference_line(
    ax,
    y: float,
    text: str,
    *,
    x: float = 0.5,
    above: bool = True,
    role: str = "reference",
    **kwargs,
) -> None:
    """A horizontal baseline, labelled *on itself*.

    The label goes next to the line rather than into a legend, for the same
    reason ``analysis_prior``'s ``constraint_line`` exists: a reference is read
    where it is drawn. It also keeps legends down to the series that actually
    vary, which on a four-panel figure is the difference between a legend that
    fits in white space and one that covers a curve. ``x`` is an axes fraction,
    so the caller places the label wherever that panel happens to be empty.

    ``role`` picks the meaning from :data:`ROLES`, and picking the right one
    matters: ``reference`` is a constraint the data must respect, ``oracle`` is a
    level that needs the ground truth to reach, and ``naive`` is a deliberately
    wrong baseline. A level that only an oracle can hit must not be drawn as if
    it were free -- that is exactly the misreading that makes "the model beats
    the baseline" sound cheaper than it is.
    """
    style = {**ROLES[role], **kwargs}
    ax.axhline(y, **style)
    ax.annotate(
        text,
        xy=(x, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3 if above else -3),
        textcoords="offset points",
        ha="center",
        va="bottom" if above else "top",
        color=style["color"],
        fontsize=FONT_SIZE_PT - 2,
    )


def panel_tag(ax, tag: str) -> None:
    """Move the panel's title left and prefix it ``(a)``, ``(b)``, ...

    Matplotlib keeps one ``Text`` per title location, so the centred title has to
    be *cleared* rather than merely re-set -- otherwise both render, on top of
    each other. Left-aligned because a tag belongs at the reading edge.
    """
    title = ax.get_title()
    ax.set_title("")
    ax.set_title(f"({tag}) {title}", loc="left", fontsize=FONT_SIZE_PT)
