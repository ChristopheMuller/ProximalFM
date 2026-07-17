"""The visual identity of the prior audits: one place to change how figures look.

Why a module rather than per-figure choices
-------------------------------------------
Two different problems, two different mechanisms:

- **Look** -- fonts, sizes, grid, spines. Global and declarative, so it belongs
  in Matplotlib's ``rcParams`` (:func:`apply_style`). Nothing else should touch
  them.
- **Meaning** -- the confounder ``U`` must be the same colour in *every* figure
  it appears in; "theory" must always read as a reference overlay. ``rcParams``
  cannot express this: its ``prop_cycle`` assigns colours by *plot order*, so
  ``d_U`` would be red in one figure and blue in the next. Meaning needs an
  explicit registry, which is what :data:`BLOCK_COLORS`, :data:`ROLES` and
  :func:`variant_styles` are.

The rule for analyses: **never name a colour, size or linestyle inline** -- ask
this module. A reader who learns "red = latent confounder" once should never
have to relearn it.

Sizing and the paper
--------------------
Figures are saved at their true final size (see :data:`TEXT_WIDTH_IN` and
:func:`figsize`) and included **without scaling**::

    \\includegraphics{figure.pdf}      % note: no width=... , by design

If a figure is saved 14in wide and then squeezed into a 6in text block, its
9pt labels render at ~4pt -- and a differently-sized figure lands at a
different effective font size. Sizing here, not in LaTeX, is what keeps type
consistent across the appendix.

For the same reason the layout engine is ``constrained`` rather than
``savefig.bbox="tight"``: *tight* crops the canvas after the fact, so the saved
width no longer matches what was asked for. Constrained layout fits labels
*within* the requested size, keeping it exact.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
from matplotlib.colors import to_hex

from .plotting import plt

# ---------------------------------------------------------------------------
# Document geometry
# ---------------------------------------------------------------------------

#: Text width of the paper, in inches. a4 (8.27in) minus the 1.1in margins set
#: by ``\geometry`` in ``docs/latex/preamble.tex``. If the geometry changes,
#: measure the truth with ``\the\textwidth`` in the document (the result is in
#: TeX points; divide by 72.27) and update this single number.
TEXT_WIDTH_IN = 6.07

#: Default panel aspect (height/width). 1/phi -- unobtrusive and conventional.
GOLDEN = 0.618


def figsize(fraction: float = 1.0, aspect: float = GOLDEN) -> Tuple[float, float]:
    """Size for a single-panel figure, as a fraction of the text width.

    ``fraction=1.0`` spans the text block; ``0.5`` sits two-up.
    """
    width = fraction * TEXT_WIDTH_IN
    return (width, width * aspect)


def figsize_grid(
    ncols: int,
    nrows: int = 1,
    fraction: float = 1.0,
    panel_aspect: float = 1.0,
    extra_height_in: float = 0.0,
) -> Tuple[float, float]:
    """Size for an ``nrows`` x ``ncols`` panel grid, fitted to the text width.

    Panel height follows from the width so that panels stay square-ish
    regardless of how many there are -- the alternative (fixing total height)
    makes panels squash as columns are added.

    Parameters
    ----------
    panel_aspect : float, default=1.0
        Height/width of one panel. ``1.0`` is square; use ``GOLDEN`` for wide.

    extra_height_in : float, default=0.0
        Headroom for a suptitle or a shared legend.
    """
    width = fraction * TEXT_WIDTH_IN
    panel_w = width / ncols
    return (width, panel_w * panel_aspect * nrows + extra_height_in)


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

#: Okabe--Ito: the standard colourblind-safe qualitative palette. Used as the
#: source for every categorical colour below, so nothing in the appendix relies
#: on hue distinctions that ~8% of male readers cannot make.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

#: Colour per structural variable. Deliberately echoes the main-text DAG
#: (``figures_main/prior_dag.tex``), where the latent confounder is drawn in red
#: and the observed nodes in blues/greens: a reader who has seen the DAG already
#: knows what red means. Keyed by bare name; :func:`block_color` also accepts
#: the ``d_U`` dimension spelling.
BLOCK_COLORS = {
    "U": OKABE_ITO["vermillion"],  # latent -- red, as in the DAG
    "X": OKABE_ITO["blue"],  # baseline covariates
    "W": OKABE_ITO["green"],  # outcome-inducing proxy
    "Z": OKABE_ITO["orange"],  # treatment-inducing proxy
    "A": OKABE_ITO["purple"],  # treatment
    "Y": OKABE_ITO["sky_blue"],  # outcome
}


def block_color(key: str) -> str:
    """Colour for a structural variable. Accepts ``"U"`` or ``"d_U"``."""
    name = key[2:] if key.startswith("d_") else key
    try:
        return BLOCK_COLORS[name]
    except KeyError:
        raise KeyError(
            f"No colour for block {key!r}. Known: {sorted(BLOCK_COLORS)}"
        ) from None


#: How recurring *roles* are drawn, as kwargs ready to splat into a plot call.
#: These encode reading conventions rather than identities:
#:
#: - ``theory``   -- an analytic law overlaid on measured data.
#: - ``reference`` -- a constraint or boundary the data must respect.
#: - ``oracle``   -- an unachievable floor (e.g. adjustment given true ``U``).
#: - ``naive``    -- a deliberately wrong baseline.
ROLES: Dict[str, Dict] = {
    "theory": dict(color="black", linestyle="--", marker="o", markersize=3, lw=1.2),
    "reference": dict(color=OKABE_ITO["vermillion"], linestyle="--", lw=1.2),
    "oracle": dict(color="black", linestyle=":", lw=1.4),
    "naive": dict(color="0.5", linestyle="-.", lw=1.2),
}

#: Sequential colormap for counts/densities (heatmaps). Perceptually uniform and
#: colourblind-safe; also survives greyscale printing.
COUNT_CMAP = "viridis"

#: Sequential colormap for ordered variant sweeps (e.g. K = 1, 5, 25, ...).
SWEEP_CMAP = "plasma"

#: Categorical colours for variants with no fixed meaning, in assignment order.
_VARIANT_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["vermillion"],
]

#: Variants whose colour is pinned across every figure and every family, because
#: they carry a fixed meaning in the appendix. Anything else is assigned from
#: ``_VARIANT_CYCLE`` in family order.
VARIANT_COLORS = {
    "triangular": OKABE_ITO["blue"],  # the prior as shipped
    "uniform": OKABE_ITO["orange"],  # pre-Option-B d_U sampling
    "unenforced": OKABE_ITO["green"],  # completeness enforcement off
    "full": OKABE_ITO["blue"],  # the unablated prior
}


def variant_styles(
    labels: Sequence[str], ordered: bool = False
) -> Dict[str, Dict[str, str]]:
    """Map variant labels to a stable visual style, in the order given.

    Two kinds of family need two kinds of palette:

    - **categorical** (default) -- ablations like triangular / uniform /
      unenforced. Known labels keep their pinned colour across every figure
      (:data:`VARIANT_COLORS`); unknown ones are assigned from a colourblind-safe
      cycle in family order, so the baseline (listed first by the
      identity-first convention) is always the first colour.
    - **ordered** (``ordered=True``) -- sweeps like ``K = 1, 5, 25, ...``, where
      the labels have a natural order and a *sequential* ramp shows it. A
      categorical palette would throw that information away.

    Returns ``{label: {"color": ...}}`` -- a dict so callers can splat it and so
    later additions (hatch, marker) do not break call sites.
    """
    labels = list(labels)
    if ordered:
        cmap = matplotlib.colormaps[SWEEP_CMAP]
        n = max(len(labels) - 1, 1)
        # Stop short of the very top of the ramp: it is pale on most sequential
        # maps and disappears against white.
        return {
            label: {"color": to_hex(cmap(0.1 + 0.75 * i / n))}
            for i, label in enumerate(labels)
        }

    styles: Dict[str, Dict[str, str]] = {}
    next_free = 0
    for label in labels:
        if label in VARIANT_COLORS:
            styles[label] = {"color": VARIANT_COLORS[label]}
            continue
        styles[label] = {"color": _VARIANT_CYCLE[next_free % len(_VARIANT_CYCLE)]}
        next_free += 1
    return styles


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

#: Base font size, in points, as rendered *in the paper*. Figures are included
#: unscaled, so this is literal: slightly below the document's 10pt body, which
#: is the usual convention for figure text.
FONT_SIZE_PT = 9

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    # Constrained layout fits labels *inside* the requested figure size;
    # savefig bbox="tight" would instead crop afterwards, silently changing the
    # width and defeating the sizing scheme documented at the top of this module.
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "font.size": FONT_SIZE_PT,
    "axes.titlesize": FONT_SIZE_PT,
    "axes.labelsize": FONT_SIZE_PT,
    "xtick.labelsize": FONT_SIZE_PT - 1,
    "ytick.labelsize": FONT_SIZE_PT - 1,
    "legend.fontsize": FONT_SIZE_PT - 1,
    "figure.titlesize": FONT_SIZE_PT + 1,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,  # grid behind the data, never over it
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "image.cmap": COUNT_CMAP,
    "axes.prop_cycle": plt.cycler(color=_VARIANT_CYCLE),
    # Match the paper's serif body text; mathtext in labels then agrees with
    # the surrounding LaTeX rather than sitting in a different typeface.
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
}


def apply_style() -> None:
    """Apply the shared figure style. Idempotent; call once per render."""
    plt.rcParams.update(_RC)


# ---------------------------------------------------------------------------
# Recurring idioms
# ---------------------------------------------------------------------------


def constraint_line(ax, lo: float, hi: float, label: str = None, **kwargs) -> None:
    """Draw a ``y = x`` boundary that the data is expected to respect.

    Used for the completeness heuristic, where all mass must sit on or below
    ``d_U = d_proxy``. Styled as :data:`ROLES`\\ ``["reference"]``.
    """
    style = {**ROLES["reference"], **kwargs}
    ax.plot([lo, hi], [lo, hi], label=label, **style)
    if label:
        ax.legend(loc="upper left")


def theory_line(ax, x: Iterable, y: Iterable, label: str = "theory", **kwargs):
    """Overlay an analytic law on measured data. Styled as ``ROLES["theory"]``."""
    style = {**ROLES["theory"], **kwargs}
    return ax.plot(list(x), list(y), label=label, **style)


def grouped_bar_offsets(n_series: int, total_width: float = 0.8) -> List[float]:
    """Bar offsets that centre ``n_series`` grouped bars on each tick.

    Returns one offset per series; the matching bar width is
    ``total_width / n_series``.
    """
    width = total_width / n_series
    return [(i - (n_series - 1) / 2) * width for i in range(n_series)]
