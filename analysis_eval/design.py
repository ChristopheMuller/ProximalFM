"""Shared visual identity for evaluation figures.

Every method has an explicit style so its identity never depends on plotting
order. New figures should call :func:`method_style` rather than naming colours,
markers, or line styles inline. Unknown methods fail loudly until registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .plotting import plt

TEXT_WIDTH_IN = 6.07
FULL_PAGE_WIDTH_IN = 9.4

# Context size is an ordered visual channel in trajectory plots. Areas rather
# than radii grow moderately, leaving method-specific marker shapes readable.
CONTEXT_MARKER_AREAS = (18.0, 27.0, 38.0, 52.0)
ISO_NPEHE_LEVELS = (0.25, 0.5, 0.75, 1.0)

# Okabe--Ito plus dark, print-safe extensions. Colour identifies the estimator
# family; marker and line style distinguish variants within that family.
COLORS = {
    "black": "#000000",
    "grey": "#777777",
    "light_grey": "#AAAAAA",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "brown": "#8C564B",
}

# Shared visual grammar. Dashed/circle denotes a simple or linear variant,
# solid/polygon a nonlinear learned variant, and dotted a marginal/reference
# output. Foundation models and principal proximal competitors are thicker.
LINEWIDTH = 1.35
EMPHASIZED_LINEWIDTH = 1.75
PROPOSED_LINEWIDTH = 2.15
MARKERSIZE = 3.6
EMPHASIZED_MARKERSIZE = 4.2


@dataclass(frozen=True)
class MethodStyle:
    """Stable display properties for one benchmark output."""

    label: str
    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = LINEWIDTH
    markersize: float = MARKERSIZE
    zorder: int = 2
    alpha: float = 0.9

    def line_kwargs(self) -> dict[str, object]:
        return {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "markersize": self.markersize,
            "zorder": self.zorder,
            "alpha": self.alpha,
        }


# Registry order is legend and rendering order: conventional learners, proxy
# marginalisation, proximal learners, foundation models, the proposed method,
# then unranked references. Later, emphasized entries render over baselines.
METHOD_STYLES: Mapping[str, MethodStyle] = {
    # Conventional meta-learners: colour identifies S/T/X; marker identifies
    # OLS/RF/TabICL.
    "S-OLS": MethodStyle("S-OLS", COLORS["sky_blue"], "o", "--"),
    "S-RF": MethodStyle("S-RF", COLORS["sky_blue"], "s"),
    "S-TabICL": MethodStyle("S-TabICL", COLORS["sky_blue"], "D", ":"),
    "T-OLS": MethodStyle("T-OLS", COLORS["orange"], "o", "--"),
    "T-RF": MethodStyle("T-RF", COLORS["orange"], "s"),
    "T-TabICL": MethodStyle("T-TabICL", COLORS["orange"], "D", ":"),
    "X-OLS": MethodStyle("X-OLS", COLORS["purple"], "o", "--"),
    "X-RF": MethodStyle("X-RF", COLORS["purple"], "D"),
    "X-TabICL": MethodStyle("X-TabICL", COLORS["purple"], "s", ":"),
    # Proxy-marginalisation variants are deliberately neutral.
    "S-RF-WZ-Kernel-128": MethodStyle(
        "S-RF + kernel marginalisation", COLORS["grey"], "s", "--"
    ),
    "S-RF-WZ-kNN-128": MethodStyle(
        "S-RF + kNN marginalisation", COLORS["grey"], "s", ":"
    ),
    "S-TabICL-WZ-kNN-16": MethodStyle(
        "S-TabICL + kNN marginalisation", COLORS["grey"], "D", "-."
    ),
    # Classical proximal learners. Each estimator family has its own colour;
    # down/up markers encode conditional/independent-W variants.
    "P2SLS": MethodStyle("P2SLS", COLORS["brown"], "P", "--"),
    "P-Kernel-RF": MethodStyle("P-Kernel-RF", COLORS["brown"], "s"),
    "P-Kernel-OLS": MethodStyle("P-Kernel-OLS", COLORS["brown"], "o", "--"),
    "P-Kernel-ATE": MethodStyle("P-Kernel-ATE", COLORS["brown"], "P", ":"),
    "KPV-X-Wcond": MethodStyle(
        "KPV (conditional W)",
        COLORS["green"],
        "v",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "KPV-X-Wind": MethodStyle(
        "KPV (independent W)",
        COLORS["green"],
        "^",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "KPV-ATE": MethodStyle("KPV (ATE)", COLORS["green"], "P", ":"),
    "PMMR-X-Wcond": MethodStyle(
        "PMMR (conditional W)",
        COLORS["brown"],
        "v",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "PMMR-X-Wind": MethodStyle(
        "PMMR (independent W)",
        COLORS["brown"],
        "^",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "PMMR-ATE": MethodStyle("PMMR (ATE)", COLORS["brown"], "H", ":"),
    "NMMR-V-fixed-Wcond": MethodStyle(
        "NMMR (conditional W)",
        COLORS["purple"],
        "v",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-V-fixed-Wind": MethodStyle(
        "NMMR (independent W)",
        COLORS["purple"],
        "^",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    # Competing foundation models share a colour but use distinct markers.
    "CausalFM": MethodStyle(
        "CausalFM",
        COLORS["blue"],
        "P",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "DoPFN": MethodStyle(
        "Do-PFN",
        COLORS["blue"],
        "D",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "CausalPFN": MethodStyle(
        "CausalPFN",
        COLORS["blue"],
        "X",
        ":",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "CausalPFN-WZ-kNN-16": MethodStyle(
        "CausalPFN + kNN marginalisation",
        COLORS["blue"],
        "D",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "ProximalFM": MethodStyle(
        "ProximalFM",
        COLORS["vermillion"],
        "*",
        linewidth=PROPOSED_LINEWIDTH,
        markersize=5.8,
        zorder=5,
        alpha=1.0,
    ),
    # References are visually quiet and always render last.
    "Oracle-CATE(X)": MethodStyle("Oracle CATE(X)", COLORS["black"], "", ":"),
    "Oracle-ATE": MethodStyle("Oracle ATE", COLORS["grey"], "", ":"),
    "Oracle-ITE": MethodStyle("Oracle ITE", COLORS["light_grey"], "", ":"),
}


# Representative CATE(X) methods for figures whose legends or panels cannot
# accommodate the full registry. The selection spans conventional, proxy-
# marginalised, proximal, and foundation-model approaches. It is fixed in
# advance: plot selection must not depend on observed scores.
RELEVANT_METHODS: tuple[str, ...] = (
    "S-TabICL-WZ-kNN-16",
    "T-OLS",
    "S-OLS",
    "X-RF",
    "X-TabICL",
    "P-Kernel-RF",
    "P-Kernel-OLS",
    "KPV-X-Wcond",
    "PMMR-X-Wcond",
    "CausalPFN",
    "ProximalFM",
)


def method_style(method: str) -> MethodStyle:
    """Return the registered style for ``method`` or fail with a useful fix."""
    try:
        return METHOD_STYLES[method]
    except KeyError:
        raise KeyError(
            f"No analysis_eval style for method {method!r}; add it to "
            "analysis_eval.design.METHOD_STYLES"
        ) from None


def ordered_methods(methods: Iterable[str]) -> list[str]:
    """Return methods in registry order, validating complete style coverage."""
    methods = set(methods)
    unknown = methods - set(METHOD_STYLES)
    if unknown:
        method_style(sorted(unknown)[0])
    return [method for method in METHOD_STYLES if method in methods]


def relevant_methods(methods: Iterable[str]) -> list[str]:
    """Return available compact-figure methods in canonical order.

    Extra available methods are intentionally ignored. An empty intersection
    fails loudly because it usually means result names and the curated compact
    set have drifted apart.
    """
    available = set(methods)
    selected = [method for method in RELEVANT_METHODS if method in available]
    if not selected:
        raise ValueError(
            "none of analysis_eval.design.RELEVANT_METHODS are available; "
            "update the curated compact-figure method list"
        )
    return ordered_methods(selected)


_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "savefig.bbox": None,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.55,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Apply the common evaluation-figure theme."""
    plt.rcParams.update(_RC)


def learning_curve_figsize() -> tuple[float, float]:
    """Final canvas size for the eight-panel landscape figure."""
    return FULL_PAGE_WIDTH_IN, 5.25


def decomposition_figsize() -> tuple[float, float]:
    """Final canvas size for the eight-panel decomposition trajectories."""
    return FULL_PAGE_WIDTH_IN, 5.55


def speed_rank_figsize() -> tuple[float, float]:
    """Final canvas size for the two-mechanism speed--accuracy comparison."""
    return FULL_PAGE_WIDTH_IN, 6.0
