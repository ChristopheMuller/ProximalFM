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
    "charcoal": "#3C3C3C",
    "grey": "#777777",
    "light_grey": "#AAAAAA",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "brown": "#8C564B",
    "red": "#D62728",
    "wine": "#882255",
}

# Shared visual grammar. Colour identifies an estimator or base-regressor
# family. Marker and line style identify variants within that family. The
# combination (colour, marker, line style) is unique across the registry, so
# every method remains identifiable in colour and in monochrome.
LINEWIDTH = 1.45
EMPHASIZED_LINEWIDTH = 1.85
PROPOSED_LINEWIDTH = 2.15
MARKERSIZE = 4.0
EMPHASIZED_MARKERSIZE = 4.6

META_COLORS = {
    "OLS": COLORS["sky_blue"],
    "RF": COLORS["orange"],
    "TabICL": COLORS["purple"],
}
META_MARKERS = {"OLS": "o", "RF": "s", "TabICL": "D"}
META_LINESTYLES = {"S": "--", "T": "-", "X": ":"}
MARGINALIZED_LINESTYLE = "-."
# Causal Chambers uses a compact, eight-method legend. These additional
# patterns separate otherwise similar curves without making the panel grid a
# field of dotted lines. They are intentionally used only by its ATE outputs.
ATE_SHORT_DASH = (0, (4.0, 2.0))
ATE_MEDIUM_DASH = (0, (6.5, 2.0))
ATE_LONG_DASH = (0, (9.0, 2.5))
ATE_DASH_DOT_DOT = (0, (5.0, 1.5, 1.0, 1.5, 1.0, 1.5))


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
    alpha: float = 0.95

    def line_kwargs(self) -> dict[str, object]:
        return {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "markersize": self.markersize,
            "markeredgewidth": 0.65,
            "zorder": self.zorder,
            "alpha": self.alpha,
            "solid_capstyle": "round",
            "dash_capstyle": "round",
        }

    def identity(self) -> tuple[str, str, str]:
        """Return the visual channels that distinguish methods in a legend."""
        return self.color, self.marker, self.linestyle


# Registry order is legend and rendering order: conventional learners, proxy
# marginalisation, proximal learners, foundation models, the proposed method,
# then unranked references. Later, emphasized entries render over baselines.
METHOD_STYLES: Mapping[str, MethodStyle] = {
    # Conventional meta-learners: colour and marker identify OLS/RF/TabICL;
    # line style identifies the S/T/X learner.
    "S-OLS": MethodStyle(
        "S-learner (ridge)",
        META_COLORS["OLS"],
        META_MARKERS["OLS"],
        META_LINESTYLES["S"],
    ),
    "S-RF": MethodStyle(
        "S-learner (RF)",
        META_COLORS["RF"],
        META_MARKERS["RF"],
        META_LINESTYLES["S"],
    ),
    "S-TabICL": MethodStyle(
        "S-learner (TabICL)",
        META_COLORS["TabICL"],
        META_MARKERS["TabICL"],
        META_LINESTYLES["S"],
    ),
    "T-OLS": MethodStyle(
        "T-learner (ridge)",
        META_COLORS["OLS"],
        META_MARKERS["OLS"],
        META_LINESTYLES["T"],
    ),
    "T-RF": MethodStyle(
        "T-learner (RF)",
        META_COLORS["RF"],
        META_MARKERS["RF"],
        META_LINESTYLES["T"],
    ),
    "T-TabICL": MethodStyle(
        "T-learner (TabICL)",
        META_COLORS["TabICL"],
        META_MARKERS["TabICL"],
        META_LINESTYLES["T"],
    ),
    "X-OLS": MethodStyle(
        "X-learner (ridge)",
        META_COLORS["OLS"],
        META_MARKERS["OLS"],
        META_LINESTYLES["X"],
    ),
    "X-RF": MethodStyle(
        "X-learner (RF)",
        META_COLORS["RF"],
        META_MARKERS["RF"],
        META_LINESTYLES["X"],
    ),
    "X-TabICL": MethodStyle(
        "X-learner (TabICL)",
        META_COLORS["TabICL"],
        "d",
        META_LINESTYLES["X"],
    ),
    # Proxy-marginalisation variants retain the parent estimator's colour.
    # Their dash-dot line distinguishes marginalised outputs from the parent;
    # the marker identifies the integration rule (diamond=kernel, hexagon=kNN).
    "S-RF-WZ-Kernel-128": MethodStyle(
        "S-learner (RF) + kernel marg.",
        META_COLORS["RF"],
        "d",
        MARGINALIZED_LINESTYLE,
    ),
    "S-RF-WZ-kNN-128": MethodStyle(
        "S-learner (RF) + kNN marg.",
        META_COLORS["RF"],
        "h",
        MARGINALIZED_LINESTYLE,
    ),
    "S-TabICL-WZ-kNN-16": MethodStyle(
        "S-learner (TabICL) + kNN marg.",
        META_COLORS["TabICL"],
        "h",
        MARGINALIZED_LINESTYLE,
    ),
    # Classical proximal learners. P-learner outputs share a colour and solid
    # line; their OLS/RF/TabICL markers match the meta-learner grammar above.
    "P2SLS": MethodStyle("P2SLS", COLORS["grey"], "P", ATE_SHORT_DASH),
    "P-Kernel-RF": MethodStyle("P-learner (RF)", COLORS["green"], "s"),
    "P-Kernel-OLS": MethodStyle("P-learner (ridge)", COLORS["green"], "o"),
    "P-Kernel-TabICL": MethodStyle("P-learner (TabICL)", COLORS["green"], "D"),
    "P-Kernel-ATE": MethodStyle("P-Kernel-ATE", COLORS["green"], "P", ":"),
    "KPV-X-Wcond": MethodStyle(
        "KPV (cond. W)",
        COLORS["charcoal"],
        "v",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "KPV-X-Wind": MethodStyle(
        "KPV (independent W)",
        COLORS["charcoal"],
        "^",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "KPV-ATE": MethodStyle(
        "KPV (ATE)", COLORS["charcoal"], "P", MARGINALIZED_LINESTYLE
    ),
    "PMMR-X-Wcond": MethodStyle(
        "PMMR (cond. W)",
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
    "PMMR-ATE": MethodStyle("PMMR (ATE)", COLORS["brown"], "H", ATE_LONG_DASH),
    "NMMR-V-fixed-Wcond": MethodStyle(
        "NMMR (cond. W)",
        COLORS["wine"],
        "v",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-V-fixed-Wind": MethodStyle(
        "NMMR (independent W)",
        COLORS["wine"],
        "^",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-V-fixed-ATE": MethodStyle(
        "NMMR (V, fixed; ATE)",
        COLORS["wine"],
        "H",
        ATE_DASH_DOT_DOT,
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-U-fixed-ATE": MethodStyle(
        "NMMR (U, fixed; ATE)",
        COLORS["wine"],
        "H",
        "--",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-V-median-ATE": MethodStyle(
        "NMMR (V, median; ATE)",
        COLORS["wine"],
        "8",
        ":",
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=3,
    ),
    "NMMR-U-median-ATE": MethodStyle(
        "NMMR (U, median; ATE)",
        COLORS["wine"],
        "8",
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
    "CausalFM-XWZ": MethodStyle(
        "CausalFM (XWZ)",
        COLORS["blue"],
        "P",
        MARGINALIZED_LINESTYLE,
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
    "DoPFN-XWZ": MethodStyle(
        "Do-PFN (XWZ)",
        COLORS["blue"],
        "D",
        MARGINALIZED_LINESTYLE,
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
    "CausalPFN-XWZ": MethodStyle(
        "CausalPFN (XWZ)",
        COLORS["blue"],
        "X",
        ATE_MEDIUM_DASH,
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "CausalPFN-WZ-kNN-16": MethodStyle(
        "CausalPFN + kNN marg.",
        COLORS["blue"],
        "h",
        MARGINALIZED_LINESTYLE,
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
        zorder=4,
    ),
    "CausalPFN-WZ-kernel-16": MethodStyle(
        "CausalPFN + kernel marg.",
        COLORS["blue"],
        "d",
        MARGINALIZED_LINESTYLE,
        linewidth=EMPHASIZED_LINEWIDTH,
        markersize=EMPHASIZED_MARKERSIZE,
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
    "ProximalFM-FT": MethodStyle(
        "ProximalFM-FT",
        COLORS["vermillion"],
        "*",
        ":",
        linewidth=PROPOSED_LINEWIDTH,
        markersize=5.8,
        zorder=5,
        alpha=1.0,
    ),
    "ProximalFM-ATE": MethodStyle(
        "ProximalFM (ATE)",
        COLORS["vermillion"],
        "p",
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


def _validate_method_styles(styles: Mapping[str, MethodStyle]) -> None:
    """Reject ambiguous identities when the registry is edited."""
    owners: dict[tuple[str, str, str], str] = {}
    labels: dict[str, str] = {}
    for method, style in styles.items():
        if style.label in labels:
            raise ValueError(
                f"methods {labels[style.label]!r} and {method!r} share label "
                f"{style.label!r}"
            )
        labels[style.label] = method

        identity = style.identity()
        if identity in owners:
            raise ValueError(
                f"methods {owners[identity]!r} and {method!r} share visual "
                f"identity {identity!r}"
            )
        owners[identity] = method


_validate_method_styles(METHOD_STYLES)


# Representative CATE(X) methods for figures whose legends or panels cannot
# accommodate the full registry. The selection spans conventional, proxy-
# marginalised, proximal, and foundation-model approaches. It is fixed in
# advance: plot selection must not depend on observed scores.
RELEVANT_METHODS: tuple[str, ...] = (
    "T-OLS",
    "S-TabICL-WZ-kNN-16",
    "P-Kernel-OLS",
    "P-Kernel-RF",
    "P-Kernel-TabICL",
    "NMMR-V-fixed-Wcond",
    "KPV-X-Wcond",
    "PMMR-X-Wcond",
    "CausalPFN",
    "CausalPFN-WZ-kNN-16",
    "ProximalFM",
    "ProximalFM-FT",
)

SELECTION_CAUSAL_CHAMERS: tuple[str, ...] = (
    "X-TabICL",
    "CausalPFN-XWZ",
    "P-Kernel-ATE",
    "P2SLS",
    "KPV-ATE",
    "PMMR-ATE",
    "NMMR-V-fixed-ATE",
    "ProximalFM-ATE",
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
    return FULL_PAGE_WIDTH_IN, 4.55


def decomposition_figsize() -> tuple[float, float]:
    """Final canvas size for the eight-panel decomposition trajectories."""
    return FULL_PAGE_WIDTH_IN, 5.55


def speed_rank_figsize() -> tuple[float, float]:
    """Final canvas size for the two-mechanism speed--accuracy comparison."""
    return FULL_PAGE_WIDTH_IN, 4.25
