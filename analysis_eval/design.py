"""Shared visual identity for evaluation figures.

Every method has an explicit style so its identity never depends on plotting
order. New figures should call :func:`method_style` rather than naming colours,
markers, or line styles inline. Unknown methods fail loudly until registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .plotting import plt

TEXT_WIDTH_IN = 6.07
FULL_PAGE_WIDTH_IN = 9.4

# Okabe--Ito plus dark, print-safe extensions. Paired variants share a colour
# and differ in marker/line pattern, preserving method-family meaning.
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


@dataclass(frozen=True)
class MethodStyle:
    """Stable display properties for one benchmark output."""

    label: str
    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = 1.35
    markersize: float = 3.6
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


# Registry order is legend and rendering order. Foundation/proximal models are
# emphasized last so their lines remain visible above classical baselines.
METHOD_STYLES: Mapping[str, MethodStyle] = {
    "S-OLS": MethodStyle("S-OLS", COLORS["sky_blue"], "o", "--"),
    "S-RF": MethodStyle("S-RF", COLORS["sky_blue"], "s"),
    "T-OLS": MethodStyle("T-OLS", COLORS["orange"], "o", "--"),
    "T-RF": MethodStyle("T-RF", COLORS["orange"], "s"),
    "X-RF": MethodStyle("X-RF", COLORS["purple"], "D"),
    "P2SLS": MethodStyle("P2SLS", COLORS["brown"], "P", "--"),
    "P-Kernel-RF": MethodStyle("P-Kernel-RF", COLORS["brown"], "s"),
    "P-Kernel-OLS": MethodStyle("P-Kernel-OLS", COLORS["brown"], "o", "--"),
    "P-Kernel-ATE": MethodStyle("P-Kernel-ATE", COLORS["brown"], "P", ":"),
    "NMMR-V-fixed-Wind": MethodStyle(
        "NMMR (independent W)", COLORS["green"], "^", "--"
    ),
    "NMMR-V-fixed-Wcond": MethodStyle("NMMR (conditional W)", COLORS["green"], "v"),
    "CausalPFN": MethodStyle(
        "CausalPFN", COLORS["blue"], "X", linewidth=1.75, zorder=3
    ),
    "ProximalFM": MethodStyle(
        "ProximalFM",
        COLORS["vermillion"],
        "*",
        linewidth=2.15,
        markersize=5.8,
        zorder=5,
        alpha=1.0,
    ),
    "Oracle-CATE(X)": MethodStyle("Oracle CATE(X)", COLORS["black"], "", ":"),
    "Oracle-ATE": MethodStyle("Oracle ATE", COLORS["grey"], "", ":"),
    "Oracle-ITE": MethodStyle("Oracle ITE", COLORS["light_grey"], "", ":"),
}


def method_style(method: str) -> MethodStyle:
    """Return the registered style for ``method`` or fail with a useful fix."""
    try:
        return METHOD_STYLES[method]
    except KeyError:
        raise KeyError(
            f"No analysis_eval style for method {method!r}; add it to "
            "analysis_eval.design.METHOD_STYLES"
        ) from None


def ordered_methods(methods: set[str] | list[str]) -> list[str]:
    """Return methods in registry order, validating complete style coverage."""
    methods = set(methods)
    unknown = methods - set(METHOD_STYLES)
    if unknown:
        method_style(sorted(unknown)[0])
    return [method for method in METHOD_STYLES if method in methods]


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
