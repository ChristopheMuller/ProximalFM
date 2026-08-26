"""plot_style.py — how the benchmark figures look, in one place.

Nothing here reads a results file or decides *what* to draw; it owns colour,
markers, ordering, the reference lines a metric is read against, and the two
mechanical chores (theme, saving). :mod:`benchmarks.plots` owns the figures and
:mod:`benchmarks.plot_frames` the numbers.

Two conventions worth keeping:

* **Colour is the method family, marker is the variant within it.** Twenty-odd
  methods do not get twenty-odd distinguishable colours, so a reader locates
  "the kernel methods" by colour and the strategy within them by marker. The
  table below is the registry; entries stay in it while a method is commented
  out of the config, so re-enabling one does not silently recolour the figures.
* **Anchors travel with the metric, not with the figure.** ``nPEHE = 1`` means
  "lost to the constant-ATE predictor" on every plot that shows an nPEHE, so it
  is declared once in :data:`METRIC_ANCHORS` and drawn by whichever figure shows
  that metric. A metric added to a :class:`~benchmarks.metrics.Task` without an
  entry here simply gets no reference line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

METHOD_STYLES = {
    "S-OLS": {"color": "#1f77b4", "marker": "o"},
    "S-OLS+AX": {"color": "#1f77b4", "marker": "s"},
    "S-OLS-WZ": {"color": "#2b5c8f", "marker": "v"},
    "S-OLS-WZ-Oracle": {"color": "#4a7bb0", "marker": "h", "linestyle": ":"},
    "S-OLS-Oracle": {"color": "#4a7bb0", "marker": "*", "s": 160, "linestyle": ":"},
    "S-OLS+AX-Oracle": {"color": "#4a7bb0", "marker": "s", "linestyle": ":"},
    "T-OLS": {"color": "#1f77b4", "marker": "^"},
    "T-OLS-WZ": {"color": "#2b5c8f", "marker": ">"},
    "T-OLS-Oracle": {"color": "#4a7bb0", "marker": "*", "s": 160, "linestyle": ":"},
    "T-OLS-WZ-Oracle": {"color": "#4a7bb0", "marker": "H", "linestyle": ":"},
    "S-RF": {"color": "#2ca02c", "marker": "o"},
    "S-RF-WZ": {"color": "#238b45", "marker": "v"},
    "S-RF-Oracle": {"color": "#41ab5d", "marker": "*", "s": 160, "linestyle": ":"},
    "S-RF-ITE-Oracle": {"color": "#005a32", "marker": "h", "s": 150, "linestyle": ":"},
    "T-RF": {"color": "#2ca02c", "marker": "^"},
    "T-RF-WZ": {"color": "#238b45", "marker": ">"},
    "T-RF-Oracle": {"color": "#41ab5d", "marker": "*", "s": 160, "linestyle": ":"},
    "X-RF": {"color": "#2ca02c", "marker": "P"},
    "X-RF-W": {"color": "#238b45", "marker": "p"},
    "X-RF-Oracle": {"color": "#41ab5d", "marker": "*", "s": 160, "linestyle": ":"},
    "X-RF-WZ-Oracle": {"color": "#005a32", "marker": "H", "s": 150, "linestyle": ":"},
    "CausalForest": {"color": "#137e36", "marker": "D"},
    "S-TabICL": {"color": "#9467bd", "marker": "o"},
    "S-TabICL-Oracle": {"color": "#756bb1", "marker": "*", "s": 180, "linestyle": ":"},
    "S-TabICL-XWU": {"color": "#542788", "marker": "h", "s": 170, "linestyle": ":"},
    "T-TabICL": {"color": "#9467bd", "marker": "^"},
    "T-TabICL-Oracle": {"color": "#756bb1", "marker": "X", "s": 160, "linestyle": ":"},
    "X-TabICL": {"color": "#9467bd", "marker": "v"},
    "X-TabICL-Oracle": {"color": "#756bb1", "marker": "P", "s": 160, "linestyle": ":"},
    "P2SLS": {"color": "#ff7f0e", "marker": "o"},
    "P2SLS+AX": {"color": "#ff7f0e", "marker": "s"},
    "P-Kernel": {"color": "#008080", "marker": "o"},
    "P-Kernel-RF": {"color": "#008080", "marker": "o"},
    "P-Kernel-OLS": {"color": "#008080", "marker": "s"},
    "P-Kernel-TabICL": {"color": "#008080", "marker": "^"},
    "P-Kernel-ATE": {"color": "#008080", "marker": "D"},
    "P-RF": {"color": "#008080", "marker": "o"},
    "P-ATE": {"color": "#008080", "marker": "D"},
    "KPV": {"color": "#17becf", "marker": "o"},
    "KPV-XW": {"color": "#17becf", "marker": "o"},
    "KPV-X-Wind": {"color": "#17becf", "marker": "s"},
    "KPV-X-Wcond": {"color": "#17becf", "marker": "^"},
    "KPV-ATE": {"color": "#17becf", "marker": "D"},
    "PMMR": {"color": "#8c564b", "marker": "o"},
    "PMMR-XW": {"color": "#8c564b", "marker": "o"},
    "PMMR-X-Wind": {"color": "#8c564b", "marker": "s"},
    "PMMR-X-Wcond": {"color": "#8c564b", "marker": "^"},
    "PMMR-ATE": {"color": "#8c564b", "marker": "D"},
    "NMMR-V": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-XW": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-Wind": {"color": "#e377c2", "marker": "s"},
    "NMMR-V-Wcond": {"color": "#e377c2", "marker": "^"},
    "NMMR-V-fixed": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-fixed-XW": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-fixed-Wind": {"color": "#e377c2", "marker": "s"},
    "NMMR-V-fixed-Wcond": {"color": "#e377c2", "marker": "^"},
    "NMMR-V-median": {"color": "#f768a1", "marker": "o"},
    "NMMR-V-median-XW": {"color": "#f768a1", "marker": "o"},
    "NMMR-V-median-Wind": {"color": "#f768a1", "marker": "s"},
    "NMMR-V-median-Wcond": {"color": "#f768a1", "marker": "^"},
    "NMMR-U": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-XW": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-Wind": {"color": "#c51b8a", "marker": "s"},
    "NMMR-U-Wcond": {"color": "#c51b8a", "marker": "^"},
    "NMMR-U-fixed": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-fixed-XW": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-fixed-Wind": {"color": "#c51b8a", "marker": "s"},
    "NMMR-U-fixed-Wcond": {"color": "#c51b8a", "marker": "^"},
    "NMMR-U-median": {"color": "#7a0177", "marker": "o"},
    "NMMR-U-median-XW": {"color": "#7a0177", "marker": "o"},
    "NMMR-U-median-Wind": {"color": "#7a0177", "marker": "s"},
    "NMMR-U-median-Wcond": {"color": "#7a0177", "marker": "^"},
    "DFPV": {"color": "#bcbd22", "marker": "o"},
    "DFPV-XW": {"color": "#bcbd22", "marker": "o"},
    "DFPV-X-Wind": {"color": "#bcbd22", "marker": "s"},
    "DFPV-X-cond": {"color": "#bcbd22", "marker": "^"},
    "DFPV-X-Wcond": {"color": "#bcbd22", "marker": "^"},
    "DFPV-ATE": {"color": "#bcbd22", "marker": "D"},
    "CausalFM": {"color": "#e6550d", "marker": "X", "s": 140},
    "DoPFN": {"color": "#d95f02", "marker": "D", "s": 130},
    "CausalPFN": {"color": "#7570b3", "marker": "P", "s": 140},
    "CausalPFN-XWZ": {"color": "#542788", "marker": "p", "s": 140},
    "CausalPFN-XWU": {"color": "#3f007d", "marker": "h", "s": 150, "linestyle": ":"},
    "ProximalFM": {"color": "#d62728", "marker": "*", "s": 220},
    "ProximalFM-75k": {"color": "#67000d", "marker": "D", "s": 150},
    "ProximalFM-50k": {"color": "#860303", "marker": "h", "s": 150},
    "ProximalFM-25k": {"color": "#a50f15", "marker": "p", "s": 140},
    "ProximalFM-10k": {"color": "#cb181d", "marker": "s", "s": 140},
    "ProximalFM-5k": {"color": "#ef3b2c", "marker": "^", "s": 130},
    "ProximalFM-3k": {"color": "#fb6a4a", "marker": "v", "s": 130},
    "ProximalFM-1": {"color": "#fc9272", "marker": "1", "s": 120},
    "ProximalFM3": {"color": "#d62728", "marker": "*", "s": 200},
    "ProximalFM4": {"color": "#b2182b", "marker": "p", "s": 160},
    "ProximalFM5": {"color": "#990000", "marker": "h", "s": 160},
    "Oracle-CATE(X)": {"color": "#000000", "marker": "*", "s": 220},
    "Oracle-ATE": {"color": "#666666", "marker": "D", "s": 130},
    "Oracle-ITE": {"color": "#999999", "marker": "v", "s": 130},
}
REFERENCE_STYLE = {"linestyle": "--", "linewidth": 1.6, "alpha": 0.9}

_FAMILIES = list(METHOD_STYLES)


def style_for(method: str) -> dict:
    """Style lookup: exact name, else longest prefix match, else grey.

    Prefix matching covers checkpoint-suffixed output names (e.g.
    ``ProximalFM3_55350`` inherits the ``ProximalFM3`` style).
    """
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    prefixes = [k for k in METHOD_STYLES if method.startswith(k)]
    if prefixes:
        return METHOD_STYLES[max(prefixes, key=len)]
    return {"color": "gray", "marker": "o"}


def color_for(method: str) -> str:
    return style_for(method).get("color", "gray")


def family_key(method: str) -> tuple[int, str]:
    """Sort key grouping a method with its family, families in registry order.

    Used where the ordering carries no information (legends, tick labels on a
    figure already sorted by something else) so that related methods sit
    together instead of alphabetically interleaved.
    """
    matches = [k for k in _FAMILIES if method.startswith(k)]
    if not matches:
        return (len(_FAMILIES), method)
    return (_FAMILIES.index(max(matches, key=len)), method)


def abbreviate(method: str, width: int = 24) -> str:
    """Tick-label spelling of a method name: long names lose their middle."""
    if len(method) <= width:
        return method
    keep = width - 1
    return method[: keep // 2] + "…" + method[-(keep - keep // 2) :]


# ─────────────────────────────────────────────────────────────────────────────
# Reference lines
# ─────────────────────────────────────────────────────────────────────────────
#: value → (label, colour) per metric key. Drawn by any figure showing that
#: metric, so "nPEHE = 1 is the constant-ATE baseline" is stated identically
#: everywhere. Metrics absent from this map simply get no reference line.
METRIC_ANCHORS: dict[str, list[tuple[float, str, str]]] = {
    "npehe": [(1.0, "constant-ATE baseline", "#d62728")],
    "shape_r2": [
        (0.0, "constant-ATE baseline", "#d62728"),
        (1.0, "perfect", "#2ca02c"),
    ],
    "cate_r2": [(0.0, "constant-ATE baseline", "#d62728"), (1.0, "perfect", "#2ca02c")],
    "ate_bias_norm": [(0.0, "unbiased", "#333333")],
    "ate_abs_bias_norm": [(0.0, "unbiased", "#333333")],
}


def draw_anchors(ax, key: str, *, axis: str = "x", label: bool = True) -> None:
    """Draw ``key``'s reference lines on ``ax`` along ``axis`` ("x" or "y")."""
    line = ax.axvline if axis == "x" else ax.axhline
    for value, text, color in METRIC_ANCHORS.get(key, []):
        line(
            value,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            zorder=0,
            label=text if label else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chores
# ─────────────────────────────────────────────────────────────────────────────


def use_theme(font_scale: float = 1.0) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=font_scale)
    # Pinned off: the figures lay themselves out explicitly, and an automatic
    # engine left on by something else in the process makes `tight_layout`
    # raise on any figure carrying a colorbar.
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def save_figure(fig, path: Path, *, dpi: int = 200, close: bool = True) -> Path:
    """Write ``fig`` to ``path``, creating the directory, and optionally close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  ✓ {path}")
    return path
