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

# ─────────────────────────────────────────────────────────────────────────────
# Method styling
# ─────────────────────────────────────────────────────────────────────────────
# One entry per `name:` in config_*.yaml (active *and* commented, so styles
# survive when a method is re-enabled).
#
# Convention: colour = method family, marker = variant within the family.
#   OLS .......... Blue    #1f77b4      KPV .......... Cyan   #17becf
#   RF ........... Green   #2ca02c      PMMR ......... Brown  #8c564b
#   TabICL ....... Purple  #9467bd      NMMR ......... Pink   #e377c2
#   P2SLS ........ Orange  #ff7f0e      DFPV ......... Olive  #bcbd22
#   Neural causal models (CausalFM / DoPFN / CausalPFN / ProximalFM) . Red #d62728
#   Oracles ...... Black/grey (never a competitor — see REFERENCE_STYLE)
#
# Proximal strategy markers: paired/XW = "o", independent/X-Wind = "s",
# conditional/X-cond = "^", marginal/ATE = "D".
METHOD_STYLES = {
    # ── OLS baselines (Blue) ──
    "S-OLS": {"color": "#1f77b4", "marker": "o"},  # circle
    "S-OLS+AX": {"color": "#1f77b4", "marker": "s"},  # square
    "T-OLS": {"color": "#1f77b4", "marker": "^"},  # triangle up
    # ── RF baselines (Green) ──
    "S-RF": {"color": "#2ca02c", "marker": "o"},
    "T-RF": {"color": "#2ca02c", "marker": "^"},
    "X-RF": {"color": "#2ca02c", "marker": "P"},  # plus (filled)
    "P-RF": {"color": "#2ca02c", "marker": "X"},  # cross (filled)
    # ── TabICL baselines (Purple) ── (Oracle variants use a star)
    "S-TabICL": {"color": "#9467bd", "marker": "o"},
    "S-TabICL-Oracle": {"color": "#9467bd", "marker": "*", "s": 180},
    "T-TabICL": {"color": "#9467bd", "marker": "^"},
    "T-TabICL-Oracle": {"color": "#9467bd", "marker": "X"},
    "X-TabICL": {"color": "#9467bd", "marker": "v"},  # triangle down
    "X-TabICL-Oracle": {"color": "#9467bd", "marker": "P"},
    # ── P2SLS proximal baselines (Orange) ──
    "P2SLS": {"color": "#ff7f0e", "marker": "o"},
    "P2SLS+AX": {"color": "#ff7f0e", "marker": "s"},
    # ── KPV (Cyan) ──
    "KPV-XW": {"color": "#17becf", "marker": "o"},
    "KPV-X-Wind": {"color": "#17becf", "marker": "s"},
    "KPV-X-Wcond": {"color": "#17becf", "marker": "^"},
    "KPV-ATE": {"color": "#17becf", "marker": "D"},
    # ── PMMR (Brown) ──
    "PMMR-XW": {"color": "#8c564b", "marker": "o"},
    "PMMR-X-Wind": {"color": "#8c564b", "marker": "s"},
    "PMMR-X-Wcond": {"color": "#8c564b", "marker": "^"},
    "PMMR-ATE": {"color": "#8c564b", "marker": "D"},
    # ── NMMR (Pinks: V-statistic light, U-statistic dark) ──
    "NMMR-V-XW": {"color": "#e377c2", "marker": "o"},
    "NMMR-V-Wind": {"color": "#e377c2", "marker": "s"},
    "NMMR-V-Wcond": {"color": "#e377c2", "marker": "^"},
    "NMMR-U-XW": {"color": "#c51b8a", "marker": "o"},
    "NMMR-U-Wind": {"color": "#c51b8a", "marker": "s"},
    "NMMR-U-Wcond": {"color": "#c51b8a", "marker": "^"},
    # ── DFPV (Olive) ──
    "DFPV-XW": {"color": "#bcbd22", "marker": "o"},
    "DFPV-X-Wind": {"color": "#bcbd22", "marker": "s"},
    "DFPV-X-cond": {"color": "#bcbd22", "marker": "^"},
    "DFPV-ATE": {"color": "#bcbd22", "marker": "D"},
    # ── Other forests ──
    "CausalForest": {"color": "#2ca02c", "marker": "D"},
    # ── Tabular & Neural Models (Red) ──
    "CausalFM": {"color": "#d62728", "marker": "X"},  # cross
    "DoPFN": {"color": "#d62728", "marker": "D"},  # diamond
    "CausalPFN": {"color": "#d62728", "marker": "P"},  # plus
    "ProximalFM3": {"color": "#d62728", "marker": "*", "s": 200},  # big star
    "ProximalFM4": {"color": "#d62728", "marker": "p", "s": 160},  # pentagon
    "ProximalFM": {"color": "#d62728", "marker": "*", "s": 200},
    # ── Reference rows (never competitors) ──
    "Oracle-CATE(X)": {"color": "#000000", "marker": "*", "s": 200},
    "Oracle-ATE": {"color": "#777777", "marker": "D"},
    "Oracle-ITE": {"color": "#aaaaaa", "marker": "v"},
}

#: How a reference row is drawn wherever it appears beside competitors. An
#: oracle is an axis annotation that happens to have per-episode numbers, so it
#: is always dashed and never filled — no reader should mistake it for a bar in
#: the ranking.
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


def save_figure(fig, path: Path, *, dpi: int = 200) -> Path:
    """Write ``fig`` to ``path``, creating the directory, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")
    return path
