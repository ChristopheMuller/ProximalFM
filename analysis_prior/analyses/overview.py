r"""Visualize the prior: what the delivered datasets look like (B.2.x).

Covers the "Visualize the prior" subsubsection (``app:prior_visualize``) with
three galleries, all read straight off the delivered data (no training):

- ``fig:prior_feature_distributions`` -- one column per block
  (``X, U, W, Z, A, Y``), ``k`` rows, each cell the marginal density of one
  feature column sampled from a random dataset (for ``A``, the propensity
  ``pi``; for ``Y``, both potential outcomes ``Y_0``/``Y_1`` overlaid).

- ``fig:prior_feature_pairs`` -- within-block dependence: one column per block,
  ``k`` rows, each cell a scatter of two features from the same block of a
  random dataset (for ``A``, ``pi`` vs the draw ``A``; for ``Y``, the two
  potential outcomes ``Y_0`` vs ``Y_1``).

- ``fig:prior_feature_pairs_cross`` -- cross-block dependence: one column per
  chosen block pair, ``k`` rows, each cell a scatter of one feature from each
  block of a random dataset (parent->child ``U``->``W`` / ``U``->``Z``, the two
  proxies ``W``--``Z``, and the confounder against the propensity ``U``--``pi``).

The galleries sample datasets with a fixed seed, so every figure is
deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..design import OKABE_ITO, apply_style, block_color, figsize_grid
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure
from .base import Analysis, AnalysisArtifacts, register

#: Blocks shown in the distribution / within-block galleries. ``A`` and ``Y``
#: are scalar "blocks": their feature is the propensity and the outcome.
GALLERY_BLOCKS = ("X", "U", "W", "Z", "A", "Y")

#: Cross-block pairs (a chosen few, not all): the two parent->child edges, the
#: two proxies (associated only through ``U``), and the confounder against the
#: propensity.
CROSS_PAIRS = (("U", "W"), ("U", "Z"), ("W", "Z"), ("U", "A"))

#: Colour block for each cross pair, one of its two endpoints. ``CROSS_PAIRS``
#: touches exactly the four blocks ``{U, W, Z, A}``, so each pair can be
#: assigned a distinct endpoint's true (unblended) colour with no repeats --
#: coloring every pair by its child alone would collide, since ``Z`` is the
#: child of both ``U-Z`` and ``W-Z``.
CROSS_PAIR_COLOR_BLOCK = {
    ("U", "W"): "U",
    ("U", "Z"): "Z",
    ("W", "Z"): "W",
    ("U", "A"): "A",
}

#: Colours for the two potential outcomes overlaid in the ``Y`` marginal panel.
Y0_COLOR = OKABE_ITO["blue"]
Y1_COLOR = OKABE_ITO["vermillion"]

#: Fallback categorical cap for records whose sampled_hp predates the knob.
_DEFAULT_MAX_CATEGORIES = 20


def _max_categories(record: PriorDatasetRecord) -> int:
    if record.sampled_hp and "max_categories" in record.sampled_hp:
        return int(record.sampled_hp["max_categories"])
    return _DEFAULT_MAX_CATEGORIES


def _categorical_mask(block: np.ndarray, max_categories: int) -> np.ndarray:
    """Boolean mask of the categorical columns (<= cap distinct values).

    Mirrors ``reconstruct_profile`` in :mod:`analysis_prior.analyses.categorical`:
    a standardising ``Corrupt`` is a monotone affine map, so it preserves the
    distinct-value count and the categorisation stays readable off the array.
    """
    if block.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    n_unique = np.array([np.unique(block[:, j]).size for j in range(block.shape[1])])
    return n_unique <= max_categories


def _feature_columns(record: PriorDatasetRecord, block: str) -> np.ndarray:
    """The ``(T, d)`` feature array of a block for density/scatter purposes.

    The two scalar blocks map to their single characteristic feature: ``A`` to
    the propensity ``A_probs``, ``Y`` to the observed outcome.
    """
    if block == "A":
        return record.A_probs[:, None]
    if block == "Y":
        return record.Y_obs[:, None]
    return getattr(record, block)


def _feature_label(block: str, col: int, n_cols: int) -> str:
    """LaTeX label for one feature: a symbol for the scalar blocks, else ``B_j``."""
    if block == "A":
        return r"$\pi$"
    if block == "Y":
        return "$Y$"
    return f"${block}_{{{col + 1}}}$" if n_cols > 1 else f"${block}$"


# ---------------------------------------------------------------------------
# Gallery sampling
# ---------------------------------------------------------------------------


def _pick_datasets(
    records: List[PriorDatasetRecord],
    k: int,
    rng: np.random.Generator,
    predicate: Callable[[PriorDatasetRecord], bool],
) -> List[int]:
    """Up to ``k`` distinct dataset indices satisfying ``predicate``.

    Deterministic given ``rng``. Prefers distinct datasets; if fewer than ``k``
    qualify, cycles through them again so the gallery still fills ``k`` rows.
    """
    candidates = [i for i, r in enumerate(records) if predicate(r)]
    if not candidates:
        return []
    order = list(rng.permutation(candidates))
    while len(order) < k:
        order += list(rng.permutation(candidates))
    return order[:k]


@dataclass
class DensityPanel:
    index: int
    values: np.ndarray
    label: str
    categorical: bool
    #: Optional second series overlaid in the same panel (the ``Y`` marginal
    #: overlays both potential outcomes ``Y_0``/``Y_1``).
    values2: Optional[np.ndarray] = None


@dataclass
class ScatterPanel:
    index: int
    x: np.ndarray
    y: np.ndarray
    xlabel: str
    ylabel: str


def sample_distributions(
    records: List[PriorDatasetRecord], rng: np.random.Generator, k: int
) -> Dict[str, List[DensityPanel]]:
    """For each block, ``k`` sampled single-feature marginals (feature gallery).

    For ``A`` the single feature is the propensity; for ``Y`` the panel overlays
    both potential outcomes ``Y_0``/``Y_1`` (as a two-series ``DensityPanel``).
    """
    out: Dict[str, List[DensityPanel]] = {}
    for block in GALLERY_BLOCKS:

        def has_feature(r: PriorDatasetRecord, b: str = block) -> bool:
            return _feature_columns(r, b).shape[1] >= 1

        panels: List[DensityPanel] = []
        for i in _pick_datasets(records, k, rng, has_feature):
            record = records[i]
            if block == "Y":
                panels.append(
                    DensityPanel(
                        index=record.index,
                        values=record.Y0,
                        values2=record.Y1,
                        label="$Y_0,\\,Y_1$",
                        categorical=False,
                    )
                )
                continue
            cols = _feature_columns(record, block)
            j = int(rng.integers(cols.shape[1]))
            cat = (
                bool(
                    _categorical_mask(getattr(record, block), _max_categories(record))[
                        j
                    ]
                )
                if block != "A"
                else False
            )
            panels.append(
                DensityPanel(
                    index=record.index,
                    values=cols[:, j],
                    label=_feature_label(block, j, cols.shape[1]),
                    categorical=cat,
                )
            )
        out[block] = panels
    return out


def sample_within_pairs(
    records: List[PriorDatasetRecord],
    rng: np.random.Generator,
    k: int,
    scatter_rows: int = 600,
) -> Dict[str, List[ScatterPanel]]:
    """For each block, ``k`` sampled same-block feature-pair scatters (Figure 3).

    ``X/U/W/Z`` need two columns; ``A`` shows the propensity against the draw,
    ``Y`` the two potential outcomes.
    """
    out: Dict[str, List[ScatterPanel]] = {}
    for block in GALLERY_BLOCKS:

        def predicate(r: PriorDatasetRecord, b: str = block) -> bool:
            # A/Y are always plottable (propensity-vs-draw, Y0-vs-Y1); the
            # feature blocks need at least two columns to pair.
            return b in ("A", "Y") or getattr(r, b).shape[1] >= 2

        panels: List[ScatterPanel] = []
        for i in _pick_datasets(records, k, rng, predicate):
            record = records[i]
            if block == "A":
                x, y = record.A_probs, _jitter(record.A, rng)
                xl, yl = r"$\pi$", "$A$"
            elif block == "Y":
                x, y = record.Y0, record.Y1
                xl, yl = "$Y_0$", "$Y_1$"
            else:
                arr = getattr(record, block)
                ja, jb = (int(v) for v in rng.choice(arr.shape[1], 2, replace=False))
                x, y = arr[:, ja], arr[:, jb]
                xl = _feature_label(block, ja, arr.shape[1])
                yl = _feature_label(block, jb, arr.shape[1])
            x, y = _subsample(x, y, scatter_rows, rng)
            panels.append(ScatterPanel(record.index, x, y, xl, yl))
        out[block] = panels
    return out


def sample_cross_pairs(
    records: List[PriorDatasetRecord],
    rng: np.random.Generator,
    k: int,
    scatter_rows: int = 600,
) -> Dict[str, List[ScatterPanel]]:
    """For each chosen block pair, ``k`` sampled cross-block scatters (Figure 4)."""
    out: Dict[str, List[ScatterPanel]] = {}
    for block_a, block_b in CROSS_PAIRS:
        key = f"{block_a}-{block_b}"

        def has_both(r: PriorDatasetRecord, a: str = block_a, b: str = block_b) -> bool:
            return (
                _feature_columns(r, a).shape[1] >= 1
                and _feature_columns(r, b).shape[1] >= 1
            )

        panels: List[ScatterPanel] = []
        for i in _pick_datasets(records, k, rng, has_both):
            record = records[i]
            ca = _feature_columns(record, block_a)
            cb = _feature_columns(record, block_b)
            ja = int(rng.integers(ca.shape[1]))
            jb = int(rng.integers(cb.shape[1]))
            x, y = _subsample(ca[:, ja], cb[:, jb], scatter_rows, rng)
            panels.append(
                ScatterPanel(
                    record.index,
                    x,
                    y,
                    _feature_label(block_a, ja, ca.shape[1]),
                    _feature_label(block_b, jb, cb.shape[1]),
                )
            )
        out[key] = panels
    return out


def _jitter(
    binary: np.ndarray, rng: np.random.Generator, width: float = 0.04
) -> np.ndarray:
    """Vertical jitter for a 0/1 array, so the two arms are legible in a scatter."""
    return binary.astype(np.float64) + rng.uniform(-width, width, size=len(binary))


def _subsample(
    x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return np.asarray(x), np.asarray(y)
    take = rng.choice(len(x), n, replace=False)
    return np.asarray(x)[take], np.asarray(y)[take]


# ---------------------------------------------------------------------------
# Results + analysis
# ---------------------------------------------------------------------------


@dataclass
class PriorOverviewResults:
    n_datasets: int
    distributions: Dict[str, List[DensityPanel]]
    within: Dict[str, List[ScatterPanel]]
    cross: Dict[str, List[ScatterPanel]]


@register
class PriorOverviewAnalysis(Analysis):
    name = "prior_overview"
    description = (
        "Visualize the prior (B.2.x): sampled galleries of feature marginals, "
        "within-block and cross-block dependence."
    )

    #: Datasets scanned when sampling the galleries.
    n_datasets = 512
    #: Rows (sampled datasets) in each gallery figure.
    k_rows = 5
    #: Rows subsampled per scatter panel, for legibility and file size.
    scatter_rows = 600
    #: Deterministic sampling seed.
    seed = 0

    def compute(self, cache: PriorCache) -> PriorOverviewResults:
        records = list(cache)[: self.n_datasets]
        rng = np.random.default_rng(self.seed)
        return PriorOverviewResults(
            n_datasets=len(records),
            distributions=sample_distributions(records, rng, self.k_rows),
            within=sample_within_pairs(records, rng, self.k_rows, self.scatter_rows),
            cross=sample_cross_pairs(records, rng, self.k_rows, self.scatter_rows),
        )

    def render(self, results: PriorOverviewResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures: List[Path] = []
        figures += self._render_distributions(results, outdir)
        figures += self._render_within(results, outdir)
        figures += self._render_cross(results, outdir)

        summary: Dict[str, Any] = {
            "n_datasets": results.n_datasets,
            "k_rows": self.k_rows,
        }
        return AnalysisArtifacts(summary=summary, figures=figures, tables=[])

    # -- Feature-marginal gallery -------------------------------------------

    def _render_distributions(
        self, results: PriorOverviewResults, outdir: Path
    ) -> List[Path]:
        blocks = list(results.distributions)
        nrows, ncols = self.k_rows, len(blocks)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize_grid(
                ncols=ncols, nrows=nrows, panel_aspect=0.8, extra_height_in=0.4
            ),
        )
        for ci, block in enumerate(blocks):
            axes[0][ci].set_title(
                f"${block}$"
                if block not in ("A", "Y")
                else ("$A$ ($\\pi$)" if block == "A" else "$Y_0,\\,Y_1$")
            )
            panels = results.distributions[block]
            for ri in range(nrows):
                ax = axes[ri][ci]
                ax.set_yticks([])
                if ri >= len(panels):
                    ax.set_axis_off()
                    continue
                panel = panels[ri]
                if panel.values2 is not None:
                    # Y panel: both potential outcomes overlaid (Y0 vs Y1).
                    ax.hist(
                        panel.values,
                        bins=30,
                        density=True,
                        histtype="step",
                        lw=1.3,
                        color=Y0_COLOR,
                    )
                    ax.hist(
                        panel.values2,
                        bins=30,
                        density=True,
                        histtype="step",
                        lw=1.3,
                        color=Y1_COLOR,
                    )
                else:
                    ls = "--" if panel.categorical else "-"
                    ax.hist(
                        panel.values,
                        bins=30,
                        density=True,
                        histtype="step",
                        lw=1.3,
                        ls=ls,
                        color=block_color(block),
                    )
                ax.set_xlabel(panel.label, labelpad=1)
                ax.locator_params(axis="x", nbins=3)
                ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] - 2)
        return save_figure(fig, outdir, "prior_feature_distributions")

    # -- Figure 3: within-block scatter gallery -----------------------------

    def _render_within(self, results: PriorOverviewResults, outdir: Path) -> List[Path]:
        return self._render_scatter_gallery(
            results.within,
            list(results.within),
            outdir,
            "prior_feature_pairs",
            {b: (f"${b}$" if b not in ("A", "Y") else b) for b in results.within},
        )

    # -- Figure 4: cross-block scatter gallery ------------------------------

    def _render_cross(self, results: PriorOverviewResults, outdir: Path) -> List[Path]:
        headers = {f"{a}-{b}": _pair_header(a, b) for a, b in CROSS_PAIRS}
        return self._render_scatter_gallery(
            results.cross,
            list(results.cross),
            outdir,
            "prior_feature_pairs_cross",
            headers,
        )

    def _render_scatter_gallery(
        self,
        data: Dict[str, List[ScatterPanel]],
        columns: List[str],
        outdir: Path,
        name: str,
        headers: Dict[str, str],
    ) -> List[Path]:
        nrows, ncols = self.k_rows, len(columns)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize_grid(
                ncols=ncols, nrows=nrows, panel_aspect=1.0, extra_height_in=0.4
            ),
        )
        axes = np.atleast_2d(axes)
        if ncols == 1:
            axes = axes.reshape(nrows, 1)
        for ci, key in enumerate(columns):
            # Colour by the block being plotted; for a cross pair "A-B" pick
            # the endpoint assigned in CROSS_PAIR_COLOR_BLOCK, so pairs that
            # share a block (e.g. "U-Z" and "W-Z") stay visually distinct.
            if "-" in key:
                block_a, block_b = key.split("-")
                color = block_color(CROSS_PAIR_COLOR_BLOCK[(block_a, block_b)])
            else:
                color = block_color(key)
            axes[0][ci].set_title(headers[key])
            panels = data[key]
            for ri in range(nrows):
                ax = axes[ri][ci]
                if ri >= len(panels):
                    ax.set_axis_off()
                    continue
                panel = panels[ri]
                ax.scatter(panel.x, panel.y, s=6, alpha=0.3, linewidths=0, color=color)
                ax.set_xlabel(panel.xlabel, labelpad=1)
                ax.set_ylabel(panel.ylabel, labelpad=1)
                ax.locator_params(nbins=3)
                ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"] - 2)
        return save_figure(fig, outdir, name)


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def _pair_header(block_a: str, block_b: str) -> str:
    """Column header for a cross-block pair, naming the propensity for ``A``."""

    def sym(b: str) -> str:
        return r"\pi" if b == "A" else b

    return f"${sym(block_a)}$--${sym(block_b)}$"
