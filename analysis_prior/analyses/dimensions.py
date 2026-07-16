"""Audit of active feature dimensions and the completeness heuristic.

Covers appendix A.2 (dimension sampling) and A.3/A.4 (completeness survival).

The prior enforces the completeness heuristic ``min(d_W, d_Z) >= d_U`` at
*sampling* time, but ``filter_constant_features_and_pad`` then drops constant
columns, which can shrink the delivered ``d_W``/``d_Z`` below ``d_U``. This audit
measures how often the constraint still holds in the data the model actually
sees -- the quantity behind the plan's Table 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..generate import PriorCache
from ..plotting import apply_style, plt, save_figure, save_table
from .base import Analysis, AnalysisArtifacts, register


def completeness_stats(
    d_U: np.ndarray, d_W: np.ndarray, d_Z: np.ndarray
) -> Dict[str, Any]:
    """Completeness-survival statistics from per-dataset active dimensions.

    Pure function of three integer arrays, so it can be tested exactly on
    hand-built inputs.

    "Completeness holds" is taken as the dimension heuristic
    ``min(d_W, d_Z) >= d_U`` evaluated on the *delivered* (post-filtering)
    widths. Datasets with ``d_U == 0`` are unconfounded and trivially satisfy it.

    Returns
    -------
    dict with keys
        ``holds`` : bool ndarray, per dataset.
        ``overall_fraction`` : float, share of datasets where it holds.
        ``by_d_U`` : DataFrame with columns
            ``d_U, n_datasets, n_hold, fraction_hold``, one row per observed
            ``d_U`` value.
    """
    d_U = np.asarray(d_U)
    d_W = np.asarray(d_W)
    d_Z = np.asarray(d_Z)
    if not (len(d_U) == len(d_W) == len(d_Z)):
        raise ValueError("d_U, d_W, d_Z must have equal length")
    if len(d_U) == 0:
        raise ValueError("need at least one dataset")

    holds = np.minimum(d_W, d_Z) >= d_U

    rows = []
    for value in sorted(np.unique(d_U)):
        mask = d_U == value
        rows.append(
            {
                "d_U": int(value),
                "n_datasets": int(mask.sum()),
                "n_hold": int(holds[mask].sum()),
                "fraction_hold": float(holds[mask].mean()),
            }
        )

    return {
        "holds": holds,
        "overall_fraction": float(holds.mean()),
        "by_d_U": pd.DataFrame(rows),
    }


def joint_counts(
    row_vals: np.ndarray,
    col_vals: np.ndarray,
    row_max: int = None,
    col_max: int = None,
) -> np.ndarray:
    """Count matrix of paired integer dimensions.

    Pure function so the heatmap contents can be tested exactly.

    Returns a grid ``g`` where ``g[i, j]`` is the number of datasets with
    ``row_vals == i`` and ``col_vals == j``. Indexing starts at 0 so that the
    (rare) ``d_U == 0`` datasets get their own row rather than being dropped.

    Parameters
    ----------
    row_vals, col_vals : array of int
        Paired per-dataset dimensions (e.g. ``d_U`` and ``d_W``).

    row_max, col_max : int, optional
        Grid extent. Defaults to the observed maxima. Pass these to align grids
        across panels that should share axes.
    """
    row_vals = np.asarray(row_vals, dtype=int)
    col_vals = np.asarray(col_vals, dtype=int)
    if len(row_vals) != len(col_vals):
        raise ValueError("row_vals and col_vals must have equal length")
    if len(row_vals) == 0:
        raise ValueError("need at least one observation")
    if (row_vals < 0).any() or (col_vals < 0).any():
        raise ValueError("dimension counts must be non-negative")

    r_max = int(row_vals.max()) if row_max is None else int(row_max)
    c_max = int(col_vals.max()) if col_max is None else int(col_max)
    if r_max < row_vals.max() or c_max < col_vals.max():
        raise ValueError("row_max/col_max must be >= the observed maxima")

    grid = np.zeros((r_max + 1, c_max + 1), dtype=int)
    np.add.at(grid, (row_vals, col_vals), 1)
    return grid


@dataclass
class DimensionResults:
    """Per-dataset active dimensions plus derived completeness statistics."""

    d_U: np.ndarray
    d_X: np.ndarray
    d_W: np.ndarray
    d_Z: np.ndarray
    completeness: Dict[str, Any]

    @property
    def n_datasets(self) -> int:
        return len(self.d_U)


@register
class DimensionsAnalysis(Analysis):
    name = "dimensions"
    description = "Active feature dimensions and completeness-heuristic survival."

    def compute(self, cache: PriorCache) -> DimensionResults:
        dims = {k: [] for k in ("d_U", "d_X", "d_W", "d_Z")}
        for record in cache:
            for key, value in record.dims.items():
                dims[key].append(value)
        arrays = {k: np.asarray(v, dtype=int) for k, v in dims.items()}

        return DimensionResults(
            completeness=completeness_stats(
                arrays["d_U"], arrays["d_W"], arrays["d_Z"]
            ),
            **arrays,
        )

    def render(self, results: DimensionResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        # --- Table: completeness survival, broken down by d_U -------------
        table = results.completeness["by_d_U"].copy()
        tables = save_table(
            table,
            outdir,
            "completeness_heuristic",
            caption=(
                "Fraction of datasets for which the completeness heuristic "
                "$\\min(d_W, d_Z) \\ge d_U$ still holds after constant-column "
                "filtering, by latent dimension $d_U$."
            ),
        )

        # --- Figure: histograms of the four active dimensions -------------
        fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
        for ax, key, colour in zip(
            axes,
            ("d_U", "d_X", "d_W", "d_Z"),
            ("tab:red", "tab:blue", "tab:green", "tab:orange"),
        ):
            values = getattr(results, key)
            # Integer-aligned bins so each dimension count gets its own bar.
            bins = np.arange(values.min(), values.max() + 2) - 0.5
            ax.hist(values, bins=bins, color=colour, alpha=0.8)
            ax.set_title(f"${key}$")
            ax.set_xlabel("active dimension")
        axes[0].set_ylabel("datasets")
        fig.suptitle("Active feature dimensions across the prior")
        figures = save_figure(fig, outdir, "dimension_histograms")

        # --- Figure: joint distribution of d_U against each other block ---
        # One figure, three heatmap panels. On the W/Z panels the completeness
        # heuristic requires all mass on/below the diagonal d_U = d_proxy; the
        # dashed line makes violations (mass above it) visible at a glance.
        # d_X has no such constraint, so no line is drawn there.
        d_U_max = int(results.d_U.max())
        pairs = (
            ("d_X", results.d_X, False),
            ("d_W", results.d_W, True),
            ("d_Z", results.d_Z, True),
        )
        jfig, jaxes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (key, col_vals, has_bound) in zip(jaxes, pairs):
            grid = joint_counts(results.d_U, col_vals, row_max=d_U_max)
            im = ax.imshow(
                grid,
                origin="lower",
                cmap="viridis",
                aspect="auto",
                interpolation="nearest",
            )
            jfig.colorbar(im, ax=ax, label="datasets")
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            ax.set_xlabel(f"${key}$")
            ax.set_ylabel("$d_U$")
            ax.set_title(f"$d_U$ vs ${key}$")
            ax.grid(False)
            if has_bound:
                m = min(grid.shape) - 1
                ax.plot([0, m], [0, m], "r--", lw=1.2, label="$d_U = " + key + "$")
                ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        jfig.suptitle("Joint distribution of $d_U$ and the other feature blocks")
        figures += save_figure(jfig, outdir, "dimension_joint_heatmaps")

        summary = {
            "n_datasets": results.n_datasets,
            "completeness_overall_fraction": results.completeness["overall_fraction"],
            "completeness_violation_fraction": 1.0
            - results.completeness["overall_fraction"],
            "mean_d_U": float(results.d_U.mean()),
            "mean_d_X": float(results.d_X.mean()),
            "mean_d_W": float(results.d_W.mean()),
            "mean_d_Z": float(results.d_Z.mean()),
            "n_unconfounded": int((results.d_U == 0).sum()),
        }

        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)
