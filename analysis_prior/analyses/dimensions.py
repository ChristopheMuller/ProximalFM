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
