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

from proximalfm.prior._dataset import SCMPrior

from ..design import (
    COUNT_CMAP,
    apply_style,
    block_color,
    constraint_line,
    figsize_grid,
    grouped_bar_offsets,
    theory_line,
    variant_styles,
)
from ..generate import PriorCache
from ..plotting import plt, save_figure, save_table
from .base import Analysis, AnalysisArtifacts, ComparisonAnalysis, register


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


def dim_U_pmf(lo: int, hi: int, distribution: str) -> Dict[int, float]:
    """Analytic pmf of the sampled confounder dimension on ``{lo..hi}``.

    Delegates the triangular case to :meth:`SCMPrior.triangular_dim_pmf` so the
    audit can never drift from what the generator actually samples.
    """
    if distribution == "uniform":
        return {d: 1.0 / (hi - lo + 1) for d in range(lo, hi + 1)}
    if distribution == "triangular":
        vals, probs = SCMPrior.triangular_dim_pmf(lo, hi)
        return {int(d): float(p) for d, p in zip(vals, probs)}
    raise ValueError(f"Unknown dim_U distribution {distribution!r}")


def proxy_marginal_pmf(
    u_pmf: Dict[int, float], min_proxy: int, max_proxy: int, max_dim_U: int
) -> Dict[int, float]:
    """Analytic marginal pmf of a proxy dimension under conditional-uniform sampling.

    Mirrors the generator's rule (``SCMPrior.get_batch``): given ``d_U = u``,
    the proxy is uniform on ``{max(min_proxy, u), ..., max(max_proxy', lower)}``
    with ``max_proxy' = max(max_proxy, max_dim_U)``. Marginalising over
    ``u_pmf`` gives the induced pmf.

    For the canonical case (``min_proxy = 1``, common bound ``m``) this
    reproduces the closed forms: a triangular ``u_pmf`` yields the linear ramp
    ``p(w) = w / (m(m+1)/2)`` and a uniform ``u_pmf`` yields the harmonic pmf
    ``p(w) = (H_m - H_{m-w}) / m`` -- both pinned by tests.
    """
    effective_max = max(max_proxy, max_dim_U)
    pmf: Dict[int, float] = {}
    for u, pu in u_pmf.items():
        lower = max(min_proxy, u)
        upper = max(effective_max, lower)
        width = upper - lower + 1
        for w in range(lower, upper + 1):
            pmf[w] = pmf.get(w, 0.0) + pu / width
    return dict(sorted(pmf.items()))


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


def extract_dimension_results(cache: PriorCache) -> DimensionResults:
    """Collect delivered (post-filter) dimensions from a cache."""
    dims = {k: [] for k in ("d_U", "d_X", "d_W", "d_Z")}
    for record in cache:
        for key, value in record.dims.items():
            dims[key].append(value)
    arrays = {k: np.asarray(v, dtype=int) for k, v in dims.items()}

    return DimensionResults(
        completeness=completeness_stats(arrays["d_U"], arrays["d_W"], arrays["d_Z"]),
        **arrays,
    )


@register
class DimensionsAnalysis(Analysis):
    name = "dimensions"
    description = "Active feature dimensions and completeness-heuristic survival."

    def compute(self, cache: PriorCache) -> DimensionResults:
        return extract_dimension_results(cache)

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
        fig, axes = plt.subplots(
            1,
            4,
            figsize=figsize_grid(ncols=4, panel_aspect=1.15, extra_height_in=0.3),
            sharey=True,
        )
        for ax, key in zip(axes, ("d_U", "d_X", "d_W", "d_Z")):
            values = getattr(results, key)
            # Integer-aligned bins so each dimension count gets its own bar.
            bins = np.arange(values.min(), values.max() + 2) - 0.5
            ax.hist(values, bins=bins, color=block_color(key), alpha=0.85)
            ax.set_title(f"${key}$")
            ax.set_xlabel("active dimension")
        axes[0].set_ylabel("datasets")
        fig.suptitle("Active feature dimensions across the prior")
        figures = save_figure(fig, outdir, "dimension_histograms")

        # --- Figure: joint distribution of d_U against each other block ---
        d_U_max = int(results.d_U.max())
        pairs = (
            ("d_X", results.d_X, False),
            ("d_W", results.d_W, True),
            ("d_Z", results.d_Z, True),
        )
        grids = {
            key: joint_counts(results.d_U, col_vals, row_max=d_U_max)
            for key, col_vals, _ in pairs
        }
        vmax = max(grid.max() for grid in grids.values())

        jfig, jaxes = plt.subplots(
            1, 3, figsize=figsize_grid(ncols=3, panel_aspect=1.0, extra_height_in=0.3)
        )
        for ax, (key, _, has_bound) in zip(jaxes, pairs):
            grid = grids[key]
            im = ax.imshow(
                grid,
                origin="lower",
                cmap=COUNT_CMAP,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=vmax,
            )
            ax.set_xticks(range(grid.shape[1]))
            ax.set_yticks(range(grid.shape[0]))
            ax.set_xlabel(f"${key}$")
            ax.set_ylabel("$d_U$")
            ax.set_title(f"$d_U$ vs ${key}$")
            ax.grid(False)
            if has_bound:
                constraint_line(ax, 0, min(grid.shape) - 1, label=f"$d_U = {key}$")
        jfig.colorbar(im, ax=jaxes, label="datasets", shrink=0.8)
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


@dataclass
class VariantDimensions:
    """One variant's delivered dimensions plus its analytic sampling laws."""

    results: DimensionResults
    #: The variant's ``dim_U_distribution`` ("triangular" or "uniform").
    scheme: str
    #: Whether post-filter completeness enforcement was on.
    enforced: bool
    #: Analytic pmfs of the *sampled* dims: key ("d_U"/"d_W"/"d_Z") -> pmf.
    theory: Dict[str, Dict[int, float]]


@dataclass
class DimensionComparisonResults:
    """Per-variant dimension results, in family (intervention) order."""

    variants: Dict[str, VariantDimensions]

    @property
    def baseline(self) -> str:
        """First label = the family's baseline, by the identity-first convention."""
        return next(iter(self.variants))


def _theory_mean(pmf: Dict[int, float]) -> float:
    return float(sum(k * p for k, p in pmf.items()))


@register
class DimensionComparisonAnalysis(ComparisonAnalysis):
    name = "dimension_comparison"
    description = (
        "Dimension marginals vs their analytic sampling laws, and "
        "completeness-heuristic survival, across prior variants."
    )

    def compute(self, caches: Dict[str, PriorCache]) -> DimensionComparisonResults:
        variants: Dict[str, VariantDimensions] = {}
        for label, cache in caches.items():
            spec = cache.spec
            fixed_hp = spec.resolved_hp()[0]
            scheme = fixed_hp.get("dim_U_distribution", "triangular")

            u_pmf = dim_U_pmf(spec.min_dim_U, spec.max_dim_U, scheme)
            theory = {
                "d_U": u_pmf,
                "d_W": proxy_marginal_pmf(
                    u_pmf, spec.min_dim_W, spec.max_dim_W, spec.max_dim_U
                ),
                "d_Z": proxy_marginal_pmf(
                    u_pmf, spec.min_dim_Z, spec.max_dim_Z, spec.max_dim_U
                ),
                # d_X has no completeness coupling: plain uniform.
                "d_X": dim_U_pmf(spec.min_dim_X, spec.max_dim_X, "uniform"),
            }

            variants[label] = VariantDimensions(
                results=extract_dimension_results(cache),
                scheme=scheme,
                enforced=bool(fixed_hp.get("enforce_completeness_heuristic", True)),
                theory=theory,
            )
        return DimensionComparisonResults(variants=variants)

    # -- rendering ---------------------------------------------------------

    def render(
        self, results: DimensionComparisonResults, outdir: Path
    ) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_marginals(results, outdir)
        figures += self._render_joint_heatmaps(results, outdir)
        tables = self._render_sampling_table(results, outdir)

        summary: Dict[str, Any] = {"baseline": results.baseline}
        for label, var in results.variants.items():
            res = var.results
            summary[label] = {
                "scheme": var.scheme,
                "enforced": var.enforced,
                "n_datasets": res.n_datasets,
                "mean_d_U": float(res.d_U.mean()),
                "mean_d_W": float(res.d_W.mean()),
                "mean_d_Z": float(res.d_Z.mean()),
                "mean_d_X": float(res.d_X.mean()),
                "violation_fraction": 1.0 - res.completeness["overall_fraction"],
                "n_unconfounded": int((res.d_U == 0).sum()),
            }

        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_marginals(
        self, results: DimensionComparisonResults, outdir: Path
    ) -> list:
        """Grouped empirical bars per variant + one analytic overlay per scheme.

        Only variants that enforce the completeness heuristic are shown: the
        figure is about the *sampling laws* and their analytic overlays, and an
        unenforced variant shares its law with its enforced counterpart, so it
        adds a third bar per group without a third curve to compare it to. Its
        effect belongs to the joint heatmaps instead. If a family has no
        enforced variant we fall back to plotting all of them.
        """
        variants = {
            label: var for label, var in results.variants.items() if var.enforced
        } or dict(results.variants)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize_grid(ncols=3, panel_aspect=0.95, extra_height_in=0.5),
            sharey=True,
        )
        n = len(variants)
        styles = variant_styles(variants)
        offsets = grouped_bar_offsets(n)
        width = 0.8 / n

        for ax, key in zip(axes, ("d_U", "d_W", "d_Z")):
            support_vals: set = set()
            for var in variants.values():
                support_vals |= set(np.unique(getattr(var.results, key)).tolist())
                support_vals |= set(var.theory[key])
            support = sorted(support_vals)
            positions = {v: i for i, v in enumerate(support)}
            x = np.arange(len(support))

            for offset, (label, var) in zip(offsets, variants.items()):
                observed = getattr(var.results, key)
                counts = np.zeros(len(support))
                for value, count in zip(*np.unique(observed, return_counts=True)):
                    counts[positions[int(value)]] = count
                ax.bar(
                    x + offset,
                    counts / max(1, len(observed)),
                    width,
                    label=label,
                    alpha=0.85,
                    **styles[label],
                )

            # One theory line per distinct sampling law (variants sharing a
            # scheme share a line; the overlay describes the *sampled* dims,
            # so delivered bars may sit slightly below it after filtering).
            # Solid/dotted separates the laws where two are shown at once.
            seen = {}
            for var in variants.values():
                pmf = var.theory[key]
                signature = (var.scheme, tuple(sorted(pmf.items())))
                if signature in seen:
                    continue
                seen[signature] = var.scheme
                theory_line(
                    ax,
                    [positions[v] for v in pmf],
                    list(pmf.values()),
                    label=f"theory ({var.scheme})",
                    linestyle="--" if len(seen) == 1 else ":",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(support)
            ax.set_xlabel(f"${key}$")
            ax.set_title(f"${key}$")
        axes[0].set_ylabel("probability")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=True)
        return save_figure(fig, outdir, "dimension_marginals")

    def _render_joint_heatmaps(
        self, results: DimensionComparisonResults, outdir: Path
    ) -> list:
        """One row per variant: d_U vs d_X / d_W / d_Z, on shared axes.

        Colour scale is shared *within* each column (across variants) rather
        than globally, since variants can differ in ``n_datasets`` and a
        single scale across all three dimensions would let the biggest one
        wash out the others. Sharing per column is what the figure is for:
        comparing a variant against the baseline on the same dimension.
        """
        n = len(results.variants)
        d_U_max = max(int(v.results.d_U.max()) for v in results.variants.values())
        columns = (("d_X", False), ("d_W", True), ("d_Z", True))
        col_max = {
            key: max(
                int(getattr(v.results, key).max()) for v in results.variants.values()
            )
            for key, _ in columns
        }
        grids = {
            (label, key): joint_counts(
                var.results.d_U,
                getattr(var.results, key),
                row_max=d_U_max,
                col_max=col_max[key],
            )
            for label, var in results.variants.items()
            for key, _ in columns
        }
        col_vmax = {
            key: max(grids[(label, key)].max() for label in results.variants)
            for key, _ in columns
        }

        fig, axes = plt.subplots(
            n,
            3,
            # Headroom for the suptitle plus the row of colorbars along the
            # bottom.
            figsize=figsize_grid(
                ncols=3, nrows=n, panel_aspect=1.0, extra_height_in=0.8
            ),
            squeeze=False,
        )
        for row, (label, var) in enumerate(results.variants.items()):
            for col, (key, has_bound) in enumerate(columns):
                ax = axes[row][col]
                grid = grids[(label, key)]
                im = ax.imshow(
                    grid,
                    origin="lower",
                    cmap=COUNT_CMAP,
                    aspect="auto",
                    interpolation="nearest",
                    vmin=0,
                    vmax=col_vmax[key],
                )
                ax.set_xticks(range(grid.shape[1]))
                ax.set_yticks(range(grid.shape[0]))
                ax.set_xlabel(f"${key}$")
                ax.set_ylabel("$d_U$")
                ax.set_title(f"{label}: $d_U$ vs ${key}$")
                ax.grid(False)
                if has_bound:
                    constraint_line(ax, 0, min(grid.shape) - 1, label=f"$d_U = {key}$")
        # One colorbar per column, placed *below* the column rather than to its
        # right: a right-side bar would be squeezed into the inter-column gap and
        # overlap the next column's panels and titles.
        for col, (key, _) in enumerate(columns):
            fig.colorbar(
                axes[0][col].images[0],
                ax=axes[:, col],
                location="bottom",
                label="datasets",
                shrink=0.8,
                aspect=40,
            )
        fig.suptitle("Joint distribution of $d_U$ and the other blocks, per variant")
        return save_figure(fig, outdir, "dimension_joint_heatmaps_comparison")

    def _render_sampling_table(
        self, results: DimensionComparisonResults, outdir: Path
    ) -> list:
        """Table 4: sampling law and theory-vs-delivered means, baseline variant.

        The theory column describes the *sampled* dimensions; the delivered
        column is measured after constant-column filtering and completeness
        enforcement, which can only *remove* columns.

        That postprocessing is, however, not the dominant term in the gap:
        filtering costs ~0.02 columns per dataset, while the Monte Carlo error
        on a mean over ``n_datasets`` draws is an order of magnitude larger
        (~0.09 for $d_X$ at n=1024, whose spread is widest because it is plain
        uniform and uncoupled from $d_U$). So the theory-delivered gap is noise
        first and postprocessing second, and it can fall either side of the
        theoretical mean. The reported standard error is what makes that
        readable -- without it a $+0.14$ gap on $d_X$ looks like a bug rather
        than the $1.5$ s.e. fluctuation it is.
        """
        var = results.variants[results.baseline]
        law_u = (
            "triangular $p(d) \\propto (d^{\\max}+1-d)$"
            if var.scheme == "triangular"
            else "uniform"
        )

        rows = []
        for key, law in (
            ("d_U", law_u),
            ("d_W", "uniform $\\{\\max(d^{\\min}, d_U), \\dots, d^{\\max}\\}$"),
            ("d_Z", "uniform $\\{\\max(d^{\\min}, d_U), \\dots, d^{\\max}\\}$"),
            ("d_X", "uniform"),
        ):
            observed = getattr(var.results, key)
            pmf = var.theory[key]
            rows.append(
                {
                    "variable": f"${key}$",
                    "support": f"{min(pmf)}--{max(pmf)}",
                    "sampling law": law,
                    "mean (theory)": _theory_mean(pmf),
                    "mean (deliv.)": float(observed.mean()),
                    "s.e.": float(observed.std(ddof=1) / np.sqrt(len(observed))),
                }
            )

        return save_table(
            pd.DataFrame(rows),
            outdir,
            "dimension_sampling",
            caption=(
                "Dimension sampling laws and their means (baseline variant "
                f"``{results.baseline}'', $n = {var.results.n_datasets}$ "
                "datasets). Theory describes the sampled dimensions; delivered "
                "values are measured after constant-column filtering and "
                "completeness enforcement, which can only remove columns "
                "($\\approx 0.02$ columns per dataset here)."
            ),
            label="tab:prior_dim_sampling",
            centering=True,
            # Cells are intentional LaTeX ($d_U$, $p(d)\propto\dots$).
            escape=False,
        )
