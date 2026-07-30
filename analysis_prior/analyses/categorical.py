"""Audit of the categorical-completeness heuristic (Track F).

The dimension heuristic ``min(d_W, d_Z) >= d_U`` has a discrete generalisation:
a proxy is *categorically complete* w.r.t. ``U`` when every column of ``U`` can be
matched one-to-one to a proxy column that is at least as informative -- a
continuous column dominating any finite categorical one (see
:func:`proximalfm.prior._reg2cls.dominates`). The shipped prior *constructs* each
proxy to dominate ``U``'s profile and additionally rejects any dataset that fails
the check after constant-column filtering, so on delivered data the heuristic
holds by construction.

This audit measures that on the data the model actually sees. Because the
categorisation metadata is not persisted in the cache, each block's profile is
**reconstructed from realised values**: a column with at most ``max_categories``
distinct values is categorical (with that many categories), otherwise continuous.
This requires ``T > max_categories`` to mean anything: on a short enough dataset
every column would look categorical.

This is *not* quite the rule ``observed_block_profile`` applies in the prior,
and the difference is why this audit reports a small nonzero violation rate on
the enforced arm. That function reads a column's type off the categorisation
metadata -- a column it did not discretise is continuous, hence an unbounded
wildcard in :func:`dominates` -- whereas here every column is typed from its
realised values. The two disagree on a never-discretised column that
nonetheless lands on a small lattice, which happens when a mechanism reads its
output close to an affine function of one of ``U``'s categorised columns
(~0.3% of never-discretised columns; ~3% of datasets carry one). Such a column
is a wildcard to the enforcement check and a low-cardinality categorical column
here, so a dataset with little profile slack can pass enforcement and still
fail this audit. The realised-value rule is the honest one for delivered data,
so it is what we report; see the "Survival" discussion in appendix B.2.4.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from proximalfm.prior._reg2cls import BlockProfile, dominates

from ..design import SWEEP_CMAP, apply_style, figsize_grid, variant_styles
from ..generate import PriorCache
from ..plotting import plt, save_figure, save_table
from .base import AnalysisArtifacts, ComparisonAnalysis, register

#: The proxy blocks whose completeness w.r.t. ``U`` we audit.
_PROXY_BLOCKS = ("W", "Z")


def reconstruct_profile(block: np.ndarray, max_categories: int) -> BlockProfile:
    """Classify a realised feature block into continuous / categorical columns.

    Pure function of an array, so the classifier can be tested exactly.

    A column is *categorical with ``k`` categories* when it has ``k <=
    max_categories`` distinct values, else *continuous*. Cache blocks are already
    trimmed to their active (non-constant) columns, so every column has ``>= 2``
    distinct values; the distinct count is measured on the realised values, which
    is exactly what :func:`dominates` must hold for on delivered data.

    Parameters
    ----------
    block : ndarray, shape (T, d)
        A realised ``U``/``W``/``Z`` block (active columns only). ``d`` may be 0
        (an unconfounded ``U``), giving an empty profile.

    max_categories : int
        Distinct-value cap below which a column counts as categorical -- the
        prior's ``max_categories`` hyperparameter.

    Returns
    -------
    BlockProfile
        ``continuous`` count and category-counts sorted descending.
    """
    block = np.asarray(block)
    cats: List[int] = []
    continuous = 0
    for col in range(block.shape[1]):
        n_unique = int(np.unique(block[:, col]).size)
        if n_unique <= max_categories:
            cats.append(n_unique)
        else:
            continuous += 1
    return BlockProfile(
        continuous=continuous, categorical=tuple(sorted(cats, reverse=True))
    )


def categorical_survival_stats(
    u_profiles: List[BlockProfile],
    w_profiles: List[BlockProfile],
    z_profiles: List[BlockProfile],
) -> Dict[str, Any]:
    """Share of datasets whose proxies categorically dominate ``U``.

    Pure function of three equal-length profile lists, so it can be tested on
    hand-built profiles.

    "Completeness holds" means *both* proxies dominate ``U``'s profile:
    ``dominates(W, U) and dominates(Z, U)`` -- the categorical analogue of
    ``min(d_W, d_Z) >= d_U``. Datasets whose ``U`` has no categorical columns are
    handled by :func:`dominates` itself (they reduce to the continuous-slot
    count), so nothing special is done here.

    Returns
    -------
    dict with keys
        ``holds`` : bool ndarray, per dataset.
        ``overall_fraction`` : float, share where both proxies dominate.
        ``by_n_cat_U`` : DataFrame with columns
            ``n_cat_U, n_datasets, n_hold, fraction_hold``, one row per observed
            count of categorical columns in ``U``.
    """
    if not (len(u_profiles) == len(w_profiles) == len(z_profiles)):
        raise ValueError("profile lists must have equal length")
    if not u_profiles:
        raise ValueError("need at least one dataset")

    holds = np.array(
        [
            dominates(w, u) and dominates(z, u)
            for u, w, z in zip(u_profiles, w_profiles, z_profiles)
        ]
    )
    n_cat_U = np.array([len(u.categorical) for u in u_profiles])

    rows = []
    for value in sorted(np.unique(n_cat_U)):
        mask = n_cat_U == value
        rows.append(
            {
                "n_cat_U": int(value),
                "n_datasets": int(mask.sum()),
                "n_hold": int(holds[mask].sum()),
                "fraction_hold": float(holds[mask].mean()),
            }
        )

    return {
        "holds": holds,
        "overall_fraction": float(holds.mean()),
        "by_n_cat_U": pd.DataFrame(rows),
    }


def category_count_distribution(
    profiles: List[BlockProfile], max_categories: int
) -> Dict[Any, int]:
    """Pool every column of a block across datasets into a category-count tally.

    Pure aggregation, tested exactly. Returns a mapping whose keys are the
    integer category counts ``2..max_categories`` observed, plus the string
    ``"continuous"``, to the number of columns at that level. Absent counts are
    simply missing (the renderer fills them with zero over the fixed support).
    """
    tally: Counter = Counter()
    for profile in profiles:
        tally["continuous"] += profile.continuous
        for k in profile.categorical:
            tally[int(k)] += 1
    return dict(tally)


def binned_category_marginals(dist: Dict[Any, int], k_bins: List[tuple]) -> List[tuple]:
    """Bin a category-count tally into display proportions of *all* columns.

    Pure function, tested exactly. Continuous columns dominate (often >85%), so a
    per-``k`` bar would crush the categorical tail; the caller passes coarse
    ``k_bins`` (inclusive ``(lo, hi)`` ranges) and this returns proportions over
    ``{continuous} + bins`` that sum to 1.

    Returns
    -------
    list of (label, proportion)
        ``("cont.", ...)`` first, then one entry per bin (label ``"k"`` for a
        singleton bin, ``"lo-hi"`` otherwise), as a share of the block's columns.
    """
    total = sum(dist.values())
    rows = [("cont.", dist.get("continuous", 0) / total if total else 0.0)]
    for lo, hi in k_bins:
        count = sum(v for k, v in dist.items() if isinstance(k, int) and lo <= k <= hi)
        label = f"{lo}" if lo == hi else f"{lo}–{hi}"
        rows.append((label, count / total if total else 0.0))
    return rows


def composition_by_demand(
    u_profiles: List[BlockProfile], proxy_profiles: List[BlockProfile]
) -> pd.DataFrame:
    """Mean proxy composition as a function of ``U``'s categorical-column count.

    Pure function of two equal-length profile lists, so it can be tested on
    hand-built inputs. For each observed value of ``|cats(U)|`` (the number of
    categorical columns in ``U``), it averages the paired proxy's number of
    categorical columns and number of continuous columns. This is the "how much
    does enforcement force the proxy's structure" view: under enforcement the
    proxy composition tracks ``U``'s demand, under the ablation it does not.

    Returns
    -------
    DataFrame with columns
        ``n_cat_U, n_datasets, mean_n_cat_P, sem_n_cat_P, mean_frac_cat_P,
        sem_frac_cat_P``, one row per observed ``|cats(U)|``. ``frac_cat_P`` is
        the fraction of proxy columns that are categorical -- unlike a raw
        continuous/categorical count it is not confounded by the proxy having
        more columns when ``U`` is larger (``d_W >= d_U``). ``sem_*`` is the
        standard error of the mean (0 where a bin has a single dataset).
    """
    if len(u_profiles) != len(proxy_profiles):
        raise ValueError("profile lists must have equal length")
    if not u_profiles:
        raise ValueError("need at least one dataset")

    n_cat_U = np.array([len(u.categorical) for u in u_profiles])
    n_cat_P = np.array([len(p.categorical) for p in proxy_profiles], dtype=float)
    d_P = np.array(
        [p.continuous + len(p.categorical) for p in proxy_profiles], dtype=float
    )
    frac_cat_P = np.divide(n_cat_P, d_P, out=np.zeros_like(n_cat_P), where=d_P > 0)

    def _sem(values: np.ndarray, n: int) -> float:
        return float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    rows = []
    for value in sorted(np.unique(n_cat_U)):
        mask = n_cat_U == value
        n = int(mask.sum())
        rows.append(
            {
                "n_cat_U": int(value),
                "n_datasets": n,
                "mean_n_cat_P": float(n_cat_P[mask].mean()),
                "sem_n_cat_P": _sem(n_cat_P[mask], n),
                "mean_frac_cat_P": float(frac_cat_P[mask].mean()),
                "sem_frac_cat_P": _sem(frac_cat_P[mask], n),
            }
        )
    return pd.DataFrame(rows)


def proxy_stats_by_u_feature(
    u_profiles: List[BlockProfile], proxy_profiles: List[BlockProfile]
) -> pd.DataFrame:
    """One row per (U column, proxy) pair: the U column's kind + proxy-block stats.

    Pure function. There is no one-to-one U-to-proxy column mapping, so each U
    column is attached to its dataset's whole proxy *block* summary. This is the
    per-U-feature conditioning view: bin U columns by their own category count and
    read off what the co-occurring proxy block looks like on average.

    Returns
    -------
    DataFrame with columns
        ``u_card`` : the U column's category count, or ``-1`` if it is continuous.
        ``prop_num`` : fraction of the proxy block's columns that are continuous.
        ``mean_card`` : mean category count of the proxy block's categorical
            columns (``NaN`` if the proxy has none).

    Because the proxy stats are block-level, every U column of a dataset carries
    the same ``prop_num``/``mean_card``; the association is with "the dataset
    contains a U column of this kind", not a specific proxy column.
    """
    if len(u_profiles) != len(proxy_profiles):
        raise ValueError("profile lists must have equal length")
    if not u_profiles:
        raise ValueError("need at least one dataset")

    u_card: List[int] = []
    prop_num: List[float] = []
    mean_card: List[float] = []
    for u, p in zip(u_profiles, proxy_profiles):
        d_p = p.continuous + len(p.categorical)
        pn = p.continuous / d_p if d_p > 0 else np.nan
        mc = float(np.mean(p.categorical)) if p.categorical else np.nan
        # -1 marks a continuous U column; categorical columns carry their count.
        kinds = [-1] * u.continuous + list(u.categorical)
        for kind in kinds:
            u_card.append(int(kind))
            prop_num.append(pn)
            mean_card.append(mc)
    return pd.DataFrame(
        {"u_card": u_card, "prop_num": prop_num, "mean_card": mean_card}
    )


@dataclass
class VariantCategories:
    """One arm's per-dataset block profiles plus derived statistics."""

    #: Reconstructed profiles, one per dataset, keyed by block name (U/W/Z).
    profiles: Dict[str, List[BlockProfile]]
    #: Whether ``U`` was categorised (``corrupt_U``).
    corrupt_U: bool
    #: Whether proxies were enforced to dominate ``U`` (``enforce_categorical_completeness``).
    enforced: bool
    #: The ``max_categories`` cap the arm was generated with.
    max_categories: int
    #: Categorical-completeness survival statistics (see the module function).
    survival: Dict[str, Any]

    @property
    def n_datasets(self) -> int:
        return len(self.profiles["U"])


@dataclass
class CategoricalComparisonResults:
    """Per-arm categorical results, in family (intervention) order."""

    variants: Dict[str, VariantCategories]

    @property
    def baseline(self) -> str:
        """First label = the family's baseline, by the identity-first convention."""
        return next(iter(self.variants))


def extract_categorical_results(
    cache: PriorCache, max_categories: int
) -> VariantCategories:
    """Reconstruct block profiles and survival stats from one cache."""
    profiles: Dict[str, List[BlockProfile]] = {"U": [], "W": [], "Z": []}
    for record in cache:
        for block in ("U", "W", "Z"):
            profiles[block].append(
                reconstruct_profile(getattr(record, block), max_categories)
            )

    fixed_hp = cache.spec.resolved_hp()[0]
    return VariantCategories(
        profiles=profiles,
        corrupt_U=bool(fixed_hp.get("corrupt_U", False)),
        enforced=bool(fixed_hp.get("enforce_categorical_completeness", False)),
        max_categories=int(max_categories),
        survival=categorical_survival_stats(
            profiles["U"], profiles["W"], profiles["Z"]
        ),
    )


@register
class CategoricalCompletenessAnalysis(ComparisonAnalysis):
    name = "categorical_completeness"
    description = (
        "Categorical-completeness heuristic: category-count marginals and "
        "proxy dominance of U's profile, across prior variants."
    )

    def compute(self, caches: Dict[str, PriorCache]) -> CategoricalComparisonResults:
        variants: Dict[str, VariantCategories] = {}
        for label, cache in caches.items():
            max_categories = cache.spec.resolved_hp()[0].get("max_categories", 50)
            variants[label] = extract_categorical_results(cache, max_categories)
        return CategoricalComparisonResults(variants=variants)

    # -- rendering ---------------------------------------------------------

    def render(
        self, results: CategoricalComparisonResults, outdir: Path
    ) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_category_marginals(results, outdir)
        figures += self._render_enforcement_vs_u_demand(results, outdir)
        tables = self._render_survival_table(results, outdir)

        # Key is "baseline_label" (not "baseline") to avoid ever colliding with
        # an arm that happens to be named "baseline".
        summary: Dict[str, Any] = {"baseline_label": results.baseline}
        for label, var in results.variants.items():
            summary[label] = {
                "corrupt_U": var.corrupt_U,
                "enforced": var.enforced,
                "max_categories": var.max_categories,
                "n_datasets": var.n_datasets,
                "dominance_fraction": var.survival["overall_fraction"],
                "violation_fraction": 1.0 - var.survival["overall_fraction"],
                "mean_n_cat_U": float(
                    np.mean([len(p.categorical) for p in var.profiles["U"]])
                ),
                "mean_n_cat_W": float(
                    np.mean([len(p.categorical) for p in var.profiles["W"]])
                ),
                "mean_n_cat_Z": float(
                    np.mean([len(p.categorical) for p in var.profiles["Z"]])
                ),
            }

        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_category_marginals(
        self, results: CategoricalComparisonResults, outdir: Path
    ) -> list:
        """One panel per block (U/W/Z); one 100%-stacked bar per arm.

        Each bar partitions the block's columns into ``cont.`` (continuous, in
        grey) plus the categorical counts binned into ``2..5`` singletons and
        coarser tail bins, coloured by a sequential ramp. Stacking to 100% keeps
        the dominant continuous share honest while the height of the coloured band
        shows how much each arm categorises -- the contrast the figure is for.
        """
        variants = results.variants
        cap = max(v.max_categories for v in variants.values())
        # Coarser bins than the raw counts: a legible stack needs few segments.
        k_bins = [(k, k) for k in range(2, 6)] + [(6, 10), (11, 20), (21, cap)]

        arms = list(variants)
        # Segment order (bottom -> top): continuous first, then ascending counts.
        seg_labels = [
            row[0]
            for row in binned_category_marginals(
                category_count_distribution(
                    variants[arms[0]].profiles["U"], variants[arms[0]].max_categories
                ),
                k_bins,
            )
        ]
        cmap = matplotlib.colormaps[SWEEP_CMAP]
        n_cat = len(seg_labels) - 1  # excluding "cont."
        seg_colors = {"cont.": "0.8"}
        for i, label in enumerate(seg_labels[1:]):
            seg_colors[label] = to_hex(cmap(0.1 + 0.8 * i / max(1, n_cat - 1)))

        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize_grid(ncols=3, panel_aspect=1.2, extra_height_in=0.6),
            sharey=True,
        )
        x = np.arange(len(arms))
        for ax, block in zip(axes, ("U", "W", "Z")):
            props = {
                arm: dict(
                    binned_category_marginals(
                        category_count_distribution(
                            variants[arm].profiles[block], variants[arm].max_categories
                        ),
                        k_bins,
                    )
                )
                for arm in arms
            }
            bottom = np.zeros(len(arms))
            for seg in seg_labels:
                heights = np.array([props[arm][seg] for arm in arms])
                ax.bar(
                    x,
                    heights,
                    bottom=bottom,
                    color=seg_colors[seg],
                    label=seg,
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottom += heights
            ax.set_xticks(x)
            ax.set_xticklabels(arms, rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"${block}$")
            ax.grid(False)
        axes[0].set_ylabel("proportion of columns")
        # Legend once, to the right; segments are ordered so it reads as a scale.
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside right upper",
            title="categories",
            frameon=True,
        )
        return save_figure(fig, outdir, "category_count_marginals")

    def _render_enforcement_vs_u_demand(
        self, results: CategoricalComparisonResults, outdir: Path
    ) -> list:
        """How enforcement makes proxy structure track ``U``'s categorical demand.

        One figure, three panels, all pooling ``W`` and ``Z`` (both proxies, built
        the same way) and all restricted to the categorised-``U`` arms
        (``corrupt_U``): the views condition on categorical demand, which would be
        ~always zero for an arm whose ``U`` is never categorised.

        Panels 1--2 condition on a *single* ``U`` column's own category count
        (``2, ..., 9``, coarser tail bins, or ``num`` if continuous) and read off a
        property of the co-occurring proxy block (see
        :func:`proxy_stats_by_u_feature`): the fraction of proxy columns that are
        continuous, then the mean category count of the proxy's categorical
        columns. Panel 3 moves to the *dataset* level, plotting the mean fraction
        of categorical proxy columns against ``|cats(U)|``; under enforcement it
        rises toward ``proxy_cat_cover_prob`` while the unconstrained
        ``no_enforcement`` arm stays flat.

        Caveat for panels 1--2: proxy stats are block-level, so they condition on
        "the dataset contains a ``U`` column of this kind", not on a matched
        column -- read the trends as association. Sparse bins are dropped as
        noise (``min_features`` columns for panels 1--2, ``min_datasets``
        datasets for panel 3).
        """
        min_features = 20
        min_datasets = 5
        # Continuous ("num") first, then category-count singletons up to 9, then
        # two coarser tail bins (the cap is 50, so a single "10+" bin would mix
        # barely-categorical columns with near-cap ones).
        order = ["num"] + [str(k) for k in range(2, 10)] + ["10-19", "20-29", "30+"]

        def _label(card: int) -> str:
            if card < 0:
                return "num"
            if card < 10:
                return str(card)
            if card < 20:
                return "10-19"
            if card < 30:
                return "20-29"
            return "30+"

        arms = {label: var for label, var in results.variants.items() if var.corrupt_U}
        styles = variant_styles(arms)

        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize_grid(ncols=3, panel_aspect=1.15, extra_height_in=0.5),
        )
        for label, var in arms.items():
            # Pool W and Z against the same U: each dataset contributes both.
            u_pool = var.profiles["U"] + var.profiles["U"]
            p_pool = var.profiles["W"] + var.profiles["Z"]
            color = styles[label]["color"]

            # Panels 1-2: per-U-column conditioning.
            df = proxy_stats_by_u_feature(u_pool, p_pool)
            df = df.assign(bin=df["u_card"].map(_label))
            for ax, col in zip(axes[:2], ("prop_num", "mean_card")):
                grouped = df.groupby("bin")[col]
                mean = grouped.mean()
                # SEM from the non-NaN count in each bin.
                count = grouped.count()
                sem = grouped.std() / np.sqrt(count.where(count > 0))
                keep = [
                    b
                    for b in order
                    if b in mean.index and count.get(b, 0) >= min_features
                ]
                xs = [order.index(b) for b in keep]
                ys = np.array([mean[b] for b in keep])
                es = np.array([sem[b] if np.isfinite(sem[b]) else 0.0 for b in keep])
                ax.plot(xs, ys, marker="o", markersize=3, label=label, color=color)
                ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.2)

            # Panel 3: per-dataset conditioning on U's total categorical demand.
            demand = composition_by_demand(u_pool, p_pool)
            demand = demand[demand["n_datasets"] >= min_datasets]
            x = demand["n_cat_U"].to_numpy()
            y = demand["mean_frac_cat_P"].to_numpy()
            err = demand["sem_frac_cat_P"].to_numpy()
            axes[2].plot(x, y, marker="o", markersize=3, label=label, color=color)
            axes[2].fill_between(x, y - err, y + err, color=color, alpha=0.2)

        # Panels 1-2 share an x-axis meaning; the label is kept terse because a
        # third of the text width is not much room, with the "num" tick and the
        # caption carrying the rest.
        for ax in axes[:2]:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_xlabel("$U$ col.: categories")
        axes[0].set_ylabel("proxy: frac. continuous cols")
        axes[1].set_ylabel("proxy: mean categories | cat.")
        axes[2].set_xlabel("$|\\mathrm{cats}(U)|$")
        axes[2].set_ylabel("proxy: frac. categorical cols")
        axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Panel letters: the appendix caption refers to (a)/(b)/(c).
        for ax, letter in zip(axes, "abc"):
            ax.set_title(f"({letter})", loc="left")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=len(arms),
            frameon=True,
        )
        return save_figure(fig, outdir, "proxy_enforcement_vs_u_demand")

    def _render_survival_table(
        self, results: CategoricalComparisonResults, outdir: Path
    ) -> list:
        """Survival of categorical completeness, by number of categorical U columns.

        The informative contrast is ``default`` (enforced -> ~100% by
        construction) vs ``no_enforcement`` (proxies unconstrained -> the
        pre-enforcement violation rate). Arms whose ``U`` is never categorised
        would contribute only the ``n_cat_U = 0`` row, so they are dropped to
        keep the table about what categorisation + enforcement buy.
        """
        shown = {
            label: var
            for label, var in results.variants.items()
            if var.corrupt_U  # U is categorised only when corrupt_U is on
        }
        if not shown:
            return []

        # Lay the arms out side by side rather than stacked: each variant becomes
        # a group of three columns (datasets/dominating/fraction) sharing the
        # single ``|cats(U)|`` key column. The arms are generated over the same
        # dataset specs, so their ``n_cat_U`` supports coincide in practice; the
        # outer merge aligns them on the shared key regardless and falls back to a
        # zero count for any value an arm happens not to reach.
        stat_names = {
            "n_datasets": "datasets",
            "n_hold": "dominating",
            "fraction_hold": "fraction",
        }
        merged = None
        for label, var in shown.items():
            frame = var.survival["by_n_cat_U"][["n_cat_U", *stat_names]].copy()
            frame.columns = ["n_cat_U", *(f"{label}::{s}" for s in stat_names.values())]
            merged = (
                frame
                if merged is None
                else merged.merge(frame, on="n_cat_U", how="outer")
            )
        merged = merged.sort_values("n_cat_U").reset_index(drop=True)
        merged["n_cat_U"] = merged["n_cat_U"].astype(int)
        for label in shown:
            for stat in ("datasets", "dominating"):
                col = f"{label}::{stat}"
                merged[col] = merged[col].fillna(0).astype(int)

        # A two-level column header: the shared key under an empty group, then one
        # ``\multicolumn`` spanner per variant over its three stat columns. The
        # table is written with ``escape=False`` (for the math in the key
        # column), so a variant label with an underscore (e.g.
        # ``no_enforcement``) needs manual escaping here or it breaks the build.
        def _split_header(col: str) -> tuple:
            arm, stat = col.split("::")
            return (arm.replace("_", "\\_"), stat)

        columns = [("", "$|\\mathrm{cats}(U)|$")]
        columns += [_split_header(col) for col in merged.columns[1:]]
        merged.columns = pd.MultiIndex.from_tuples(columns)
        table = merged

        return save_table(
            table,
            outdir,
            "categorical_completeness_survival",
            caption=(
                "Survival of the categorical-completeness heuristic: share of "
                "datasets at each number of categorical $U$ columns whose "
                "delivered proxies both dominate $U$'s categorical profile "
                "($\\texttt{dominates}(W, U)$ and $\\texttt{dominates}(Z, U)$), "
                "measured on realised values after constant-column filtering. "
                "The \\emph{no\\_enforcement} arm (proxies unconstrained) gives "
                "the pre-enforcement violation rate; \\emph{default} holds by "
                "construction."
            ),
            label="tab:prior_categorical_completeness_survival",
            escape=False,
            centering=True,
            # Rule off the shared key column from each variant's 3-column group;
            # ``|r`` spanners keep the rules unbroken through the group header.
            column_format="r" + "|rrr" * len(shown),
            multicolumn_format="|r",
        )
