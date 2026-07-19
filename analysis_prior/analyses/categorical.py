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
At the shipped cap this is unambiguous -- a continuous column has ~``T`` distinct
values, far above the cap. This mirrors ``observed_block_profile`` in the prior,
but reads from the cached arrays rather than the live tensors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from proximalfm.prior._reg2cls import BlockProfile, dominates

from ..design import apply_style
from ..generate import PriorCache
from ..plotting import save_table
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

        tables = self._render_survival_table(results, outdir)

        # Key is "baseline_label" (not "baseline") because an arm is itself named
        # "baseline"; a bare "baseline" key would collide with its per-arm entry.
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

        return AnalysisArtifacts(summary=summary, figures=[], tables=tables)

    def _render_survival_table(
        self, results: CategoricalComparisonResults, outdir: Path
    ) -> list:
        """Survival of categorical completeness, by number of categorical U columns.

        The informative contrast is ``full`` (enforced -> ~100% by construction)
        vs ``ablation`` (proxies unconstrained -> the pre-enforcement violation
        rate). ``baseline`` has a continuous ``U`` (no categorical columns), so it
        contributes only the ``n_cat_U = 0`` row and is dropped to keep the table
        about what categorisation + enforcement buy.
        """
        shown = {
            label: var
            for label, var in results.variants.items()
            if var.corrupt_U  # U is categorised only when corrupt_U is on
        }
        if not shown:
            return []

        frames = []
        for label, var in shown.items():
            frame = var.survival["by_n_cat_U"].copy()
            frame.insert(0, "variant", label)
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)

        if table["variant"].nunique() == 1:
            table = table.drop(columns="variant")
        table = table.rename(
            columns={
                "n_cat_U": "$|\\mathrm{cats}(U)|$",
                "n_datasets": "datasets",
                "n_hold": "dominating",
                "fraction_hold": "fraction",
            }
        )

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
                "The \\emph{ablation} arm (proxies unconstrained) gives the "
                "pre-enforcement violation rate; \\emph{full} (the default) holds "
                "by construction."
            ),
            label="tab:prior_categorical_completeness_survival",
            escape=False,
        )
