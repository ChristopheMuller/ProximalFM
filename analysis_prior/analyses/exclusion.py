"""Proxy exclusion restrictions (B.3.2) -- the audit of ``as:proxy_exclusion``.

The assumption has two clauses, ``Y indep Z | (A, X, U)`` and
``W indep (A, Z) | (X, U)``, and both hold **by construction**: ``_dataset.py``
never wires ``Z`` into ``f_W``/``f_Y`` nor ``W`` into ``f_Z``/``f_A``
(``eq:uwz``). So this module is not looking for a violation -- it asks the two
questions that a "by construction" claim cannot answer on its own:

1. **Does the delivered data still show no leakage?** Post-processing (outlier
   clipping, categorisation, standardisation) happens after the wiring, and the
   audit regression sees the realised columns, not the graph. Measured as the
   cross-fitted incremental ``R^2`` of the excluded block:

   - **Y-side**: predict ``Y_obs`` from ``{A, X, U, Z}``; report
     ``dR2_Z = R2(all) - R2(all \\ Z)``. Expected ``~0``.
   - **A-side**: predict the raw treatment logit ``logit(A_probs)`` (the
     ``mechanism_outputs`` convention -- an affine image of the ``f_A`` output,
     unclamped in practice) from ``{X, U, W}``; report ``dR2_W``. Expected
     ``~0``.

   Each conditioning set is the one the clause names, and nothing more. In
   particular ``Z`` is *not* in the A-side set: the clause conditions on
   ``(X, U)``, and ``Z`` is itself a proxy of ``U``, so leaving it in would let
   it absorb the confounder and blunt the positive control below -- the check
   would then be near-zero for the wrong reason.

2. **Would the test have caught a leak?** A ``dR2`` of zero is only evidence if
   the estimator could have produced a nonzero one. The **positive control** re-
   runs both checks with ``U`` dropped from the predictor set (``Y_obs`` from
   ``{A, X, Z}``; the logit from ``{X, W}``). ``U`` is the *only* path linking
   ``Z`` to ``Y`` and ``W`` to ``A``, so removing it makes the excluded block
   genuinely predictive: the control must light up clearly nonzero. The gap
   between the two conditions is the result; the near-zero number alone is not.

**The four fits are a 2x2 lattice, and the figure reads two different edges of
it.** Per dataset per side (Y-side shown; the A-side is the same with
``{X,U,W}`` and ``W`` in place of ``Z``)::

                     includes Z            drops Z
    includes U   (1) R2(Y|A,X,U,Z)   (2) R2(Y|A,X,U)
    drops U      (3) R2(Y|A,X,Z)     (4) R2(Y|A,X)

- ``check``   = (1) - (2): what **Z** adds, with ``U`` present -- the claim.
- ``control`` = (3) - (4): what **Z** adds, with ``U`` gone -- the power check.
- ``u_dependence`` = (1) - (3): what **U** adds -- how much confounding this
  draw actually has, i.e. how much there is for the proxy to leak.

The first two are the figure's y-axis (horizontal edges: drop the *proxy*), the
third is its colour and point ordering (a vertical edge: drop *U*). Note that
``control`` and ``u_dependence`` share the term (3) with *opposite* signs, which
biases their correlation downward -- the positive association the figure shows
is therefore not an artifact of reusing the same fits.

Two caveats, stated in the caption:

- **Supersets.** The cached ``U``/``X``/``Z``/``W`` are the full delivered
  blocks, supersets of the columns each mechanism actually consumed. Extra
  columns cannot manufacture signal that is absent, so a near-zero ``dR2`` is
  the conservative reading; conditioning on the *full* ``U`` is also exactly
  what the assumption asks for.
- **Scope of the W clause.** ``W indep (A, Z) | (X, U)`` has two halves; the
  audit tests the ``A`` half (``W`` adds nothing to the treatment logit given
  ``X, U``). The ``Z`` half is the same wiring fact -- neither block is an input
  of the other's mechanism -- and is not separately regressed.
- **Conditional independence vs. mean independence.** A cross-fitted ``R^2``
  detects leakage into the conditional *mean*. A dependence that leaves the mean
  untouched (pure variance/shape) would pass. That is the standard scope of a
  regression-based exclusion check and is not tightened here.

Datasets with ``d_U = 0`` are excluded: with no confounder there is no path for
the positive control to open, so the control is vacuous rather than informative.
Their count is reported in ``summary.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from tqdm import tqdm

from ..design import (
    COUNT_CMAP,
    FONT_SIZE_PT,
    ROLES,
    apply_style,
    block_color,
    figsize_grid,
)
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from ..regression import DEFAULT_REGRESSION_CACHE_DIR, cached_oof
from .base import Analysis, AnalysisArtifacts, register

#: The two exclusion clauses, as (predictor blocks, excluded block, target).
#: ``blocks`` is the *full* conditioning set including ``U``; the positive
#: control is this same set with ``U`` removed, so it is derived, not listed.
#: Block order is fixed so the predictor-set labels -- and hence the regression
#: cache keys -- are stable.
CHECKS: Dict[str, Dict[str, Any]] = {
    "Y": {"blocks": ("A", "X", "U", "Z"), "dropped": "Z", "target": "Y_obs"},
    "A": {"blocks": ("X", "U", "W"), "dropped": "W", "target": "A_logit"},
}

#: The two conditions, in plotting order: the assumption's own conditioning set,
#: then the ``U``-free positive control.
CONDITIONS = ("with_U", "without_U")

#: ``dR2`` above which we call the incremental signal *detectable*. Cross-fitted
#: R^2 differences are noisy around zero (both terms are estimates on the same
#: folds), so a bare ``> 0`` share would report fold noise as leakage; 2% of
#: outcome variance is comfortably above that and still far below anything a
#: downstream estimator would notice.
LEAK_THRESHOLD = 0.02

#: Top fraction of datasets, by how much of the target ``U`` uniquely explains,
#: on which the headline read is taken. Over half the prior's draws route almost
#: no ``U`` into ``Y`` or ``A`` (``confounding_share_of_U_in_*`` is Beta(1.2,2.8),
#: small most of the time), and on those the positive control has nothing to
#: expose -- its near-zero value there says the confounding is weak, not that the
#: test is blind. Restricting to the most-confounded third is what gives the
#: control power and hence gives the near-zero check its meaning.
TOP_TERCILE = 1.0 / 3.0

#: Width, in x-axis units, of the point spread inside one column. The points
#: are laid out across it by confounding rank (see ``_strip_panel``), so this is
#: the length of that implicit axis; 0.6 keeps the two columns of a panel
#: clearly separated at a spacing of 1.0.
SPREAD = 0.6


def _target(record: PriorDatasetRecord, side: str) -> np.ndarray:
    """The regression target for a check: ``Y_obs`` (Y-side), logit (A-side)."""
    if side == "A":
        # Clipped only to keep the logit finite; A_probs is already clamped to
        # [eps, 1-eps] by sample_A, so this is a no-op in practice.
        p = np.clip(record.A_probs.astype(np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p)).astype(np.float32)
    return record.Y_obs.astype(np.float32)


def _blocks(record: PriorDatasetRecord) -> Dict[str, np.ndarray]:
    """Every predictor block, as ``(T, d)`` float32 matrices (``A`` is a column)."""
    return {
        "A": np.asarray(record.A, dtype=np.float32).reshape(len(record.A), 1),
        "X": np.asarray(record.X, dtype=np.float32),
        "U": np.asarray(record.U, dtype=np.float32),
        "W": np.asarray(record.W, dtype=np.float32),
        "Z": np.asarray(record.Z, dtype=np.float32),
    }


def _r2(
    blocks: Mapping[str, np.ndarray],
    names: Sequence[str],
    y: np.ndarray,
    tag_prefix: str,
    model: str,
    n_splits: int,
    seed: int,
    cache_dir: Path,
) -> float:
    """Cross-fitted ``R^2`` of ``y`` on the concatenated ``names`` blocks.

    The predictor-set label is the block names joined in the order given, so a
    set reachable from two conditions (e.g. ``A+X+Z``, both the positive
    control's full set and the ``with_U`` set minus ``U``) hits the *same* cache
    entry and is fitted once.
    """
    X = np.hstack([blocks[name] for name in names])
    return cached_oof(
        f"{tag_prefix}|predictors={'+'.join(names)}",
        X,
        y,
        model=model,
        n_splits=n_splits,
        seed=seed,
        cache_dir=cache_dir,
    )["r2"]


def exclusion_frame(
    records: List[PriorDatasetRecord],
    cache_key: str,
    model: str = "rf",
    n_splits: int = 3,
    seed: int = 0,
    max_rows: int | None = 1024,
    regression_cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per (dataset, side, condition): the incremental ``dR2``.

    Columns: ``index``, ``side`` (``"Y"``/``"A"``), ``condition``
    (``"with_U"``/``"without_U"``), ``dropped`` (the excluded block), ``r2_full``
    (that condition's full-set ``R^2``), ``delta_r2`` and ``dim_U``.

    ``cache_key`` scopes the regression cache to the dataset collection; each
    fit is tagged with it, the record index, the target and the predictor set,
    so re-renders are free and specs never collide. Datasets with ``d_U = 0``
    are skipped (see the module docstring).

    Four cross-fitted regressions per side per dataset (two conditions x
    full/reduced, minus the one set the two conditions share -- three distinct
    fits, the fourth served from cache).
    """
    rows: List[Dict[str, Any]] = []
    records = list(records)
    for record in tqdm(
        records,
        desc="exclusion regressions",
        unit="dataset",
        disable=not progress or len(records) <= 1,
    ):
        blocks = _blocks(record)
        dim_U = blocks["U"].shape[1]
        if dim_U == 0:
            continue

        n_rows = len(record.A)
        if max_rows is not None and n_rows > max_rows:
            n_rows = max_rows
            blocks = {k: v[:n_rows] for k, v in blocks.items()}

        for side, check in CHECKS.items():
            y = _target(record, side)[:n_rows]
            dropped = check["dropped"]
            # Row count in the tag: a different max_rows is a different
            # regression and must not hit the same cache entry.
            tag = (
                f"{cache_key}|i={record.index}|target={check['target']}"
                f"|rows={n_rows}"
            )
            for condition in CONDITIONS:
                names = [
                    name
                    for name in check["blocks"]
                    if not (condition == "without_U" and name == "U")
                ]
                r2_full = _r2(
                    blocks, names, y, tag, model, n_splits, seed, regression_cache_dir
                )
                reduced = [name for name in names if name != dropped]
                r2_reduced = _r2(
                    blocks, reduced, y, tag, model, n_splits, seed, regression_cache_dir
                )
                rows.append(
                    {
                        "index": record.index,
                        "side": side,
                        "condition": condition,
                        "dropped": dropped,
                        "r2_full": r2_full,
                        "delta_r2": r2_full - r2_reduced,
                        "dim_U": int(dim_U),
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class ExclusionResults:
    """Per-(dataset, side, condition) incremental ``dR2``, plus provenance."""

    frame: pd.DataFrame
    model: str
    n_skipped_no_u: int

    def series(self, side: str, condition: str) -> np.ndarray:
        """The ``dR2`` values for one (side, condition) cell, as an array."""
        frame = self.frame
        mask = (frame["side"] == side) & (frame["condition"] == condition)
        return frame.loc[mask, "delta_r2"].to_numpy()

    def paired(self, side: str) -> pd.DataFrame:
        """One row per dataset: ``check``, ``control`` and ``u_dependence``.

        ``u_dependence`` is the unique contribution of ``U`` to the target --
        ``R2(with U) - R2(without U)``, i.e. the difference between the two
        conditions' *full*-set fits. It costs nothing: both terms are already
        computed for the check, and it is precisely the quantity the positive
        control's power depends on. A dataset whose target barely responds to
        ``U`` has no leak for the control to expose, so its near-zero control
        value is a statement about the confounding in that draw, not about the
        test.
        """
        data = self.frame[self.frame["side"] == side]
        wide = data.pivot(
            index="index", columns="condition", values=["delta_r2", "r2_full"]
        )
        return pd.DataFrame(
            {
                "check": wide[("delta_r2", "with_U")],
                "control": wide[("delta_r2", "without_U")],
                "u_dependence": (
                    wide[("r2_full", "with_U")] - wide[("r2_full", "without_U")]
                ),
            }
        ).reset_index()

    def strongly_confounded(self, side: str) -> pd.DataFrame:
        """The :data:`TOP_TERCILE` of datasets by ``u_dependence``, for ``side``.

        The subset where the positive control is *entitled* to have power, and
        so the subset the exclusion claim is really tested on.
        """
        data = self.paired(side)
        threshold = data["u_dependence"].quantile(1.0 - TOP_TERCILE)
        return data[data["u_dependence"] > threshold]


@register
class ExclusionAnalysis(Analysis):
    name = "exclusion"
    description = (
        "Exclusion-restriction audit (B.3.2): incremental R2 of the excluded "
        "proxy (Z->Y, W->A), with a U-dropped positive control."
    )

    #: Regression knobs; override on an instance for tests (``model="linear"``)
    #: or for the TabICL robustness pass. ``rf`` at the full 1024 rows is the
    #: headline setting, matching the other B.3 audits.
    model = "rf"
    n_splits = 3
    seed = 0
    max_rows = 1024
    regression_cache_dir = DEFAULT_REGRESSION_CACHE_DIR

    def compute(self, cache: PriorCache) -> ExclusionResults:
        records = list(cache)
        n_skipped = sum(1 for r in records if np.asarray(r.U).shape[1] == 0)
        if self.model == "tabicl" and len(records) > 64:
            # ~n * 2 sides * 3 distinct predictor sets * n_splits TabICL fits.
            print(
                f"exclusion: {len(records)} datasets x 6 predictor sets with "
                f"TabICL -- slow on first run (predictions are then cached). "
                f"Lower --n-datasets for a quicker pass."
            )
        frame = exclusion_frame(
            records,
            cache_key=cache.spec.cache_key,
            model=self.model,
            n_splits=self.n_splits,
            seed=self.seed,
            max_rows=self.max_rows,
            regression_cache_dir=self.regression_cache_dir,
        )
        return ExclusionResults(
            frame=frame, model=self.model, n_skipped_no_u=int(n_skipped)
        )

    def render(self, results: ExclusionResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_figure(results, outdir)
        tables = self._render_table(results, outdir)

        summary: Dict[str, Any] = {
            "n_datasets": int(results.frame["index"].nunique()),
            "n_skipped_no_u": results.n_skipped_no_u,
            "model": results.model,
            "leak_threshold": LEAK_THRESHOLD,
        }
        for side, check in CHECKS.items():
            for condition in CONDITIONS:
                values = results.series(side, condition)
                key = f"{check['dropped']}_to_{side}_{condition}"
                summary[f"median_delta_r2_{key}"] = float(np.median(values))
                summary[f"p90_delta_r2_{key}"] = float(np.quantile(values, 0.9))
                summary[f"share_above_threshold_{key}"] = float(
                    np.mean(values > LEAK_THRESHOLD)
                )

            # How much confounding there is to leak, and the read restricted to
            # the third of datasets where the control can actually have power.
            paired = results.paired(side)
            top = results.strongly_confounded(side)
            summary[f"median_u_dependence_{side}"] = float(
                paired["u_dependence"].median()
            )
            summary[f"share_u_dependence_below_threshold_{side}"] = float(
                np.mean(paired["u_dependence"] < LEAK_THRESHOLD)
            )
            summary[f"spearman_u_dependence_control_{side}"] = float(
                spearmanr(paired["u_dependence"], paired["control"]).statistic
            )
            for column, condition in (("check", "with_U"), ("control", "without_U")):
                key = f"{check['dropped']}_to_{side}_{condition}_top_tercile"
                summary[f"median_delta_r2_{key}"] = float(top[column].median())
                summary[f"share_above_threshold_{key}"] = float(
                    np.mean(top[column] > LEAK_THRESHOLD)
                )
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    # -- figure -------------------------------------------------------------

    def _render_figure(self, results: ExclusionResults, outdir: Path) -> List[Path]:
        """One panel per clause, each a pair of columns: the assumption's own
        conditioning set against the ``U``-dropped positive control. Points are
        coloured *and* horizontally ordered by ``u_dependence`` (see
        :meth:`_strip_panel`); a median tick summarises each column. Read as a
        pair: the left column sitting on zero is the claim, the right one
        lifting off it -- and fanning upward with confounding strength -- is
        what makes that zero mean something.
        """
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize_grid(ncols=2, nrows=1, panel_aspect=1.0),
            sharey=True,
        )
        titles = {
            "Y": r"(a) $Z$ excluded from $f_Y$: $\Delta R^2_Z$",
            "A": r"(b) $W$ excluded from $f_A$: $\Delta R^2_W$",
        }

        values = results.frame["delta_r2"].to_numpy()
        finite = values[np.isfinite(values)]
        lo, hi = np.percentile(finite, [0.5, 99.5])
        pad = 0.05 * max(hi - lo, 1e-3)

        dependence = np.concatenate(
            [results.paired(side)["u_dependence"].to_numpy() for side in CHECKS]
        )
        vmax = max(float(np.percentile(dependence, 95)), 1e-3)
        norm = Normalize(vmin=0.0, vmax=vmax)

        for ax, side in zip(axes, CHECKS):
            self._strip_panel(ax, results, side, norm)
            ax.set_title(titles[side])
            ax.set_ylim(min(lo, 0.0) - pad, hi + pad)
        # The y-axis and the colour are two different edges of the same 2x2 fit
        # lattice {with/without U} x {with/without the excluded proxy}: the
        # y-axis is the increment from dropping the *proxy*, the colour the
        # increment from dropping *U*. Both labels name their dropped block, or
        # a reader has no way to tell two "incremental R^2" scales apart.
        axes[0].set_ylabel(r"$\Delta R^2$ of the excluded proxy")

        bar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=COUNT_CMAP),
            ax=axes,
            fraction=0.04,
            pad=0.02,
            extend="both",
        )
        bar.set_label(r"confounding: $\Delta R^2$ of $U$", fontsize=FONT_SIZE_PT - 1)
        return save_figure(fig, outdir, "exclusion")

    @staticmethod
    def _strip_panel(ax, results: ExclusionResults, side: str, norm) -> None:
        colours = {"with_U": block_color(side), "without_U": block_color("U")}
        labels = {
            "with_U": "conditioning\non $U$",
            "without_U": "control:\n$U$ dropped",
        }

        paired = results.paired(side).sort_values("u_dependence")
        dependence = paired["u_dependence"].to_numpy()
        columns = {"with_U": "check", "without_U": "control"}

        offsets = np.argsort(np.argsort(dependence)) / max(len(dependence) - 1, 1)
        offsets = SPREAD * (offsets - 0.5)

        data = [paired[columns[condition]].to_numpy() for condition in CONDITIONS]

        for position, (condition, values) in enumerate(zip(CONDITIONS, data)):
            ax.scatter(
                position + offsets,
                values,
                s=5,
                alpha=0.75,
                linewidths=0,
                c=dependence,
                cmap=COUNT_CMAP,
                norm=norm,
                zorder=3,
            )

            median = float(np.median(values))
            ax.plot(
                [position - 0.32, position + 0.32],
                [median, median],
                color=colours[condition],
                lw=1.6,
                zorder=4,
            )

        ax.axhline(0.0, **ROLES["reference"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([labels[c] for c in CONDITIONS])
        ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="x")

    # -- table --------------------------------------------------------------

    def _render_table(self, results: ExclusionResults, outdir: Path) -> List[Path]:
        """Median / 90th-percentile ``dR2`` and the detectable-leak share, one
        row per (clause, condition)."""
        clause = {
            "Y": r"$Z$ excluded from $f_Y$",
            "A": r"$W$ excluded from $f_A$",
        }
        predictors = {
            ("Y", "with_U"): r"$\{A, X, U, Z\}$",
            ("Y", "without_U"): r"$\{A, X, Z\}$",
            ("A", "with_U"): r"$\{X, U, W\}$",
            ("A", "without_U"): r"$\{X, W\}$",
        }
        condition_label = {
            "with_U": "conditions on $U$",
            "without_U": "positive control ($U$ dropped)",
        }
        rows = []
        for side in CHECKS:
            top = results.strongly_confounded(side)
            position = 0
            for condition, column in (("with_U", "check"), ("without_U", "control")):
                for subset, values in (
                    ("all", results.series(side, condition)),
                    ("strongly confounded (top third)", top[column].to_numpy()),
                ):
                    rows.append(
                        {
                            "clause": clause[side] if position == 0 else "",
                            "conditioning set": predictors[(side, condition)],
                            "role": condition_label[condition],
                            "datasets": subset,
                            "median $\\Delta R^2$": float(np.median(values)),
                            "90th pct": float(np.quantile(values, 0.9)),
                            f"share $> {LEAK_THRESHOLD}$": float(
                                np.mean(values > LEAK_THRESHOLD)
                            ),
                        }
                    )
                    position += 1
        return save_table(
            pd.DataFrame(rows),
            outdir,
            "exclusion_table",
            caption=(
                "Exclusion-restriction audit. For each clause of "
                "\\Cref{as:proxy_exclusion}, the cross-fitted incremental "
                "$\\Delta R^2$ of the excluded block -- $R^2$ with the block "
                "minus $R^2$ without it, over the stated conditioning set. "
                "Targets: $Y_{\\mathrm{obs}}$ for the $Y$ clause, "
                "$\\mathrm{logit}(\\pi)$ for the $A$ clause. The positive "
                "control drops $U$, the only path linking the excluded block to "
                "the target, and so measures the power of the same test to "
                "detect a leak that is really there. Because that power is only "
                "as large as the confounding in a given draw, each read is "
                "reported both over all datasets and over the third with the "
                "largest unique $U$-dependence of the target -- the subset on "
                "which the control is entitled to fire, and hence the subset "
                "that tests the exclusion claim."
            ),
            label="tab:prior_exclusion",
            group_column="clause",
            escape=False,
        )
