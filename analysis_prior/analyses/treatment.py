"""Treatment assignment: the ``f_A`` pipeline (B.2.5) and the overlap audit (B.3.1).

Two analyses share this module because they share a subject -- the propensity
``pi`` -- at two different scales. :class:`TreatmentWalkthroughAnalysis` shows
*how one dataset's* ``pi`` is built (``fig:prior_treatment_walkthrough``);
:class:`OverlapAnalysis` asks what the resulting ``pi`` looks like *across the
whole prior* (``fig:prior_overlap``).

Covers appendix B.2.5 (``fig:prior_treatment_walkthrough``). ``sample_A`` turns
a raw ``f_A`` logit into a treatment draw through a fixed five-step pipeline
(see the equations of ``app:prior_treatment``):

1. **raw** ``ell = f_A(U^[A], X_A, Z*, eps_A)`` -- an uncontrolled MLP output,
   whose location and spread drift with depth/width/input scale;
2. **standardise** ``ell_hat = (ell - mean) / (sd + 1e-8)`` (eq:A_normalise),
   pinning the batch to zero mean and unit variance;
3. **shift + scale** ``ell' = s * ell_hat + b`` (eq:A_affine), the two per-dataset
   overlap knobs setting prevalence (``b``) and sharpness (``s``);
4. **sigmoid + clamp** ``pi = clip(sigmoid(ell'), eps, 1 - eps)``
   (eq:A_sigmoid / eq:A_clamp), the propensity, floored away from 0/1;
5. **draw** ``A ~ Bernoulli(pi)``.

This analysis illustrates the pipeline on a *single* representative dataset,
tracking one highlighted unit (drawn in red) through all five stages. It is a
pure function of the cached raw logits (``record.a_logits``, retained only when
the cache carries mechanism diagnostics -- regenerate with ``--force`` if not)
and the per-dataset propensity knobs.

The production overlap clamp ``eps = 1e-3`` sits flush against 0/1 and is
invisible on a ``[0, 1]`` axis, so panel (d) draws the clamp at an *enlarged*
:data:`DISPLAY_EPSILON` purely for legibility -- the pipeline is otherwise
exact, and the reconstruction is validated against the stored ``A_probs`` at the
true ``eps`` (``reconstruction_residual`` in ``summary.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from matplotlib.patches import Patch

from ..design import FONT_SIZE_PT, ROLES, apply_style, block_color, figsize_grid
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure
from .base import Analysis, AnalysisArtifacts, register

#: The four distributional stages, in order, as (key, panel title, x symbol).
#: The fifth stage (the Bernoulli draw) is rendered separately.
STAGES = (
    ("raw", "raw logit", r"$\ell = f_A(\cdot)$"),
    ("standardised", "standardise", r"$\hat\ell$"),
    ("affine", "shift $+$ scale", r"$\ell' = s\,\hat\ell + b$"),
    ("propensity", "sigmoid $+$ clamp", r"$\pi = \sigma(\ell')$"),
)

#: Quantile of the raw logit picked out as the tracked (red) unit. Above the
#: median so it moves visibly under a positive shift and lands clearly on one
#: side of the sigmoid, without being a saturated tail point.
TRACKED_QUANTILE = 0.85

#: Clamp used *only* for drawing panel (d)/(e). The production clamp
#: (``propensity_epsilon``, 1e-3) is invisible against the axis; enlarging it
#: makes the floor -- and the mass piled on it -- legible. Everything else in
#: the figure, and the A_probs validation, uses the true production clamp.
DISPLAY_EPSILON = 0.05

#: Target treatment prevalence for the illustrated dataset. Deliberately off the
#: balanced split so panel (e)'s two arms are visibly unequal -- a 50/50 example
#: makes the final draw look like a coin and hides that ``b`` moves prevalence.
TARGET_PREVALENCE = 0.68


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def reconstruct_pipeline(record: PriorDatasetRecord) -> Dict[str, Any]:
    """Re-apply ``sample_A``'s transform to one dataset's raw logits.

    Pure function of the stored raw logits and the per-dataset propensity knobs.
    Returns the per-row array at each pipeline stage (``propensity`` clamped at
    the enlarged :data:`DISPLAY_EPSILON` for the figure, ``propensity_true`` at
    the production clamp), the scalar knobs, and ``recon_residual`` -- the max
    absolute gap between the true-clamp propensity and the stored ``A_probs``, a
    self-check that the reconstruction reproduces ``sample_A`` exactly.

    Raises if the record predates the raw-logit diagnostics -- the cache must be
    regenerated with ``--force`` (same data, logits added).
    """
    if record.a_logits is None or record.sampled_hp is None:
        raise ValueError(
            f"dataset {record.index} has no stored f_A logits; the cache "
            "predates them. Regenerate with --force (same data, logits added)."
        )

    raw = np.asarray(record.a_logits, dtype=np.float64)
    # The mean/std sample_A standardises by. f_A now runs directly on the T
    # observed rows (Z/A are leaves w.r.t. MU0/MU1, generator no longer
    # replicates them K-fold), so these coincide with raw's own moments for
    # any current record; stored explicitly only for older records generated
    # before that change, where sample_A standardised over the full T*K
    # population and raw held just the observed-row slice of it.
    mean = record.a_logit_mean if record.a_logit_mean is not None else float(raw.mean())
    std = (
        record.a_logit_std if record.a_logit_std is not None else float(raw.std(ddof=1))
    )
    shift = float(record.sampled_hp["propensity_shift"])
    scale = float(record.sampled_hp["propensity_scale"])
    epsilon = float(record.sampled_hp.get("propensity_epsilon", 1e-3))

    standardised = (raw - mean) / (std + 1e-8)
    affine = scale * standardised + shift
    prob = _sigmoid(affine)
    propensity_true = np.clip(prob, epsilon, 1.0 - epsilon)
    propensity = np.clip(prob, DISPLAY_EPSILON, 1.0 - DISPLAY_EPSILON)

    residual = float(np.max(np.abs(propensity_true - np.asarray(record.A_probs))))
    lo = float(np.mean(prob < DISPLAY_EPSILON))
    hi = float(np.mean(prob > 1.0 - DISPLAY_EPSILON))
    return {
        "raw": raw,
        "standardised": standardised,
        "affine": affine,
        "propensity": propensity,
        "propensity_true": propensity_true,
        "shift": shift,
        "scale": scale,
        "epsilon": epsilon,
        "display_epsilon": DISPLAY_EPSILON,
        "prevalence": float(np.mean(record.A)),
        "clamp_lo": lo,
        "clamp_hi": hi,
        "recon_residual": residual,
    }


def select_dataset(records: Sequence[PriorDatasetRecord]) -> int:
    """Index of the dataset to illustrate.

    Wants a legible walkthrough: an *un*balanced prevalence near
    :data:`TARGET_PREVALENCE` (so panel (e)'s arms are visibly unequal and the
    shift ``b`` is seen to do something) together with a visible overlap clamp.
    Among datasets whose bounds carry a legible-but-not-degenerate mass
    (total in ``[0.06, 0.45]``), we pick the one whose prevalence is closest to
    the target on either side of the split; failing that we fall back to the
    most clamped dataset, then to the sharpest overall. Deterministic.
    """
    scored = []
    for i, record in enumerate(records):
        if record.a_logits is None:
            continue
        pipe = reconstruct_pipeline(record)
        scored.append(
            (i, pipe["prevalence"], pipe["scale"], pipe["clamp_lo"], pipe["clamp_hi"])
        )
    if not scored:
        raise ValueError(
            "no dataset carries f_A logits; regenerate the cache with --force."
        )

    # Distance to the target imbalance, symmetric about 0.5 so a rare-treatment
    # dataset (prevalence ~0.32) is as acceptable as a common-treatment one.
    def imbalance_gap(prevalence: float) -> float:
        return abs(abs(prevalence - 0.5) - abs(TARGET_PREVALENCE - 0.5))

    legible = [s for s in scored if 0.06 <= (s[3] + s[4]) <= 0.45]
    if legible:
        return min(legible, key=lambda s: imbalance_gap(s[1]))[0]
    return max(scored, key=lambda s: s[3] + s[4])[0]


@dataclass
class TreatmentWalkthroughResults:
    """The illustrated dataset, its per-stage arrays and the tracked unit."""

    index: int
    seed: int
    pipeline: Dict[str, Any]
    tracked_row: int
    #: Treatment draw of the tracked unit (0/1).
    tracked_A: int

    @property
    def tracked_values(self) -> Dict[str, float]:
        """The tracked unit's value at each of the four distributional stages."""
        return {key: float(self.pipeline[key][self.tracked_row]) for key, *_ in STAGES}


@register
class TreatmentWalkthroughAnalysis(Analysis):
    name = "treatment_walkthrough"
    description = (
        "Treatment assignment on one SCM: raw f_A logit -> standardise -> "
        "shift/scale -> sigmoid+clamp -> Bernoulli draw, tracking one unit."
    )

    def compute(self, cache: PriorCache) -> TreatmentWalkthroughResults:
        index = select_dataset(cache.records)
        record = cache.records[index]
        pipeline = reconstruct_pipeline(record)

        # Track a clearly-above-average unit through the pipeline. Chosen on the
        # raw logit so the choice is independent of the (per-dataset) knobs.
        order = np.argsort(pipeline["raw"], kind="stable")
        tracked_row = int(order[int(TRACKED_QUANTILE * (len(order) - 1))])
        return TreatmentWalkthroughResults(
            index=index,
            seed=record.seed,
            pipeline=pipeline,
            tracked_row=tracked_row,
            tracked_A=int(record.A[tracked_row]),
        )

    def render(
        self, results: TreatmentWalkthroughResults, outdir: Path
    ) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_walkthrough(results, outdir)

        pipe = results.pipeline
        summary: Dict[str, Any] = {
            "dataset_index": results.index,
            "dataset_seed": results.seed,
            "propensity_shift": pipe["shift"],
            "propensity_scale": pipe["scale"],
            "propensity_epsilon": pipe["epsilon"],
            "display_epsilon": pipe["display_epsilon"],
            "prevalence": pipe["prevalence"],
            "clamp_lo_fraction": pipe["clamp_lo"],
            "clamp_hi_fraction": pipe["clamp_hi"],
            # Should be ~1e-6: confirms the reconstruction reproduces sample_A.
            "reconstruction_residual": pipe["recon_residual"],
            "tracked_row": results.tracked_row,
            "tracked_A": results.tracked_A,
            **{f"tracked_{k}": v for k, v in results.tracked_values.items()},
        }
        return AnalysisArtifacts(summary=summary, figures=figures, tables=[])

    def _render_walkthrough(
        self, results: TreatmentWalkthroughResults, outdir: Path
    ) -> List[Path]:
        """Five panels left-to-right: the four distributions then the draw, with
        one unit (red) carried through. The red line sits at the tracked unit's
        value in each panel, so a reader can watch a single logit become a
        propensity and then a coin flip.
        """
        pipe = results.pipeline
        tracked = results.tracked_row
        highlight = block_color("U")  # vermillion; the tracked unit (no U here)
        fill = "0.72"

        fig, axes = plt.subplots(
            1,
            5,
            figsize=figsize_grid(
                ncols=5, nrows=1, panel_aspect=1.5, extra_height_in=0.25
            ),
        )
        for ax, (key, title, xlabel) in zip(axes, STAGES):
            values = pipe[key]
            xlim = (0.0, 1.0) if key == "propensity" else None
            self._density_panel(
                ax, values, float(values[tracked]), fill, highlight, xlim=xlim
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            if key == "propensity":
                for bound in (pipe["display_epsilon"], 1.0 - pipe["display_epsilon"]):
                    ax.axvline(bound, color="0.35", linestyle=":", lw=1.0)
        axes[0].set_ylabel("density")

        self._draw_panel(axes[4], results, fill, highlight)
        return save_figure(fig, outdir, "treatment_walkthrough")

    @staticmethod
    def _density_panel(ax, values, tracked_value, fill, highlight, xlim=None) -> None:
        rng = xlim if xlim is not None else (float(values.min()), float(values.max()))
        ax.hist(
            values,
            bins=40,
            range=rng,
            density=True,
            color=fill,
            edgecolor="none",
        )
        ax.axvline(tracked_value, color=highlight, lw=1.6)
        # A dot on the baseline marks the tracked unit's position explicitly.
        ax.plot(
            [tracked_value],
            [0.0],
            marker="o",
            markersize=4,
            color=highlight,
            clip_on=False,
            zorder=5,
        )
        if xlim is not None:
            ax.set_xlim(*xlim)

    @staticmethod
    def _draw_panel(ax, results: TreatmentWalkthroughResults, fill, highlight) -> None:
        """The realised treatment across the *batch* -- the population fraction
        in each arm, so this panel is the same population view as (a)-(d) rather
        than a single unit's Bernoulli PMF. The tracked unit's realised arm is
        marked in red, echoing the baseline dot the other panels use for it.
        """
        prevalence = float(results.pipeline["prevalence"])
        heights = [1.0 - prevalence, prevalence]
        ax.bar([0, 1], heights, width=0.6, color=fill, edgecolor="0.3")
        # The tracked unit landed in one arm; mark it like the other panels.
        ax.plot(
            [results.tracked_A],
            [0.0],
            marker="o",
            markersize=4,
            color=highlight,
            clip_on=False,
            zorder=5,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$A=0$", r"$A=1$"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(r"sample $A$")
        ax.set_xlabel(r"$A\sim\mathrm{Bern}(\pi)$")
        ax.set_ylabel("fraction")
        ax.grid(axis="x")


# ---------------------------------------------------------------------------
# B.3.1 -- overlap audit
# ---------------------------------------------------------------------------

#: Conventional practical-overlap band. Positivity holds *structurally* for
#: every dataset (the clamp puts pi in [eps, 1-eps] by fiat), so a pass/fail
#: check would be vacuous; this band is the threshold for *practical* overlap,
#: below which a downstream estimator has little to work with.
OVERLAP_BAND = (0.05, 0.95)

#: Quantile grid for the per-dataset propensity curves of panel (b). 201 points
#: resolves the tails (where the interesting mass is) without making the
#: cross-dataset percentile arrays large.
QUANTILE_GRID = np.linspace(0.0, 1.0, 201)

#: Cross-dataset bands drawn around the median curve, as (lo, hi) percentile
#: pairs from widest to narrowest. Drawn in that order so the narrow (darker)
#: band paints over the wide one.
BANDS = ((10, 90), (25, 75))

#: Number of log-spaced bins for the pooled min(pi, 1-pi) histogram.
N_POOLED_BINS = 60

#: Slack when testing whether a propensity sits *on* a clamp bound. The cache
#: stores ``A_probs`` as float32, so the clamp's exact value does not survive the
#: round trip bit-for-bit; comfortably above float32 resolution at 1e-3, and far
#: below any spacing that would catch an unclamped row.
CLAMP_TOL = 1e-6


@dataclass
class OverlapResults:
    """Overlap statistics, pooled over rows and summarised per dataset.

    Attributes
    ----------
    pooled_margin : ndarray, shape (n_datasets * T,)
        ``min(pi, 1 - pi)`` for every row of every dataset -- the distance from
        each unit's propensity to the nearest degenerate assignment. Folding the
        two tails into one statistic makes "how close to no-overlap" a single
        axis, which is the quantity positivity is actually about.

    quantile_curves : ndarray, shape (n_datasets, len(QUANTILE_GRID))
        Per-dataset quantile function of ``pi``. Row ``i`` column ``j`` is the
        ``QUANTILE_GRID[j]``-quantile of dataset ``i``'s propensities.

    poor_fraction, clamp_fraction, prevalence : ndarray, shape (n_datasets,)
        Per dataset: share of rows outside :data:`OVERLAP_BAND`; share of rows
        sitting *on* the clamp bounds (positivity held by fiat rather than by
        the mechanism); and the realised treated fraction ``mean(A)``.

    epsilon : float
        The production clamp, ``propensity_epsilon``.
    """

    pooled_margin: np.ndarray
    quantile_curves: np.ndarray
    poor_fraction: np.ndarray
    clamp_fraction: np.ndarray
    prevalence: np.ndarray
    epsilon: float


@register
class OverlapAnalysis(Analysis):
    name = "treatment_overlap"
    description = (
        "Overlap audit (B.3.1): distribution of the true propensity pi across "
        "the prior, pooled over rows and as per-dataset quantile bands."
    )

    def compute(self, cache: PriorCache) -> OverlapResults:
        """Read ``A_probs`` straight off the cache -- no regression needed.

        ``A_probs`` *is* the ground-truth ``P(A=1 | X, U)`` that ``as:positivity``
        is stated over, so the audit is a pure summary of cached values; fitting
        a propensity model on ``(X, U)`` would only re-estimate it with error.
        """
        margins: List[np.ndarray] = []
        curves: List[np.ndarray] = []
        poor: List[float] = []
        clamped: List[float] = []
        prevalence: List[float] = []
        epsilon = 1e-3

        lo_band, hi_band = OVERLAP_BAND
        for record in cache.records:
            pi = np.asarray(record.A_probs, dtype=np.float64)
            if record.sampled_hp is not None:
                epsilon = float(record.sampled_hp.get("propensity_epsilon", epsilon))

            margins.append(np.minimum(pi, 1.0 - pi))
            curves.append(np.quantile(pi, QUANTILE_GRID))
            poor.append(float(np.mean((pi < lo_band) | (pi > hi_band))))
            # The clamp is a hard clip, so a clamped row's propensity is exactly
            # the bound; the tolerance only absorbs the cache's float32 storage
            # (1 - eps round-trips to 0.99900001, not 0.999).
            on_bound = (pi <= epsilon + CLAMP_TOL) | (pi >= 1.0 - epsilon - CLAMP_TOL)
            clamped.append(float(np.mean(on_bound)))
            prevalence.append(float(np.mean(record.A)))

        return OverlapResults(
            pooled_margin=np.concatenate(margins),
            quantile_curves=np.stack(curves),
            poor_fraction=np.asarray(poor),
            clamp_fraction=np.asarray(clamped),
            prevalence=np.asarray(prevalence),
            epsilon=epsilon,
        )

    def render(self, results: OverlapResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        fig, axes = plt.subplots(
            1, 2, figsize=figsize_grid(ncols=2, nrows=1, panel_aspect=0.85)
        )
        self._pooled_panel(axes[0], results)
        self._quantile_band_panel(axes[1], results)
        figures = save_figure(fig, outdir, "overlap")

        lo_band, _ = OVERLAP_BAND
        summary: Dict[str, Any] = {
            "n_datasets": int(results.quantile_curves.shape[0]),
            "propensity_epsilon": results.epsilon,
            "overlap_band": list(OVERLAP_BAND),
            # Pooled over rows: what the prior serves on average.
            "pooled_fraction_outside_band": float(
                np.mean(results.pooled_margin < lo_band)
            ),
            "pooled_margin_p01": float(np.quantile(results.pooled_margin, 0.01)),
            "pooled_margin_median": float(np.median(results.pooled_margin)),
            # Per dataset: how concentrated the poor overlap is.
            "median_dataset_fraction_outside_band": float(
                np.median(results.poor_fraction)
            ),
            "share_datasets_any_mass_outside_band": float(
                np.mean(results.poor_fraction > 0.0)
            ),
            "share_datasets_materially_poor": float(
                np.mean(results.poor_fraction > 0.1)
            ),
            # How much work the clamp -- i.e. positivity by fiat -- is doing.
            "median_dataset_clamp_fraction": float(np.median(results.clamp_fraction)),
            "share_datasets_any_clamped": float(np.mean(results.clamp_fraction > 0.0)),
            "median_prevalence": float(np.median(results.prevalence)),
        }
        return AnalysisArtifacts(summary=summary, figures=figures, tables=[])

    @staticmethod
    def _pooled_panel(ax, results: OverlapResults) -> None:
        """(a) Pooled histogram of ``min(pi, 1-pi)`` over every row of every
        dataset, on a log axis: the production clamp sits at ``eps = 1e-3`` and
        the mass that matters piles up against it, which a linear axis hides
        entirely. Heights are the share of *all* rows, so the panel reads as the
        marginal law of overlap under the prior.

        Caveat worth stating in the caption: bars are per-bin shares on a *log*
        grid, so bin width grows to the right and part of the rightward rise is
        that widening rather than extra mass. The panel is meant to be read for
        where mass sits (the pile-up on the clamp) rather than as a density.
        """
        margin = results.pooled_margin
        lo = results.epsilon
        bins = np.logspace(np.log10(lo * 0.9), np.log10(0.5), N_POOLED_BINS)
        ax.hist(
            np.clip(margin, bins[0], bins[-1]),
            bins=bins,
            weights=np.full(margin.shape, 1.0 / margin.size),
            color=block_color("A"),
            edgecolor="none",
        )
        ax.axvline(lo, color="0.35", linestyle=":", lw=1.0)
        ax.axvline(OVERLAP_BAND[0], **ROLES["reference"])
        ax.annotate(
            r"clamp $\varepsilon$",
            xy=(lo, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=FONT_SIZE_PT - 2,
            color="0.35",
        )
        ax.set_xscale("log")
        ax.set_xlim(bins[0], 0.5)
        ax.set_xlabel(r"$\min(\pi,\,1-\pi)$")
        ax.set_ylabel("share of rows")
        ax.set_title("(a) pooled overlap margin")

    @staticmethod
    def _quantile_band_panel(ax, results: OverlapResults) -> None:
        """(b) Per-dataset quantile function of ``pi``, summarised across
        datasets: median curve with 25-75 and 10-90 percentile bands. Panel (a)
        pools rows and so cannot distinguish "every dataset is mildly bad" from
        "a few datasets are degenerate"; the width of these bands is exactly that
        distinction.
        """
        curves = results.quantile_curves
        colour = block_color("A")
        for (lo_pct, hi_pct), alpha in zip(BANDS, (0.18, 0.34)):
            lo, hi = np.percentile(curves, [lo_pct, hi_pct], axis=0)
            ax.fill_between(
                QUANTILE_GRID, lo, hi, color=colour, alpha=alpha, linewidth=0
            )
        ax.plot(QUANTILE_GRID, np.median(curves, axis=0), color=colour, lw=1.6)

        for bound in OVERLAP_BAND:
            ax.axhline(bound, **ROLES["reference"])

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("row quantile")
        ax.set_ylabel(r"propensity $\pi$")
        ax.set_title("(b) per-dataset quantile curves")

        handles = [
            plt.Line2D([], [], color=colour, lw=1.6, label="median dataset"),
            Patch(facecolor=colour, alpha=0.34, label="25–75%"),
            Patch(facecolor=colour, alpha=0.18, label="10–90%"),
            plt.Line2D([], [], label=r"$\pi=0.05,\,0.95$", **ROLES["reference"]),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=FONT_SIZE_PT - 2,
            framealpha=0.92,
        )
