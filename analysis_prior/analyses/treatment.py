"""Treatment-assignment walkthrough: one SCM's logits from ``f_A`` to ``A``.

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

from ..design import apply_style, block_color, figsize_grid
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
    # The full-population mean/std sample_A standardises by. Stored explicitly so
    # this is exact even when K > 1 (observed slice != whole population); fall
    # back to the slice's own moments for older records that only kept the slice.
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
