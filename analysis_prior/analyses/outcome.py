"""Heterogeneity dial: does ``gamma`` control the spread of the delivered effect?

Covers appendix B.2.6 (``fig:prior_heterogeneity_spread``). The outcome remix of
``eq:remix`` recombines the raw potential outcomes with a per-dataset dial
``gamma ~ U(0,1)`` and a per-row weight ``alpha ~ U(0,1)``. Substituting the two
lines of that equation collapses the per-row weights, leaving the individual
effect exactly

    tau = Y1 - Y0 = gamma * tau_raw + (1 - gamma) * tau_bar,                (*)

i.e. **alpha-free**, and linear in ``gamma`` between the raw effect
(``gamma = 1``, fully heterogeneous) and the dataset's raw average effect
``tau_bar`` (``gamma = 0``, a single constant effect). Hence
``sd(tau) = gamma * sd(tau_raw)``: the dial is a pure spread control.

What this analysis can and cannot show
--------------------------------------
``gamma`` is drawn *inside* the generator (``torch.rand(1)`` in
``_dataset.py``), not from the sampled-HP config, so it cannot be pinned from a
:class:`~analysis_prior.config.PriorSpec` -- there is no way to hold an SCM
fixed and turn the dial. But ``gamma`` is *recorded* per dataset
(``record.gamma_hetero``) and drawn uniformly, so a single prior cache is
already a sweep: this analysis reads the delivered ``gamma`` and asks whether
effect dispersion tracks it **across** the prior. That answers "does the dial
work in the data we ship" rather than the counterfactual "what would this SCM
look like at another gamma"; the caption says so.

One consequence: the ATE-preservation half of the claim (the remix is anchored
at ``tau_bar``, so the dial reshapes spread only) is *not* visible this way --
the ATE varies by orders of magnitude across SCMs, swamping any gamma effect.
Only the spread claim is testable population-style.

Scale-free statistics
---------------------
``Y`` is **not** standardised by ``Corrupt``, so each SCM's outcome carries its
own arbitrary scale and a raw ``sd(tau)`` compared across datasets measures
outcome scale, not heterogeneity. Every statistic here is therefore a ratio:

- **dispersion** ``sd(tau) / sd([Y0; Y1])`` -- effect spread in units of the
  outcome scale. By (*) this is ``gamma * c`` with ``c = sd(tau_raw) / sd(Y)``
  a gamma-independent property of the SCM, so it should rise proportionally
  with ``gamma`` through the origin. The denominator is only *approximately*
  gamma-neutral (the remix reshuffles ``Y0``/``Y1`` too), which is why
  ``summary.json`` reports ``spearman_gamma_vs_scale`` as a check: near zero
  means the proportionality overlay is sound.
- **corr(Y0, Y1)** -- at ``gamma = 0`` the effect is the constant ``tau_bar``,
  so ``Y1 = Y0 + tau_bar`` and the correlation is *exactly* 1. That endpoint is
  an analytic anchor the measured trend must approach, and it needs no scale
  reference at all.

Effect definition. Statistics use the ITE ``Y1 - Y0`` of the realised
replicate, which satisfies (*) exactly at any ``K``. At ``K = 1`` (the
``general_ite`` workhorse) the Monte Carlo CATE target ``MU1 - MU0`` *is* that
same quantity, so on such a cache the panels read equally as a CATE view; at
``K > 1`` the CATE target would additionally average out the within-``x``
replicate noise, which is the separate subject of the MC-error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

from ..design import (
    ROLES,
    apply_style,
    block_color,
    figsize_grid,
    theory_line,
    variant_styles,
)
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure
from .base import Analysis, AnalysisArtifacts, register
from .mechanism_inputs import _add_trend

#: Edges of the gamma bins the dispersion violins are grouped into.
GAMMA_BINS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: Bins holding fewer datasets than this are dropped rather than drawn from a
#: handful of points (same convention as the categorical-completeness figures).
MIN_BIN_COUNT = 5

#: The two gammas the contrast panel illustrates. Deliberately *not* the
#: extremes of the prior: by (*) the density heights scale like ``1/gamma``, so
#: a 0.03-vs-0.97 pair would put a spike ~30x taller than the broad density it
#: is meant to be compared against and flatten the latter into the axis. A 9:1
#: ratio still reads unambiguously as "gathered vs spread" while keeping both
#: densities legible on one linear axis.
CONTRAST_GAMMAS: Tuple[float, float] = (0.1, 0.9)

#: How many datasets nearest each target gamma are considered when picking the
#: contrast pair. The pair is then chosen for matched SCM dispersion (below),
#: so this is the width of the search, not of the accepted gamma window.
CONTRAST_CANDIDATES = 25

#: Below this gamma the per-SCM dispersion constant ``c = dispersion / gamma``
#: is a ratio of two near-zero numbers and numerically meaningless; such
#: datasets are excluded from the proportionality overlay and the pair search
#: (they still appear in every scatter and violin).
GAMMA_FLOOR = 1e-3


def heterogeneity_frame(records: Sequence[PriorDatasetRecord]) -> pd.DataFrame:
    """One row per dataset: delivered ``gamma`` and the scale-free statistics.

    Pure function of the records, so the arithmetic can be tested against
    hand-built potential outcomes. Raises if a record predates the ``gamma``
    diagnostic -- such a cache must be regenerated with ``--force`` (identical
    data, the diagnostics are simply recorded alongside).
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        if record.gamma_hetero is None:
            raise ValueError(
                f"dataset {record.index} has no stored gamma_hetero; the cache "
                "predates it. Regenerate with --force (same data, diagnostics "
                "added)."
            )
        gamma = float(record.gamma_hetero)
        ite = np.asarray(record.ite, dtype=np.float64)
        y0 = np.asarray(record.Y0, dtype=np.float64)
        y1 = np.asarray(record.Y1, dtype=np.float64)

        # Pooled over both arms: a single outcome-scale reference, steadier
        # than either arm alone under the remix's per-row mixing.
        scale = float(np.std(np.concatenate([y0, y1])))
        sd_ite = float(np.std(ite))
        dispersion = sd_ite / scale if scale > 0 else np.nan

        # Exactly 1 at gamma = 0 (Y1 = Y0 + tau_bar); undefined if an arm is
        # constant, which a degenerate SCM can produce.
        corr = (
            float(np.corrcoef(y0, y1)[0, 1])
            if np.std(y0) > 0 and np.std(y1) > 0
            else np.nan
        )

        rows.append(
            {
                "index": record.index,
                "seed": record.seed,
                "gamma": gamma,
                "ate": float(np.mean(ite)),
                "sd_ite": sd_ite,
                "scale": scale,
                "dispersion": dispersion,
                "corr_y0y1": corr,
                # The SCM's own dispersion constant c, with gamma divided out:
                # gamma-independent by (*), hence what the contrast pair is
                # matched on and what the proportionality overlay averages.
                "raw_dispersion": (
                    dispersion / gamma if gamma > GAMMA_FLOOR else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def select_contrast_pair(
    frame: pd.DataFrame,
    targets: Tuple[float, float] = CONTRAST_GAMMAS,
    n_candidates: int = CONTRAST_CANDIDATES,
) -> Tuple[int, int]:
    """Positions in ``frame`` of the low- and high-gamma datasets to contrast.

    Two SCMs drawn at different gammas differ in *both* gamma and their own raw
    dispersion ``c``, so the naive "closest to each target" pick would show a
    contrast that is partly SCM luck. Among the ``n_candidates`` datasets
    nearest each target we therefore choose the pair whose ``c`` matches most
    closely (smallest ``|log(c_low / c_high)|``), leaving gamma as the only
    material difference between the two panels. Deterministic.
    """
    usable = frame[frame["raw_dispersion"] > 0].dropna(subset=["raw_dispersion"])
    if len(usable) < 2:
        raise ValueError(
            "need at least two datasets with a finite dispersion constant to "
            f"build the contrast pair; got {len(usable)}"
        )

    pools = []
    for target in targets:
        nearest = usable.assign(gap=(usable["gamma"] - target).abs()).nsmallest(
            n_candidates, "gap", keep="all"
        )
        pools.append(nearest)

    best: Tuple[float, int, int] | None = None
    for _, low in pools[0].iterrows():
        for _, high in pools[1].iterrows():
            if low["index"] == high["index"]:
                continue
            mismatch = abs(
                np.log(low["raw_dispersion"]) - np.log(high["raw_dispersion"])
            )
            key = (mismatch, int(low["index"]), int(high["index"]))
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError("could not form a contrast pair from distinct datasets")
    return best[1], best[2]


@dataclass
class HeterogeneityResults:
    """Per-dataset statistics plus the two illustrated effect distributions.

    ``contrast`` maps ``"low"``/``"high"`` to that dataset's row of ``frame``
    (as a dict) together with ``standardised_ite`` -- the effect centred on the
    dataset's own ``tau_bar`` and divided by its outcome scale, which is what
    makes two SCMs comparable on one axis.
    """

    frame: pd.DataFrame
    contrast: Dict[str, Dict[str, Any]]

    def binned(self, edges: Tuple[float, ...] = GAMMA_BINS) -> List[Dict[str, Any]]:
        """Dispersion grouped into gamma bins, sparse bins dropped."""
        out: List[Dict[str, Any]] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Half-open except at the top, so gamma = 1 lands in the last bin.
            in_bin = (self.frame["gamma"] >= lo) & (
                (self.frame["gamma"] < hi) if hi < edges[-1] else True
            )
            values = self.frame.loc[in_bin, "dispersion"].dropna().to_numpy()
            if len(values) < MIN_BIN_COUNT:
                continue
            out.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "centre": 0.5 * (lo + hi),
                    "values": values,
                    "median": float(np.median(values)),
                    "n": int(len(values)),
                }
            )
        return out


@register
class HeterogeneitySpreadAnalysis(Analysis):
    name = "heterogeneity"
    description = (
        "Heterogeneity dial: does the delivered effect spread track the "
        "per-dataset gamma of the outcome remix?"
    )

    def compute(self, cache: PriorCache) -> HeterogeneityResults:
        frame = heterogeneity_frame(cache.records)
        low, high = select_contrast_pair(frame)

        contrast: Dict[str, Dict[str, Any]] = {}
        for label, index in (("low", low), ("high", high)):
            row = frame[frame["index"] == index].iloc[0].to_dict()
            ite = np.asarray(cache.records[index].ite, dtype=np.float64)
            # Centre on the dataset's own tau_bar and divide by its outcome
            # scale: the two SCMs then share one axis, on which the only
            # remaining difference is how wide the effect distribution is.
            row["standardised_ite"] = (ite - ite.mean()) / row["scale"]
            contrast[label] = row
        return HeterogeneityResults(frame=frame, contrast=contrast)

    def render(self, results: HeterogeneityResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_spread(results, outdir)

        frame = results.frame
        summary: Dict[str, Any] = {
            "n_datasets": int(len(frame)),
            # The headline: dispersion should track gamma almost perfectly.
            "spearman_gamma_vs_dispersion": _spearman(
                frame["gamma"], frame["dispersion"]
            ),
            # Should be strongly negative, approaching corr = 1 as gamma -> 0.
            "spearman_gamma_vs_corr": _spearman(frame["gamma"], frame["corr_y0y1"]),
            # QA on the denominator: near zero means the outcome-scale
            # reference is gamma-neutral, so the proportionality overlay in
            # panel (c) is comparing like with like.
            "spearman_gamma_vs_scale": _spearman(frame["gamma"], frame["scale"]),
            "median_raw_dispersion": _median(frame["raw_dispersion"]),
        }

        bins = results.binned()
        summary["dispersion_median_by_gamma_bin"] = {
            f"{b['lo']:.1f}-{b['hi']:.1f}": b["median"] for b in bins
        }
        summary["n_by_gamma_bin"] = {
            f"{b['lo']:.1f}-{b['hi']:.1f}": b["n"] for b in bins
        }
        if bins:
            lowest = frame["gamma"] < bins[0]["hi"]
            highest = frame["gamma"] >= bins[-1]["lo"]
            summary["median_corr_lowest_gamma_bin"] = _median(
                frame.loc[lowest, "corr_y0y1"]
            )
            summary["median_corr_highest_gamma_bin"] = _median(
                frame.loc[highest, "corr_y0y1"]
            )

        for label, row in results.contrast.items():
            for key in ("index", "seed"):
                summary[f"contrast_{label}_{key}"] = int(row[key])
            for key in ("gamma", "dispersion", "raw_dispersion", "ate"):
                summary[f"contrast_{label}_{key}"] = float(row[key])

        return AnalysisArtifacts(summary=summary, figures=figures, tables=[])

    def _render_spread(self, results: HeterogeneityResults, outdir: Path) -> List[Path]:
        """Three panels: the contrast, the analytic anchor, the population trend.

        (a) is the picture the section asks for -- one gathered and one spread
        effect density. (b) and (c) are what stop it being an anecdote: (b)
        checks the one endpoint the algebra pins exactly, (c) checks the whole
        prior against the proportionality (*) implies.
        """
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize_grid(
                ncols=3, nrows=1, panel_aspect=1.0, extra_height_in=0.3
            ),
        )
        self._contrast_panel(axes[0], results)
        self._correlation_panel(axes[1], results)
        self._dispersion_panel(axes[2], results)
        return save_figure(fig, outdir, "heterogeneity_spread")

    @staticmethod
    def _contrast_panel(ax, results: HeterogeneityResults) -> None:
        """Two effect densities on a shared, outcome-scaled axis.

        Both are centred on their own dataset's raw average effect, drawn at
        ``x = 0``: the remix anchors the collapse at ``tau_bar`` rather than at
        zero, so the low-gamma spike sitting on that line is the visual of the
        anchoring, not an artefact of the centring.
        """
        styles = variant_styles(["low", "high"], ordered=True)
        high = results.contrast["high"]["standardised_ite"]

        # A kernel density rather than a histogram: the two distributions differ
        # in width by ~10x, and shared bins wide enough for the broad one give
        # the narrow one three cells, turning a spike into a blocky artefact.
        # Scott's bandwidth adapts to each, so both are resolved on their own
        # terms -- at the cost of the spike standing ~1/gamma times taller,
        # which is the honest rendering of the ratio.
        span = float(np.quantile(np.abs(high), 0.995))
        grid = np.linspace(-span, span, 512)
        for label in ("low", "high"):
            values = results.contrast[label]["standardised_ite"]
            gamma = results.contrast[label]["gamma"]
            density = gaussian_kde(values)(grid)
            color = styles[label]["color"]
            ax.fill_between(grid, density, color=color, alpha=0.3, linewidth=0)
            ax.plot(grid, density, color=color, lw=1.4, label=rf"$\gamma={gamma:.2f}$")
        ax.axvline(0.0, zorder=0, **ROLES["reference"])
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(-span, span)
        ax.set_title("(a) effect density")
        ax.set_xlabel(r"$(\tau - \bar\tau)\,/\,\mathrm{sd}(Y)$")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")

    @staticmethod
    def _correlation_panel(ax, results: HeterogeneityResults) -> None:
        """corr(Y0, Y1) against gamma, with the exact gamma -> 0 anchor."""
        frame = results.frame
        ax.scatter(
            frame["gamma"],
            frame["corr_y0y1"],
            s=6,
            alpha=0.35,
            linewidths=0,
            color=block_color("Y"),
        )
        _add_trend(ax, frame["gamma"], frame["corr_y0y1"])
        # The algebra pins this line exactly at gamma = 0: Y1 = Y0 + tau_bar.
        ax.axhline(1.0, **ROLES["reference"])
        ax.set_title("(b) outcome coupling")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\mathrm{corr}(Y_0, Y_1)$")

    @staticmethod
    def _dispersion_panel(ax, results: HeterogeneityResults) -> None:
        """Dispersion densities per gamma bin, against the proportionality (*).

        This is the population form of the dial: each violin is the spread of
        ``sd(tau)/sd(Y)`` over the datasets in one gamma bin, so the bins
        should march up a straight line through the origin. Within-bin spread
        is SCM variation (the constant ``c``), not gamma.
        """
        bins = results.binned()
        if not bins:
            return
        styles = variant_styles([str(i) for i in range(len(bins))], ordered=True)
        centres = [b["centre"] for b in bins]

        parts = ax.violinplot(
            [b["values"] for b in bins],
            positions=centres,
            widths=0.16,
            showextrema=False,
            showmedians=True,
        )
        for body, key in zip(parts["bodies"], styles):
            body.set_facecolor(styles[key]["color"])
            body.set_edgecolor("none")
            body.set_alpha(0.75)
        parts["cmedians"].set_color("0.15")
        parts["cmedians"].set_linewidth(1.2)

        # dispersion = gamma * c, so the medians should sit on a line through
        # the origin with slope the prior's median c.
        slope = _median(results.frame["raw_dispersion"])
        if slope is not None:
            grid = np.linspace(0.0, 1.0, 6)
            theory_line(
                ax, grid, slope * grid, label=r"$\gamma\cdot\mathrm{median}(c)$"
            )
            ax.legend(loc="upper left")

        pooled = results.frame["dispersion"].dropna()
        if len(pooled):
            ax.set_ylim(0.0, float(np.quantile(pooled, 0.99)))
        ax.set_xlim(0.0, 1.0)
        ax.set_title("(c) delivered spread")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\mathrm{sd}(\tau)\,/\,\mathrm{sd}(Y)$")


def _spearman(x, y) -> float | None:
    """Spearman rho over the rows where both are finite, or None if degenerate."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None
    return float(spearmanr(data["x"], data["y"]).statistic)


def _median(values) -> float | None:
    clean = pd.Series(values).dropna()
    return float(clean.median()) if len(clean) else None
