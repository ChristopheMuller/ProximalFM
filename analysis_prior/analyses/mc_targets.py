"""Monte Carlo CATE targets: how much of the training target is signal?

Covers appendix B.2.6 (``fig:prior_cate_vs_ite``, ``tab:prior_mc_signal_share``).
Every unit is replicated ``K`` times -- same covariate ``x``, fresh exogenous
noise -- and the CATE target is the replicate average
``Q_hat_K(x) = MU1 - MU0``. That average is unbiased for ``Q(x)`` at every
``K``, so the cost of a small ``K`` is never bias: it is **variance**, which
this analysis quantifies.

The decomposition
-----------------
Write ``sigma^2(x) = Var(tau | x)`` for the within-unit replicate variance (the
part the average is integrating out) and ``Var_x(Q)`` for the between-unit
variation of the true CATE (the signal a CATE learner is meant to find). Since
the replicates are i.i.d. given ``x``, the target's dispersion across rows
splits exactly:

    Var_rows(Q_hat_K) = Var_x(Q) + E[sigma^2] / K

so with ``NSR = E[sigma^2] / Var_x(Q)`` the **signal share**

    S(K) = Var_x(Q) / Var_rows(Q_hat_K) = 1 / (1 + NSR / K)                (*)

is the fraction of the target's variance that is real -- equivalently the R^2
of the target against the true CATE, equivalently its reliability. ``S(1)`` is
what a ``K = 1`` prior delivers; ``S(K) -> 1`` as ``K`` grows.

Estimating it from one cache
----------------------------
The cache stores the ``K``-average *and* the single retained replicate, and the
retained replicate is one of the ``K`` that went into the average. The mean of
the **other** ``K - 1`` is therefore recoverable exactly,

    Q_loo = (K * Q_hat_K - ITE) / (K - 1),

and -- being a disjoint set of i.i.d. draws -- is statistically **independent**
of ``ITE`` given ``x``. That gives two independent measurements of the same
latent ``Q(x)``, i.e. textbook parallel-forms reliability, and hence two
subtraction-free estimators:

    Var_x(Q)     = Cov_rows(ITE, Q_loo)                    (shared signal only)
    E[sigma^2]   = mean_rows[(ITE - Q_loo)^2] * (K-1)/K    (paired contrasts)

Neither differences two nearly-equal variances, which is what makes the result
usable at a short ``seq_len``: precision comes from ``T`` paired contrasts.
``Var_rows(ITE) ~= Var_x(Q) + E[sigma^2]`` and
``Var_rows(Q_hat_K) ~= Var_x(Q) + E[sigma^2]/K`` are then *implied* rather than
used, and both are reported as ``*_residual`` in ``summary.json``. They are not
equally informative: expanding the two estimators shows the first reduces to
``Var(ITE) + sigma^2/(K-1)``, i.e. it is near-algebraic and holds almost
regardless of whether the replicate model is right. The **second** is the real
check -- nothing forces ``Var_rows(Q_hat_K)`` to land on the predicted
``Var_x(Q) + E[sigma^2]/K``. Its residual is floored by the sampling error of a
variance over ``T`` rows (``~sqrt(2/T)``, so ~3% at the shipped ``T=2048``),
which is the argument for raising ``seq_len`` if the check needs to be tighter.

What is measured and what is extrapolated
-----------------------------------------
Only ``NSR`` is measured, once per dataset. The curve ``S(K)`` of (*) is then
**analytic** -- it is the variance of a sample mean, exact under the i.i.d.
replicate structure the generator guarantees, but it is an extrapolation, not a
sweep. The cache pins measured anchors only at ``K = 1`` (the retained
replicate), ``K - 1`` (the leave-one-out mean) and ``K`` (the full average);
intermediate ``K`` would need the per-replicate matrix, which is not cached.
That is why the ``S(K)`` grid is reported as a *table* rather than drawn as a
band of curves: the form should not suggest a measurement that did not happen.

Note that ``gamma`` cancels from ``NSR``. The remix gives
``tau = gamma * tau_raw + (1-gamma) * tau_bar``, so both ``E[sigma^2]`` and
``Var_x(Q)`` carry a factor ``gamma^2`` that divides out -- the ``K`` question
is a property of the mechanisms, not of the heterogeneity dial. The one
exception is numerical: as ``gamma -> 0`` both variances vanish and their ratio
is estimated from near-zero quantities, so those datasets are excluded (see
:data:`MIN_GAMMA`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..design import (
    ROLES,
    apply_style,
    block_color,
    figsize_grid,
    trend_line,
    variant_styles,
)
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from .base import Analysis, AnalysisArtifacts, register

#: ``K`` values the signal-share table reports. Spans the production range
#: (training runs use 100-200) down to the K=1 regime the MC targets replaced.
K_GRID: Tuple[int, ...] = (1, 5, 10, 25, 50, 100, 200, 500)

#: ``K`` values the signal-share histogram overlays. Deliberately independent
#: of the cache's own ``mc_cate``: ``S(K)`` follows analytically from the
#: measured ``NSR``, so any ``K`` can be displayed off any ``K>1`` cache --
#: edit this tuple to re-target the panel at a different production setting
#: without regenerating anything. Kept a subset of :data:`K_GRID` so every
#: plotted value is also quotable from ``summary.json`` and the table.
SHARE_PANEL_K: Tuple[int, ...] = (1, 10, 100)

#: Datasets with a smaller heterogeneity dial are dropped: at ``gamma -> 0``
#: the effect is constant, so signal *and* noise both vanish and ``NSR`` is a
#: ratio of two numerical zeros. ``NSR`` is otherwise gamma-free, so this
#: excludes a numerical failure mode, not a slice of the claim.
MIN_GAMMA = 0.05

#: Signal share the illustrated dataset is picked closest to. A dataset at
#: ``S(1) ~ 0.5`` shows the attenuation unambiguously -- the fitted slope sits
#: visibly below the identity without the cloud degenerating into noise.
TARGET_SIGNAL_SHARE = 0.5


def signal_share(nsr: np.ndarray | float, k: int) -> np.ndarray | float:
    """Fraction of a ``K``-replicate target's variance that is real signal."""
    return 1.0 / (1.0 + np.asarray(nsr, dtype=np.float64) / k)


def leave_one_out_cate(ite: np.ndarray, cate: np.ndarray, k: int) -> np.ndarray:
    """Mean of the ``K-1`` replicates the observed one is *not* among.

    Independent of ``ite`` given ``x`` (disjoint sets of i.i.d. draws), which is
    what makes it usable both as the near-noiseless reference axis of the
    scatter and as the second measurement in the reliability estimators. Its
    own noise is ``sigma^2 / (K-1)``, i.e. ~1% of the single replicate's at
    ``K = 100``.
    """
    if k < 2:
        raise ValueError(f"leave-one-out needs K >= 2; got K={k}")
    return (k * cate - ite) / (k - 1)


def target_frame(records: Sequence[PriorDatasetRecord], k: int) -> pd.DataFrame:
    """One row per dataset: the signal/noise split of its Monte Carlo target.

    Pure function of the records and ``K``, so the estimators can be tested
    against replicate structure built by hand.
    """
    if k < 2:
        raise ValueError(
            f"the Monte Carlo target analysis needs a cache with K >= 2; this "
            f"spec has mc_cate={k}. Use --spec mc_targets (K=100); a K=1 "
            "cache carries no replicate structure to measure."
        )

    rows: List[Dict[str, Any]] = []
    for record in records:
        ite = np.asarray(record.ite, dtype=np.float64)
        cate = np.asarray(record.cate, dtype=np.float64)
        q_loo = leave_one_out_cate(ite, cate, k)

        # Two independent measurements of the same Q(x): their covariance is
        # the signal, their squared difference the replicate noise. Neither
        # subtracts near-equal quantities.
        var_q = float(np.cov(ite, q_loo, ddof=1)[0, 1])
        sigma2 = float(np.mean((ite - q_loo) ** 2) * (k - 1) / k)

        var_ite = float(np.var(ite, ddof=1))
        var_cate = float(np.var(cate, ddof=1))
        usable = var_q > 0

        rows.append(
            {
                "index": record.index,
                "seed": record.seed,
                "gamma": (
                    float(record.gamma_hetero)
                    if record.gamma_hetero is not None
                    else np.nan
                ),
                "var_q": var_q,
                "sigma2": sigma2,
                "var_ite": var_ite,
                "var_cate": var_cate,
                "nsr": sigma2 / var_q if usable else np.nan,
                # Direct regression slope of Q_loo on ITE. Equals S(1) up to
                # sampling error, and is the line drawn on the scatter. Because
                # the noisy target is the *regressor*, its noise dilutes the
                # slope: slope 1 would mean the K=1 target carries no replicate
                # noise at all, i.e. is already the CATE.
                "slope": var_q / var_ite if var_ite > 0 else np.nan,
                # Implied identities, used nowhere above. The ITE one is
                # near-algebraic given the estimators (see module docstring);
                # the CATE one is the informative check.
                "var_ite_residual": (
                    (var_q + sigma2) / var_ite - 1.0 if var_ite > 0 else np.nan
                ),
                "var_cate_residual": (
                    (var_q + sigma2 / k) / var_cate - 1.0 if var_cate > 0 else np.nan
                ),
            }
        )

    frame = pd.DataFrame(rows)
    # gamma may be absent (older caches); keep such rows rather than dropping
    # the whole analysis, hence the negated comparison.
    frame["usable"] = (frame["var_q"] > 0) & ~(frame["gamma"] < MIN_GAMMA)
    for k_value in K_GRID:
        frame[f"S_{k_value}"] = signal_share(frame["nsr"], k_value)
    # K needed for a 90%-signal target: S(K) = 0.9  <=>  K = 9 * NSR.
    frame["k90"] = 9.0 * frame["nsr"]
    return frame


@dataclass
class McTargetResults:
    """Per-dataset signal/noise split, plus the illustrated dataset's rows."""

    frame: pd.DataFrame
    k: int
    #: Row of ``frame`` for the illustrated dataset, plus its ``ite``/``q_loo``.
    contrast: Dict[str, Any]

    @property
    def usable(self) -> pd.DataFrame:
        return self.frame[self.frame["usable"]]


def select_dataset(frame: pd.DataFrame) -> Optional[int]:
    """Index of the dataset to illustrate: signal share nearest the target."""
    usable = frame[frame["usable"]].dropna(subset=["S_1"])
    if usable.empty:
        return None
    gap = (usable["S_1"] - TARGET_SIGNAL_SHARE).abs()
    return int(usable.loc[gap.idxmin(), "index"])


@register
class McTargetsAnalysis(Analysis):
    name = "mc_targets"
    description = (
        "Monte Carlo CATE targets: what share of the K-replicate target's "
        "variance is real CATE signal rather than replicate noise?"
    )

    def compute(self, cache: PriorCache) -> McTargetResults:
        k = int(cache.spec.mc_cate)
        frame = target_frame(cache.records, k)

        contrast: Dict[str, Any] = {}
        index = select_dataset(frame)
        if index is not None:
            record = cache.records[index]
            ite = np.asarray(record.ite, dtype=np.float64)
            contrast = frame[frame["index"] == index].iloc[0].to_dict()
            contrast["ite"] = ite
            contrast["q_loo"] = leave_one_out_cate(
                ite, np.asarray(record.cate, dtype=np.float64), k
            )
        return McTargetResults(frame=frame, k=k, contrast=contrast)

    def render(self, results: McTargetResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_figure(results, outdir)
        tables = self._render_table(results, outdir)

        frame, usable = results.frame, results.usable
        summary: Dict[str, Any] = {
            "K": results.k,
            "n_datasets": int(len(frame)),
            "n_usable": int(len(usable)),
            # Datasets whose signal variance did not come out positive: the
            # estimator's failure mode when the true CATE barely varies.
            "n_nonpositive_signal": int((frame["var_q"] <= 0).sum()),
            "n_below_min_gamma": int((frame["gamma"] < MIN_GAMMA).sum()),
            "min_gamma": MIN_GAMMA,
        }
        for label, column in (("nsr", "nsr"), ("k90", "k90")):
            summary[f"{label}_quantiles"] = _quantiles(usable[column])
        for k_value in K_GRID:
            summary[f"signal_share_K{k_value}"] = _quantiles(usable[f"S_{k_value}"])
        # Implied identities. The CATE one is the informative check that the
        # replicate model the estimators assume is the one the generator
        # delivered; its floor is the sampling error of a variance over T rows.
        for column in ("var_ite_residual", "var_cate_residual"):
            summary[f"median_abs_{column}"] = _median_abs(usable[column])
        if results.contrast:
            for key in ("index", "seed"):
                summary[f"illustrated_{key}"] = int(results.contrast[key])
            for key in ("nsr", "slope", "S_1", "gamma"):
                summary[f"illustrated_{key}"] = float(results.contrast[key])

        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_figure(self, results: McTargetResults, outdir: Path) -> List[Path]:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize_grid(
                ncols=2, nrows=1, panel_aspect=0.85, extra_height_in=0.3
            ),
        )
        self._scatter_panel(axes[0], results)
        self._share_panel(axes[1], results)
        return save_figure(fig, outdir, "cate_vs_ite")

    @staticmethod
    def _scatter_panel(ax, results: McTargetResults) -> None:
        """The CATE a K=1 target is meant to estimate, against that target.

        Axes: the K=1 target (the retained replicate) on the **horizontal**, the
        leave-one-out replicate mean on the **vertical**. The latter is
        independent of the plotted single replicate and carries ~1/(K-1) of its
        noise, so it stands in for the truth and the vertical scatter about the
        identity *is* the Monte Carlo error a K=1 target would hand the model.

        The informative feature is the **slope**, not the diagonal cloud. Noise
        in a *regressor* dilutes a slope (noise in the response does not), and
        the noisy target is on the horizontal, so the slope comes out at exactly
        that target's reliability, ``Var(Q)/Var(ITE) = S(1)``. A noise-free
        target would instead give slope 1, so the gap between the fitted line
        and the identity is the replicate noise that averaging removes.
        """
        if not results.contrast:
            ax.set_axis_off()
            return
        ite = results.contrast["ite"]
        q_loo = results.contrast["q_loo"]
        slope = float(results.contrast["slope"])

        ax.scatter(ite, q_loo, s=6, alpha=0.35, linewidths=0, color=block_color("Y"))
        lo = float(min(ite.min(), q_loo.min()))
        hi = float(max(ite.max(), q_loo.max()))
        ax.plot([lo, hi], [lo, hi], label="identity", **ROLES["reference"])
        # Fitted attenuation through the joint mean.
        x_mid, y_mid = float(ite.mean()), float(q_loo.mean())
        grid = np.array([lo, hi])
        trend_line(
            ax,
            grid,
            y_mid + slope * (grid - x_mid),
            label=rf"slope $={slope:.2f}$",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title("(a) one dataset")
        ax.set_xlabel(r"$K{=}1$ target $\;Y_1 - Y_0$")
        ax.set_ylabel(r"held-out CATE $\;\hat Q_{\mathrm{loo}}$")
        ax.legend(loc="upper left")

    @staticmethod
    def _share_panel(ax, results: McTargetResults) -> None:
        """Signal share across the prior at each ``K`` of :data:`SHARE_PANEL_K`.

        Bounded in [0, 1] and directly readable as "this fraction of the
        training target is real"; the gap between the curves is what the Monte
        Carlo averaging buys, and the gap between the two largest ``K`` is the
        marginal return on doubling generation compute.
        """
        usable = results.usable
        nsr = usable["nsr"].to_numpy()
        bins = np.linspace(0.0, 1.0, 41)
        # Ordered ramp, not the block palette: K is a sweep with a natural
        # order, which a sequential colormap shows and a categorical one hides.
        styles = variant_styles([str(k) for k in SHARE_PANEL_K], ordered=True)
        for k_value, key in zip(SHARE_PANEL_K, styles):
            values = pd.Series(signal_share(nsr, k_value)).dropna()
            color = styles[key]["color"]
            # Outline plus light fill: three filled histograms alone would mud
            # together where the larger K overlap near 1.
            ax.hist(values, bins=bins, color=color, alpha=0.25, edgecolor="none")
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                lw=1.3,
                color=color,
                label=rf"$K={k_value}$",
            )
        ax.set_title("(b) signal share")
        ax.set_xlabel(r"$\mathrm{Var}(Q)\,/\,\mathrm{Var}(\hat Q_K)$")
        ax.set_ylabel("datasets")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper center")

    def _render_table(self, results: McTargetResults, outdir: Path) -> List[Path]:
        """Signal share by ``K``: median and the 10-90% spread of the prior.

        A table rather than a curve, deliberately: only ``NSR`` is measured,
        and ``S(K)`` is the exact analytic consequence of it (see the module
        docstring), so the presentation should not imply a measured sweep.
        """
        usable = results.usable
        rows = []
        for k_value in K_GRID:
            quantiles = _quantiles(usable[f"S_{k_value}"])
            rows.append(
                {
                    "$K$": k_value,
                    "median": quantiles["p50"] if quantiles else np.nan,
                    "10th pct": quantiles["p10"] if quantiles else np.nan,
                    "90th pct": quantiles["p90"] if quantiles else np.nan,
                }
            )
        return save_table(
            pd.DataFrame(rows),
            outdir,
            "mc_signal_share",
            caption=(
                "Share of the Monte Carlo CATE target's variance that is real "
                "CATE signal, $S(K)=\\mathrm{Var}(Q(X))/\\mathrm{Var}(\\hat "
                "Q_K(X))=1/(1+\\mathrm{NSR}/K)$, across the prior. The "
                "noise-to-signal ratio $\\mathrm{NSR}$ is measured once per "
                "dataset from the replicate structure; $S(K)$ then follows "
                "analytically, being the variance of a sample mean."
            ),
            label="tab:prior_mc_signal_share",
            escape=False,
        )


def _quantiles(values) -> Optional[Dict[str, float]]:
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    return {
        "p10": float(clean.quantile(0.10)),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
    }


def _median_abs(values) -> Optional[float]:
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().abs()
    return float(clean.median()) if len(clean) else None
