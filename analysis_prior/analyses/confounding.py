"""Confounding strength (B.3.3) -- the audit of ``as:latent_ignorability``.

This is not a named assumption's QA so much as the empirical premise the whole
paper rests on. Two claims, one ladder:

1. **The synthetic confounding is real.** An estimator that adjusts only for the
   measured covariates is *biased* for the true ATE. If it were not, the proxies
   would be decorative and there would be nothing for a proximal method to buy.
2. **Conditioning on the true confounder fixes it.** Adjusting for ``(X, U)``
   recovers the truth -- which is exactly what ``as:latent_ignorability``
   claims. The assumption is asserted by construction (``U`` is the only
   backdoor path); this shows it *delivers* on the realised data.

Five rungs, all scored against the same ground truth ``mean(MU1 - MU0)`` -- a
Monte Carlo target read straight off the cache, not something estimated here:

- **naive** -- ``mean(Y | A=1) - mean(Y | A=0)``, no adjustment at all.
- **backdoor-X** -- T-learner on ``X``: the correct adjustment set *if* there
  were no latent confounder.
- **backdoor-XWZ** -- T-learner on ``(X, W, Z)``: the DAG-blind move of throwing
  every measured column into the adjustment set. Included because a reviewer
  will ask whether the proxies alone already fix things by acting as noisy
  substitutes for ``U``; they partially can, which is why proximal methods use
  them *structurally* rather than as ordinary covariates.
- **oracle-XU** -- T-learner on ``(X, U)``, possible only because the data is
  synthetic and ``U`` is retained. This is the *identification* oracle: the
  minimal set ``as:latent_ignorability`` actually names.
- **oracle-XUW** -- T-learner on ``(X, U, W)``: the *efficiency* oracle. ``W`` is
  not needed for identification -- once ``U`` is conditioned on, the backdoor
  path is already blocked -- but it **is a genuine parent of** ``f_Y``
  (``_dataset.py`` feeds ``W_direct``, a per-dataset column subsample of ``W``,
  into the outcome mechanism). Omitting it therefore leaves the oracle's outcome
  model *misspecified relative to the true mechanism*, inflating its residual
  variance. Since the oracle rung is read as the ladder's floor, that inflation
  matters: the ``XU``-to-``XUW`` gap separates "irreducible" from "we left a real
  parent out of the regression".

  ``Z`` is deliberately **not** added to either oracle. It is a parent of ``f_A``
  only and is excluded from ``f_Y`` by construction (``as:proxy_exclusion``), so
  in an outcome-regression estimator it can contribute nothing but noise -- and
  an instrument-like variable in an adjustment set is the textbook
  bias-amplification risk. Its place is rung 3, where throwing in everything
  measured is the whole point.

  Adding ``W`` cannot open a path: it is pre-treatment (``f_W`` never sees
  ``A``), and its only parents are ``U`` and ``X``, both already conditioned on,
  so the collider it sits on is already blocked.

Errors are ``|ATE_rung - ATE_true| / sd(Y_obs)``. The normalisation is what
makes datasets comparable: outcome scales are arbitrary per SCM (the same
reasoning ``fig:prior_heterogeneity_spread`` already uses), so a raw error is
uninterpretable across the prior.

**A normalisation caveat, worth knowing before reading the share grid.** ``sd_y``
is computed from the very ``Y_obs`` whose ``U``-composition ``share_Y``
describes, so the two are mechanically entangled (measured: Spearman
``sd_y`` vs ``share_Y`` ~0.06, vs ``sd_y`` vs ``share_A`` ~0.04 -- weak, but not
zero, and asymmetric in the direction that matters). Concretely: on the
``rf``/``n=1024`` headline run, the *unnormalised* naive bias correlates with
``share_Y`` (Spearman ~0.07) at least as strongly as with ``share_A`` (~0.06) --
if anything slightly more. Dividing by ``sd_y`` then partially cancels the
``share_Y`` signal (an SCM where ``U`` dominates ``f_Y`` tends to have a larger
``sd_y`` too, shrinking the ratio) while leaving the ``share_A`` signal mostly
untouched (``share_A`` describes a different mechanism, only weakly entangled
with ``sd_y``). So the share grid's near-flat columns are not evidence that the
outcome-side confounding share doesn't matter -- the raw effect is there, just
muted by the very normalisation that makes the figure comparable across SCMs.
Both ``raw_error`` (this asymmetry) and ``abs_error`` (what is plotted, for
cross-SCM comparability) are kept in the returned frame.

**Conditioning on delivered confounding strength.** The confounding dial
``confounding_share_of_U_in_{A,Y}`` is a *sampled* hyperparameter
(``Beta(1.2, 2.8)``), so it cannot be swept while holding the SCM fixed: pinning
it would change ``PriorSpec.generative_digest()`` and hence every dataset seed,
and the arms would be different SCMs rather than the same SCM at different
confounding. But it does not need to be swept -- 1024 draws from the prior *are*
the sweep, and the realised share is recorded per dataset. The second figure
therefore bins datasets on the **delivered** ``U`` share of ``f_A`` and ``f_Y``
(first-layer injected variance, the estimator ``mechanism_inputs`` settled on
for B.2.3 -- what the mechanism actually got, not what was asked for) and reports
the median backdoor-``X`` error per cell.

The grid is two-dimensional on purpose. Confounding bias requires ``U`` to enter
**both** mechanisms: a large ``s_Y`` with ``s_A = 0`` is an outcome that depends
on ``U`` under a treatment that does not, which is unconfounded. A marginal view
in either share alone would average that interaction away.

**What the delivered data actually shows does not match that interaction story.**
The elevation in backdoor-``X`` error tracks ``share_A`` alone: the top row
(delivered ``U`` share of ``f_A`` in its top tercile) sits at 0.045-0.046
regardless of column, against 0.028-0.041 in the two rows below, and is flat
across ``share_Y`` (Spearman of delivered share vs. backdoor error: 0.057 for
``share_A``, 0.042 for ``share_Y``; measured on the ``general_ite`` n=1024
headline run). There is no top-right-corner effect requiring both shares
jointly -- the grid does not falsify the two-dimensional rationale above (a
large ``s_Y``/``s_A=0`` cell is still unconfounded by construction), but the
*empirical* signal in this prior's delivered draws is one-dimensional, driven
by how strongly ``U`` enters treatment assignment rather than by the
interaction. Part of ``share_Y``'s apparent silence is the normalisation
artifact described above (its raw-bias correlation, 0.072, is comparable to
``share_A``'s 0.062, before ``sd_y``'s stronger entanglement with ``share_Y``
partially cancels it) -- but even on the raw scale there is no visible
corner effect, just two axes with comparable marginal correlations.

The oracle rung is binned on the same grid as a control. It must stay flat: an
oracle error that *grows* with confounding strength would be finite-sample
regression difficulty (``U`` matters more, so mis-fitting it costs more), not a
failure of ``as:latent_ignorability``.

Datasets with ``d_U = 0`` are **kept**, unlike in the exclusion audit. There they
made the positive control vacuous; here they are the unconfounded end of the
very axis being measured, and dropping them would hide the fact that the prior
spans "no confounding at all" as well as "severe".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from ..design import COUNT_CMAP, FONT_SIZE_PT, ROLES, apply_style, figsize_grid
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from ..regression import DEFAULT_REGRESSION_CACHE_DIR, cached_t_learner
from .base import Analysis, AnalysisArtifacts, register
from .mechanism_inputs import mechanism_frame

#: The adjustment ladder, in reading order: the predictor blocks each rung
#: regresses on. ``naive`` has no regression at all and is handled separately.
#: Block order is fixed so the predictor-set labels -- and hence the regression
#: cache keys -- are stable.
RUNGS: Dict[str, Tuple[str, ...]] = {
    "naive": (),
    "backdoor_X": ("X",),
    "backdoor_XWZ": ("X", "W", "Z"),
    "oracle_XU": ("X", "U"),
    "oracle_XUW": ("X", "U", "W"),
}

#: The two oracle rungs, identification-minimal first. Their gap is a pure
#: estimation-quality quantity (see the module docstring), so several summary
#: statistics are reported for both.
ORACLES = ("oracle_XU", "oracle_XUW")

#: Normalised absolute ATE error above which a rung is called *materially
#: biased*: the estimate is off by more than a tenth of an outcome standard
#: deviation. Far above cross-fitting noise, far below "the sign is wrong".
BIAS_THRESHOLD = 0.10

#: Number of quantile bins per axis of the confounding-strength grid. Three
#: gives ~113 datasets per cell at the full `general_ite` n=1024 -- as coarse
#: as the grid can go and still resolve a 3x3 interaction, which matters because
#: (per the normalisation caveat above) the per-cell signal is weak and a finer
#: grid would mostly resolve noise. Raise it alongside ``--n-datasets``, not
#: independently.
N_SHARE_BINS = 3

#: Floor applied before taking log10 of the errors, so an exactly-zero error
#: (possible for the naive rung on an unconfounded draw) stays on the axis.
ERROR_FLOOR = 1e-4


def _blocks(record: PriorDatasetRecord) -> Dict[str, np.ndarray]:
    """The predictor blocks, as ``(T, d)`` float32 matrices."""
    return {
        name: np.asarray(getattr(record, name), dtype=np.float32)
        for name in ("X", "U", "W", "Z")
    }


def confounding_frame(
    records: List[PriorDatasetRecord],
    cache_key: str,
    model: str = "rf",
    n_splits: int = 3,
    seed: int = 0,
    max_rows: int | None = 1024,
    regression_cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per (dataset, rung): the ATE estimate and its normalised error.

    Columns: ``index``, ``rung``, ``ate_hat``, ``ate_true``, ``abs_error``
    (normalised by ``sd(Y_obs)``), ``raw_error`` (unnormalised
    ``|ate_hat - ate_true|`` -- kept alongside ``abs_error`` because ``sd_y``
    is itself entangled with the delivered ``U`` share of ``f_Y``, see the
    module docstring's normalisation caveat), ``dim_U``, ``sd_y``,
    ``prevalence``.

    ``cache_key`` scopes the regression cache to the dataset collection; each
    fit is tagged with it, the record index, the row count and the predictor
    set, so re-renders are free and specs never collide.
    """
    rows: List[Dict[str, Any]] = []
    records = list(records)
    for record in tqdm(
        records,
        desc="confounding ladder",
        unit="dataset",
        disable=not progress or len(records) <= 1,
    ):
        blocks = _blocks(record)
        A = np.asarray(record.A).ravel()
        Y = np.asarray(record.Y_obs, dtype=np.float32).ravel()
        cate = np.asarray(record.cate, dtype=np.float64).ravel()

        n_rows = len(A)
        if max_rows is not None and n_rows > max_rows:
            n_rows = max_rows
            blocks = {k: v[:n_rows] for k, v in blocks.items()}
            A, Y, cate = A[:n_rows], Y[:n_rows], cate[:n_rows]

        # The ground truth is the ATE of the rows actually being estimated on,
        # so a truncated run is scored against its own target rather than
        # against the full draw's.
        ate_true = float(cate.mean())
        sd_y = float(np.std(Y))
        if not np.isfinite(sd_y) or sd_y == 0:
            continue

        for rung, names in RUNGS.items():
            if rung == "naive":
                treated, control = Y[A == 1], Y[A == 0]
                ate_hat = (
                    float(treated.mean() - control.mean())
                    if treated.size and control.size
                    else np.nan
                )
            else:
                # Row count in the tag: a different max_rows is a different
                # regression and must not hit the same cache entry.
                tag = (
                    f"{cache_key}|i={record.index}|rows={n_rows}"
                    f"|predictors={'+'.join(names)}"
                )
                ate_hat = cached_t_learner(
                    tag,
                    np.hstack([blocks[name] for name in names]),
                    A,
                    Y,
                    model=model,
                    n_splits=n_splits,
                    seed=seed,
                    cache_dir=regression_cache_dir,
                )["ate"]
            raw_error = abs(ate_hat - ate_true)
            rows.append(
                {
                    "index": record.index,
                    "rung": rung,
                    "ate_hat": ate_hat,
                    "ate_true": ate_true,
                    "raw_error": raw_error,
                    "abs_error": raw_error / sd_y,
                    "dim_U": int(blocks["U"].shape[1]),
                    "sd_y": sd_y,
                    "prevalence": float(np.mean(A)),
                }
            )
    return pd.DataFrame(rows)


def delivered_shares(records: List[PriorDatasetRecord]) -> pd.DataFrame:
    """Per-dataset delivered ``U`` share of ``f_A`` and ``f_Y``.

    Reuses ``mechanism_inputs.mechanism_frame`` rather than re-deriving the
    share arithmetic, so B.2.3 and this audit are guaranteed to be talking about
    the same number. Columns: ``index``, ``share_A``, ``share_Y``,
    ``share_min`` (the binding one -- see the module docstring on why bias needs
    both), ``target_share_A``, ``target_share_Y``.
    """
    frame = mechanism_frame(records)
    wide = frame.pivot(
        index="index",
        columns="mechanism",
        values=["first_layer_share_signal", "target_share"],
    )
    out = pd.DataFrame(
        {
            "index": wide.index,
            "share_A": wide[("first_layer_share_signal", "A")].to_numpy(),
            "share_Y": wide[("first_layer_share_signal", "Y")].to_numpy(),
            "target_share_A": wide[("target_share", "A")].to_numpy(),
            "target_share_Y": wide[("target_share", "Y")].to_numpy(),
        }
    ).reset_index(drop=True)
    out["share_min"] = out[["share_A", "share_Y"]].min(axis=1)
    return out


@dataclass
class ConfoundingResults:
    """Per-(dataset, rung) ATE errors, joined to the delivered share dials."""

    frame: pd.DataFrame
    shares: pd.DataFrame
    model: str

    def errors(self, rung: str) -> np.ndarray:
        """The normalised absolute errors for one rung, NaNs dropped."""
        values = self.frame.loc[self.frame["rung"] == rung, "abs_error"].to_numpy()
        return values[np.isfinite(values)]

    def joined(self, rung: str) -> pd.DataFrame:
        """One rung's errors with the delivered shares attached, per dataset."""
        rung_frame = self.frame[self.frame["rung"] == rung]
        merged = rung_frame.merge(self.shares, on="index", how="inner")
        return merged[np.isfinite(merged["abs_error"])]


def share_grid(
    joined: pd.DataFrame, n_bins: int = N_SHARE_BINS
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Median error and dataset count on a quantile grid of ``(share_Y, share_A)``.

    Quantile bins, not equal-width: ``Beta(1.2, 2.8)`` puts most of its mass
    below 0.5, so an equal-width grid would leave the top cells empty and spend
    its resolution where nothing sits.

    Returns ``(median_error, counts, y_edges, a_edges)`` with the grid indexed
    ``[a_bin, y_bin]`` -- row index is ``share_A``, so the array is already in
    ``imshow`` orientation with ``origin="lower"``.
    """
    edges = {
        axis: list(
            np.unique(
                np.quantile(joined[f"share_{axis}"], np.linspace(0, 1, n_bins + 1))
            )
        )
        for axis in ("A", "Y")
    }
    bins = {
        axis: np.clip(
            np.digitize(joined[f"share_{axis}"], edges[axis][1:-1]),
            0,
            len(edges[axis]) - 2,
        )
        for axis in ("A", "Y")
    }
    n_a, n_y = len(edges["A"]) - 1, len(edges["Y"]) - 1
    median = np.full((n_a, n_y), np.nan)
    counts = np.zeros((n_a, n_y), dtype=int)
    errors = joined["abs_error"].to_numpy()
    for i in range(n_a):
        for j in range(n_y):
            cell = errors[(bins["A"] == i) & (bins["Y"] == j)]
            counts[i, j] = cell.size
            if cell.size:
                median[i, j] = float(np.median(cell))
    return median, counts, edges["Y"], edges["A"]


@register
class ConfoundingStrengthAnalysis(Analysis):
    name = "confounding_strength"
    description = (
        "Confounding-strength ladder (B.3.3): ATE error under naive / "
        "backdoor-X / backdoor-XWZ / oracle-(X,U) / oracle-(X,U,W) adjustment, "
        "and how it tracks the delivered confounding share."
    )

    #: Regression knobs; override on an instance for tests (``model="linear"``)
    #: or for the TabICL robustness pass. ``rf`` at the full 1024 rows is the
    #: headline setting, matching the other B.3 audits.
    model = "rf"
    n_splits = 3
    seed = 0
    max_rows = 1024
    regression_cache_dir = DEFAULT_REGRESSION_CACHE_DIR

    def compute(self, cache: PriorCache) -> ConfoundingResults:
        records = list(cache)
        if self.model == "tabicl" and len(records) > 64:
            # ~n * 3 rungs * 2 arms * (n_splits + 1) TabICL fits.
            print(
                f"confounding_strength: {len(records)} datasets x 3 regression "
                f"rungs with TabICL -- slow on first run (predictions are then "
                f"cached). Lower --n-datasets for a quicker pass."
            )
        frame = confounding_frame(
            records,
            cache_key=cache.spec.cache_key,
            model=self.model,
            n_splits=self.n_splits,
            seed=self.seed,
            max_rows=self.max_rows,
            regression_cache_dir=self.regression_cache_dir,
        )
        return ConfoundingResults(
            frame=frame, shares=delivered_shares(records), model=self.model
        )

    def render(self, results: ConfoundingResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_ladder(results, outdir)
        figures += self._render_share_grid(results, outdir)
        tables = self._render_table(results, outdir)

        summary: Dict[str, Any] = {
            "n_datasets": int(results.frame["index"].nunique()),
            "n_unconfounded": int(
                (results.frame.drop_duplicates("index")["dim_U"] == 0).sum()
            ),
            "model": results.model,
            "bias_threshold": BIAS_THRESHOLD,
        }
        for rung in RUNGS:
            errors = results.errors(rung)
            summary[f"median_error_{rung}"] = float(np.median(errors))
            summary[f"p90_error_{rung}"] = float(np.quantile(errors, 0.9))
            summary[f"share_materially_biased_{rung}"] = float(
                np.mean(errors > BIAS_THRESHOLD)
            )
        # The headline contrast, against each oracle: the identification-minimal
        # one is the literal assumption, the (X,U,W) one the correctly-specified
        # outcome model, hence the tighter floor and the larger ratio.
        for oracle in ORACLES:
            summary[f"backdoor_X_over_{oracle}_median"] = float(
                summary["median_error_backdoor_X"] / summary[f"median_error_{oracle}"]
            )
        # How much of the identification oracle's residual was simply leaving W,
        # a real parent of f_Y, out of the regression.
        summary["oracle_XU_over_oracle_XUW_median"] = float(
            summary["median_error_oracle_XU"] / summary["median_error_oracle_XUW"]
        )
        for rung in ("backdoor_X", *ORACLES):
            joined = results.joined(rung)
            summary[f"spearman_share_min_vs_error_{rung}"] = float(
                spearmanr(joined["share_min"], joined["abs_error"]).statistic
            )
            quartiles = np.quantile(joined["share_min"], [0.25, 0.75])
            for label, mask in (
                ("bottom_quartile", joined["share_min"] <= quartiles[0]),
                ("top_quartile", joined["share_min"] >= quartiles[1]),
            ):
                summary[f"median_error_{rung}_{label}_share_min"] = float(
                    np.median(joined.loc[mask, "abs_error"])
                )
            # The grid's top-right cell: both shares in their top bin. This is
            # the one place the ladder's premise is under real load, and pooling
            # it away is exactly what the 2-D grid exists to prevent.
            # `None`, not NaN, when the cell is empty (possible on a tiny cache
            # where the quantile bins collapse): NaN is not valid JSON.
            median, counts, _, _ = share_grid(joined)
            summary[f"median_error_{rung}_top_cell"] = (
                float(median[-1, -1]) if counts[-1, -1] else None
            )
            summary[f"n_top_cell_{rung}"] = int(counts[-1, -1])

        # The normalisation-caveat evidence (see module docstring): sd_y is
        # entangled with share_Y specifically, which is why the share grid's
        # columns look flatter than its rows even though the raw bias is not.
        naive_joined = results.joined("naive")
        for axis in ("share_A", "share_Y"):
            summary[f"spearman_sd_y_vs_{axis}"] = float(
                spearmanr(naive_joined[axis], naive_joined["sd_y"]).statistic
            )
        for rung in ("naive", "backdoor_X"):
            joined = results.joined(rung)
            for axis in ("share_A", "share_Y"):
                summary[f"spearman_{axis}_vs_raw_error_{rung}"] = float(
                    spearmanr(joined[axis], joined["raw_error"]).statistic
                )
                summary[f"spearman_{axis}_vs_abs_error_{rung}"] = float(
                    spearmanr(joined[axis], joined["abs_error"]).statistic
                )
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    # -- figure: the ladder --------------------------------------------------

    def _render_ladder(self, results: ConfoundingResults, outdir: Path) -> List[Path]:
        """One violin per rung, on a log error axis.

        Log, because the rungs are expected to differ by *orders of magnitude*
        -- on a linear axis the oracle and backdoor violins would both be pinned
        to the bottom and the gap that carries the whole result would be
        unreadable.
        """
        fig, ax = plt.subplots(figsize=figsize_grid(ncols=1, nrows=1, panel_aspect=0.5))

        labels = {
            "naive": "naive\n(no adjustment)",
            "backdoor_X": "backdoor\n$X$",
            "backdoor_XWZ": "backdoor\n$(X, W, Z)$",
            "oracle_XU": "oracle\n$(X, U)$",
            "oracle_XUW": "oracle\n$(X, U, W)$",
        }
        # Colour by role, not by block: these are five adjustment strategies,
        # and the reading is naive/biased -> oracle. `ROLES` already owns that
        # vocabulary for line styles; here it is only the colours that carry it.
        # The two oracles share a colour -- they are one role, at two levels of
        # regression specification.
        colours = {
            "naive": ROLES["naive"]["color"],
            "backdoor_X": "0.35",
            "backdoor_XWZ": "0.35",
            "oracle_XU": ROLES["oracle"]["color"],
            "oracle_XUW": ROLES["oracle"]["color"],
        }

        rungs = list(RUNGS)
        data = [np.log10(np.maximum(results.errors(r), ERROR_FLOOR)) for r in rungs]
        parts = ax.violinplot(
            data,
            positions=list(range(len(rungs))),
            widths=0.75,
            showextrema=False,
            showmedians=False,
        )
        for body, rung in zip(parts["bodies"], rungs):
            body.set_facecolor(colours[rung])
            body.set_alpha(0.35)
            body.set_edgecolor("none")

        rng = np.random.default_rng(0)  # jitter only; fixed for reproducibility
        for position, (rung, values) in enumerate(zip(rungs, data)):
            ax.scatter(
                position + rng.uniform(-0.12, 0.12, size=values.size),
                values,
                s=3,
                alpha=0.25,
                linewidths=0,
                color=colours[rung],
                zorder=3,
            )
            median = float(np.median(values))
            ax.plot(
                [position - 0.28, position + 0.28],
                [median, median],
                color="0.15",
                lw=1.6,
                zorder=4,
            )
            ax.annotate(
                f"{10 ** median:.3f}",
                xy=(position, median),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=FONT_SIZE_PT - 2,
                color="0.15",
                zorder=5,
            )

        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels([labels[r] for r in rungs])
        ax.set_xlim(-0.6, len(rungs) - 0.4)
        _log_ticks(ax, np.concatenate(data))
        ax.set_ylabel(r"$|\widehat{ATE} - ATE| \,/\, \mathrm{sd}(Y)$")
        ax.grid(axis="x")
        return save_figure(fig, outdir, "confounding_ladder")

    # -- figure: error vs delivered confounding share ------------------------

    def _render_share_grid(
        self, results: ConfoundingResults, outdir: Path
    ) -> List[Path]:
        """Backdoor-``X`` and oracle error over the delivered ``(s_Y, s_A)`` grid.

        The two panels share one colour scale -- that is the whole point. On the
        ``general_ite`` n=1024 headline run the backdoor panel darkens along the
        top row (high delivered ``U`` share of ``f_A``, i.e. the *treatment*
        side alone) rather than toward the top-right corner -- it is essentially
        flat across the ``f_Y``-share columns, not the interaction the grid was
        built to expose (see the module docstring). The oracle panel should stay
        without that row-wise structure, though its range is not far below the
        backdoor panel's, so the contrast is modest rather than stark at this
        sample size.
        """
        panels = [
            ("backdoor_X", r"(a) backdoor $X$"),
            ("oracle_XU", "(b) oracle $(X, U)$ — control"),
        ]
        grids = {rung: share_grid(results.joined(rung)) for rung, _ in panels}
        finite = np.concatenate(
            [grid[0][np.isfinite(grid[0])].ravel() for grid in grids.values()]
        )
        vmin, vmax = float(finite.min()), float(np.quantile(finite, 0.98))

        fig, axes = plt.subplots(
            1, 2, figsize=figsize_grid(ncols=2, nrows=1, panel_aspect=1.05)
        )
        for ax, (rung, title) in zip(axes, panels):
            median, counts, y_edges, a_edges = grids[rung]
            image = ax.imshow(
                median,
                origin="lower",
                aspect="auto",
                cmap=COUNT_CMAP,
                vmin=vmin,
                vmax=vmax,
                extent=(0, median.shape[1], 0, median.shape[0]),
            )
            for i in range(median.shape[0]):
                for j in range(median.shape[1]):
                    if not np.isfinite(median[i, j]):
                        continue
                    # White on the dark end of viridis, black on the pale end.
                    shade = (median[i, j] - vmin) / max(vmax - vmin, 1e-12)
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{median[i, j]:.2f}\n$n$={counts[i, j]}",
                        ha="center",
                        va="center",
                        fontsize=FONT_SIZE_PT - 3.5,
                        color="white" if shade < 0.6 else "0.1",
                    )
            _share_axis(ax, y_edges, axis="x")
            _share_axis(ax, a_edges, axis="y")
            ax.set_title(title)
            ax.set_xlabel(r"delivered $U$ share of $f_Y$")
            ax.grid(False)
        axes[0].set_ylabel(r"delivered $U$ share of $f_A$")
        fig.colorbar(
            image,
            ax=axes,
            label=r"median $|\widehat{ATE} - ATE| \,/\, \mathrm{sd}(Y)$",
            fraction=0.045,
        )
        return save_figure(fig, outdir, "confounding_vs_share")

    # -- table ---------------------------------------------------------------

    def _render_table(self, results: ConfoundingResults, outdir: Path) -> List[Path]:
        """Median / 90th-percentile normalised error and the materially-biased
        share, one row per rung."""
        adjustment = {
            "naive": "none",
            "backdoor_X": r"$X$",
            "backdoor_XWZ": r"$(X, W, Z)$",
            "oracle_XU": r"$(X, U)$",
            "oracle_XUW": r"$(X, U, W)$",
        }
        role = {
            "naive": "unadjusted",
            "backdoor_X": "correct if no latent confounder",
            "backdoor_XWZ": "DAG-blind: everything measured",
            "oracle_XU": "identification: minimal sufficient set",
            "oracle_XUW": r"efficiency: adds $W$, a parent of $f_Y$",
        }
        rows = []
        for rung in RUNGS:
            errors = results.errors(rung)
            rows.append(
                {
                    "rung": rung.replace("_", " "),
                    "adjustment set": adjustment[rung],
                    "role": role[rung],
                    "median error": float(np.median(errors)),
                    "90th pct": float(np.quantile(errors, 0.9)),
                    f"share $> {BIAS_THRESHOLD:g}$": float(
                        np.mean(errors > BIAS_THRESHOLD)
                    ),
                }
            )
        return save_table(
            pd.DataFrame(rows),
            outdir,
            "confounding_table",
            caption=(
                "Confounding-strength ladder. Absolute ATE error against the "
                "exact in-sample ground truth $\\mathrm{mean}(Y_1-Y_0)$, "
                "normalised by $\\mathrm{sd}(Y_{\\mathrm{obs}})$ "
                "for cross-SCM comparability. Every rung but naive is a "
                "cross-fitted T-learner. The oracle rungs condition on the true "
                "latent confounder $U$, available only because the data is "
                "synthetic; their gap to backdoor-$X$ is the confounding "
                "\\Cref{as:latent_ignorability} claims the proxies remove. "
                "$(X,U,W)$ additionally conditions on $W$, a genuine parent of "
                "$f_Y$, so its gap to $(X,U)$ isolates outcome-model "
                "misspecification rather than residual confounding."
            ),
            label="tab:prior_confounding",
            escape=False,
            wrap_columns=["role"],
            small=True,
        )


def _log_ticks(ax, log_values: np.ndarray) -> None:
    """Label a log10-transformed axis with the original (linear) magnitudes."""
    lo = int(np.floor(np.nanmin(log_values)))
    hi = int(np.ceil(np.nanmax(log_values)))
    ticks = list(range(lo, hi + 1))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_ylim(lo - 0.15, hi + 0.35)


def _share_axis(ax, edges: List[float], axis: str) -> None:
    """Tick a quantile-binned share axis at the bin edges."""
    positions = np.arange(len(edges))
    labels = [f"{edge:.2f}" for edge in edges]
    if axis == "x":
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_PT - 2.5)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_PT - 2.5)
