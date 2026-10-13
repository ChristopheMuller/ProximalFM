"""Proxy relevance (B.3.4) -- the empirical audit of ``as:completeness``.

Completeness asks that the proxies carry *enough variation in* ``U`` for the
bridge functions to be pinned down. The generator only enforces a **counting**
surrogate of it: the categorical-completeness check of ``_reg2cls.py``
(``dominates``), which generalises ``min(d_W, d_Z) >= d_U`` by requiring every
``U`` column to be matched by a distinct, at-least-as-informative proxy column.
That check is structural -- it looks at widths and category counts, never at
values -- and in the shipped prior it passes for **every** dataset by
construction.

So the question this module asks is not "does the heuristic hold?" (it does, by
fiat) but the one the heuristic cannot answer: **does holding it buy functional
recoverability of ``U``?** That is the "heuristic passes while completeness
quietly fails" side of the audit.

**How it is measured.** For each column ``u_j`` of ``U``, cross-fitted
:func:`~analysis.analysis_prior.regression.leave_one_out_r2` over the two blocks
``{X, P}`` for each proxy ``P in {W, Z}``::

    dR2_P(u_j) = R2(u_j | X, P) - R2(u_j | X)

i.e. what the proxy adds *over the covariates*, which is the right increment:
``X`` is a parent of ``U`` (``eq:uwz``), so a proxy that merely re-encoded ``X``
would be useless for identification even with a high raw ``R2``. Two
aggregations over ``U``'s columns are carried side by side, and they say
different things:

- the unweighted **mean** -- the headline, treating every ``U`` column as
  equally important (a simplification worth flagging rather than fixing: the
  columns enter ``f_A``/``f_Y`` with different weights, and any weighting would
  have to commit to one mechanism's view);
- the **minimum** -- the weakest direction of ``U``. Completeness is a statement
  about *all* of ``U``, so one unrecoverable direction is enough to break it,
  and the gap between the two aggregations is a result in its own right.

**The figure is one y-axis read three ways.** Every panel plots the same
``dR2``; only the x-axis changes, and the two proxies are the two series
throughout (``W`` and ``Z`` in their usual block colours, so no colour channel is
spent on anything else):

- **(a) pooled relevance** -- how much the proxies add at all, mean and weakest
  column per proxy. ``dR2 > 0`` is nearly free, so the readable claim is the
  *level* and the size of the weak tail, not a sign test.
- **(b) by dimension slack** ``d_P - d_U`` -- does the counting heuristic's
  spirit hold, i.e. does spare proxy width buy recoverability?
- **(c) by mechanism noise** (``noise_share_of_input_in_P``) -- does diluting the
  proxy's mechanism input with exogenous noise cost recoverability?

Each proxy is audited against its **own** slack, never the joint
``min(d_W, d_Z) - d_U``: the two clauses of the assumption are separate (``W``
for the outcome bridge, ``Z`` for the treatment bridge), and ``d_W``/``d_Z`` are
drawn independently and disagree in 83% of ``general_ite``'s datasets, so the
joint statistic would attribute a proxy's relevance to the other proxy's width.
Both correlations are in the table for the contrast.

Two caveats on the x-axes, both stated in the caption:

- **Slack conflates two things.** ``d_P - d_U`` rises when the proxy is wide
  *and* when ``U`` is narrow, and "narrow ``U`` is easier to recover" is a
  different claim from "wide proxy helps". The table therefore reports Spearman
  against ``d_P`` and ``d_U`` separately as well.
- **Panel (c) plots the sampled knob, not the delivered composition.**
  ``record.mechanisms`` carries realised input composition for ``f_A``/``f_Y``
  only, so for ``f_W``/``f_Z`` the target share is all there is. The exclusion
  audit measured knobs associating ~3x more weakly than delivered quantities, so
  read (c) as an attenuated view of the noise effect. ``u_share_expected`` (the
  ``U`` share of the proxy's input width implied by the knobs) is derived in
  :func:`_knobs` and reported in ``summary.json`` as the combined-dilution
  version of the same story.

Datasets with ``d_U = 0`` are excluded: there is nothing to recover, so a
``dR2`` of zero would be a correct answer to a vacuous question. Their count is
reported in ``summary.json``.

Two further caveats on the measurement itself:

- **The proxy blocks are supersets.** The cached ``W``/``Z`` are the full
  delivered blocks. Extra columns cannot manufacture signal, so a *low* ``dR2``
  is the conservative reading -- exactly the direction this audit cares about.
- **Mean recoverability, not distributional.** A cross-fitted ``R^2`` sees the
  conditional mean. Completeness is a stronger, distributional condition; a
  proxy could in principle pin down ``U`` through features a regression's mean
  cannot see. The ``dR2`` is therefore a *lower* bound on relevance: high
  ``dR2`` implies real relevance, low ``dR2`` implies weak *mean* relevance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from ..design import FONT_SIZE_PT, apply_style, block_color, figsize_grid
from ..frame_cache import DEFAULT_FRAME_CACHE_DIR, cached_frame
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from ..regression import DEFAULT_REGRESSION_CACHE_DIR, leave_one_out_r2
from .base import Analysis, AnalysisArtifacts, register

#: The two proxies, audited independently. Order fixes the series order in every
#: panel and the row order of the table.
PROXIES = ("W", "Z")

#: The two per-dataset aggregations over ``U``'s columns, with their labels.
#: ``mean`` is the headline; ``min`` is the weakest direction, which is what
#: completeness actually stands or falls on.
AGGREGATIONS = {"delta_r2": "mean", "delta_r2_min": "weakest"}

#: ``dR2`` below which we call the proxy's contribution *functionally weak*. The
#: counting heuristic passes for every dataset in the shipped prior, so the share
#: of datasets under this line is precisely the rate at which the cheap check and
#: the assumption it stands in for disagree. 0.1 is a deliberate, generous line:
#: a proxy explaining under a tenth of the residual variation of the confounder
#: is not going to pin down a bridge function.
WEAK_THRESHOLD = 0.1

#: Smallest group that gets a point on the conditional panels. Slack reaches 9
#: but the far tail holds a handful of datasets, and a median over 3 of them is
#: noise drawn with the authority of a line.
MIN_BIN_COUNT = 10

#: Quantile bins for the continuous noise knob in panel (c). Six keeps ~170
#: datasets per bin at ``general_ite``'s n=1024 -- enough for a stable median and
#: an IQR band.
N_NOISE_BINS = 6

#: Dimension slack above which panel (b) merges every dataset into one bin. The
#: slack tail is thin (52/34/20 datasets at slack 7/8/9 in ``general_ite``), and
#: a median over twenty datasets drawn as a line point reads as a trend it is
#: not -- the un-merged version showed a dramatic jump at slack 9 that is three
#: draws wide.
SLACK_TOP_BIN = 7


def _slack(record: PriorDatasetRecord) -> Dict[str, int]:
    """Dimension slack ``d_P - d_U`` per proxy, on the *delivered* widths.

    Post-filter widths, not the sampled ones: a constant column is dropped
    before the data is handed over, and the heuristic has to hold for what was
    actually delivered.
    """
    dims = record.dims
    return {p: int(dims[f"d_{p}"] - dims["d_U"]) for p in PROXIES}


def _knobs(record: PriorDatasetRecord, proxy: str) -> Dict[str, float]:
    """The two sampled knobs that dilute ``U`` out of ``f_P``, and their product.

    ``f_P``'s input is ``[U, X_P, eps]``: the ``U`` columns, a Binomial
    subsample of ``X`` sized by ``covariate_share_of_X_in_P``, and an exogenous
    noise block sized so that noise takes a ``noise_share_of_input_in_P`` share
    of the total width (``confounding_input_dim``, clamped at 0.75). Both knobs
    push ``U``'s share of the input width down, so the single quantity that
    summarises the dilution is

        u_share_expected = d_U (1 - s) / (d_U + p * d_X)

    with ``s`` the (clamped) noise share and ``p`` the covariate share. It is an
    *expectation*: the covariate subsample is a Binomial draw whose realisation
    is not recorded, and ``confounding_input_dim`` rounds and floors. Returns
    ``NaN`` for caches written before ``sampled_hp`` existed.
    """
    hp = record.sampled_hp or {}
    noise = float(hp.get(f"noise_share_of_input_in_{proxy}", np.nan))
    covariate = float(hp.get(f"covariate_share_of_X_in_{proxy}", np.nan))
    dim_U = float(hp.get("dim_U", np.nan))
    dim_X = float(hp.get("dim_X", np.nan))
    signal = dim_U + covariate * dim_X
    u_share = dim_U * (1.0 - min(noise, 0.75)) / signal if signal > 0 else np.nan
    return {
        "noise_share": noise,
        "covariate_share": covariate,
        "u_share_expected": u_share,
    }


def relevance_frame(
    records: List[PriorDatasetRecord],
    cache_key: str,
    model: str = "rf",
    n_splits: int = 3,
    seed: int = 0,
    max_rows: int | None = 1024,
    regression_cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per (dataset, ``U`` column, proxy): the incremental ``dR2``.

    Columns: ``index``, ``u_column``, ``proxy``, ``r2_full`` (``R2(u_j | X, P)``),
    ``r2_covariates`` (``R2(u_j | X)``), ``r2_proxy_only`` (``R2(u_j | P)``),
    ``delta_r2``, ``dim_U``, ``dim_proxy``, ``slack``, and the three
    :func:`_knobs` fields.

    ``r2_covariates`` is shared between the two proxies -- same predictor set,
    same tag, so it is fitted once and served from cache the second time. Five
    distinct cross-fitted regressions per ``U`` column: ``X``, ``X+W``, ``W``,
    ``X+Z``, ``Z``.

    ``cache_key`` scopes the regression cache to the dataset collection; each
    fit is tagged with it, the record index, the target column and the predictor
    set, so re-renders are free and specs never collide. Datasets with
    ``d_U = 0`` are skipped (see the module docstring).
    """
    rows: List[Dict[str, Any]] = []
    records = list(records)
    for record in tqdm(
        records,
        desc="relevance regressions",
        unit="dataset",
        disable=not progress or len(records) <= 1,
    ):
        U = np.asarray(record.U, dtype=np.float32)
        dim_U = U.shape[1]
        if dim_U == 0:
            continue

        n_rows = len(record.A)
        if max_rows is not None and n_rows > max_rows:
            n_rows = max_rows
        blocks = {
            name: np.asarray(getattr(record, name), dtype=np.float32)[:n_rows]
            for name in ("X",) + PROXIES
        }
        slack = _slack(record)
        knobs = {proxy: _knobs(record, proxy) for proxy in PROXIES}

        for j in range(dim_U):
            u_j = U[:n_rows, j]
            # Row count in the tag: a different max_rows is a different
            # regression and must not hit the same cache entry.
            tag = f"{cache_key}|i={record.index}|target=U{j}|rows={n_rows}"
            for proxy in PROXIES:
                r2 = leave_one_out_r2(
                    {"X": blocks["X"], proxy: blocks[proxy]},
                    u_j,
                    tag,
                    model=model,
                    n_splits=n_splits,
                    seed=seed,
                    cache_dir=regression_cache_dir,
                )
                rows.append(
                    {
                        "index": record.index,
                        "u_column": j,
                        "proxy": proxy,
                        "r2_full": r2["full"],
                        # leave_one_out_r2 keys each entry by the block it
                        # *drops*: the set without the proxy is X alone, the set
                        # without X is the proxy alone.
                        "r2_covariates": r2[proxy],
                        "r2_proxy_only": r2["X"],
                        "delta_r2": r2["full"] - r2[proxy],
                        "dim_U": dim_U,
                        "dim_proxy": blocks[proxy].shape[1],
                        "slack": slack[proxy],
                        **knobs[proxy],
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class RelevanceResults:
    """Per-(dataset, ``U`` column, proxy) ``dR2``, plus provenance."""

    frame: pd.DataFrame
    model: str
    n_skipped_no_u: int

    def per_dataset(self) -> pd.DataFrame:
        """One row per (dataset, proxy): both aggregations plus the drivers.

        Everything but the two ``dR2`` aggregations is constant within a
        (dataset, proxy) group, so ``first`` is exact rather than a summary.
        """
        return self.frame.groupby(["index", "proxy"], as_index=False).agg(
            delta_r2=("delta_r2", "mean"),
            delta_r2_min=("delta_r2", "min"),
            r2_full=("r2_full", "mean"),
            r2_covariates=("r2_covariates", "mean"),
            r2_proxy_only=("r2_proxy_only", "mean"),
            slack=("slack", "first"),
            dim_U=("dim_U", "first"),
            dim_proxy=("dim_proxy", "first"),
            noise_share=("noise_share", "first"),
            covariate_share=("covariate_share", "first"),
            u_share_expected=("u_share_expected", "first"),
        )

    def proxy_frame(self, proxy: str) -> pd.DataFrame:
        """:meth:`per_dataset` restricted to one proxy, indexed by dataset."""
        per = self.per_dataset()
        return per[per["proxy"] == proxy].set_index("index")

    def slack_min(self) -> pd.Series:
        """The counting heuristic's own joint statistic, per dataset.

        ``min(d_W, d_Z) - d_U``. Not used by the figure -- each proxy is drawn
        against its own slack -- but kept for the table's correlation contrast.
        """
        per = self.per_dataset()
        return per.groupby("index")["slack"].min()

    def series(self, proxy: str, column: str = "delta_r2") -> np.ndarray:
        """One value per dataset for ``proxy``, as an array."""
        return self.proxy_frame(proxy)[column].to_numpy()


def _binned_quantiles(
    x: np.ndarray, y: np.ndarray, bins: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and quartiles of ``y`` within bins of ``x``; empty bins dropped.

    Returns ``(x_centre, q25, median, q75)``, where the centre is the *median
    of the x values in the bin* rather than the bin midpoint -- with quantile
    bins on a skewed knob the midpoint would sit where no dataset does.
    Bins holding fewer than :data:`MIN_BIN_COUNT` datasets are dropped.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    which = np.digitize(x, np.asarray(bins)[1:-1])
    centres, lows, medians, highs = [], [], [], []
    for b in range(len(bins) - 1):
        mask = which == b
        if mask.sum() < MIN_BIN_COUNT:
            continue
        centres.append(float(np.median(x[mask])))
        low, mid, high = np.quantile(y[mask], [0.25, 0.5, 0.75])
        lows.append(float(low))
        medians.append(float(mid))
        highs.append(float(high))
    return (np.array(centres), np.array(lows), np.array(medians), np.array(highs))


@register
class ProxyRelevanceAnalysis(Analysis):
    name = "proxy_relevance"
    description = (
        "Proxy-relevance audit (B.3.4): incremental R2 of each proxy for the "
        "latent U, against dimension slack and the mechanism noise knob."
    )

    #: Regression knobs; override on an instance for tests (``model="linear"``)
    #: or for the TabICL robustness pass. ``rf`` at the full 1024 rows is the
    #: headline setting, matching the other B.3 audits.
    model = "rf"
    n_splits = 3
    seed = 0
    max_rows = 1024
    regression_cache_dir = DEFAULT_REGRESSION_CACHE_DIR
    #: Where the assembled frame is cached, so a re-render costs one file read
    #: rather than ~5 regression-cache reads per ``U`` column. Set to ``None``
    #: to bypass the layer entirely.
    frame_cache_dir = DEFAULT_FRAME_CACHE_DIR

    def compute(self, cache: PriorCache) -> RelevanceResults:
        records = list(cache)
        n_skipped = sum(1 for r in records if np.asarray(r.U).shape[1] == 0)
        if self.model == "tabicl" and len(records) > 32:
            # ~n * mean(d_U) * 5 predictor sets * n_splits TabICL fits.
            print(
                f"proxy_relevance: {len(records)} datasets x d_U targets x 5 "
                f"predictor sets with TabICL -- slow on first run (predictions "
                f"are then cached). Lower --n-datasets for a quicker pass."
            )

        def build(indices=None) -> pd.DataFrame:
            wanted = records if indices is None else [by_index[i] for i in indices]
            return relevance_frame(
                wanted,
                cache_key=cache.spec.cache_key,
                model=self.model,
                n_splits=self.n_splits,
                seed=self.seed,
                max_rows=self.max_rows,
                regression_cache_dir=self.regression_cache_dir,
            )

        if self.frame_cache_dir is None:
            frame = build()
        else:
            by_index = {r.index: r for r in records}
            # Only the datasets that produce rows: ``relevance_frame`` skips
            # ``d_U = 0``, and an index with no rows would look permanently
            # missing to the frame cache and be rebuilt on every call.
            frame = cached_frame(
                tag=(
                    f"proxy_relevance|{cache.spec.cache_key}|model={self.model}"
                    f"|splits={self.n_splits}|seed={self.seed}|rows={self.max_rows}"
                ),
                indices=[r.index for r in records if np.asarray(r.U).shape[1] > 0],
                build=build,
                # Editing any of these invalidates the cached frames; the
                # regression cache underneath is unaffected and still serves the
                # fits, so a rebuild is bookkeeping, not refitting.
                depends_on=(relevance_frame, _knobs, _slack),
                cache_dir=self.frame_cache_dir,
            )
        return RelevanceResults(
            frame=frame, model=self.model, n_skipped_no_u=int(n_skipped)
        )

    def render(self, results: RelevanceResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_figure(results, outdir)
        tables = self._render_table(results, outdir)

        slack_min = results.slack_min()
        summary: Dict[str, Any] = {
            "n_datasets": int(results.frame["index"].nunique()),
            "n_skipped_no_u": results.n_skipped_no_u,
            "model": results.model,
            "weak_threshold": WEAK_THRESHOLD,
            # The counting heuristic is enforced at generation time, so this is
            # expected to be 1.0; it is reported rather than assumed, because
            # every "heuristic passes but ..." statement below is conditional
            # on it and an unenforced ablation would move it.
            "share_heuristic_passes": float((slack_min >= 0).mean()),
            "share_both_weak": float(
                np.mean(
                    (results.series("W") < WEAK_THRESHOLD)
                    & (results.series("Z") < WEAK_THRESHOLD)
                )
            ),
        }
        for proxy in PROXIES:
            data = results.proxy_frame(proxy)
            for column, label in AGGREGATIONS.items():
                values = data[column].to_numpy()
                summary[f"median_delta_r2_{label}_{proxy}"] = float(np.median(values))
                summary[f"share_weak_{label}_{proxy}"] = float(
                    np.mean(values < WEAK_THRESHOLD)
                )
                summary[f"share_positive_{label}_{proxy}"] = float(np.mean(values > 0))
            # The two halves of the increment, kept because a small dR2 has two
            # very different causes: the covariates already explain U (high
            # r2_covariates), or the proxy carries little about U in the first
            # place (low r2_proxy_only). Only the second is a relevance failure.
            summary[f"median_r2_covariates_{proxy}"] = float(
                data["r2_covariates"].median()
            )
            summary[f"median_r2_proxy_only_{proxy}"] = float(
                data["r2_proxy_only"].median()
            )
            for name, driver in self._drivers(data, slack_min).items():
                summary[f"spearman_{name}_{proxy}"] = self._spearman(
                    driver, data["delta_r2"].to_numpy()
                )
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    # -- drivers ------------------------------------------------------------

    @staticmethod
    def _drivers(data: pd.DataFrame, slack_min: pd.Series) -> Dict[str, np.ndarray]:
        """The candidate explanations for a proxy's relevance, per dataset.

        ``slack`` is the panel-(b) axis; ``dim_proxy`` and ``dim_U`` are the two
        halves it conflates (slack rises both when the proxy is wide and when
        ``U`` is narrow, and those are different claims); ``slack_min`` is the
        counting heuristic's joint statistic, reported to justify not colouring
        by it; ``noise_share`` is the panel-(c) axis and ``u_share_expected``
        its combined-dilution version.
        """
        return {
            "dim_proxy": data["dim_proxy"].to_numpy(),
            "dim_U": data["dim_U"].to_numpy(),
            "slack": data["slack"].to_numpy(),
            "slack_min": slack_min.reindex(data.index).to_numpy(),
            "noise_share": data["noise_share"].to_numpy(),
            "u_share_expected": data["u_share_expected"].to_numpy(),
        }

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman correlation over the pairs where both are finite.

        ``NaN``-tolerant because the knob-derived drivers are ``NaN`` on caches
        written before ``sampled_hp`` existed, and a constant driver (a spec
        that pins a knob) has no rank correlation to report.
        """
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
            return float("nan")
        return float(spearmanr(x[finite], y[finite]).statistic)

    # -- figure -------------------------------------------------------------

    def _render_figure(self, results: RelevanceResults, outdir: Path) -> List[Path]:
        """One quantity on the y-axis, three x-axes: the marginal, then the two
        generative knobs that should move it.

        The panels share the y-axis literally (``sharey``), so the reader learns
        the ``dR2`` scale once in (a) and then reads (b) and (c) against it. The
        two proxies are the only colour-carrying distinction, in their standard
        block colours -- there is no third variable competing for the channel.
        """
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize_grid(ncols=3, nrows=1, panel_aspect=1.15),
            sharey=True,
        )
        self._marginal_panel(axes[0], results)
        self._slack_panel(axes[1], results)
        self._noise_panel(axes[2], results)

        axes[0].set_ylabel(r"$\Delta R^2$ about $U$")
        # Headroom for the two in-panel legends, which would otherwise sit on
        # the topmost whisker of (a) and the first band of (b).
        low, high = axes[0].get_ylim()
        axes[0].set_ylim(low, high + 0.22 * (high - low))
        for ax in axes:
            # The weak line is the figure's one decision boundary and the only
            # dashed rule; zero is a scale landmark, drawn thinner.
            # ax.axhline(WEAK_THRESHOLD, color="0.6", lw=0.8, linestyle="--", zorder=1)
            ax.axhline(0.0, color="0.8", lw=0.6, zorder=1)
        axes[1].legend(loc="upper left", handlelength=1.4, ncol=2)
        return save_figure(fig, outdir, "proxy_relevance")

    @staticmethod
    def _marginal_panel(ax, results: RelevanceResults) -> None:
        """(a) Distribution of ``dR2`` per proxy, mean and weakest ``U`` column.

        Boxes rather than a strip: at n=1024 a point cloud is a smear, and what
        the panel has to carry is the median, the spread and the size of the
        weak tail. The mean/weakest pair is *the* result of the aggregation
        choice -- a dataset can look healthy on average while one direction of
        ``U`` stays unrecoverable -- so both are drawn, not just tabulated.
        """
        positions, colours, data = [], [], []
        for i, proxy in enumerate(PROXIES):
            frame = results.proxy_frame(proxy)
            for k, column in enumerate(AGGREGATIONS):
                positions.append(2.5 * i + k)
                colours.append(block_color(proxy))
                data.append(frame[column].to_numpy())

        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            whis=(10, 90),
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="0.15", lw=1.2),
        )
        for patch, colour, column in zip(
            boxes["boxes"], colours, list(AGGREGATIONS) * len(PROXIES)
        ):
            # The weakest-column box is the same colour, lighter: it is the same
            # measurement read pessimistically, not a different series.
            patch.set_facecolor(colour)
            patch.set_alpha(0.45 if column == "delta_r2" else 0.18)
            patch.set_edgecolor(colour)
        for element in ("whiskers", "caps"):
            for line in boxes[element]:
                line.set_color("0.4")
                line.set_linewidth(0.8)

        # Ticks name the proxy (one label per pair, centred); the aggregation is
        # the fill alpha, explained by the legend. Four tick labels under four
        # boxes collide at this panel width, and "mean"/"weakest" repeated twice
        # is the least informative thing that could occupy that space.
        ax.set_xticks([2.5 * i + 0.5 for i in range(len(PROXIES))])
        ax.set_xticklabels([f"${p}$" for p in PROXIES])
        ax.set_title("(a) pooled relevance")
        ax.grid(axis="x")
        ax.legend(
            handles=[
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="0.35",
                    alpha=0.45 if column == "delta_r2" else 0.18,
                    edgecolor="0.35",
                    label=label,
                )
                for column, label in AGGREGATIONS.items()
            ],
            loc="upper center",
            ncol=2,
            frameon=False,
            handlelength=1.1,
            columnspacing=0.8,
            fontsize=FONT_SIZE_PT - 2,
            title="aggregation over $U$",
            title_fontsize=FONT_SIZE_PT - 2,
        )

    @staticmethod
    def _slack_panel(ax, results: RelevanceResults) -> None:
        """(b) ``dR2`` against the proxy's *own* dimension slack ``d_P - d_U``.

        Integer bins, since slack is a small integer; the far tail is dropped by
        :data:`MIN_BIN_COUNT` rather than drawn as a median over three datasets.
        """
        for proxy in PROXIES:
            frame = results.proxy_frame(proxy)
            slack = np.minimum(frame["slack"].to_numpy(), SLACK_TOP_BIN)
            edges = np.arange(slack.min(), SLACK_TOP_BIN + 2) - 0.5
            x, low, median, high = _binned_quantiles(
                slack, frame["delta_r2"].to_numpy(), edges
            )
            # x is the median slack within the bin, which for the merged top bin
            # lands wherever its mass is; pin it to the tick so the two proxies'
            # last points sit on the same x and under the "7+" label.
            x = np.where(x >= SLACK_TOP_BIN, SLACK_TOP_BIN, x)
            ProxyRelevanceAnalysis._trend(ax, x, low, median, high, proxy)
        ticks = list(range(0, SLACK_TOP_BIN + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks[:-1]] + [f"{SLACK_TOP_BIN}+"])
        ax.set_xlabel(r"dimension slack $d_P - d_U$")
        ax.set_title("(b) by dimension slack")

    @staticmethod
    def _noise_panel(ax, results: RelevanceResults) -> None:
        """(c) ``dR2`` against the proxy mechanism's sampled noise share.

        Quantile bins (:data:`N_NOISE_BINS`), because the knob is
        ``Beta(1.2, 2.8)`` and equal-width bins would leave the top ones nearly
        empty. Plots the *sampled* knob -- see the module docstring on why the
        delivered composition is unavailable for ``f_W``/``f_Z``.
        """
        for proxy in PROXIES:
            frame = results.proxy_frame(proxy)
            noise = frame["noise_share"].to_numpy()
            if not np.isfinite(noise).any():
                continue
            edges = np.nanquantile(noise, np.linspace(0, 1, N_NOISE_BINS + 1))
            x, low, median, high = _binned_quantiles(
                noise, frame["delta_r2"].to_numpy(), edges
            )
            ProxyRelevanceAnalysis._trend(ax, x, low, median, high, proxy)
        ax.set_xlabel("noise share of the $f_P$ input")
        ax.set_title("(c) by mechanism noise")

    @staticmethod
    def _trend(ax, x, low, median, high, proxy: str) -> None:
        """A median line with an interquartile band, in the proxy's colour."""
        colour = block_color(proxy)
        ax.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(
            x,
            median,
            color=colour,
            lw=1.6,
            marker="o",
            markersize=3,
            label=f"${proxy}$",
            zorder=3,
        )

    # -- table --------------------------------------------------------------

    def _render_table(self, results: RelevanceResults, outdir: Path) -> List[Path]:
        """Per proxy x aggregation: the level, the weak share, and Spearman
        against every candidate driver, including the two halves slack
        conflates. The joint ``min`` slack the figure deliberately avoids is
        reported in ``summary.json`` for that contrast, but left out of the
        table -- it is not a driver the figure or the prose argues for, and a
        column arguing against itself is not worth the width."""
        slack_min = results.slack_min()
        driver_labels = {
            "dim_proxy": "$d_P$",
            "dim_U": "$d_U$",
            "slack": "$d_P - d_U$",
            "noise_share": "noise",
        }
        rows = []
        for proxy in PROXIES:
            data = results.proxy_frame(proxy)
            drivers = self._drivers(data, slack_min)
            for position, (column, label) in enumerate(AGGREGATIONS.items()):
                values = data[column].to_numpy()
                row = {
                    "proxy": f"${proxy}$" if position == 0 else "",
                    "aggregation over $U$": label,
                    "median $\\Delta R^2$": float(np.median(values)),
                    f"share $< {WEAK_THRESHOLD}$": float(
                        np.mean(values < WEAK_THRESHOLD)
                    ),
                }
                for name, label_tex in driver_labels.items():
                    row[f"$\\rho$ {label_tex}"] = self._spearman(drivers[name], values)
                rows.append(row)
        return save_table(
            pd.DataFrame(rows),
            outdir,
            "proxy_relevance_table",
            caption=(
                "Proxy-relevance audit (\\Cref{as:completeness}): cross-fitted "
                "incremental $\\Delta R^2 = R^2(u_j \\mid X, P) - R^2(u_j \\mid X)$ "
                "of each proxy $P$ for the latent columns, aggregated over $U$ "
                "by the mean or by the weakest column. The counting heuristic "
                "(\\Cref{app:prior_postprocess}) passes for every dataset, so "
                "the \\emph{share} column is the rate at which it and functional "
                "recoverability disagree. $\\rho$ is Spearman's correlation with "
                "each candidate driver: the proxy's own width $d_P$, $d_U$, the "
                "slack $d_P - d_U$ that conflates the two, and the sampled noise "
                "share of the $f_P$ input."
            ),
            label="tab:prior_relevance",
            group_column="proxy",
            escape=False,
            small=True,
        )
