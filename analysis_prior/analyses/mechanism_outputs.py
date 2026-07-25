"""Output-side companion to the mechanism input composition (B.2.3).

Where ``mechanism_inputs`` shows the share knobs steer what each mechanism is
*fed* (wiring / first-layer variance), this shows they steer what each
mechanism's output *depends on*, measured on the delivered data with a
predictor rather than the generator internals. For the two headline
mechanisms:

- ``f_A``: predict the raw treatment logit ``logit(A_probs)`` from the cached
  parent blocks ``{U, X, Z}``.
- ``f_Y``: predict the potential outcome ``Y0`` from ``{U, X, W}``.

Each block's *unique* contribution is the cross-fitted leave-one-out drop
``R²(all blocks) - R²(all but this block)`` -- order-free, and (unlike a
sequential build-up) it never attributes shared signal to whichever block
happened to go first. The exogenous-noise channel has no block to drop, so its
readout is the unexplained share ``1 - R²(all blocks)``: the fraction of the
output variance no parent can predict. Plotting each channel's readout against
its own sampled knob mirrors ``fig:prior_mechanism_input_composition`` panel
for panel.

Two honest caveats, stated in the caption:

- **Supersets.** The cached blocks are the full delivered ``U``/``X``/``Z``/
  ``W``, supersets of the subsampled columns each mechanism actually consumed;
  extra columns cannot add signal that is not there, so the drop is a valid
  lower bound on the true dependence.
- **Remix noise on the Y side.** ``Y0`` is post heterogeneity-remix, so its
  ``1 - R²`` mixes the exogenous ``eps`` with the per-row ``alpha`` remix
  variation (which vanishes as ``gamma -> 1``); the treatment side has no such
  contamination, ``logit(A_probs)`` being an affine image of the raw ``f_A``
  output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from ..design import apply_style, block_color, figsize_grid
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from ..regression import DEFAULT_REGRESSION_CACHE_DIR, leave_one_out_r2
from .base import Analysis, AnalysisArtifacts, register
from .mechanism_inputs import CHANNELS, CHANNEL_KNOBS, _add_trend

#: Parent blocks feeding each mechanism, and the record fields holding them.
#: "proxy" is the treatment-inducing Z for f_A, the outcome-inducing W for f_Y.
MECHANISM_BLOCKS = {
    "A": {"U": "U", "X": "X", "proxy": "Z"},
    "Y": {"U": "U", "X": "X", "proxy": "W"},
}


def _target(record: PriorDatasetRecord, mechanism: str) -> np.ndarray:
    """The prediction target for a mechanism: raw logit for A, Y0 for Y."""
    if mechanism == "A":
        # A_probs is clamped to [eps, 1-eps], so the logit is finite and is an
        # affine image of the raw f_A output (sample_A standardises then shifts
        # a logit before the sigmoid).
        p = np.clip(record.A_probs.astype(np.float64), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p)).astype(np.float32)
    return record.Y0.astype(np.float32)


def output_frame(
    records: List[PriorDatasetRecord],
    cache_key: str,
    model: str = "tabicl",
    n_splits: int = 3,
    seed: int = 0,
    max_rows: int | None = 1024,
    regression_cache_dir: Path = DEFAULT_REGRESSION_CACHE_DIR,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per (dataset, mechanism): leave-one-out ΔR² per channel + knobs.

    ``cache_key`` scopes the regression cache to the dataset collection; each
    regression is tagged with it, the record index, the target and the
    predictor set, so predictions persist across re-renders and are never
    confused between specs.

    ``progress`` shows a per-dataset tqdm bar; each step is 8 cross-fitted
    regressions (2 mechanisms x 4 predictor sets), instant on a cache hit and
    the dominant cost on a miss.
    """
    rows: List[Dict[str, Any]] = []
    records = list(records)
    for record in tqdm(
        records,
        desc="mechanism_outputs regressions",
        unit="dataset",
        disable=not progress or len(records) <= 1,
    ):
        if record.sampled_hp is None:
            raise ValueError(
                f"dataset {record.index} has no sampled_hp; the cache predates "
                "the diagnostics. Regenerate with --force (same data)."
            )
        for mechanism, block_fields in MECHANISM_BLOCKS.items():
            y = _target(record, mechanism)
            blocks = {
                channel: getattr(record, field)
                for channel, field in block_fields.items()
            }
            if max_rows is not None and len(y) > max_rows:
                y = y[:max_rows]
                blocks = {k: v[:max_rows] for k, v in blocks.items()}

            r2 = leave_one_out_r2(
                blocks,
                y,
                # Row count in the tag: truncating to a different max_rows is a
                # different regression and must not hit the same cache entry.
                tag_prefix=(
                    f"{cache_key}|i={record.index}|target={mechanism}" f"|rows={len(y)}"
                ),
                model=model,
                n_splits=n_splits,
                seed=seed,
                cache_dir=regression_cache_dir,
            )

            row: Dict[str, Any] = {
                "index": record.index,
                "mechanism": mechanism,
                "dim_U": int(record.sampled_hp["dim_U"]),
                "r2_full": r2["full"],
                # Unexplained share: the eps channel's readout.
                "residual": 1.0 - r2["full"],
                "knob_eps": float(record.sampled_hp[CHANNEL_KNOBS[mechanism]["eps"]]),
            }
            for channel in ("U", "X", "proxy"):
                # Unique contribution: full minus leave-this-block-out. The
                # leave-one-out keys are the block-dict keys (channel names),
                # not the record field names.
                row[f"delta_r2_{channel}"] = r2["full"] - r2[channel]
                row[f"knob_{channel}"] = float(
                    record.sampled_hp[CHANNEL_KNOBS[mechanism][channel]]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _robust_limits(values: np.ndarray, pad: float) -> tuple[float, float]:
    """Percentile-based y-limits that always include 0, ignoring outliers."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-pad, pad)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    return (min(lo, 0.0) - pad, hi + pad)


@dataclass
class MechanismOutputResults:
    """Per-(dataset, mechanism) leave-one-out ΔR² frame."""

    frame: pd.DataFrame
    model: str

    def per_mechanism(self, mechanism: str) -> pd.DataFrame:
        return self.frame[self.frame["mechanism"] == mechanism]


@register
class MechanismOutputsAnalysis(Analysis):
    name = "mechanism_outputs"
    description = (
        "Output side of B.2.3: do the share knobs steer what f_A / f_Y "
        "outputs depend on (cross-fitted leave-one-block-out R2)?"
    )

    #: Regression knobs; override on an instance for tests (e.g. model="linear")
    #: or a heavier run. The *number of datasets* is not a knob here -- like
    #: every other analysis, this uses all datasets in the cache it is given, so
    #: it is controlled by the spec's ``n_datasets`` (the CLI ``--n-datasets``),
    #: which the content-addressed cache serves as a prefix without regenerating.
    model = "tabicl"
    n_splits = 3
    seed = 0
    max_rows = 1024
    regression_cache_dir = DEFAULT_REGRESSION_CACHE_DIR

    def compute(self, cache: PriorCache) -> MechanismOutputResults:
        records = list(cache)
        if self.model == "tabicl" and len(records) > 64:
            # ~n * 2 mechanisms * 4 predictor sets * n_splits TabICL fits; slow
            # on the first pass, then served from cache/regression/. Flag it so
            # a large --n-datasets is a deliberate choice, not a surprise.
            print(
                f"mechanism_outputs: {len(records)} datasets x 8 predictor sets "
                f"with TabICL -- slow on first run (predictions are then cached). "
                f"Lower --n-datasets for a quicker pass."
            )
        frame = output_frame(
            records,
            cache_key=cache.spec.cache_key,
            model=self.model,
            n_splits=self.n_splits,
            seed=self.seed,
            max_rows=self.max_rows,
            regression_cache_dir=self.regression_cache_dir,
        )
        return MechanismOutputResults(frame=frame, model=self.model)

    def render(
        self, results: MechanismOutputResults, outdir: Path
    ) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_grid(results, outdir)
        tables = self._render_table(results, outdir)

        summary: Dict[str, Any] = {
            "n_datasets": int(results.frame["index"].nunique()),
            "model": results.model,
        }
        for mechanism in ("A", "Y"):
            data = results.per_mechanism(mechanism)
            summary[f"{mechanism}_median_r2_full"] = float(data["r2_full"].median())
            for channel in ("U", "X", "proxy"):
                rho = spearmanr(
                    data[f"knob_{channel}"], data[f"delta_r2_{channel}"]
                ).statistic
                summary[f"{mechanism}_spearman_{channel}"] = float(rho)
            rho_eps = spearmanr(data["knob_eps"], data["residual"]).statistic
            summary[f"{mechanism}_spearman_eps"] = float(rho_eps)
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_grid(self, results: MechanismOutputResults, outdir: Path) -> List[Path]:
        """2x4 grid mirroring the input composition: each channel's output
        readout vs its own knob, with a moving average. Columns 1-3 plot the
        unique leave-one-out ΔR²; column 4 plots the unexplained share
        1 - R²(all parents) against the noise knob.
        """
        fig, axes = plt.subplots(
            2,
            4,
            figsize=figsize_grid(
                ncols=4, nrows=2, panel_aspect=1.0, extra_height_in=0.4
            ),
        )
        # Two y-scales: the three ΔR² columns share one (comparable
        # magnitudes), the residual column keeps its own [0,1]-ish scale --
        # sharing across both would flatten the ΔR² trends against the larger
        # residual range. Limits are robust (1st/99th percentile, not min/max):
        # cross-fitted R² can still produce the odd extreme value, and one
        # outlier must not blow up the axis. Points beyond the limits are simply
        # off-view; the moving-average trend is unaffected.
        frame = results.frame
        delta_cols = [f"delta_r2_{c}" for c in ("U", "X", "proxy")]
        delta_vals = frame[delta_cols].to_numpy().ravel()
        pad = 0.03
        delta_lim = _robust_limits(delta_vals, pad)
        resid_lim = _robust_limits(frame["residual"].to_numpy(), pad)

        for row, mechanism in enumerate(("A", "Y")):
            data = results.per_mechanism(mechanism)
            proxy_block = "Z" if mechanism == "A" else "W"
            for col, channel in enumerate(CHANNELS):
                ax = axes[row][col]
                if channel == "eps":
                    knob, value = data["knob_eps"], data["residual"]
                    color_key = "eps"
                    title = r"$\varepsilon$"
                    ax.set_ylim(*resid_lim)
                else:
                    knob = data[f"knob_{channel}"]
                    value = data[f"delta_r2_{channel}"]
                    color_key = proxy_block if channel == "proxy" else channel
                    title = {
                        "U": "$U$ channel",
                        "X": r"$\tilde X$",
                        "proxy": f"${proxy_block}^*$",
                    }[channel]
                    ax.set_ylim(*delta_lim)
                ax.scatter(
                    knob,
                    value,
                    s=6,
                    alpha=0.35,
                    linewidths=0,
                    color=block_color(color_key),
                )
                _add_trend(ax, knob, value)
                ax.axhline(0.0, color="0.7", lw=0.6, zorder=0)
                ax.set_title(f"$f_{mechanism}$: {title}")
                if row == 1:
                    ax.set_xlabel("sampled knob")
            axes[row][0].set_ylabel("unique $\\Delta R^2$")
            axes[row][3].set_ylabel("unexplained $1-R^2$")
        return save_figure(fig, outdir, "mechanism_output_dependence")

    def _render_table(
        self, results: MechanismOutputResults, outdir: Path
    ) -> List[Path]:
        """Median unique ΔR² per channel and the residual, per mechanism."""
        pretty = {
            "U": "$U$ channel",
            "X": "$\\tilde X$",
            "eps": "$\\varepsilon$ (residual)",
        }
        rows = []
        for mechanism in ("A", "Y"):
            data = results.per_mechanism(mechanism)
            proxy_block = "Z" if mechanism == "A" else "W"
            for position, channel in enumerate(CHANNELS):
                name = pretty.get(channel, f"${proxy_block}^*$")
                if channel == "eps":
                    median = data["residual"].median()
                else:
                    median = data[f"delta_r2_{channel}"].median()
                rows.append(
                    {
                        "mechanism": f"$f_{mechanism}$" if position == 0 else "",
                        "channel": name,
                        "median": median,
                    }
                )
        return save_table(
            pd.DataFrame(rows),
            outdir,
            "mechanism_output_dependence_table",
            caption=(
                "Median cross-fitted output dependence per mechanism input "
                "channel: for $U$, $\\tilde X$ and the proxy, the unique "
                "leave-one-out $\\Delta R^2$ (drop in $R^2$ when that block is "
                "removed from the predictors); for $\\varepsilon$, the "
                "unexplained share $1 - R^2$ from all parents. Targets: "
                "$\\mathrm{logit}(\\pi)$ for $f_A$, $Y_0$ for $f_Y$."
            ),
            label="tab:prior_mechanism_outputs",
            group_column="mechanism",
            escape=False,
        )
