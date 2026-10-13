"""Audit of the mechanism input composition: do the share knobs steer it?

Covers appendix B.2.3 (``fig:prior_mechanism_input_composition``,
``tab:prior_mechanism_inputs``): for the two headline mechanisms ``f_A`` and
``f_Y``, how much of the input signal comes from each parent block --
U-channel (raw ``U`` columns plus the ``f_aug`` re-encoding), covariates
``X̃``, the direct proxy edge (``Z*`` into ``f_A``, ``W*`` into ``f_Y``) and
the exogenous noise block ``eps`` -- and whether the sampled share knobs
actually steer it.

Two estimators of "share" (computed inside the generator, see
``proximalfm.prior._diagnostics``):

- **wiring** -- column-count share. Parent blocks are standardised before
  concatenation, so this equals the input-variance share; it tracks the target
  by construction, up to the floor (``k >= 2``), the clamp (``share <= 0.75``),
  rounding, and the binomial edge draws.
- **first-layer** -- each block's exact injected variance through the
  mechanism's first linear layer. Captures what the block-sparse / dropout
  initialisation actually zeroes out.

Share conventions: the confounding target ``s`` is defined against the
*signal* width (U-channel / (U-channel + X + proxy), ``eps`` excluded), and
the ``*_share_signal`` columns use that denominator; the composition figure
and table use shares of the *total* input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..design import (
    apply_style,
    block_color,
    figsize_grid,
    trend_line,
)
from ..generate import PriorCache, PriorDatasetRecord
from ..plotting import plt, save_figure, save_table
from .base import Analysis, AnalysisArtifacts, register

#: The four input channels of a mechanism, in plotting order. "U" pools the raw
#: U columns with the f_aug re-encoding (one confounding channel by design);
#: "proxy" is Z* for f_A and W* for f_Y.
CHANNELS = ("U", "X", "proxy", "eps")

#: The sampled hyperparameter steering each channel's share, per mechanism.
CHANNEL_KNOBS = {
    "A": {
        "U": "confounding_share_of_U_in_A",
        "X": "covariate_share_of_X_in_A",
        "proxy": "direct_edge_share_of_Z_in_A",
        "eps": "noise_share_of_input_in_A",
    },
    "Y": {
        "U": "confounding_share_of_U_in_Y",
        "X": "covariate_share_of_X_in_Y",
        "proxy": "direct_edge_share_of_W_in_Y",
        "eps": "noise_share_of_input_in_Y",
    },
}


def _channel_values(mech: Dict[str, Any], key: str) -> Dict[str, float]:
    """Pool a per-block measurement dict into the four channels."""
    values = mech[key]
    return {
        "U": float(values["U_sel"]) + float(values["U_aug"]),
        "X": float(values["X"]),
        "proxy": float(values["proxy"]),
        "eps": float(values["eps"]),
    }


def _share(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def mechanism_frame(records: Iterable[PriorDatasetRecord]) -> pd.DataFrame:
    """One row per (dataset, mechanism) with targets and delivered shares.

    Pure function of the records so the share arithmetic can be tested on
    hand-built diagnostics. Raises if any record lacks diagnostics -- caches
    written before they existed must be regenerated (``--force`` reproduces
    the identical data, plus the diagnostics).
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        if record.mechanisms is None or record.sampled_hp is None:
            raise ValueError(
                f"dataset {record.index} has no mechanism diagnostics; the "
                "cache predates them. Regenerate with --force (same data, "
                "diagnostics added)."
            )
        for mechanism in ("A", "Y"):
            mech = record.mechanisms[mechanism]
            widths = _channel_values(mech, "widths")
            flvar = _channel_values(mech, "first_layer_var")

            signal_w = widths["U"] + widths["X"] + widths["proxy"]
            signal_v = flvar["U"] + flvar["X"] + flvar["proxy"]
            total_w = signal_w + widths["eps"]
            total_v = signal_v + flvar["eps"]

            row: Dict[str, Any] = {
                "index": record.index,
                "mechanism": mechanism,
                "dim_U": int(record.sampled_hp["dim_U"]),
                # The U dial, in its own (signal-width) denominator.
                "target_share": float(record.sampled_hp[CHANNEL_KNOBS[mechanism]["U"]]),
                "wiring_share_signal": _share(widths["U"], signal_w),
                "first_layer_share_signal": _share(flvar["U"], signal_v),
            }
            for channel in CHANNELS:
                row[f"knob_{channel}"] = float(
                    record.sampled_hp[CHANNEL_KNOBS[mechanism][channel]]
                )
                row[f"wiring_share_total_{channel}"] = _share(widths[channel], total_w)
                row[f"first_layer_share_total_{channel}"] = _share(
                    flvar[channel], total_v
                )
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class MechanismInputResults:
    """The per-(dataset, mechanism) share frame."""

    frame: pd.DataFrame

    def per_mechanism(self, mechanism: str) -> pd.DataFrame:
        return self.frame[self.frame["mechanism"] == mechanism]


def _add_trend(ax, x, y, window_frac: float = 0.1) -> None:
    """Overlay a moving average of ``y`` over sorted ``x`` on a scatter."""
    ordered = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    window = max(15, int(len(ordered) * window_frac))
    smooth = ordered["y"].rolling(window, center=True, min_periods=window // 2).mean()
    trend_line(ax, ordered["x"], smooth)


@register
class MechanismInputsAnalysis(Analysis):
    name = "mechanism_inputs"
    description = (
        "Mechanism input composition: does the sampled confounding share "
        "steer the delivered U share of f_A / f_Y, decoupled from d_U?"
    )

    def compute(self, cache: PriorCache) -> MechanismInputResults:
        return MechanismInputResults(frame=mechanism_frame(cache))

    def render(self, results: MechanismInputResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_composition(results, outdir)
        tables = self._render_table(results, outdir)

        summary: Dict[str, Any] = {"n_datasets": int(results.frame["index"].nunique())}
        for mechanism in ("A", "Y"):
            data = results.per_mechanism(mechanism)
            for label, column in (
                ("wiring", "wiring_share_signal"),
                ("first_layer", "first_layer_share_signal"),
            ):
                rho = spearmanr(
                    data["target_share"], data[column], nan_policy="omit"
                ).statistic
                summary[f"{mechanism}_spearman_target_vs_{label}"] = float(rho)
            rho_du = spearmanr(
                data["dim_U"], data["first_layer_share_signal"], nan_policy="omit"
            ).statistic
            summary[f"{mechanism}_spearman_d_U_vs_first_layer"] = float(rho_du)
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_composition(
        self, results: MechanismInputResults, outdir: Path
    ) -> List[Path]:
        """Every dial: each channel's first-layer share of the total input vs
        its own knob, with a moving average making the trend legible. No
        identity overlay: the knobs act through column counts against varying
        competitors (and eps enters unstandardised), so a monotone trend --
        not a line to hit -- is the claim.
        """
        fig, axes = plt.subplots(
            2,
            4,
            figsize=figsize_grid(
                ncols=4, nrows=2, panel_aspect=1.0, extra_height_in=0.4
            ),
            sharey=True,
        )
        for row, mechanism in enumerate(("A", "Y")):
            data = results.per_mechanism(mechanism)
            proxy_block = "Z" if mechanism == "A" else "W"
            for col, channel in enumerate(CHANNELS):
                ax = axes[row][col]
                color_key = proxy_block if channel == "proxy" else channel
                ax.scatter(
                    data[f"knob_{channel}"],
                    data[f"first_layer_share_total_{channel}"],
                    s=6,
                    alpha=0.35,
                    linewidths=0,
                    color=block_color(color_key),
                )
                _add_trend(
                    ax,
                    data[f"knob_{channel}"],
                    data[f"first_layer_share_total_{channel}"],
                )
                title = {
                    "U": "$U$ channel",
                    "X": r"$\tilde X$",
                    "proxy": f"${proxy_block}^*$",
                    "eps": r"$\varepsilon$",
                }[channel]
                ax.set_title(f"$f_{mechanism}$: {title}")
                if row == 1:
                    ax.set_xlabel("sampled knob")
                if col == 0:
                    ax.set_ylabel("first-layer share\nof total input")
        return save_figure(fig, outdir, "mechanism_input_composition")

    def _render_table(self, results: MechanismInputResults, outdir: Path) -> List[Path]:
        """Median share of the total input per channel, all three estimators."""
        pretty = {"U": "$U$ channel", "X": "$\\tilde X$", "eps": "$\\varepsilon$"}
        rows = []
        for mechanism in ("A", "Y"):
            data = results.per_mechanism(mechanism)
            proxy_block = "Z" if mechanism == "A" else "W"
            for position, channel in enumerate(CHANNELS):
                name = pretty.get(channel, f"${proxy_block}^*$")
                rows.append(
                    {
                        # Sparsified: save_table renders the group label as one
                        # full-width header row per block.
                        "mechanism": f"$f_{mechanism}$" if position == 0 else "",
                        "input channel": name,
                        "wiring": data[f"wiring_share_total_{channel}"].median(),
                        "first layer": data[
                            f"first_layer_share_total_{channel}"
                        ].median(),
                    }
                )
        return save_table(
            pd.DataFrame(rows),
            outdir,
            # "_table" suffix keeps the stem distinct from the figure of the
            # same name, so the tex can \input one and \includegraphics the other.
            "mechanism_input_composition_table",
            caption=(
                "Median delivered share of each mechanism input channel, as a "
                "fraction of the total input, under the two estimators: "
                "column-count share (wiring) and exact first-layer injected "
                "variance."
            ),
            label="tab:prior_mechanism_inputs",
            group_column="mechanism",
            escape=False,
        )
