"""Definitional artifacts for the confounding-dim control (appendix B.2.5).

The confounding dial converts a target *share* ``s`` of a mechanism's input
width into a $U$-derived column count via
:meth:`proximalfm.prior._dataset.SCMPrior.confounding_input_dim`. This analysis
draws the resulting share-to-share map and tabulates the knobs. It is
**analytic**: it reads only the spec's configuration (the ``Beta`` law of the
share, the floor/cap constants of the conversion, and the block-dimension
ranges that set the competing input widths ``n``), not the generated datasets --
so it is cheap to run on any spec, e.g.::

    PYTHONPATH=. uv run python -m analysis_prior.main --spec smoke --analysis confounding_dim

The delivered-vs-analytic QA of the realised share (which *does* need the
generated data) is a separate future analysis; the cache is intentionally
unused here beyond its spec.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from proximalfm.prior._dataset import SCMPrior

from ..design import (
    apply_style,
    block_color,
    figsize,
    variant_styles,
    ROLES,
    FONT_SIZE_PT,
)
from ..generate import PriorCache
from ..plotting import plt, save_figure, save_table
from .base import Analysis, AnalysisArtifacts, register

#: Default share law when the spec neither pins nor overrides it. Mirrors
#: DEFAULT_SAMPLED_HP["confounding_share_of_U_in_Y"].
_DEFAULT_BETA: Tuple[float, float] = (1.2, 2.8)


def _conversion_constants() -> Tuple[int, float]:
    """Read the floor and cap straight off ``confounding_input_dim``'s defaults.

    Keeping these in sync with the generator by introspection, rather than
    duplicating the literals, means the figure can never quietly disagree with
    what the prior actually does.
    """
    params = inspect.signature(SCMPrior.confounding_input_dim).parameters
    return int(params["floor"].default), float(params["max_share"].default)


def _share_beta(spec) -> Tuple[float, float]:
    """The ``(a, b)`` of the share's Beta law, from the spec (Y side).

    Falls back to the shipped default if the spec pins the share to something
    other than a Beta (e.g. a uniform used to ablate the dial), since the
    definitional figure documents the shipped law.
    """
    sampled = spec.resolved_hp()[1]
    cfg = sampled.get("confounding_share_of_U_in_Y")
    if isinstance(cfg, dict) and cfg.get("distribution") == "beta":
        return float(cfg["a"]), float(cfg["b"])
    return _DEFAULT_BETA


def representative_widths(spec) -> List[int]:
    """A low / mid / high competing input width ``n = d_X + |proxy*| + 1``.

    The confounding share is targeted against the *realised* competing width,
    which ranges from ``min_dim_X + 1`` (no proxy edges kept) up to
    ``max_dim_X + max_dim_W + 1`` (all kept). Three representative widths keep
    the figure legible while bracketing that range.
    """
    n_lo = max(4, spec.min_dim_X + 1)
    n_hi = spec.max_dim_X + spec.max_dim_W + 1
    n_mid = round((n_lo + n_hi) / 2)
    return sorted({n_lo, n_mid, n_hi})


@dataclass
class ConfoundingResults:
    """Analytic description of the confounding-share dial."""

    beta_a: float
    beta_b: float
    floor: int
    cap: float
    n_values: List[int]
    #: Per-``n`` scalars worth reporting: the floor atom P(k=floor) and the
    #: floor share value 2/(2+n).
    per_n: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def mean_share(self) -> float:
        return self.beta_a / (self.beta_a + self.beta_b)


@register
class ConfoundingDimAnalysis(Analysis):
    name = "confounding_dim"
    description = (
        "Confounding-share dial: the analytic share-to-share map and knob table "
        "(appendix B.2.5). Analytic -- reads the spec, not the data."
    )

    def compute(self, cache: PriorCache) -> ConfoundingResults:
        spec = cache.spec
        a, b = _share_beta(spec)
        floor, cap = _conversion_constants()
        n_values = representative_widths(spec)

        dist = stats.beta(a, b)
        per_n: Dict[int, Dict[str, float]] = {}
        for n in n_values:
            # k = floor whenever round(s/(1-s) * n) <= floor, i.e.
            # s < (floor + 0.5) / (n + floor + 0.5).
            thr = (floor + 0.5) / (n + floor + 0.5)
            per_n[n] = {
                "floor_atom": float(dist.cdf(thr)),
                "floor_share": floor / (floor + n),
            }

        return ConfoundingResults(
            beta_a=a,
            beta_b=b,
            floor=floor,
            cap=cap,
            n_values=n_values,
            per_n=per_n,
        )

    def render(self, results: ConfoundingResults, outdir: Path) -> AnalysisArtifacts:
        outdir = Path(outdir)
        apply_style()

        figures = self._render_curve(results, outdir)
        tables = self._render_knob_table(results, outdir)

        dist = stats.beta(results.beta_a, results.beta_b)
        summary = {
            "beta_a": results.beta_a,
            "beta_b": results.beta_b,
            "mean_share": results.mean_share,
            "p_share_gt_0.5": float(dist.sf(0.5)),
            "p_share_gt_cap": float(dist.sf(results.cap)),
            "floor": results.floor,
            "cap": results.cap,
            "n_values": results.n_values,
            "floor_atom_by_n": {
                n: results.per_n[n]["floor_atom"] for n in results.n_values
            },
        }
        return AnalysisArtifacts(summary=summary, figures=figures, tables=tables)

    def _render_curve(self, results: ConfoundingResults, outdir: Path) -> list:
        """Realised share rho = k/(k+n) vs the target share s, one line per n."""
        fig, ax = plt.subplots(figsize=figsize(fraction=0.8, aspect=0.9))

        s_grid = np.linspace(1e-3, 1 - 1e-3, 800)
        labels = [f"n={n}" for n in results.n_values]
        styles = variant_styles(labels, ordered=True)

        for n in results.n_values:
            rho = np.array(
                [(k := SCMPrior.confounding_input_dim(s, n)) / (k + n) for s in s_grid]
            )
            color = styles[f"n={n}"]["color"]
            ax.plot(s_grid, rho, label=rf"$n={n}$", color=color, lw=1.4)
            # Faint marker of this n's floor level rho = floor/(floor+n).
            ax.axhline(
                results.floor / (results.floor + n),
                color=color,
                lw=0.6,
                ls=":",
                alpha=0.5,
            )

        # Identity target rho = s.
        ax.plot([0, 1], [0, 1], **{**ROLES["oracle"], "label": r"identity $\rho=s$"})

        # Universal cap: rho saturates at `cap` for every n (k = 3n).
        ax.axhline(results.cap, **ROLES["reference"])
        ax.axvline(
            results.cap, color=ROLES["reference"]["color"], ls="--", lw=0.8, alpha=0.6
        )
        ax.text(
            results.cap + 0.01,
            0.02,
            rf"$s^{{\max}}={results.cap:g}$",
            fontsize=FONT_SIZE_PT - 2,
            color=ROLES["reference"]["color"],
            rotation=90,
            va="bottom",
            ha="left",
        )

        # Beta density of s, faint band along the bottom.
        dens = stats.beta(results.beta_a, results.beta_b).pdf(s_grid)
        dens = dens / dens.max() * 0.22
        ax.fill_between(s_grid, 0, dens, color="0.6", alpha=0.25, lw=0, zorder=0)
        ax.text(
            results.mean_share,
            0.05,
            rf"$\mathrm{{Beta}}({results.beta_a:g},{results.beta_b:g})$",
            fontsize=FONT_SIZE_PT - 2,
            color="0.4",
            ha="center",
            va="bottom",
        )

        ax.set_xlabel(r"target share $s$")
        ax.set_ylabel(r"realised share $\rho = k/(k+n)$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=FONT_SIZE_PT - 2)
        return save_figure(fig, outdir, "confounding_dim_curve")

    def _render_knob_table(self, results: ConfoundingResults, outdir: Path) -> list:
        """The knob-specification table for B.2.5 (\\Cref{tab:prior_conf_dim})."""
        dist = stats.beta(results.beta_a, results.beta_b)
        beta = rf"$\mathrm{{Beta}}({results.beta_a:g}, {results.beta_b:g})$"

        rows = [
            {
                "knob": r"\texttt{confounding\_share\_of\_U\_in\_\{Y,A\}}",
                "law / value": rf"{beta}, mean {results.mean_share:.2f}",
                "role": r"target $U$ share $s$ of the mechanism input (\eqref{eq:conf_dim})",
            },
            {
                "knob": "floor",
                "law / value": rf"$k \ge {results.floor}$",
                "role": rf"at least {results.floor} $U$-columns per mechanism",
            },
            {
                "knob": r"cap $s^{\max}$",
                "law / value": rf"${results.cap:g}$ ($k \le 3n$)",
                "role": rf"augment-network memory bound; clamp mass ${dist.sf(results.cap):.3f}$",
            },
            {
                "knob": r"\texttt{direct\_edge\_share\_of\_\{W\_in\_Y,Z\_in\_A\}}",
                "law / value": r"$\mathrm{Beta}(1,1)$",
                "role": r"per-column keep prob.; kept count $\sim\mathrm{Unif}\{0,\dots,d\}$",
            },
            {
                "knob": r"\texttt{confounding\_dim\_control}, \texttt{proxy\_edge\_subsampling}",
                "law / value": "on (default)",
                "role": "off-settings are the B.2.5 ablation arms",
            },
        ]

        return save_table(
            pd.DataFrame(rows),
            outdir,
            "confounding_dim_control",
            caption=(
                "Knobs governing mechanism wiring (\\Cref{app:prior_duplication}): "
                "the share of $f_A$/$f_Y$ inputs drawn from the confounder, and the "
                "per-column presence of the direct proxy edges. Shares are drawn "
                "once per dataset and per side."
            ),
            label="tab:prior_conf_dim",
            escape=False,
            wrap_columns=["knob", "role"],
            centering=True,
        )
