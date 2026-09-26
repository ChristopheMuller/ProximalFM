"""Does method error track the ill-posedness of the episode?

    uv run python -m analysis_spectral.error_vs_illposedness \
        --bench data/prior_eval/prior_val_v2_smalldim_2000rows_subsample/bench \
        --spectral analysis_spectral/results/prior_val_v2_smalldim_20000rows_L16.csv

The figure is a *contrast*, not a leaderboard.  ``sigma2`` is not randomised
across episodes, so "the proximal methods do worse when sigma2 is small" is on
its own compatible with ``sigma2`` merely standing in for some other difficulty
the prior sampled.  What licenses the causal reading is the control group:
methods that never see ``W`` or ``Z`` have no channel through which ``sigma2``
could help them, so their error must be *flat* in it.  Both panels are built
around that comparison.

Three choices worth knowing before reading the result.

**The y-axis defaults to ``pehe``, the unnormalised RMSE against tau*(x).**  On
a ``d_X = 0`` set ``tau*(x)`` is constant, ``target_var`` is exactly 0, and
every scale-free metric (``npehe`` and the whole R2 family) is ``NaN`` by
construction -- so ``nPEHE`` is simply unavailable here.  ``pehe`` is defined
anyway, is already in standardised-outcome units (``data.py`` divides by the
context ``y`` std) hence comparable across episodes, and unlike ``|ate_bias|``
it charges a method for heterogeneity it invents: ``pehe^2 = bias^2 +
Var(predictions)``, where the level metrics see only the first term.

**The x-axis comes from the 20k-row parent set, not from the 2k rows the
methods saw.**  Deliberately: an estimate of ``sigma2`` computed on the same
2000 rows would share its sampling noise with the error being plotted against
it, which manufactures correlation.  The parent's estimate is dominated by
18000 rows the methods never touched.

**Episodes are grouped, not regressed.**  A quarter of them sit at the noise
floor with no recoverable ``W``-``Z`` dependence at all; a line fitted through
that spike would be leveraged by a group that is qualitatively different rather
than merely lower on the axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from benchmarks.plot_style import color_for  # noqa: E402

#: Methods that cannot read a proxy at predict *or* fit time -- the control
#: group. Their curve is the null: whatever slope they show in sigma2 is what
#: the axis produces for reasons unrelated to inverting the operator.
CONTROLS = {"S-OLS", "T-OLS", "S-RF", "T-RF", "X-RF", "S-TabICL", "X-TabICL"}
#: Oracles: drawn as reference lines, never ranked (benchmarks/plot_frames.py).
ORACLE_PREFIXES = ("Oracle-",)
#: Handed the true confounder -- the floor, not a competitor.
U_ORACLES = {"T-OLS-Oracle", "S-RF-Oracle", "Oracle-ITE"}
#: Dropped before anything is drawn.  ``Oracle-ITE`` is the realized
#: ``Y1 - Y0``: on the ``ite`` task it *is* the target, so it scores 0 and is a
#: genuine floor, but on the ``cate`` task it is a single noisy draw around
#: ``tau*(x)`` and its error is the irreducible per-row noise.  That is not a
#: floor for anything -- a method estimating ``tau*(x)`` should beat it, by
#: averaging out exactly the noise the oracle carries -- so drawing it as one
#: invites the reader to treat an unbeatable-looking line as a target.
EXCLUDED = {"Oracle-ITE"}
#: Controls are drawn out of their method family's palette on purpose: the eye
#: has to find the null curve first, before comparing anything to it.
CONTROL_COLOR = "#000000"
#: Below this median |bias| a method is reproducing the target exactly.
#: Oracle-ATE lands at ~6e-8 -- float32 dust from summing the replicate mean,
#: not an error, so the tolerance sits well above machine epsilon.
EXACT_TOL = 1e-6


#: The x-axes this figure understands, and the two things that differ between
#: them.  ``better`` is which end of the axis is the *better conditioned*
#: operator, and it is the whole reason this is a table rather than a string
#: parameter: for ``sigma2``/``eff_rank`` a larger value means more signal
#: transmitted, while for a decay rate a larger value means the spectrum falls
#: away *faster* -- more ill-posed, the opposite reading.  Wiring a rate in
#: without flipping it would invert the conclusion while leaving the figure
#: looking perfectly sensible.  ``partial`` marks the columns that exist only
#: for episodes with a fit (``measurable``), which is a biased subset.
X_AXES = {
    "sigma2": {
        "better": "high",
        "partial": False,
        "desc": "leading canonical correlation of W,Z given A",
    },
    "sigma2_min": {
        "better": "high",
        "partial": False,
        "desc": "sigma2 of the worse treatment cell",
    },
    "eff_rank": {
        "better": "high",
        "partial": False,
        "desc": "directions above the permutation null",
    },
    "poly_rate": {
        "better": "low",
        "partial": True,
        "desc": "p in sigma_j ~ j^-p; larger = faster decay",
    },
    "exp_rate": {
        "better": "low",
        "partial": True,
        "desc": "c in sigma_j ~ e^-cj; larger = faster decay",
    },
}


#: The y-axes this figure understands.  ``pehe`` is the unnormalised
#: root-mean-square error against tau*(x): unlike ``npehe`` it survives a
#: homogeneous episode (``d_X = 0`` makes ``Var(tau)`` exactly 0, which blanks
#: every scale-free metric), and unlike ``abs_bias`` it sees a method inventing
#: heterogeneity that is not there -- ``pehe^2 = bias^2 + Var(predictions)``.
Y_COLS = {
    "pehe": "median PEHE   (standardised-y units)",
    "abs_bias": "median |ATE bias|   (standardised-y units)",
    "npehe": "median nPEHE",
}
#: Which task's rows carry which y column.  ``pehe`` is on the effect tasks;
#: the ``ate`` task is level-only by design, so it offers ``abs_bias`` alone.
DEFAULT_Y = {"cate": "pehe", "ite": "pehe", "ate": "abs_bias"}


def is_reference(method: str) -> bool:
    return method.startswith(ORACLE_PREFIXES) or method in U_ORACLES


def load(
    bench: str | Path,
    spectral: str | Path,
    *,
    task: str = "cate",
    y_col: str | None = None,
    exclude: Sequence[str] = (),
) -> tuple[pd.DataFrame, str]:
    """Join one task's rows of ``results.csv`` to the spectral table.

    Returns the frame and the resolved y column.  ``task`` decides which
    methods are even present: the ``cate`` table admits only ``cate(x)``
    outputs, so asking for it drops every proxy-conditioning method -- that is
    the tag algebra working as designed, not data loss, but it does thin the
    method list the mechanism test runs on.  ``task="ate"`` admits everyone at
    the cost of seeing the level only.
    """
    res = pd.read_csv(Path(bench) / "results" / "results.csv")
    if "target" not in res.columns:
        raise ValueError(
            "results.csv has no `target` column -- it predates the task split. "
            "Re-run the benchmark rather than guessing which task a row is."
        )
    drop = EXCLUDED | set(exclude)
    a = res[(res.target == task) & (~res.method.isin(drop))].copy()
    if a.empty:
        raise ValueError(
            f"no rows with target == {task!r} in results.csv "
            f"(present: {sorted(res.target.unique())})"
        )
    a["abs_bias"] = a.ate_bias.abs()

    y_col = y_col or DEFAULT_Y.get(task, "abs_bias")
    if y_col not in a.columns:
        raise ValueError(
            f"results.csv has no {y_col!r} column. It was scored before that "
            "metric existed; re-run benchmarks/run.py with the same config -- "
            "the preds/ cache is reused, so only the scoring is redone."
        )
    if a[y_col].isna().all():
        raise ValueError(
            f"every {y_col!r} is NaN on the {task!r} task. If this is a "
            "homogeneous set (d_X = 0, so Var(tau) = 0) the scale-free metrics "
            "are undefined by construction -- use --y_col pehe."
        )

    sp = pd.read_csv(spectral).set_index("index")
    missing = set(a.episode.unique()) - set(sp.index)
    if missing:
        raise ValueError(
            f"{len(missing)} benchmark episodes have no spectral row "
            f"(e.g. {sorted(missing)[:5]}). The two must describe the same set."
        )
    have = [c for c in X_AXES if c in sp.columns]
    if not have:
        raise ValueError(
            f"{spectral} carries none of the known x-axis columns "
            f"({sorted(X_AXES)}); is it an analysis_spectral.scan output?"
        )
    return a.join(sp[have], on="episode"), y_col


def strata(v: pd.Series, n_groups: int = 4, *, better: str = "high") -> pd.Series:
    """Quantile groups of an ill-posedness column, labelled by their range.

    Categories come back ordered **most ill-posed first**, whichever column is
    being cut.  Keeping the reading of the x-axis fixed across axes is what
    lets two of these figures be compared at a glance; leaving each axis in its
    own natural order would silently mirror half of them.
    """
    q = pd.qcut(v, n_groups, duplicates="drop")
    labels = {c: f"{c.left:.2f}–{c.right:.2f}" for c in q.cat.categories}
    q = q.cat.rename_categories(labels)
    if better == "low":
        q = q.cat.reorder_categories(list(q.cat.categories)[::-1], ordered=True)
    return q


def _boot_median_ci(x: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return (np.nan, np.nan)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(draws, axis=1)
    return tuple(np.quantile(meds, [0.25, 0.75]))


def figure(
    df: pd.DataFrame,
    *,
    n_groups: int = 4,
    x_col: str = "sigma2",
    y_col: str = "pehe",
) -> plt.Figure:
    if x_col not in df.columns:
        raise ValueError(
            f"the spectral table has no column {x_col!r}; it carries "
            f"{sorted(c for c in X_AXES if c in df.columns)}"
        )
    spec = X_AXES[x_col]

    ep_x = df.groupby("episode")[x_col].first()
    n_all = len(ep_x)
    ep_x = ep_x.dropna()
    dropped = n_all - len(ep_x)
    if len(ep_x) < n_groups * 2:
        raise ValueError(
            f"only {len(ep_x)} of {n_all} episodes have a {x_col}; too few to "
            f"cut into {n_groups} groups. The decay rates exist only for the "
            "`measurable` episodes -- use sigma2, which is defined everywhere."
        )
    df = df[df.episode.isin(ep_x.index)]

    groups = strata(ep_x, n_groups, better=spec["better"])
    df = df.assign(stratum=df.episode.map(groups))
    order = list(groups.cat.categories)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # ── panel 1: error by ill-posedness stratum ──────────────────────────────
    methods = [m for m in df.method.unique() if not is_reference(m)]
    methods.sort(key=lambda m: (m in CONTROLS, m))
    for m in methods:
        g = df[df.method == m]
        med, lo, hi = [], [], []
        for s in order:
            v = g.loc[g.stratum == s, y_col].to_numpy()
            med.append(np.median(v))
            l, h = _boot_median_ci(v)
            lo.append(l)
            hi.append(h)
        ctl = m in CONTROLS
        ax.plot(
            range(len(order)),
            med,
            "o--" if ctl else "o-",
            color=CONTROL_COLOR if ctl else color_for(m),
            lw=3.0 if ctl else 2.0,
            ms=7,
            alpha=0.95,
            label=f"{m}  (control)" if ctl else m,
            zorder=3 if ctl else 2,
        )
        ax.fill_between(
            range(len(order)),
            lo,
            hi,
            color=CONTROL_COLOR if ctl else color_for(m),
            alpha=0.12,
            lw=0,
        )

    for m in [m for m in df.method.unique() if is_reference(m)]:
        g = df[df.method == m]
        if g[y_col].median() <= 0:
            continue  # Oracle-CATE/ATE are exactly 0 here; nothing to draw
        ax.axhline(
            g[y_col].median(),
            color="#555555",
            ls=":",
            lw=1.4,
            zorder=1,
        )
        ax.annotate(
            f"{m} (floor)",
            (len(order) - 1, g[y_col].median()),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#555555",
        )

    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel(
        f"{x_col} stratum  ({spec['desc']})\n"
        "left = more ill-posed"
        + (
            "   [axis reversed: larger rate = faster decay]"
            if spec["better"] == "low"
            else ""
        ),
        fontsize=10,
    )
    ax.set_ylabel(Y_COLS.get(y_col, f"median {y_col}"))
    title = "Error vs ill-posedness\nbands: bootstrap IQR of the median"
    if dropped:
        title += (
            f"\n{dropped} of {n_all} episodes dropped: no {x_col}. "
            "These are the ones with too little\nsignal to fit a decay — "
            "i.e. the most ill-posed. Read the remainder as a biased subset."
        )
    ax.set_title(title, fontsize=11 if not dropped else 9)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── panel 2: the mechanism test ──────────────────────────────────────────
    rows = []
    for m in df.method.unique():
        g = df[df.method == m]
        if g[y_col].median() < EXACT_TOL:
            # Oracle-ATE/Oracle-CATE reproduce the target exactly here, so what
            # is left is float dust; ranking it yields a meaningless rho.
            continue
        r = spearmanr(g[x_col], g[y_col])
        rows.append((m, r.statistic, r.pvalue))
    rho = pd.DataFrame(rows, columns=["method", "rho", "p"]).sort_values("rho")

    colors = [
        CONTROL_COLOR
        if m in CONTROLS
        else ("#888888" if is_reference(m) else color_for(m))
        for m in rho.method
    ]
    ax2.barh(range(len(rho)), rho.rho, color=colors, alpha=0.9)
    for i, (_, r) in enumerate(rho.iterrows()):
        star = "*" if r.p < 0.05 else ""
        off = -0.004 if r.rho < 0 else 0.004
        ax2.text(
            r.rho + off,
            i,
            f"{r.rho:+.3f}{star}",
            va="center",
            ha="right" if r.rho < 0 else "left",
            fontsize=8,
        )
    ax2.axvline(0, color="k", lw=1)
    ax2.set_yticks(
        range(len(rho)),
        [
            f"{m}  (control)"
            if m in CONTROLS
            else (f"{m}  (oracle)" if is_reference(m) else m)
            for m in rho.method
        ],
        fontsize=9,
    )
    ax2.set_xlabel(f"Spearman  rho( {y_col} , {x_col} )")
    expected = (
        "negative = error falls as the operator becomes\nbetter conditioned"
        if spec["better"] == "high"
        else "POSITIVE = error rises as the spectrum decays\nfaster, i.e. as the "
        "problem gets more ill-posed"
    )
    ax2.set_title(f"Mechanism test\n{expected}.  * = p < 0.05", fontsize=11)
    ax2.margins(x=0.22)
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench",
        default="data/prior_eval/prior_val_v2_smalldim_50sub_2000/bench",
    )
    p.add_argument(
        "--spectral",
        default="analysis_spectral/results/prior_val_v2_smalldim_50k_L24.csv",
    )
    p.add_argument(
        "--x_col",
        default="sigma2",
        choices=tuple(X_AXES),
        help=(
            "ill-posedness column to stratify on. The rates are defined only "
            "for `measurable` episodes; sigma2 is defined for all of them."
        ),
    )
    p.add_argument(
        "--task",
        default="cate",
        choices=("cate", "ite", "ate"),
        help=(
            "which task's rows to score. `cate` is scored against tau*(x) but "
            "admits only cate(x) methods; `ate` admits every method at the cost "
            "of seeing the level only."
        ),
    )
    p.add_argument(
        "--y_col",
        default=None,
        choices=tuple(Y_COLS),
        help="error metric (default: pehe on cate/ite, abs_bias on ate)",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help=f"extra methods to drop (always dropped: {sorted(EXCLUDED)})",
    )
    p.add_argument("--n_groups", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df, y_col = load(
        args.bench,
        args.spectral,
        task=args.task,
        y_col=args.y_col,
        exclude=args.exclude,
    )
    fig = figure(df, n_groups=args.n_groups, x_col=args.x_col, y_col=y_col)
    out = Path(
        args.out or f"analysis_spectral/results/{y_col}_vs_{args.x_col}_{args.task}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
