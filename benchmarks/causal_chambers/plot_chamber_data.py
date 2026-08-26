"""
plot_chamber_data.py — Visual audit of the Causal Chambers benchmark data.

Renders two figures from the canonical CSVs written by ``prepare_chamber_csv.py``:

``chamber_<name>.pdf/.png`` — a six-panel audit of ONE file, laid out to answer
the two questions that decide how its benchmark results may be read:

    Is there heterogeneity to estimate?   (a) tau histogram
                                          (b) Y0 vs Y1 -- level moves, gap doesn't
                                          (c) tau vs X -- is any of it predictable?
                                          (d) tau vs U -- not even by the confounder?
    Is the confounding removable?         (e) U by treatment arm  (U -> A)
                                          (f) overlap: e(U,Z) = P(A=1|U,Z).
                                              Mass at 0/1 means the effect is not identified
                                              there, however good the estimator.
                                          (g) Z vs U -- treatment-proxy relevance
                                          (h) W vs U -- outcome-proxy relevance

``chamber_<name>_ite_signal.pdf/.png`` — a four-panel diagnostic of the
realized individual treatment effect, ``ITE = Y1 - Y0``, against each observed
block (X, W, Z) and the latent U.  It makes any signal available to a CATE
learner visually explicit, rather than inferring it from the outcome-level
plots in the main audit.

``chamber_comparison.pdf/.png`` — every file side by side on the three numbers
that matter: the unadjusted confounding bias (what a method must remove),
SD(tau)/ATE (whether CATE metrics mean anything at all), and the share of rows
with e(U,Z) outside [0.05, 0.95] — where the effect is not identified at all.

Style comes from ``analysis_prior.design`` so the block colours match the paper:
red = latent U, blue = X, green = W, orange = Z. Treatment arms use the
Okabe--Ito blue/vermillion pair (validated: CVD dE 21.9, normal-vision 31.2).

Usage
─────
    python benchmarks/causal_chambers/plot_chamber_data.py            # all files
    python benchmarks/causal_chambers/plot_chamber_data.py --file strong_confounding_linear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis_prior.design import (  # noqa: E402
    OKABE_ITO,
    ROLES,
    apply_style,
    block_color,
    figsize_grid,
)
from analysis_prior.plotting import plt, save_figure  # noqa: E402

PREPARED_DIR = Path(__file__).resolve().parent / "data" / "prepared"
RESULTS_DIR = Path(__file__).resolve().parent / "figures"

# Treatment arms. Two hues rather than a grey baseline so both arms read as
# real categories; validated all-pairs in light mode.
ARM_COLORS = {0: OKABE_ITO["blue"], 1: OKABE_ITO["vermillion"]}
ARM_LABELS = {0: "control (A=0)", 1: "treated (A=1)"}

SCATTER = dict(s=7, alpha=0.45, linewidths=0)

#: Annotations sit on top of data, so give them an opaque plate to read against.
NOTE = dict(
    xycoords="axes fraction",
    va="top",
    fontsize="x-small",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
)


def _panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize="small")


def _note(ax, text: str) -> None:
    ax.annotate(text, xy=(0.04, 0.95), **NOTE)


def _arms_by_size(treated: np.ndarray) -> list[int]:
    """Arm codes, largest first — so the smaller arm is never drawn under it."""
    return sorted((0, 1), key=lambda a: -int((treated if a else ~treated).sum()))


def _binned_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 12):
    """Equal-count binned mean of y against x, for a trend overlay."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, len(xs), n_bins + 1).astype(int)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            bx.append(xs[lo:hi].mean())
            by.append(ys[lo:hi].mean())
    return np.array(bx), np.array(by)


#: Rows outside this band have a treatment that is effectively deterministic
#: given U: no estimator recovers the effect there without extrapolating.
POSITIVITY_CUT = 0.05


def _propensity(df: pd.DataFrame, treated: np.ndarray) -> np.ndarray:
    """Fitted e(U, Z) = P(A=1 | U, Z), by logistic regression.

    Conditioning on Z as well as U is not a refinement of the same quantity, it
    is a different one, and it is the one that matters here: in the chamber the
    treatment is a hard threshold on ``ir_2``, which Z (``led_2_uv``) drives
    directly, so a row can look perfectly overlapped in U alone and still have a
    deterministic treatment once Z is held fixed. A proximal estimator sees Z,
    so this is the overlap it actually faces.

    Columns are standardized before the fit: U (0-255) and Z (0-4095) live on
    scales two orders apart, and the default L2 penalty is not scale-free, so
    the raw fit would shrink Z's coefficient for a reason that is pure units.
    """
    f = df[[c for c in ("U_1", "Z_1") if c in df.columns]].to_numpy(dtype=np.float64)
    f = (f - f.mean(0)) / np.where(f.std(0) > 0, f.std(0), 1.0)
    return LogisticRegression().fit(f, treated.astype(int)).predict_proba(f)[:, 1]


def _extreme_share(e_hat: np.ndarray, cut: float = POSITIVITY_CUT) -> float:
    """Fraction of rows with e in [0, cut] u [1 - cut, 1]."""
    return float(((e_hat < cut) | (e_hat > 1 - cut)).mean())


def plot_ite_signal(df: pd.DataFrame, name: str, outdir: Path) -> list[Path]:
    """Plot realized ITE against the first X, W, Z, and U coordinates.

    Prepared Causal Chambers files currently have one coordinate per block.
    Naming the coordinate explicitly keeps this diagnostic honest if a
    multi-coordinate mapping is later supplied to ``prepare_chamber_csv.py``.
    """
    tau = df["Y1"].to_numpy() - df["Y0"].to_numpy()
    ate = tau.mean()
    blocks = (
        ("X", "covariate"),
        ("W", "outcome proxy"),
        ("Z", "treatment proxy"),
        ("U", "latent confounder"),
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize_grid(
            4, 1, fraction=1.7, panel_aspect=0.95, extra_height_in=0.4
        ),
    )
    for ax, (block, description), letter in zip(axes, blocks, "abcd"):
        col = f"{block}_1"
        if col not in df:
            raise ValueError(f"{name}: ITE diagnostic requires column {col!r}")
        values = df[col].to_numpy()
        corr = np.corrcoef(values, tau)[0, 1]
        ax.scatter(values, tau, color=block_color(block), **SCATTER)
        ax.plot(*_binned_mean(values, tau), **ROLES["trend"])
        ax.axhline(ate, color="0.6", lw=0.8, ls="--")
        _note(ax, f"corr({block}, ITE) = {corr:+.3f}")
        _panel_label(ax, letter, rf"ITE vs {description} ${block}$")
        ax.set_xlabel(f"${col}$")
        ax.set_ylabel(r"ITE $= Y_1-Y_0$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    fig.suptitle(f"{name.replace('_', ' ')} — ITE signal", fontsize="medium")
    return save_figure(fig, outdir, f"chamber_{name}_ite_signal")


def plot_dataset(df: pd.DataFrame, name: str, outdir: Path) -> list[Path]:
    """Six-panel audit of one prepared chamber file."""
    A = df["A"].to_numpy()
    Y, Y0, Y1 = (df[c].to_numpy() for c in ("Y", "Y0", "Y1"))
    tau = Y1 - Y0
    ate, sd_tau = tau.mean(), tau.std()
    treated = A > 0.5

    fig, axes = plt.subplots(
        2,
        4,
        figsize=figsize_grid(
            4, 2, fraction=1.7, panel_aspect=0.95, extra_height_in=0.6
        ),
    )

    # (a) Is there heterogeneity at all? -------------------------------------
    ax = axes[0, 0]
    ax.hist(tau, bins=40, color=block_color("Y"), edgecolor="white", linewidth=0.3)
    ax.axvline(ate, **ROLES["trend"])
    _note(ax, f"ATE {ate:,.0f}\nSD {sd_tau:,.0f}  ({sd_tau/ate:.1%} of ATE)")
    _panel_label(ax, "a", r"effect $\tau=Y_1-Y_0$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("count")

    # (b) Where does the variation live? -------------------------------------
    ax = axes[0, 1]
    # Single series: with corr(Y0,Y1) ~ 0.99 the points collapse onto a line, so
    # colouring by arm would just paint whichever arm is drawn last over the
    # other. Panel (e) carries the arm contrast instead.
    ax.scatter(Y0, Y1, color=block_color("Y"), **SCATTER)
    lo, hi = Y0.min(), Y0.max()
    # A straight reference line, so drop the marker the "theory" role carries.
    ax.plot(
        [lo, hi],
        [lo + ate, hi + ate],
        label=r"$Y_1=Y_0+\mathrm{ATE}$",
        **{**ROLES["theory"], "markersize": 0},
    )
    _note(ax, f"corr = {np.corrcoef(Y0, Y1)[0,1]:.3f}")
    _panel_label(ax, "b", "level moves, gap does not")
    ax.set_xlabel(r"$Y_0$")
    ax.set_ylabel(r"$Y_1$")
    # Arms get one shared legend for the whole figure; only the reference line
    # is panel-specific, so that is all this legend carries.
    ax.legend(
        handles=[ax.lines[-1]], loc="lower right", fontsize="xx-small", framealpha=0.9
    )

    # (c) Is the residual variation predictable? -----------------------------
    ax = axes[0, 2]
    x1 = df["X_1"].to_numpy()
    ax.scatter(x1, tau, color=block_color("X"), **SCATTER)
    ax.plot(*_binned_mean(x1, tau), **ROLES["trend"])
    ax.axhline(ate, color="0.6", lw=0.8, ls="--")
    _note(ax, f"corr(X, $\\tau$) = {np.corrcoef(x1, tau)[0,1]:+.3f}")
    _panel_label(ax, "c", r"$\tau$ vs covariate $X$")
    ax.set_xlabel("$X_1$")
    ax.set_ylabel(r"$\tau$")

    # (d) ...not even by the confounder itself? ------------------------------
    ax = axes[0, 3]
    u = df["U_1"].to_numpy()
    ax.scatter(u, tau, color=block_color("U"), **SCATTER)
    ax.plot(*_binned_mean(u, tau), **ROLES["trend"])
    ax.axhline(ate, color="0.6", lw=0.8, ls="--")
    _note(ax, f"corr(U, $\\tau$) = {np.corrcoef(u, tau)[0,1]:+.3f}")
    _panel_label(ax, "d", r"$\tau$ vs latent $U$")
    ax.set_xlabel("$U_1$")
    ax.set_ylabel(r"$\tau$")

    # (e) Is there confounding? ----------------------------------------------
    ax = axes[1, 0]
    bins = np.histogram_bin_edges(u, bins=28)
    # Step outlines rather than filled bars: the overlap region of two alpha
    # fills reads as a third colour that means nothing.
    for a in (0, 1):
        m = treated if a else ~treated
        ax.hist(
            u[m],
            bins=bins,
            color=ARM_COLORS[a],
            histtype="step",
            linewidth=1.3,
            label=ARM_LABELS[a],
        )
    _note(ax, f"corr(U, A) = {np.corrcoef(u, A)[0,1]:+.3f}")
    _panel_label(ax, "e", r"$U$ by arm  ($U\to A$)")
    ax.set_xlabel("$U_1$")
    ax.set_ylabel("count")

    # (f) Is the effect even identified? -------------------------------------
    # Confounding you can adjust away requires both arms to be observed at the
    # same (U, Z). Where e is ~0 or ~1 the treatment is effectively deterministic
    # and NO estimator recovers the effect there without extrapolating.
    ax = axes[1, 1]
    e_hat = _propensity(df, treated)
    extreme = _extreme_share(e_hat)
    for a in (0, 1):
        m = treated if a else ~treated
        ax.hist(
            e_hat[m],
            bins=np.linspace(0, 1, 26),
            color=ARM_COLORS[a],
            histtype="step",
            linewidth=1.3,
        )
    for lo_, hi_ in ((0.0, POSITIVITY_CUT), (1 - POSITIVITY_CUT, 1.0)):
        ax.axvspan(lo_, hi_, color=OKABE_ITO["vermillion"], alpha=0.13, lw=0)
    _note(ax, f"{extreme:.0%} of rows in\nthe shaded zone")
    _panel_label(ax, "f", r"overlap: $\hat e(U, Z)$")
    ax.set_xlabel(r"$\hat e(U, Z)$")
    ax.set_ylabel("count")

    # (g, h) Are the proxies relevant to U? ----------------------------------
    for col, letter, blk, desc in (
        ("Z_1", "g", "Z", r"proxy $Z$ vs $U$"),
        ("W_1", "h", "W", r"proxy $W$ vs $U$"),
    ):
        ax = axes[1, 2] if letter == "g" else axes[1, 3]
        v = df[col].to_numpy()
        ax.scatter(u, v, color=block_color(blk), **SCATTER)
        ax.plot(*_binned_mean(u, v), **ROLES["trend"])
        _note(ax, f"corr = {np.corrcoef(u, v)[0,1]:+.3f}")
        _panel_label(ax, letter, desc)
        ax.set_xlabel("$U_1$")
        ax.set_ylabel(f"${col[0]}_1$")

    for ax in axes.ravel():
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    fig.suptitle(name.replace("_", " "), fontsize="medium")
    # "outside" lets the constrained layout engine reserve a strip for the
    # legend instead of dropping it on top of the bottom row's axis labels.
    # Proxy handles so the swatches are opaque, unlike the alpha-blended marks.
    fig.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=6, color=ARM_COLORS[a]
            )
            for a in (0, 1)
        ],
        labels=[ARM_LABELS[0], ARM_LABELS[1]],
        loc="outside lower center",
        ncol=2,
        fontsize="small",
        frameon=False,
    )
    return save_figure(fig, outdir, f"chamber_{name}")


def plot_comparison(frames: dict[str, pd.DataFrame], outdir: Path) -> list[Path]:
    """Every file on the three numbers that decide how to read its results."""
    rows = []
    for name, df in frames.items():
        A = df["A"].to_numpy() > 0.5
        Y, Y0, Y1 = (df[c].to_numpy() for c in ("Y", "Y0", "Y1"))
        tau = Y1 - Y0
        ate = tau.mean()
        naive = Y[A].mean() - Y[~A].mean()
        # Same e(U, Z) fit as panel (f) of the per-file audit, so the two figures
        # cannot disagree about a file.
        e_hat = _propensity(df, A)
        rows.append(
            {
                "name": (
                    name.replace("_confounding", "")
                    .replace("_proxy_", " / ")
                    .replace("_", " ")
                ),
                "bias_pct": 100 * (naive - ate) / ate,
                "sd_over_ate_pct": 100 * tau.std() / ate,
                "extreme_pct": 100 * _extreme_share(e_hat),
            }
        )
    d = pd.DataFrame(rows).sort_values("bias_pct")
    y = np.arange(len(d))

    fig, axes = plt.subplots(
        1, 3, figsize=figsize_grid(3, 1, panel_aspect=0.72, extra_height_in=0.3)
    )

    ax = axes[0]
    # Diverging about zero: sign is the whole point (over- vs under-estimate).
    colors = [
        OKABE_ITO["vermillion"] if v > 0 else OKABE_ITO["blue"] for v in d.bias_pct
    ]
    ax.barh(y, d.bias_pct, color=colors, height=0.6)
    ax.axvline(0, color="0.3", lw=1)
    for yi, v in zip(y, d.bias_pct):
        ax.annotate(
            f"{v:+.1f}%",
            xy=(v, yi),
            xytext=(4 if v > 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize="x-small",
        )
    ax.set_yticks(y, d.name, fontsize="x-small")
    ax.set_xlabel("naive ATE bias (% of true ATE)")
    _panel_label(ax, "a", "confounding to remove")
    ax.margins(x=0.22)

    ax = axes[1]
    ax.barh(y, d.sd_over_ate_pct, color=block_color("Y"), height=0.6)
    ax.axvline(5, **ROLES["reference"])
    ax.annotate(
        "5% — below this,\nCATE metrics are\nnoise-normalised",
        xy=(5, len(d) - 0.6),
        xytext=(6, 0),
        textcoords="offset points",
        va="top",
        fontsize="xx-small",
        color=OKABE_ITO["vermillion"],
    )
    for yi, v in zip(y, d.sd_over_ate_pct):
        ax.annotate(
            f"{v:.1f}%",
            xy=(v, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize="x-small",
        )
    ax.set_yticks(y, [""] * len(d))
    ax.set_xlim(0, 8)
    ax.set_xlabel(r"SD($\tau$) / ATE  (%)")
    _panel_label(ax, "b", "heterogeneity to estimate")

    # (c) Is it identified at all? -------------------------------------------
    # Bias and heterogeneity say what there is to estimate; this says where no
    # estimator can. Rows with e pinned at 0 or 1 contribute an
    # extrapolation, not an estimate, however good the method.
    ax = axes[2]
    ax.barh(y, d.extreme_pct, color=block_color("A"), height=0.6)
    for yi, v in zip(y, d.extreme_pct):
        ax.annotate(
            f"{v:.1f}%",
            xy=(v, yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize="x-small",
        )
    ax.set_yticks(y, [""] * len(d))
    # A common scale across files is the point of the panel, but a run where
    # every file is well overlapped should not magnify a 0.2% difference.
    ax.set_xlim(0, max(10.0, 1.15 * d.extreme_pct.max()))
    ax.set_xlabel(
        rf"rows with $\hat e(U, Z)$ outside"
        "\n"
        rf"$[{POSITIVITY_CUT:g},\,{1 - POSITIVITY_CUT:g}]$  (%)"
    )
    _panel_label(ax, "c", "positivity violations")

    return save_figure(fig, outdir, "chamber_comparison")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize the Causal Chambers data.")
    p.add_argument("--file", help="Stem of one prepared CSV (default: all).")
    p.add_argument("--prepared-dir", default=str(PREPARED_DIR))
    p.add_argument("--outdir", default=str(RESULTS_DIR))
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    paths = (
        [prepared / f"{args.file}.csv"] if args.file else sorted(prepared.glob("*.csv"))
    )
    if not paths or not all(q.exists() for q in paths):
        raise SystemExit(
            f"No prepared CSVs in {prepared}. "
            "Run: python benchmarks/causal_chambers/prepare_chamber_csv.py --all"
        )

    apply_style()
    outdir = Path(args.outdir)
    frames = {q.stem: pd.read_csv(q) for q in paths}

    for name, df in frames.items():
        for out in plot_dataset(df, name, outdir):
            print(f"  ✓ {out}")
        for out in plot_ite_signal(df, name, outdir):
            print(f"  ✓ {out}")
    if len(frames) > 1:
        for out in plot_comparison(frames, outdir):
            print(f"  ✓ {out}")


if __name__ == "__main__":
    main()
