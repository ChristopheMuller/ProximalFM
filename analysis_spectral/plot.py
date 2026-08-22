"""One episode, end to end: raw joint -> categories -> matrix -> metric.

    uv run python -m analysis_spectral.plot --index 1

Five panels, laid out two-up so the discretization can be read against the data
it came from.  By default only the ``A = 0`` cell is drawn (``--a 0 1`` shows
both): the two cells tell the same story on this prior, and one of them is
worth the width.

1. **Marginals.**  The raw ``W`` and ``Z`` samples with the quantile bin edges
   drawn on top -- the only panel where the discretization is visible as a
   choice rather than as a fait accompli.
2. **The raw joint.**  A 2-D histogram of ``(W, Z)`` *before* any binning, with
   the same edges overlaid as a grid.  This is the ground truth the rest of the
   figure approximates: whatever dependence the operator can transmit has to be
   visible here first, and the grid shows how much of it the chosen ``L``
   survives.  A round blob is independence; a tilted or curved ridge is signal.
3. **Categories.**  What the edges produced.  Equal-frequency binning makes
   these bars flat *by construction over the whole episode*; departures from
   flat here are the cell's own selection, i.e. how much ``A`` moves the proxy.
4. **The matrix.**  The same joint after binning, divided by the independence
   reference -- panel 2 quantized and normalized.
5. **The metric.**  The spectrum against its own permutation null.  Points
   inside the shaded band are indistinguishable from noise at this ``n`` and
   ``L``, and the summary numbers are printed beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm, TwoSlopeNorm  # noqa: E402

from analysis_spectral._load import SpectralEpisode, load_episodes  # noqa: E402
from analysis_spectral.decay import (  # noqa: E402
    CellSpectrum,
    cell_spectrum,
    episode_summary,
)
from analysis_spectral.matrices import (  # noqa: E402
    CellMatrix,
    _proxy_labels,
    _x_cells,
    normalized_matrix,
)

W_COLOR = "#1f77b4"
Z_COLOR = "#ff7f0e"
CELL_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
#: Resolution of the raw joint histogram. Far finer than any usable ``L`` on
#: purpose -- panel 2 is meant to show what the binning throws away.
JOINT_BINS = 64


def _density(ax, v: np.ndarray, edges: np.ndarray, color: str, label: str) -> None:
    """Sample density with the bin edges overlaid.

    A KDE needs a non-degenerate sample; the prior emits categorical proxies, so
    a few-valued column is drawn as a stem plot instead of being forced through
    a smoother that would either fail or invent a shape it does not have.
    """
    uniq = np.unique(v)
    if len(uniq) <= 12:
        counts = np.array([(v == u).sum() for u in uniq], dtype=float)
        ax.stem(
            uniq,
            counts / counts.sum(),
            linefmt=color,
            markerfmt="o",
            basefmt=" ",
            label=f"{label} (discrete, {len(uniq)} levels)",
        )
    else:
        from scipy.stats import gaussian_kde

        grid = np.linspace(v.min(), v.max(), 256)
        dens = gaussian_kde(v)(grid)
        ax.plot(grid, dens, color=color, lw=1.6, label=label)
        ax.fill_between(grid, dens, color=color, alpha=0.15)
    for e in edges:
        ax.axvline(e, color=color, ls=":", lw=0.7, alpha=0.6)


def _hist_bins(v: np.ndarray, max_bins: int = JOINT_BINS) -> np.ndarray:
    """Histogram edges for one axis, collapsing to the levels of a discrete column.

    Spraying 64 bins over a 5-valued proxy produces stripes of empty columns
    that read as structure; one bin per level reads as what it is.
    """
    uniq = np.unique(v)
    if len(uniq) <= 20:
        step = np.diff(uniq).min() if len(uniq) > 1 else 1.0
        return np.append(uniq - step / 2, uniq[-1] + step / 2)
    return np.linspace(v.min(), v.max(), max_bins + 1)


def _draw_joint(
    ax, w: np.ndarray, z: np.ndarray, w_edges: np.ndarray, z_edges: np.ndarray
) -> None:
    """2-D histogram of the raw ``(W, Z)`` sample, with the bin edges as a grid.

    Counts are shown on a square-root colour scale.  A joint density is usually
    dominated by its mode, and on a linear scale the tails -- which is where a
    dependence structure is actually legible -- render as uniform dark.  The
    scale is stated on the colourbar because it is a real distortion, not a
    neutral default.
    """
    counts, xe, ye = np.histogram2d(w, z, bins=[_hist_bins(w), _hist_bins(z)])
    im = ax.pcolormesh(
        xe,
        ye,
        counts.T,
        cmap="magma",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=counts.max()),
        shading="flat",
    )
    for e in w_edges:
        ax.axvline(e, color="#66ccff", ls=":", lw=0.8, alpha=0.85)
    for e in z_edges:
        ax.axhline(e, color="#66ccff", ls=":", lw=0.8, alpha=0.85)
    ax.set_xlabel("W  (raw)")
    ax.set_ylabel("Z  (raw)")
    ax.set_title(
        "Raw joint of (W, Z) — dotted lines are the quantile bin edges",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, label="count (sqrt colour scale)")


def _prepare(
    ep: SpectralEpisode,
    *,
    n_bins: int,
    x_bins: int,
    n_perm: int,
    quantile: float,
    min_cell_rows: Optional[int],
    seed: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Per-cell labels, matrix and spectrum -- everything the figure draws."""
    if min_cell_rows is None:
        min_cell_rows = 2 * n_bins**2
    rng = np.random.default_rng(seed)

    w_lab, w_edges = _proxy_labels(ep.W, n_bins)
    z_lab, z_edges = _proxy_labels(ep.Z, n_bins)
    n_w, n_z = len(w_edges) + 1, len(z_edges) + 1
    x_cell, _ = _x_cells(ep.X, x_bins)

    cells = []
    for a in np.unique(ep.A):
        for c in np.unique(x_cell):
            sel = (ep.A == a) & (x_cell == c)
            if int(sel.sum()) < min_cell_rows:
                continue
            M, joint, p_w, p_z, w_kept, z_kept = normalized_matrix(
                w_lab[sel], z_lab[sel], n_w=n_w, n_z=n_z
            )
            cm = CellMatrix(
                a=int(a),
                x_cell=int(c),
                n=int(sel.sum()),
                M=M,
                joint=joint,
                p_w=p_w,
                p_z=p_z,
                w_categories=w_kept,
                z_categories=z_kept,
            )
            spec = cell_spectrum(
                cm,
                w_lab[sel],
                z_lab[sel],
                n_w=n_w,
                n_z=n_z,
                n_perm=n_perm,
                quantile=quantile,
                rng=rng,
            )
            cells.append({"sel": sel, "cm": cm, "spec": spec})
    return cells, w_edges, z_edges


def _draw_matrix(ax, cm: CellMatrix, kind: str = "lift") -> None:
    """The matrix panel, in one of two units.

    ``M`` itself is the object whose singular values the whole analysis is
    about, but it is a poor thing to *look* at: under independence its entries
    are ``sqrt(p_w p_z)``, i.e. ``1/L`` for equal-frequency bins, so a diverging
    scale around any fixed centre reads as "uniformly low" and the structure is
    invisible.  ``kind="lift"`` therefore divides the reference out and shows
    ``P(w,z) / (P(w) P(z))``, where 1 *is* independence, red is co-occurrence
    and blue is repulsion.  The two differ by the fixed positive factor
    ``sqrt(p_w p_z)``, so the visible structure is the same one the SVD sees.
    """
    if kind == "lift":
        M = cm.M / np.sqrt(np.outer(cm.p_w, cm.p_z))
        label = "$P(w,z)\\,/\\,P(w)P(z)$   (1 = independence)"
        span = float(np.abs(M - 1.0).max())
        norm = (
            TwoSlopeNorm(vmin=1 - span, vcenter=1.0, vmax=1 + span)
            if span > 0
            else None
        )
        cmap = "RdBu_r"
    else:
        M = cm.M
        label = "$M$   (independence: $\\sqrt{p_w p_z}$)"
        norm, cmap = None, "viridis"
    im = ax.imshow(M.T, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_xlabel("W category")
    ax.set_ylabel("Z category")
    ax.set_xticks(range(M.shape[0]), cm.w_categories, fontsize=6)
    ax.set_yticks(range(M.shape[1]), cm.z_categories, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, label=label)


def _draw_spectrum(ax, cells: list[dict]) -> None:
    for k, cell in enumerate(cells):
        spec: CellSpectrum = cell["spec"]
        color = CELL_COLORS[k % len(CELL_COLORS)]
        sig = spec.sigma[1:]
        if len(sig) == 0:
            continue
        j = np.arange(1, len(sig) + 1)
        ax.fill_between(
            j,
            spec.null_mean[1:],
            spec.null_q[1:],
            color=color,
            alpha=0.18,
            lw=0,
            label=f"a={spec.a} null (mean..q95)",
        )
        ax.plot(
            j,
            sig,
            "o-",
            color=color,
            lw=1.6,
            ms=5,
            label=f"a={spec.a}  n={spec.n}",
        )
        above = np.flatnonzero(spec.above_floor)
        if len(above):
            ax.plot(above + 1, sig[above], "o", mfc="none", mec="k", ms=10, mew=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("j  (index of the non-trivial spectrum; $\\sigma_1=1$ dropped)")
    ax.set_ylabel("$\\sigma_{j+1}$")
    ax.set_title(
        "Spectrum vs permutation null — ringed points are above the floor",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")


def plot_episode(
    ep: SpectralEpisode,
    *,
    n_bins: int = 8,
    x_bins: int = 1,
    n_perm: int = 200,
    quantile: float = 0.95,
    min_cell_rows: Optional[int] = None,
    seed: int = 0,
    matrix: str = "lift",
    a_values: Optional[Sequence[int]] = (0,),
) -> plt.Figure:
    """The figure for one episode; ``a_values=None`` draws every cell."""
    cells, w_edges, z_edges = _prepare(
        ep,
        n_bins=n_bins,
        x_bins=x_bins,
        n_perm=n_perm,
        quantile=quantile,
        min_cell_rows=min_cell_rows,
        seed=seed,
    )
    if not cells:
        raise ValueError(
            f"Episode {ep.index}: no conditioning cell has enough rows for an "
            f"{n_bins}x{n_bins} table. Lower --n_bins."
        )
    if a_values is not None:
        kept = [c for c in cells if c["cm"].a in set(a_values)]
        if not kept:
            raise ValueError(
                f"Episode {ep.index}: no cell with A in {sorted(set(a_values))}; "
                f"available: {sorted({c['cm'].a for c in cells})}."
            )
        cells = kept

    # Two panels per cell across, three rows down: marginals | raw joint,
    # categories | matrix, and the spectrum along the bottom.
    ncol = 2 * len(cells)
    fig = plt.figure(figsize=(5.8 * ncol, 13.0))
    gs = fig.add_gridspec(
        3, ncol, height_ratios=[1.0, 1.15, 0.9], hspace=0.32, wspace=0.28
    )

    for k, cell in enumerate(cells):
        sel, cm, spec = cell["sel"], cell["cm"], cell["spec"]
        c0 = 2 * k
        title = f"A = {cm.a}" + (f", X-cell {cm.x_cell}" if cm.x_cell else "")

        # 1 — marginals with the bin edges drawn on
        ax = fig.add_subplot(gs[0, c0])
        _density(ax, ep.W[sel, 0], w_edges, W_COLOR, "W")
        _density(ax, ep.Z[sel, 0], z_edges, Z_COLOR, "Z")
        ax.set_title(f"{title}   (n = {cm.n})", fontsize=11, weight="bold")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

        # 2 — the raw joint, before any binning
        ax = fig.add_subplot(gs[0, c0 + 1])
        _draw_joint(ax, ep.W[sel, 0], ep.Z[sel, 0], w_edges, z_edges)

        # 3 — realized categories
        ax = fig.add_subplot(gs[1, c0])
        width = 0.4
        ax.bar(
            cm.w_categories - width / 2, cm.p_w, width, color=W_COLOR, label="P(W=w)"
        )
        ax.bar(
            cm.z_categories + width / 2, cm.p_z, width, color=Z_COLOR, label="P(Z=z)"
        )
        ax.axhline(
            1 / n_bins, color="k", ls="--", lw=0.8, label=f"uniform (1/{n_bins})"
        )
        ax.set_xlabel("category")
        ax.set_ylabel("in-cell frequency")
        ax.set_ylim(0, max(cm.p_w.max(), cm.p_z.max()) * 1.45)
        ax.legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.9)

        # 4 — the matrix: panel 2, quantized and normalized
        ax = fig.add_subplot(gs[1, c0 + 1])
        _draw_matrix(ax, cm, kind=matrix)
        ax.set_title(
            f"Binned & normalized   $\\chi^2/n$ = {(spec.sigma**2).sum() - 1:.3f}   "
            f"$\\sigma_2$ = {spec.sigma2:.3f}",
            fontsize=9,
        )

    # 5 — the metric, across the full width
    ax = fig.add_subplot(gs[2, :])
    _draw_spectrum(ax, cells)

    summary = episode_summary([c["spec"] for c in cells])
    fig.suptitle(
        f"Episode {ep.index}   |   "
        f"$\\sigma_2$ = {summary['sigma2']:.3f}   "
        f"effective rank = {summary['eff_rank']:.2f}   "
        f"poly rate = {summary['poly_rate']:.2f}   "
        f"exp rate = {summary['exp_rate']:.2f}   "
        f"{'measurable' if summary['measurable'] else 'AT THE NOISE FLOOR'}",
        fontsize=13,
        y=0.995,
    )
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val_dir", default="data/prior_eval/prior_val_v2_smalldim_50k/val")
    p.add_argument("--index", type=int, nargs="+", default=[1], help="episode(s)")
    p.add_argument("--n_bins", type=int, default=8)
    p.add_argument("--x_bins", type=int, default=1)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument(
        "--a",
        type=int,
        nargs="+",
        default=[0],
        help="which treatment cell(s) to draw; -1 draws all of them",
    )
    p.add_argument(
        "--matrix",
        choices=("lift", "M"),
        default="lift",
        help="units of the matrix panel (see _draw_matrix)",
    )
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    args = p.parse_args()

    a_values = None if -1 in args.a else args.a

    eps = load_episodes(args.val_dir, limit=max(args.index) + 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in args.index:
        fig = plot_episode(
            eps[i],
            n_bins=args.n_bins,
            x_bins=args.x_bins,
            n_perm=args.n_perm,
            quantile=args.quantile,
            seed=i,
            matrix=args.matrix,
            a_values=a_values,
        )
        path = out_dir / f"episode_{i:04d}_L{args.n_bins}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
