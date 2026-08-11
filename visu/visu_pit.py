# %%
"""PIT diagnostics for one checkpoint on one evaluation set.

The ``*/dist/*`` surface reports the PIT as two numbers (``pit_ks_pooled`` vs
its band, ``pit_ks/p90`` vs the per-episode band). Those decide *whether* the
posterior is calibrated; they say nothing about *how* it fails. This script
draws the shape behind them, using the same functions the trainer scores with
(``proximalfm.train._dist_metrics.pit`` / ``pit_ks`` and
``_dist_eval.episode_predictive``), so nothing here can drift from the logged
metrics.

Run standalone:

    uv run python visu/visu_pit.py --ckpt mc_run_6_TuneToOneHead/best_model.ckpt \\
        --dir data/prior_eval/frozen_prod_v1 --max-episodes 100

Panels:
  A  Pooled PIT histogram, with the uniform reference and a per-bin 95%
     binomial envelope. Shape reads the failure: U-shaped = posteriors too
     narrow, dome = too wide, tilted = biased.
  B  Pooled PIT ECDF *deviation* ``ECDF(u) - u`` with the KS band. Plotting the
     deviation rather than the ECDF is what makes a KS distance of 0.02 visible
     at all; the band is the decision line ``score`` prints.
  C  Per-episode KS statistics against the per-episode band, with p50/p90
     marked. The pooled statistic passing while this tail fails is
     opposite-direction miscalibration cancelling in the pool -- the failure
     ``pit_ks/p90`` exists to catch, and the reason panel A alone is not enough.
  D  Per-episode mean PIT vs per-episode KS: separates *bias* (mean off 0.5,
     horizontal spread) from *width* miscalibration (KS with mean ~0.5).
  E  Per-episode PIT histograms for a handful of episodes, picked at quantiles
     of the per-episode KS (best -> worst), so the panel shows the range rather
     than the extremes only.

Everything is scored in one pass and cached in memory (``PIT``), so the plotting
cell can be re-run without touching the model. ``--n-ctx`` scores at a shorter
context prefix (the ``sweep`` layout); note ``grid_scale`` < 1 there, so widths
at small ``n`` are not comparable to widths at ``n_max``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from prior_eval.build_prior_frozen_set import (
    frozen_content_digest,
    load_frozen_meta,
)
from prior_eval.report import load_model
from proximalfm.train._dist_eval import episode_predictive, iter_frozen_episodes
from proximalfm.train._dist_metrics import (
    out_of_grid_fraction,
    pit,
    pit_ks,
    predictive_mean_var,
)

REPO_ROOT = (
    Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
)

# %%
# 1. Configuration
#
# Edit these when running cell-by-cell; the CLI at the bottom overrides them.
CKPT = REPO_ROOT / "mc_run_6_TuneToOneHead" / "best_model.ckpt"
SET_DIR = REPO_ROOT / "data" / "prior_eval" / "frozen_prod_v1"
MAX_EPISODES = 100  # None = the whole set (slow: ~0.3 s/episode on mps)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_CTX = None  # None = n_max (the full context; grid_scale == 1)
ALPHA = 0.05
NBINS_HIST = 25  # PIT histogram resolution; 25 bins x 128 rows/episode pools well
N_EXAMPLES = 6  # episodes shown individually in panel E
OUT_PNG = REPO_ROOT / "visu" / "pit.png"


# %%
# 2. Scoring pass: one forward per episode, PIT values kept per episode
def _load_meta(set_dir: Path, meta_path: Path | None = None) -> dict:
    """The sidecar for ``set_dir``, with an explicit escape hatch.

    ``score``/``verify`` require ``<root>/frozen_meta.pt`` and are right to:
    that adjacency is what pins a scale to a set. This script falls back to
    ``<root>/../frozen_meta.pt`` (where a couple of local copies of the
    production sets ended up) and *says so*, because the fallback is only safe
    when the parent holds one set -- with two, it silently scores episode i of
    one set against the tau_star of another. Pass ``--meta`` to be explicit.
    """
    set_dir = Path(set_dir)
    if meta_path is not None:
        return torch.load(Path(meta_path), weights_only=False)
    if (set_dir / "frozen_meta.pt").exists():
        return load_frozen_meta(set_dir)
    parent = set_dir.parent / "frozen_meta.pt"
    if parent.exists():
        print(
            f"! no frozen_meta.pt in {set_dir}; using {parent}. "
            "Verify it describes THIS set (episode count and n_max below) -- "
            "a sidecar from a sibling set would score against the wrong "
            "tau_star without erroring."
        )
        return torch.load(parent, weights_only=False)
    raise FileNotFoundError(
        f"no frozen_meta.pt in {set_dir} or {set_dir.parent}; a set without "
        "its sidecar carries no tau_star and cannot be scored distributionally"
    )


def collect_pit(
    ckpt: Path,
    set_dir: Path,
    *,
    max_episodes: int | None = MAX_EPISODES,
    device: str = DEVICE,
    n_ctx: int | None = N_CTX,
    meta_path: Path | None = None,
) -> dict:
    """Score ``ckpt`` on the frozen set at ``set_dir``; return per-episode PIT.

    ``set_dir`` is the set's *root* (holding both ``val/`` and
    ``frozen_meta.pt``) for the same reason ``score_frozen_set`` insists on it:
    without the sidecar there is no frozen comparison scale and no
    ``tau_star``, and a PIT scored against sampled ITEs instead would be a
    picture of the noise floor rather than of the posterior.
    """
    set_dir = Path(set_dir)
    meta = _load_meta(set_dir, meta_path)
    n_max = int(meta["global"]["n_max"])
    n_ctx = n_max if n_ctx is None else int(n_ctx)
    if not 1 <= n_ctx <= n_max:
        raise ValueError(f"n_ctx={n_ctx} must be in [1, n_max={n_max}]")
    records = meta["episodes"]

    model, step = load_model(Path(ckpt), device=device)
    model.eval()

    per_episode: list[np.ndarray] = []
    oog_rows, n_rows = 0.0, 0
    z_scores: list[np.ndarray] = []  # (mean_P - tau*) / sd_P, for reference

    with torch.inference_mode():
        for i, batch in enumerate(iter_frozen_episodes(set_dir / "val", max_episodes)):
            record = records[i]
            probs, edges = episode_predictive(
                model, batch, record, n_ctx=n_ctx, n_max=n_max, device=device
            )
            tau_star = record["tau_star_std"].to(probs.device).float().reshape(-1)

            u = pit(probs, edges, tau_star).flatten().float().cpu()
            per_episode.append(u.numpy())

            mean, var = predictive_mean_var(probs, edges)
            sd = var.clamp_min(1e-12).sqrt()
            z_scores.append(((mean - tau_star) / sd).float().cpu().numpy())

            oog_rows += out_of_grid_fraction(edges, tau_star) * tau_star.numel()
            n_rows += int(tau_star.numel())

    return {
        "pit": per_episode,
        "z": z_scores,
        "ckpt": str(ckpt),
        "step": step,
        "set_dir": str(set_dir),
        "digest": frozen_content_digest(meta),
        "set_kind": meta["global"].get("set_kind", "prior"),
        "n_ctx": n_ctx,
        "n_max": n_max,
        "out_of_grid_frac": oog_rows / max(n_rows, 1),
        "n_rows": n_rows,
    }


def summarize(pit_data: dict, alpha: float = ALPHA) -> dict:
    """Pooled and per-episode KS, matching ``PosteriorEvaluator.compute``."""
    per_episode = pit_data["pit"]
    pooled = torch.tensor(np.concatenate(per_episode), dtype=torch.float64)
    stat, band = pit_ks(pooled, alpha=alpha)

    ep_stat, ep_band = [], []
    for u in per_episode:
        if u.size < 2:
            continue
        s, b = pit_ks(torch.tensor(u, dtype=torch.float64), alpha=alpha)
        ep_stat.append(float(s))
        ep_band.append(float(b))

    ep_stat_t = torch.tensor(ep_stat, dtype=torch.float64)
    return {
        "pooled": pooled.numpy(),
        "pooled_ks": float(stat),
        "pooled_band": float(band),
        "episode_ks": np.asarray(ep_stat),
        "episode_band": float(np.median(ep_band)) if ep_band else float("nan"),
        "episode_ks_p50": float(ep_stat_t.median()),
        "episode_ks_p90": float(
            ep_stat_t.quantile(torch.tensor(0.9, dtype=torch.float64))
        ),
        "episode_mean_pit": np.asarray([float(u.mean()) for u in per_episode]),
    }


# %%
# 3. Plotting
def _ecdf_deviation(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(grid, ECDF(grid) - grid)`` on the sorted sample itself."""
    s = np.sort(u)
    n = s.size
    return s, np.arange(1, n + 1) / n - s


def plot_pit(pit_data: dict, summary: dict, *, alpha: float = ALPHA):
    per_episode = pit_data["pit"]
    pooled = summary["pooled"]

    fig = plt.figure(figsize=(15, 10.5))
    gs = GridSpec(2, 3, figure=fig, top=0.88, bottom=0.38, hspace=0.55, wspace=0.28)

    # --- A: pooled PIT histogram -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    counts, edges = np.histogram(pooled, bins=NBINS_HIST, range=(0.0, 1.0))
    dens = counts / counts.sum() * NBINS_HIST
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, dens, width=1.0 / NBINS_HIST, color="#4C72B0", alpha=0.85)
    ax.axhline(1.0, color="k", lw=1.2, label="uniform")
    # Per-bin 95% envelope: each bin count is Binomial(n, 1/B) under uniformity.
    n = pooled.size
    p = 1.0 / NBINS_HIST
    se = math.sqrt(p * (1 - p) / n) * NBINS_HIST
    ax.fill_between([0, 1], 1 - 1.96 * se, 1 + 1.96 * se, color="k", alpha=0.12, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1.6, dens.max() * 1.15))
    ax.set_xlabel("PIT  $u = F(\\tau^\\star)$")
    ax.set_ylabel("density")
    ax.set_title(f"A. pooled PIT ({n:,} rows)\nU-shape = too narrow, dome = too wide")
    ax.legend(fontsize=8, loc="upper center")

    # --- B: pooled ECDF deviation vs KS band --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    grid, dev = _ecdf_deviation(pooled)
    ax.plot(grid, dev, color="#C44E52", lw=1.4)
    ax.axhline(0.0, color="k", lw=1.0)
    band = summary["pooled_band"]
    ax.fill_between(
        [0, 1],
        -band,
        band,
        color="k",
        alpha=0.12,
        lw=0,
        label=f"KS band ($\\alpha$={alpha})",
    )
    ks = summary["pooled_ks"]
    ax.axhline(ks, color="#C44E52", ls=":", lw=1.0)
    ax.axhline(-ks, color="#C44E52", ls=":", lw=1.0)
    ax.set_xlim(0, 1)
    lim = max(band, ks) * 1.6
    ax.set_ylim(-lim, lim)
    verdict = "inside" if ks < band else "OUTSIDE"
    ax.set_xlabel("u")
    ax.set_ylabel("ECDF(u) $-$ u")
    ax.set_title(f"B. pooled ECDF deviation\nKS {ks:.4f} vs band {band:.4f}: {verdict}")
    ax.legend(fontsize=8, loc="upper right")

    # --- C: per-episode KS --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    eks = summary["episode_ks"]
    ax.hist(eks, bins=30, color="#55A868", alpha=0.85)
    eband = summary["episode_band"]
    ax.axvline(eband, color="k", lw=1.4, label=f"episode band {eband:.3f}")
    ax.axvline(
        summary["episode_ks_p50"],
        color="#4C72B0",
        ls="--",
        lw=1.2,
        label=f"p50 {summary['episode_ks_p50']:.3f}",
    )
    ax.axvline(
        summary["episode_ks_p90"],
        color="#C44E52",
        ls="--",
        lw=1.2,
        label=f"p90 {summary['episode_ks_p90']:.3f}",
    )
    frac_out = float((eks >= eband).mean())
    ax.set_xlabel("per-episode KS")
    ax.set_ylabel("episodes")
    ax.set_title(
        f"C. per-episode KS ({len(eks)} episodes)\n{frac_out:.0%} outside their band"
    )
    ax.legend(fontsize=8)

    # --- D: bias vs width, per episode --------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    mean_pit = summary["episode_mean_pit"]
    sc = ax.scatter(mean_pit, eks, c=eks, cmap="viridis", s=18, alpha=0.85)
    ax.axvline(0.5, color="k", lw=1.0)
    ax.axhline(eband, color="k", ls="--", lw=1.0, label="episode band")
    ax.set_xlabel("episode mean PIT   (0.5 = unbiased)")
    ax.set_ylabel("episode KS")
    ax.set_title("D. bias (x) vs miscalibration (y)")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    # --- D2: pooled z-scores, the width story in one axis --------------------
    ax = fig.add_subplot(gs[1, 1])
    z = np.concatenate(pit_data["z"])
    zclip = np.clip(z, -5, 5)
    ax.hist(
        zclip,
        bins=60,
        range=(-5, 5),
        density=True,
        color="#8172B2",
        alpha=0.85,
        label="$(\\mu_P - \\tau^\\star)/\\mathrm{sd}_P$",
    )
    xs = np.linspace(-5, 5, 400)
    ax.plot(
        xs, np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi), "k", lw=1.2, label="N(0,1)"
    )
    ax.set_xlabel("standardized error (clipped to $\\pm 5$)")
    ax.set_ylabel("density")
    ax.set_title(
        f"D2. z-scores: sd(z) = {z.std():.3f}\n>1 = over-confident, <1 = too wide"
    )
    ax.legend(fontsize=8)

    # --- summary text --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    oog = pit_data["out_of_grid_frac"]
    lines = [
        f"checkpoint  {Path(pit_data['ckpt']).parent.name}/{Path(pit_data['ckpt']).name}",
        f"step        {pit_data['step']}",
        f"set         {Path(pit_data['set_dir']).name}  ({pit_data['set_kind']})",
        f"digest      {pit_data['digest']}",
        f"n_ctx       {pit_data['n_ctx']} / n_max {pit_data['n_max']}",
        f"episodes    {len(per_episode)}   rows {pit_data['n_rows']:,}",
        "",
        f"out_of_grid_frac  {oog:.5f}",
        f"pit_ks_pooled     {ks:.5f}  (band {band:.5f})",
        f"pit_ks/p50        {summary['episode_ks_p50']:.5f}",
        f"pit_ks/p90        {summary['episode_ks_p90']:.5f}  (band {eband:.5f})",
        f"mean PIT          {pooled.mean():.5f}  (0.5)",
        f"sd(z)             {z.std():.5f}  (1.0)",
    ]
    if oog > 1e-3:
        lines += [
            "",
            "! out_of_grid_frac > 1e-3: the CATE grid is too tight;",
            "  targets pile on PIT 0/1 and every panel above is suspect.",
        ]
    if ks < band <= summary["episode_ks_p90"]:
        lines += [
            "",
            "! pooled passes but the per-episode p90 does not:",
            "  opposite-direction errors cancelling in the pool.",
        ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax.transAxes,
    )

    # --- E: per-episode PIT histograms at KS quantiles -----------------------
    order = np.argsort(eks)
    picks = np.unique(np.linspace(0, len(order) - 1, N_EXAMPLES).round().astype(int))
    # Own GridSpec with explicit bounds: the row needs a shared y-scale and a
    # heading of its own, neither of which the main grid can give it.
    sub = GridSpec(1, len(picks), figure=fig, top=0.26, bottom=0.06, wspace=0.12)
    hists = [
        np.histogram(per_episode[int(order[k])], bins=10, range=(0, 1), density=True)[0]
        for k in picks
    ]
    ymax = max(2.5, max(float(h.max()) for h in hists) * 1.1)
    for j, k in enumerate(picks):
        idx = int(order[k])
        ax = fig.add_subplot(sub[0, j])
        u = per_episode[idx]
        ax.hist(u, bins=10, range=(0, 1), density=True, color="#DD8452", alpha=0.9)
        ax.axhline(1.0, color="k", lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_title(f"ep {idx}  KS {eks[idx]:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("density", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("u", fontsize=8)
    fig.text(
        0.5,
        0.30,
        "E. per-episode PIT, episodes at KS quantiles (best → worst)",
        ha="center",
        fontsize=11,
    )

    fig.suptitle(
        f"PIT diagnostics — {Path(pit_data['ckpt']).parent.name} "
        f"step {pit_data['step']} on {Path(pit_data['set_dir']).name}",
        fontsize=13,
    )
    return fig


# %%
# 4. Run
def main(argv=None) -> int:
    global MAX_EPISODES, DEVICE, N_CTX, ALPHA, NBINS_HIST, N_EXAMPLES

    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=Path, default=CKPT)
    p.add_argument(
        "--dir",
        type=Path,
        default=SET_DIR,
        help="frozen set root (holding val/ and frozen_meta.pt)",
    )
    p.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--n-ctx", type=int, default=N_CTX)
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="explicit frozen_meta.pt (default: <dir>/frozen_meta.pt)",
    )
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--hist-bins", type=int, default=NBINS_HIST)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--out", type=Path, default=OUT_PNG)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)

    ALPHA, NBINS_HIST, N_EXAMPLES = args.alpha, args.hist_bins, args.n_examples

    data = collect_pit(
        args.ckpt,
        args.dir,
        max_episodes=args.max_episodes,
        device=args.device,
        n_ctx=args.n_ctx,
        meta_path=args.meta,
    )
    summary = summarize(data, alpha=args.alpha)
    fig = plot_pit(data, summary, alpha=args.alpha)
    if args.out is not None:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    # Cell-by-cell use: score once, then re-run the plotting cell freely.
    PIT = None  # set by: PIT = collect_pit(CKPT, SET_DIR)
