"""Sanity check: one prior dataset, one query row, its CATE posterior against context size.

    uv run python visu/visu_posterior_vs_context.py <ckpt>
    uv run python visu/visu_posterior_vs_context.py <ckpt_dir>            # latest step-*.ckpt
    uv run python visu/visu_posterior_vs_context.py <ckpt> --sizes 10,50,100,500,3000
    uv run python visu/visu_posterior_vs_context.py <ckpt> --n_ctx 3000 --seed 7

The companion of ``visu_posterior_evolution.py``: that one holds the data fixed
and sweeps *checkpoints*, this one holds the checkpoint fixed and sweeps the
amount of context the model is shown. One dataset of ``--n_ctx`` context rows is
generated once, and the model is then run on nested prefixes of it (10, 50, 100,
...) with the same single query row appended each time -- so every point on the
x-axis sees a strict subset of the data the point to its right saw.

Main panel: the predictive density over ``tau`` drawn as a violin at each
context size (both halves are the same density, scaled to a fixed pixel width),
with the posterior mean as a dot and +/- 1 sd as a bar. The true CATE of that
row is a dashed red line; the naive (confounded) observed contrast,
``mean(Y|A=1) - mean(Y|A=0)`` computed **on that same prefix**, is a dotted grey
curve -- it moves with ``n``, which is the point: it converges to the wrong
number while the posterior should contract onto the red line. Right panel: the
posterior sd against ``n`` on log-log axes, the contraction rate.

The x-axis is categorical (equally spaced), not a log scale, so violins keep a
constant width regardless of how the sizes are spaced.

Only the single-head losses (``cate_single*``, ``cate_decomposition``) have a
posterior over ``tau``; ``bins_po``/``hl`` are refused outright, since two
per-arm marginals do not identify the distribution of their difference.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from prior_eval.report import load_model
from proximalfm.prior import PriorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_episode(args, device: str):
    """One prior dataset with ``args.n_ctx`` context rows and ``--n_query`` query rows."""
    # the generator draws from all three
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prior = PriorDataset(
        batch_size=1,
        batch_size_per_gp=1,
        # min=None makes the context size exactly max_seq_len
        min_seq_len=None,
        max_seq_len=args.n_ctx,
        n_query=args.n_query,
        prior_type="mlp_scm",
        mc_cate=args.mc_cate,
        device=device,
        # loky workers carry their own RNG state, so --seed only means
        # something in-process
        n_jobs=1,
    )
    batch = prior.get_batch(batch_size=1)
    X, W, Z, A, Y_obs = batch[1], batch[2], batch[3], batch[4], batch[6]
    MU0, MU1 = batch[9], batch[10]
    d_X, d_W, d_Z = batch[11], batch[13], batch[14]
    train_size = int(batch[16][0].item())

    return {
        "X": X,
        "W": W,
        "Z": Z,
        "A": A.unsqueeze(-1).float(),
        "Y_obs": Y_obs,
        "d_X": d_X,
        "d_W": d_W,
        "d_Z": d_Z,
        "true_cate": (MU1 - MU0)[0, train_size:],
        "train_size": train_size,
    }


def prefix_inputs(ep: dict, n: int) -> dict:
    """The first ``n`` context rows, with every query row appended unchanged."""
    c = ep["train_size"]

    def cut(t):
        return torch.cat([t[:, :n], t[:, c:]], dim=1)

    return {
        "X": cut(ep["X"]),
        "Z": cut(ep["Z"]),
        "W": cut(ep["W"]),
        "A": cut(ep["A"]),
        "y_train": ep["Y_obs"][:, :n],
    }


def naive_contrast(ep: dict, n: int) -> float:
    """``mean(Y|A=1) - mean(Y|A=0)`` over the first ``n`` context rows.

    Biased by construction under the prior's unobserved ``U`` -- it is here as
    the "do nothing" reference the posterior should be away from, in the
    direction of the truth. NaN when a prefix happens to hold only one arm.
    """
    a = ep["A"][0, :n, 0].float()
    y = ep["Y_obs"][0, :n].float()
    hi, lo = a > 0.5, a <= 0.5
    if not hi.any() or not lo.any():
        return float("nan")
    return float(y[hi].mean() - y[lo].mean())


def pool_bins(vals: np.ndarray, w: np.ndarray, n_target: int):
    """Pool a histogram down to ``n_target`` groups of adjacent bins.

    Probability mass adds, so this is exact for the CDF and every functional of
    it; only the within-group shape is thrown away, which is the high-frequency
    ripple the loss never constrains. Returns densities (mass / group width),
    since ``array_split`` leaves group sizes differing by one.
    """
    n = len(vals)
    dv0 = float(np.diff(vals).mean()) if n > 1 else 1.0
    if n_target <= 0 or n_target >= n:
        return vals, w / max(dv0, 1e-12)
    idx = np.array_split(np.arange(n), n_target)
    centres = np.array([vals[g].mean() for g in idx])
    dens = np.array([w[g].sum() / (len(g) * dv0) for g in idx])
    return centres, dens


def parse_sizes(tokens, n_ctx: int) -> list[int]:
    """``--sizes 10,50 100`` -> [10, 50, 100]; nothing given -> a log-spaced default."""
    if not tokens:
        sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, n_ctx]
    else:
        sizes = []
        for tok in tokens:
            for part in str(tok).replace(",", " ").split():
                sizes.append(int(part))
    kept = sorted({s for s in sizes if 2 <= s <= n_ctx})
    if not kept:
        raise SystemExit(f"no requested context size falls in [2, {n_ctx}]")
    return kept


def resolve_ckpt(path: Path) -> Path:
    """A file is taken as is; a directory means its highest-numbered step."""
    if path.is_file():
        return path
    by_step = {
        # unpadded legacy names parse the same way, so a mixed directory works
        int(p.stem.split("-")[1]): p
        for p in path.glob("step-*.ckpt")
    }
    if not by_step:
        raise SystemExit(f"No step-*.ckpt found in {path}")
    return by_step[max(by_step)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt", type=Path, help="a .ckpt file, or a dir of step-*.ckpt")
    p.add_argument("--row", type=int, default=0, help="query row to look at")
    p.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        metavar="N",
        help="context sizes, e.g. --sizes 10,50,100 500 (default: log-spaced to --n_ctx)",
    )
    p.add_argument(
        "--agg_bins",
        type=int,
        default=200,
        help="pool the head's bins to this many before drawing a violin (default: 200)",
    )
    p.add_argument("--n_ctx", type=int, default=3000)
    p.add_argument("--n_query", type=int, default=1)
    p.add_argument("--mc_cate", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--out", type=Path, default=REPO_ROOT / "visu" / "posterior_vs_context.png"
    )
    args = p.parse_args()

    if not 0 <= args.row < args.n_query:
        raise SystemExit(f"--row must be in [0, {args.n_query})")

    sizes = parse_sizes(args.sizes, args.n_ctx)
    ep = make_episode(args, args.device)
    true_cate = float(ep["true_cate"][args.row])
    print(
        f"context {ep['train_size']} rows, query row {args.row}, "
        f"true CATE {true_cate:.4f}"
    )

    model, step = load_model(resolve_ckpt(args.ckpt), device=args.device)
    if model.loss_type in ("bins_po", "hl"):
        raise SystemExit(
            f"loss_type={model.loss_type}: two per-arm marginals do not identify "
            "the posterior of their difference, so there is nothing to draw"
        )

    curves, means, sds, naive = [], [], [], []
    for n in sizes:
        inp = prefix_inputs(ep, n)
        with torch.inference_mode():
            out = model.predict(
                **inp,
                target="cate",
                output_type="distribution",
                d_X=ep["d_X"],
                d_W=ep["d_W"],
                d_Z=ep["d_Z"],
            )
        # the grid is shared across rows, so "values" may come back with a
        # broadcast row dimension of 1
        v = out["values"]
        vals = v[0, args.row if v.shape[1] > 1 else 0].float().cpu().numpy()
        w = out["weights"][0, args.row].float().cpu().numpy()
        mean = float((w * vals).sum())
        sd = float(np.sqrt(max((w * (vals - mean) ** 2).sum(), 0.0)))
        curves.append((vals, w))
        means.append(mean)
        sds.append(sd)
        naive.append(naive_contrast(ep, n))
        print(f"  n_ctx {n:>6d}  mean {mean:+.4f}  sd {sd:.4f}  naive {naive[-1]:+.4f}")

    means_a, sds_a, naive_a = np.array(means), np.array(sds), np.array(naive)
    xs = np.arange(len(sizes), dtype=float)

    fig, (ax, ax_sd) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.4, 1]}
    )
    cmap = plt.get_cmap("viridis")

    # Violins share one density scale so their *areas* stay comparable: a
    # posterior twice as tight is drawn twice as tall, not renormalised away.
    pooled = [pool_bins(vals, w, args.agg_bins) for vals, w in curves]
    dmax = max(float(d.max()) for _, d in pooled) or 1.0
    half_w = 0.42  # in x units, i.e. fraction of the gap between two sizes

    for i, (centres, dens) in enumerate(pooled):
        color = cmap(i / max(len(pooled) - 1, 1))
        off = half_w * dens / dmax
        ax.fill_betweenx(
            centres, xs[i] - off, xs[i] + off, color=color, alpha=0.65, lw=0
        )
        ax.plot([xs[i], xs[i]], [means[i] - sds[i], means[i] + sds[i]], color="k", lw=1)
        ax.plot([xs[i]], [means[i]], "o", color="k", ms=3.5, zorder=4)

    ax.plot(xs, means_a, "-", color="k", lw=1, alpha=0.5, label="posterior mean +/- sd")
    ax.axhline(true_cate, color="red", ls="--", lw=1.5, label="true CATE")
    ax.plot(
        xs,
        naive_a,
        ":",
        color="gray",
        lw=1.8,
        marker="s",
        ms=3.5,
        label="naive contrast on the prefix",
    )

    # Frame the largest-context posterior: at small n the density is near-flat
    # over the whole grid and any window covering it leaves everything else an
    # invisible spike. Widened only enough to keep the references on screen.
    s_last = sds_a[-1] if sds_a[-1] > 0 else abs(means_a[-1]) * 0.5 + 1e-3
    finite_naive = naive_a[np.isfinite(naive_a)]
    lo = min(means_a[-1] - 3 * s_last, true_cate, *finite_naive)
    hi = max(means_a[-1] + 3 * s_last, true_cate, *finite_naive)
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_xlabel("context size (rows)")
    ax.set_ylabel("tau")
    ax.set_title(f"CATE posterior vs context, query row {args.row}, step {step}")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="best")

    ax_sd.plot(sizes, sds_a, "o-", color="C0")
    ax_sd.set_xscale("log")
    ax_sd.set_yscale("log")
    ax_sd.set_xlabel("context size (rows)")
    ax_sd.set_ylabel("posterior sd")
    ax_sd.set_title("contraction")
    ax_sd.grid(alpha=0.3, which="both")

    fig.tight_layout()
    # --out may be a directory ("--out .") as well as a filename
    out = args.out / "posterior_vs_context.png" if args.out.is_dir() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()
