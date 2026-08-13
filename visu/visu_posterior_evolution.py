"""Sanity check: one prior dataset, one query row, its CATE posterior over checkpoints.

    uv run python visu/visu_posterior_evolution.py mc_run_7_NoiseAwareTuning
    uv run python visu/visu_posterior_evolution.py <ckpt_dir> --row 3 --seed 1
    uv run python visu/visu_posterior_evolution.py <ckpt_dir> --steps 1,50,100,250

Deliberately small and dumb -- the point is to see that a directory of
``step-*.ckpt`` files loads, runs, and moves in a sensible direction, not to
score anything. For real numbers use ``python -m prior_eval.score``.

Top-left panel: the predictive density over ``tau`` for a single query row, one
curve per checkpoint (dark = early, bright = late), with the true CATE as a
dashed red line and the naive (confounded) observed ATE, ``mean(Y|A=1) -
mean(Y|A=0)`` over the context, as a dotted grey one. Top-right: the same
densities with the head's bins first pooled down to ``--agg_bins`` groups --
a linear projection, exact for anything CDF-based, which drops the
high-frequency ripple the loss never constrains. Middle row: the same
posteriors as CDFs -- no bin jitter, and the height at the true CATE is that
row's PIT -- raw on the left, Savitzky-Golay smoothed on the full 2000-bin
grid on the right. Bottom-left: the density implied by that smoothed CDF, the
second route to a ripple-free pdf. Bottom-right: the posterior's mean +/- sd
against step, against the same two references.

Only the single-head losses (``cate_single*``, ``cate_decomposition``) have a
posterior over ``tau``; for ``bins_po``/``hl`` the two per-arm marginals do not
identify the distribution of their difference, so the density panel is skipped
and only the mean curve is drawn.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter

from prior_eval.report import load_model
from proximalfm.prior import PriorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_episode(args, device: str):
    """One prior dataset, with the row split the model expects."""
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

    # Naive (confounded) ATE: the difference in observed means between the arms
    # of the context. Under the prior's unobserved U it is biased by
    # construction -- it is here as the "do nothing" reference the posterior
    # should be away from, in the direction of the truth.
    a_ctx = A[0, :train_size].float()
    y_ctx = Y_obs[0, :train_size].float()
    naive_ate = float(y_ctx[a_ctx > 0.5].mean() - y_ctx[a_ctx <= 0.5].mean())

    return {
        "naive_ate": naive_ate,
        "X": X,
        "W": W,
        "Z": Z,
        "A": A.unsqueeze(-1).float(),
        "y_train": Y_obs[:, :train_size],
        "d_X": d_X,
        "d_W": d_W,
        "d_Z": d_Z,
        "true_cate": (MU1 - MU0)[0, train_size:],
        "train_size": train_size,
    }


def pool_bins(vals: np.ndarray, w: np.ndarray, n_target: int):
    """Pool a histogram down to ``n_target`` groups of adjacent bins.

    Probability mass adds, so this is exact for the CDF and every functional
    of it; only the within-group shape is thrown away. Group sizes differ by
    at most one when the counts do not divide, hence the per-group width.
    """
    n = len(vals)
    dv0 = float(np.diff(vals).mean()) if n > 1 else 1.0
    if n_target <= 0 or n_target >= n:
        return vals, w, np.full(n, dv0)
    idx = np.array_split(np.arange(n), n_target)
    centres = np.array([vals[g].mean() for g in idx])
    mass = np.array([w[g].sum() for g in idx])
    widths = np.array([len(g) * dv0 for g in idx])
    return centres, mass, widths


def smooth_cdf(vals: np.ndarray, w: np.ndarray, window: int, polyorder: int = 3):
    """Savitzky-Golay the CDF on the full grid, then differentiate it back.

    Smoothing the *CDF* rather than the density is the better-posed half of
    the job: the CDF is monotone and already visually clean, so a local
    polynomial fit has an easy target, and its analytic derivative is a much
    smoother density than differencing a boxcar-averaged CDF would give.
    Monotonicity is not guaranteed by the filter, so it is imposed afterwards
    (clip, then running max) and the density is clipped at zero -- both are
    no-ops on a well-behaved fit and keep the panel honest when they are not.
    """
    n = len(vals)
    cdf = np.cumsum(w)
    window = min(int(window), n if n % 2 else n - 1)
    window = max(window if window % 2 else window - 1, polyorder + 2)
    if window >= n:
        return cdf, np.gradient(cdf, vals)
    cdf_s = savgol_filter(cdf, window, polyorder)
    cdf_s = np.maximum.accumulate(np.clip(cdf_s, 0.0, 1.0))
    pdf_s = np.clip(np.gradient(cdf_s, vals), 0.0, None)
    return cdf_s, pdf_s


def parse_steps(tokens) -> list[int] | None:
    """``--steps 1,50 100`` -> [1, 50, 100]; nothing given -> None (all)."""
    if not tokens:
        return None
    out = []
    for tok in tokens:
        for part in str(tok).replace(",", " ").split():
            out.append(int(part))
    return sorted(set(out))


def find_checkpoints(ckpt_dir: Path, steps: list[int] | None = None) -> list[Path]:
    by_step = {
        # unpadded legacy names parse the same way, so a mixed directory works
        int(p.stem.split("-")[1]): p
        for p in ckpt_dir.glob("step-*.ckpt")
    }
    if not by_step:
        raise SystemExit(f"No step-*.ckpt found in {ckpt_dir}")

    if steps is None:
        return [by_step[s] for s in sorted(by_step)]

    missing = [s for s in steps if s not in by_step]
    if missing:
        available = ", ".join(str(s) for s in sorted(by_step))
        raise SystemExit(
            f"No checkpoint for step(s) {missing} in {ckpt_dir}.\n"
            f"Available: {available}"
        )
    return [by_step[s] for s in steps]


def main() -> None:
    # uv run python visu/visu_posterior_evolution.py whole_1 --out . --steps 1,150,500,1000,1500,2000,3000,4750,5000,6000,7000,8000,9000,10000,11000,12550 --seed 123 --n_ctx 2000
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ckpt_dir", type=Path, help="directory holding step-*.ckpt")
    p.add_argument("--row", type=int, default=0, help="query row to look at")
    p.add_argument(
        "--steps",
        nargs="+",
        default=None,
        metavar="STEP",
        help="steps to plot, e.g. --steps 1,50,100 250 (default: every one found)",
    )
    p.add_argument(
        "--agg_bins",
        type=int,
        default=500,
        help="number of pooled bins for the smoothed pdf panel (default: 500)",
    )
    p.add_argument(
        "--cdf_window",
        type=int,
        default=101,
        help="Savitzky-Golay window, in bins, for the smoothed-cdf panels",
    )
    p.add_argument("--n_ctx", type=int, default=256)
    p.add_argument("--n_query", type=int, default=2)
    p.add_argument("--mc_cate", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "visu" / "posterior_evo.png")
    args = p.parse_args()

    ep = make_episode(args, args.device)
    if not 0 <= args.row < args.n_query:
        raise SystemExit(f"--row must be in [0, {args.n_query})")
    true_cate = float(ep["true_cate"][args.row])
    naive_ate = ep["naive_ate"]
    print(
        f"context {ep['train_size']} rows, query row {args.row}, "
        f"true CATE {true_cate:.4f}, naive ATE {naive_ate:+.4f}"
    )

    ckpts = find_checkpoints(args.ckpt_dir, parse_steps(args.steps))
    steps, means, sds, curves = [], [], [], []

    for path in ckpts:
        try:
            model, step = load_model(path, device=args.device)
        except Exception as exc:  # a half-written checkpoint is the usual cause
            print(f"skipping {path.name}: {exc}")
            continue
        step = int(step) if step is not None else int(path.stem.split("-")[1])

        with torch.inference_mode():
            out = model.predict(
                X=ep["X"],
                Z=ep["Z"],
                W=ep["W"],
                A=ep["A"],
                y_train=ep["y_train"],
                target="cate",
                output_type=(
                    "mean" if model.loss_type in ("bins_po", "hl") else "distribution"
                ),
                d_X=ep["d_X"],
                d_W=ep["d_W"],
                d_Z=ep["d_Z"],
            )

        if isinstance(out, dict):
            # the grid is shared across rows, so "values" may come back with a
            # broadcast row dimension of 1
            v = out["values"]
            vals = v[0, args.row if v.shape[1] > 1 else 0].float().cpu().numpy()
            w = out["weights"][0, args.row].float().cpu().numpy()
            mean = float((w * vals).sum())
            sd = float(np.sqrt(max((w * (vals - mean) ** 2).sum(), 0.0)))
            curves.append((step, vals, w))
        else:
            mean, sd = float(out[0, args.row]), float("nan")

        steps.append(step)
        means.append(mean)
        sds.append(sd)
        print(
            f"  step {step:>7d}  mean {mean:+.4f}  sd {sd:.4f}  loss_type={model.loss_type}"
        )

    if not steps:
        raise SystemExit("no checkpoint loaded")

    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
    (ax_d, ax_a), (ax_c, ax_cs), (ax_ds, ax_t) = axes
    cmap = plt.get_cmap("viridis")

    if curves:
        for i, (step, vals, w) in enumerate(curves):
            color = cmap(i / max(len(curves) - 1, 1))
            width = float(np.diff(vals).mean()) if len(vals) > 1 else 1.0
            # a histogram head is a piecewise-uniform density, so divide by the
            # bin width to make curves with different grids comparable
            ax_d.plot(
                vals, w / max(width, 1e-12), color=color, lw=1.2, label=f"step {step}"
            )
            # The CDF is the same object integrated, and it is the easier read
            # of the two: no bin-to-bin jitter, and the value it takes at the
            # true CATE is that row's PIT -- 0 or 1 means the truth is outside
            # the posterior, 0.5 means dead centre.
            ax_c.plot(vals, np.cumsum(w), color=color, lw=1.2, label=f"step {step}")
            # Same density, adjacent bins pooled. Mass adds, so the CDF is
            # untouched; what goes is the sub-kernel ripple the loss cannot
            # see. Each group is divided by its own width because array_split
            # leaves group sizes differing by one when the counts do not divide.
            c_a, m_a, wid_a = pool_bins(vals, w, args.agg_bins)
            ax_a.plot(
                c_a,
                m_a / np.maximum(wid_a, 1e-12),
                color=color,
                lw=1.2,
                label=f"step {step}",
            )
            # The other route to the same end: keep all 2000 bins, smooth the
            # cdf, and read the density off its derivative.
            cdf_s, pdf_s = smooth_cdf(vals, w, args.cdf_window)
            ax_cs.plot(vals, cdf_s, color=color, lw=1.2, label=f"step {step}")
            ax_ds.plot(vals, pdf_s, color=color, lw=1.2, label=f"step {step}")

        # Frame the *final* posterior: the early checkpoints are near-flat over
        # the whole grid, so any window covering them leaves the converged
        # density an invisible spike. Last checkpoint's mean +/- 2 sd, widened
        # only enough to keep the two reference lines on screen.
        m_last, s_last = means[-1], sds[-1]
        if s_last != s_last or s_last <= 0:  # NaN or degenerate
            s_last = abs(m_last) * 0.5 + 1e-3
        lo = min(m_last - 2 * s_last, true_cate, naive_ate)
        hi = max(m_last + 2 * s_last, true_cate, naive_ate)
        pad = 0.05 * (hi - lo + 1e-6)

        for ax in (ax_d, ax_a, ax_c, ax_cs, ax_ds):
            ax.axvline(true_cate, color="red", ls="--", lw=1.5, label="true CATE")
            ax.axvline(
                naive_ate, color="gray", ls=":", lw=1.5, label="naive ATE (observed)"
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_xlabel("tau")
        n_bins = len(curves[0][1])
        ax_d.set_ylabel("density")
        ax_d.set_title(f"CATE posterior (pdf, {n_bins} bins), query row {args.row}")
        ax_a.set_ylabel("density")
        ax_a.set_title(f"same pdf, bins pooled to {min(args.agg_bins, n_bins)}")
        ax_c.set_ylabel("P(tau <= t)")
        ax_c.set_ylim(0, 1)
        ax_c.set_title("same posterior, cdf")
        ax_c.grid(alpha=0.3)
        ax_cs.set_ylabel("P(tau <= t)")
        ax_cs.set_ylim(0, 1)
        ax_cs.set_title(f"cdf, savgol-smoothed (window {args.cdf_window} bins)")
        ax_cs.grid(alpha=0.3)
        ax_ds.set_ylabel("density")
        ax_ds.set_title("pdf implied by the smoothed cdf")
        if len(curves) <= 12:
            ax_d.legend(fontsize=7)
    else:
        for ax in (ax_d, ax_a, ax_c, ax_cs, ax_ds):
            ax.text(
                0.5,
                0.5,
                "two-marginal loss:\nno posterior over tau",
                ha="center",
                va="center",
            )
            ax.set_axis_off()

    means_a, sds_a = np.array(means), np.array(sds)
    ax_t.plot(steps, means_a, "o-", color="C0", label="posterior mean", zorder=3)
    if np.isfinite(sds_a).any():
        # 1->2 sd first, so the darker inner band draws over it
        for lo_b, hi_b in (
            (means_a + sds_a, means_a + 2 * sds_a),
            (means_a - 2 * sds_a, means_a - sds_a),
        ):
            ax_t.fill_between(steps, lo_b, hi_b, color="C0", alpha=0.12, lw=0)
        # one proxy handle rather than two identical legend entries
        ax_t.fill_between([], [], [], color="C0", alpha=0.12, lw=0, label="+/- 2 sd")
        ax_t.fill_between(
            steps,
            means_a - sds_a,
            means_a + sds_a,
            color="C0",
            alpha=0.3,
            lw=0,
            label="+/- 1 sd",
        )
        # The 2 sd band of an early, near-flat checkpoint covers the whole
        # grid and would flatten everything else, so the window is the average
        # of the widest and the last checkpoint's 2 sd -- the outer band is
        # then clipped early on and intact where it matters. Reference lines
        # are kept on screen; the mean curve too, since it is the subject.
        sd_ok = sds_a[np.isfinite(sds_a)]
        # (2*widest + 2*last) / 2
        half = float(sd_ok.max() + sd_ok[-1])
        centre = 0.5 * (means_a.min() + means_a.max())
        t_lo = min(centre - half, means_a.min(), true_cate, naive_ate)
        t_hi = max(centre + half, means_a.max(), true_cate, naive_ate)
        t_pad = 0.05 * (t_hi - t_lo + 1e-6)
        ax_t.set_ylim(t_lo - t_pad, t_hi + t_pad)
    ax_t.axhline(true_cate, color="red", ls="--", lw=1.5, label="true CATE")
    ax_t.axhline(naive_ate, color="gray", ls=":", lw=1.5, label="naive ATE (observed)")
    ax_t.set_xlabel("step")
    ax_t.set_ylabel("tau")
    ax_t.set_title("evolution over checkpoints")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)

    fig.tight_layout()
    # --out may be a directory ("--out .") as well as a filename
    out = args.out / "posterior_evo.png" if args.out.is_dir() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()
