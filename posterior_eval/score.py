"""Score a checkpoint's CATE posterior on a frozen set (EVAL_PLAN step 5).

    uv run python -m posterior_eval.score --dir data/posterior_eval/frozen_v1 \
        --ckpt path/to/model.ckpt

Offline sibling of the trainer wiring that follows in step 6: the same
:class:`proximalfm.train._dist_eval.PosteriorEvaluator`, driven from the CLI so
its numbers can be read and argued with before they start scrolling past in a
training log.

Read the table in this order, because each entry conditions the next:

1. ``out_of_grid_frac`` -- if it is not ~0 the grid is too tight and every other
   number below is scored against targets no forecast on this grid can cover.
2. ``sd_over_rmse`` -- 1.0 is calibrated, below 1 over-confident. Never read a
   width without it.
3. ``pit_ks`` vs its band, and then ``pit_ks_episode_p90`` vs the per-episode
   band: the pooled statistic passing while the per-episode tail fails is
   cancellation, not calibration.
4. ``crps`` only against ``crps_over_oracle`` -- as that ratio approaches 1 the
   remaining CRPS is the bin grid, and the remedy is more ``--nbins``, not more
   training steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from proximalfm.train._dist_eval import UnsupportedLossTypeError, score_frozen_set


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    return f"{value:.5f}"


def _verdict(metrics: Dict[str, float]) -> List[str]:
    """The three readings that are easy to get backwards, spelled out."""
    lines = []

    oog = metrics.get("prior_dist/out_of_grid_frac")
    if oog is not None and oog > 1e-3:
        lines.append(
            f"  ! out_of_grid_frac = {oog:.4f}: the CATE grid "
            "(--cate_vmin/--cate_vmax) is too tight; the numbers below are "
            "scored against targets no forecast on this grid can cover."
        )

    sdr = metrics.get("prior_dist/sd_over_rmse")
    if sdr is not None and sdr == sdr:
        if sdr < 0.9:
            lines.append(
                f"  sd_over_rmse = {sdr:.3f} < 1: over-confident "
                f"(intervals ~{100 * (1 - sdr):.0f}% too narrow)."
            )
        elif sdr > 1.1:
            lines.append(f"  sd_over_rmse = {sdr:.3f} > 1: too wide.")
        else:
            lines.append(f"  sd_over_rmse = {sdr:.3f}: calibrated width.")

    ks, band = metrics.get("prior_dist/pit_ks"), metrics.get("prior_dist/pit_ks_band")
    if ks is not None and band is not None:
        inside = "inside" if ks < band else "OUTSIDE"
        lines.append(f"  pooled PIT-KS {ks:.4f} vs band {band:.4f}: {inside}.")

    p90 = metrics.get("prior_dist/pit_ks_episode_p90")
    ep_band = metrics.get("prior_dist/pit_ks_episode_band")
    if p90 is not None and ep_band is not None:
        if p90 >= ep_band and ks is not None and band is not None and ks < band:
            lines.append(
                f"  ! per-episode KS p90 = {p90:.4f} >= band {ep_band:.4f} while "
                "the pooled statistic passes: opposite-direction miscalibration "
                "is cancelling in the pool."
            )
        else:
            lines.append(
                f"  per-episode KS p90 {p90:.4f} vs band {ep_band:.4f}: "
                f"{'inside' if p90 < ep_band else 'OUTSIDE'}."
            )

    ratio = metrics.get("prior_dist/crps_over_oracle")
    if ratio is not None and ratio == ratio and ratio < 1.5:
        lines.append(
            f"  crps_over_oracle = {ratio:.2f}: close to the bin-resolution "
            "ceiling -- the remedy is more --nbins, not more steps."
        )

    s_over_sd = metrics.get("prior_dist/s_over_sd")
    if s_over_sd is not None and s_over_sd == s_over_sd:
        note = "negligible" if s_over_sd < 0.2 else "worth correcting (step 7)"
        lines.append(f"  s/sd = {s_over_sd:.4f}: target noise {note}.")

    return lines


def run(
    out: Path,
    ckpt: Path,
    device: str = "cpu",
    max_episodes: Optional[int] = None,
    n_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    from posterior_eval.report import load_model

    model, step = load_model(Path(ckpt), device=device)
    metrics = score_frozen_set(
        model,
        Path(out),
        n_ctx=n_ctx,
        device=device,
        max_episodes=max_episodes,
    )

    from posterior_eval.build_prior_frozen_set import (
        frozen_content_digest,
        load_frozen_meta,
    )

    digest = frozen_content_digest(load_frozen_meta(Path(out))) or "n/a"

    print()
    # The digest, not the path, is what makes this table quotable later: a
    # directory name can be reused, the digest cannot (ROADMAP A4).
    print(f"Frozen set: {out}  [content_digest={digest}]")
    print(f"Checkpoint: {ckpt} (step {step}, loss_type={model.loss_type})")
    print(
        f"  episodes {int(metrics.get('prior_point/eval/episodes', 0))}"
        f"   rows {int(metrics.get('prior_dist/eval/rows', 0))}"
        f"   n_ctx {int(metrics.get('prior_dist/n_ctx', 0))}"
        f"   grid_scale {metrics.get('prior_dist/grid_scale', float('nan')):.4f}"
    )
    print()
    for group in ("prior_point/", "prior_dist/"):
        keys = sorted(k for k in metrics if k.startswith(group))
        if not keys:
            continue
        print(f"  {group}")
        for key in keys:
            print(f"    {key[len(group):]:<28} {_fmt(metrics[key])}")
        print()

    for line in _verdict(metrics):
        print(line)
    print()
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m posterior_eval.score",
        description="Score a checkpoint's CATE posterior on a frozen eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="frozen set root (holds val/ and frozen_meta.pt)",
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument(
        "--n_ctx",
        type=int,
        default=None,
        help="context prefix to score at; defaults to the set's n_max",
    )
    args = p.parse_args(argv)

    try:
        run(
            args.dir,
            args.ckpt,
            device=args.device,
            max_episodes=args.max_episodes,
            n_ctx=args.n_ctx,
        )
    except UnsupportedLossTypeError as exc:
        print(f"\nRefusing to score this checkpoint:\n  {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
