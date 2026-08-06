"""ROADMAP A2 -- score a checkpoint on each stratified cut of a frozen set.

    uv run python -m posterior_eval.score_strata \\
        --dir data/posterior_eval/frozen_prod_v1 --ckpt path/to/model.ckpt

Offline sibling of :mod:`posterior_eval.score`, which reports one pooled number
per metric. This reports the same metrics per **cut** defined in
:mod:`posterior_eval.strata`, each scored both *inside* and *outside* the cut.

One pass over the data, whatever the number of cuts:
:class:`~proximalfm.train._dist_eval.PosteriorEvaluator` accumulates row-weighted
sums, so a stratum's metrics are exactly what a separate pass over only that
stratum's rows would produce. The model predicts once per episode and
``score_episode`` fans that single forward out to every accumulator.

Reading the table
=================

**Read the complement, not the pool.** Every cut is reported ``in`` vs ``out``
because the pooled number *contains* the stratum -- comparing a stratum against
a pool it is one third of understates every difference.

**``crps`` is not comparable across strata.** Strata differ in intrinsic
difficulty *and* in the scale of ``tau``. A worse CRPS under weak proxies is the
expected consequence of a harder problem and says nothing about the posterior.
Treating it as a finding is the single most likely misuse of this module, which
is why the printed table separates the two blocks.

**Comparable across strata**: ``sd_over_rmse``, ``cov50_pooled``/``cov90_pooled``
and the per-episode ``cov90/p25``--``p75`` spread, ``crps_over_oracle``, and
``pit_ks`` -- the last strictly against *its own* band, never the pooled one.
The band scales as ``1/sqrt(n)``, so a smaller stratum earns a wider band and a
naive comparison manufactures failures. The per-episode coverage quantiles carry
the same caveat for a different reason: their binomial floor scales with the
*rows per episode inside the cut*, so a row-level cut that keeps a third of each
episode's rows widens the IQR on sampling noise alone.

**The headline for A2** is ``sd_over_rmse`` on ``weak_proxy`` against its
complement. Pooled ~1 with ``weak_proxy`` well below 1 is the failure the
roadmap names: the posterior is *not* widening where proximal identification is
fragile, which is worse than being merely inaccurate there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from posterior_eval.build_prior_frozen_set import (
    frozen_content_digest,
    load_frozen_meta,
)
from posterior_eval.report import load_model
from posterior_eval.strata import MIN_EPISODES, load_manifest
from proximalfm.train._dist_eval import (
    PosteriorEvaluator,
    UnsupportedLossTypeError,
    iter_frozen_episodes,
    score_episode,
)
from proximalfm.train._eval import CausalEvaluator

#: Metrics whose value is a ratio or a probability, hence comparable between
#: strata of different intrinsic difficulty.
COMPARABLE = (
    "sd_over_rmse",
    "cov50_pooled",
    "cov90_pooled",
    "cov90/p25",
    "cov90/p75",
    "crps_over_oracle",
    "pit_ks",
    "pit_ks_band",
)

#: Metrics on the scale of ``tau``, hence NOT comparable between strata.
SCALE_BOUND = ("crps", "mean_sd", "width50", "width90")

#: ``sd_over_rmse`` below this is genuinely over-confident: intervals narrower
#: than the errors warrant. **The A2 alarm is gated on this absolute level, not
#: on the stratum-vs-complement gap alone.** A model that is 3x too wide
#: everywhere is over-*dispersed*; one of its strata being relatively less wide
#: is not the failure ROADMAP A2 describes, and reporting it as such would fire
#: on every early checkpoint.
OVERCONFIDENT = 0.9

#: Minimum stratum-vs-complement ``sd_over_rmse`` gap worth remarking on.
RELATIVE_GAP = 0.05


class _Bucket:
    """A named subset of rows, with its own pair of evaluators."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.dist = PosteriorEvaluator()
        self.point = CausalEvaluator()
        self.episodes = 0

    def compute(self) -> Dict[str, float]:
        metrics = dict(self.dist.compute(prefix="", include_count=True))
        point = self.point.compute(prefix="", include_count=False)
        if "cate/rmse" in point:
            metrics["cate/rmse"] = point["cate/rmse"]
        metrics["episodes"] = float(self.episodes)
        return metrics


def _buckets_for(manifest: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """``(bucket_name, cut_name, side)`` for the pool and every cut's two sides."""
    out = [("pooled", "", "")]
    for name in manifest["cuts"]:
        out.append((f"{name}/in", name, "in"))
        out.append((f"{name}/out", name, "out"))
    return out


def score_strata(
    model,
    out: Path,
    *,
    device: str = "cpu",
    max_episodes: Optional[int] = None,
    n_ctx: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Any]] = None,
    progress: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Score every stratum in one pass; returns ``{bucket_name: metrics}``.

    Single process by design -- the stratified pass is an offline milestone
    tool, so it carries none of ``score_frozen_set``'s rank/world_size
    machinery. Adding DDP here would mean one collective pair per bucket, which
    every rank would have to enter in the same order.

    That single-process choice is also why ``progress`` defaults to on: the pass
    is one forward per episode over the whole set, minutes to tens of minutes,
    with nothing printed until the table at the end.
    """
    out = Path(out)
    meta = load_frozen_meta(out) if meta is None else meta
    manifest = load_manifest(out, meta=meta) if manifest is None else manifest

    n_max = int(meta["global"]["n_max"])
    n_ctx = n_max if n_ctx is None else int(n_ctx)
    records = meta["episodes"]
    cuts = manifest["cuts"]

    total = len(records) if max_episodes is None else min(len(records), max_episodes)
    specs = _bucket_specs(manifest)
    buckets = {name: _Bucket(name, kind) for name, kind, _ in specs}

    episodes = tqdm(
        enumerate(iter_frozen_episodes(out / "val", max_episodes)),
        total=total,
        desc="episodes",
        unit="ep",
        disable=not progress,
    )
    with torch.inference_mode():
        for i, batch in episodes:
            fanout: List[Tuple[PosteriorEvaluator, Optional[torch.Tensor]]] = []
            point_fanout: List[Tuple[CausalEvaluator, Optional[torch.Tensor]]] = []

            for bucket_name, _kind, spec in specs:
                mask = _episode_mask(cuts, spec, i)
                if mask is _SKIP:
                    continue
                bucket = buckets[bucket_name]
                bucket.episodes += 1
                fanout.append((bucket.dist, mask))
                point_fanout.append((bucket.point, mask))

            score_episode(
                model,
                batch,
                records[i],
                n_ctx=n_ctx,
                n_max=n_max,
                device=device,
                dist_evaluators=fanout,
                point_evaluators=point_fanout,
            )

    return {name: b.compute() for name, b in buckets.items() if b.episodes}


#: Sentinel for "this episode contributes nothing to this bucket".
_SKIP = object()


def _bucket_specs(manifest: Dict[str, Any]) -> List[Tuple[str, str, Optional[tuple]]]:
    specs: List[Tuple[str, str, Optional[tuple]]] = [("pooled", "episode", None)]
    for name, cut in manifest["cuts"].items():
        specs.append((f"{name}/in", cut["kind"], (name, True)))
        specs.append((f"{name}/out", cut["kind"], (name, False)))
    return specs


def _episode_mask(cuts, spec, i: int):
    """Row mask for one bucket on episode ``i``; ``_SKIP`` if it contributes none.

    ``None`` means "every row", which is both the pooled bucket and an
    episode-level cut the episode belongs to -- the distinction between an
    episode cut and a row cut disappears at this point, which is exactly why
    both kinds cost the same.
    """
    if spec is None:
        return None
    name, inside = spec
    cut = cuts[name]
    mask = cut["mask"]
    if cut["kind"] == "episode":
        member = bool(mask[i])
        return None if member == inside else _SKIP
    rows = mask[i]
    rows = rows if inside else ~rows
    return rows if bool(rows.any()) else _SKIP


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "     -"
    if value != value:
        return "   nan"
    return f"{value:.4f}"


def _print_report(results: Dict[str, Dict[str, float]], manifest: Dict[str, Any]):
    pooled = results.get("pooled", {})
    print("\n" + "=" * 78)
    print("Stratified posterior evaluation")
    print("=" * 78)
    print(
        f"\npooled: {int(pooled.get('episodes', 0))} episodes, "
        f"{int(pooled.get('eval_rows', 0))} rows"
    )

    print("\n--- COMPARABLE across strata (ratios and probabilities) ---")
    header = f"  {'bucket':<26} {'eps':>5} {'rows':>7} " + " ".join(
        f"{k[:12]:>12}" for k in COMPARABLE
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in _report_order(manifest, results):
        m = results[name]
        row = " ".join(_fmt(m.get(k)).rjust(12) for k in COMPARABLE)
        print(
            f"  {name:<26} {int(m.get('episodes', 0)):>5} "
            f"{int(m.get('eval_rows', 0)):>7} {row}"
        )

    print("\n--- SCALE-BOUND: do NOT compare between strata ---")
    print("    (different difficulty AND different tau scale; shown for the record)")
    header = f"  {'bucket':<26} " + " ".join(f"{k[:9]:>9}" for k in SCALE_BOUND)
    header += f" {'cate/rmse':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in _report_order(manifest, results):
        m = results[name]
        row = " ".join(_fmt(m.get(k)).rjust(9) for k in SCALE_BOUND)
        print(f"  {name:<26} {row} {_fmt(m.get('cate/rmse')):>10}")

    _print_verdict(results, manifest)


def _report_order(manifest, results) -> List[str]:
    order = ["pooled"]
    for name in manifest["cuts"]:
        order += [f"{name}/in", f"{name}/out"]
    return [n for n in order if n in results]


def _print_verdict(results: Dict[str, Dict[str, float]], manifest: Dict[str, Any]):
    print("\n--- Reading ---")
    pooled = results.get("pooled", {})
    pooled_sdr = pooled.get("sd_over_rmse")

    for name in manifest["cuts"]:
        inside, outside = results.get(f"{name}/in"), results.get(f"{name}/out")
        if not inside or not outside:
            continue
        a, b = inside.get("sd_over_rmse"), outside.get("sd_over_rmse")
        if a is None or b is None or a != a or b != b:
            continue
        gap = a - b
        flag = ""
        if a < OVERCONFIDENT and a < b:
            flag = "  <-- narrower AND over-confident in absolute terms"
        print(f"  {name:<26} sd_over_rmse  in={a:.3f}  out={b:.3f}  d={gap:+.3f}{flag}")

    if pooled_sdr is not None and pooled_sdr == pooled_sdr:
        print(f"  {'pooled':<26} sd_over_rmse  {pooled_sdr:.3f}")

    weak = results.get("weak_proxy/in", {}).get("sd_over_rmse")
    weak_out = results.get("weak_proxy/out", {}).get("sd_over_rmse")
    if weak is not None and weak_out is not None and weak == weak:
        print()
        relatively_narrower = weak < weak_out - RELATIVE_GAP
        if relatively_narrower and weak < OVERCONFIDENT:
            print(
                "  ! weak_proxy is over-confident in absolute terms AND narrower "
                "than its\n    complement. This is the ROADMAP A2 failure: the "
                "posterior does not widen\n    where proximal identification is "
                "fragile."
            )
        elif relatively_narrower:
            # The distinction that makes this tool honest rather than alarming.
            # sd_over_rmse >> 1 everywhere means the model is over-*dispersed*;
            # "relatively narrower" inside that is not over-confidence, and
            # calling it the A2 failure would cry wolf on every early checkpoint.
            print(
                f"  weak_proxy is narrower than its complement "
                f"({weak:.2f} vs {weak_out:.2f}), but both are\n"
                f"  above {OVERCONFIDENT} -- the model is over-DISPERSED here, not "
                "over-confident. This\n  is a relative statement inside a globally "
                "too-wide posterior, not the A2\n  failure. Re-read it once pooled "
                "sd_over_rmse approaches 1; the gap is what\n  to watch, since it "
                "is what would survive the width coming down."
            )
        else:
            print(
                "  weak_proxy is not narrower than its complement -- the posterior "
                "is not\n  *specifically* blind to weak identification."
            )

    small = [
        n
        for n, m in results.items()
        if m.get("episodes", 0) < MIN_EPISODES and n != "pooled"
    ]
    if small:
        print(
            f"\n  ! under {MIN_EPISODES} episodes, read pit_ks against its own "
            f"(wider) band only: {', '.join(sorted(small))}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m posterior_eval.score_strata",
        description="Score a checkpoint per stratified cut of a frozen set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", type=Path, required=True, help="frozen set ROOT")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument(
        "--n_ctx",
        type=int,
        default=None,
        help="context prefix; defaults to n_max (the frozen comparison point)",
    )
    p.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="also write the full metric dict here",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="suppress the progress bar (it writes to stderr, so redirecting "
        "stdout keeps it on screen; this is for logs that capture both)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    meta = load_frozen_meta(args.dir)
    manifest = load_manifest(args.dir, meta=meta)
    model, step = load_model(args.ckpt, device=args.device)
    try:
        results = score_strata(
            model,
            args.dir,
            device=args.device,
            max_episodes=args.max_episodes,
            n_ctx=args.n_ctx,
            meta=meta,
            manifest=manifest,
            progress=not args.no_progress,
        )
    except UnsupportedLossTypeError as exc:
        print(f"\n{exc}\n")
        return 2

    print(f"\ncheckpoint: {args.ckpt}" + (f" (step {step})" if step else ""))
    # Names the set by content, not by a reusable directory name (ROADMAP A4) --
    # these tables get quoted into the roadmap and the paper.
    print(f"frozen set: {args.dir}  [content_digest={frozen_content_digest(meta)}]")
    _print_report(results, manifest)

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"Written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
