"""Integrity and sanity check for a frozen posterior-evaluation set.

    uv run python -m eval_datasets.prior_eval.verify --dir data/prior_eval/frozen_prod_v1

Needs no checkpoint. Sibling of :mod:`eval_datasets.prior_eval.report`, which answers
"is ``K`` big enough *given a model*"; this one answers "is the set intact and
well-formed at all" -- the question you have after building on one machine and
copying to another.

Six groups, ordered so each conditions the next. A failure in an early group
makes the later numbers meaningless, so the first FAIL is the one to read.

1. **Structure.** Every ``batch_*.pt`` loads, the episode count matches the
   sidecar, no ``.tmp`` files survived. Catches a truncated copy.
2. **Per-episode invariants.** ``T = n_max + n_q``, ``train_size = n_max``,
   ``K_eff = K``, and the NaN layout of ``MU0``/``MU1``/``S2`` (NaN on the
   context prefix, finite on the query rows -- the existing convention).
3. **Sidecar-vs-disk agreement.** ``tau_star`` and ``y_context_*`` recomputed
   from the ``val/`` files must match what ``frozen_meta.pt`` claims. This is
   the check that catches a sidecar paired with the wrong ``val/``, or an
   episode order that drifted -- neither of which shows up as a crash later,
   only as quietly wrong metrics.
4. **Target noise.** ``s`` against the spread of ``tau*``. Not the final word
   (that needs a model, hence ``report.py``), but a ``K`` that is too small is
   visible here without one.
5. **Grid coverage.** ``tau_star_std`` against ``--cate_vmin``/``--cate_vmax``.
   At ``n_ctx = n_max`` the scoring grid is essentially the head's own grid, so
   a target outside it is one no forecast can ever cover -- and
   ``out_of_grid_frac`` at score time would be the first thing to invalidate
   every other number.
6. **Stratification readiness.** The per-episode hyperparameters the cuts need are
   present and actually vary.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from eval_datasets.prior_eval.build_prior_frozen_set import (
    Y_STD_FLOOR,
    frozen_content_digest,
    load_frozen_meta,
)
from eval_datasets.prior_eval.report import iter_episodes

# Agreement tolerance for group 3. The sidecar stores what the generator
# produced in memory; the val/ files went through the sparse round-trip and a
# float32 cast, so exact equality is not the right bar -- but anything above
# this is a different episode, not a rounding difference.
RECON_ATOL = 1e-4
RECON_RTOL = 1e-4


class Check:
    """One named check plus its verdict, so the report can be printed at once."""

    def __init__(self, name: str):
        self.name = name
        self.failures: List[str] = []
        self.notes: List[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures


def _recompute_from_config(g: Dict[str, Any]) -> Optional[str]:
    """Digest of ``g``'s config alone, ignoring any digest stored in it.

    The point of verifying is to catch a stored value that disagrees with the
    config beside it, so the stored value must not be consulted here.
    """
    return frozen_content_digest(
        {"global": {k: v for k, v in g.items() if k != "content_digest"}}
    )


def _quantiles(t: torch.Tensor, qs: Sequence[float]) -> List[float]:
    q = torch.tensor(qs, dtype=torch.float64)
    return torch.quantile(t.double().reshape(-1), q).tolist()


def verify(
    out_dir: Path,
    cate_vmin: float = -3.0,
    cate_vmax: float = 3.0,
    max_episodes: Optional[int] = None,
    strict_recon: bool = True,
) -> Dict[str, Any]:
    """Run every group over the set at ``out_dir``; returns the collected stats."""
    out_dir = Path(out_dir)
    val_dir = out_dir / "val"

    structure = Check("1. structure")
    invariants = Check("2. per-episode invariants")
    agreement = Check("3. sidecar vs disk")
    noise = Check("4. target noise")
    grid = Check("5. grid coverage")
    strata = Check("6. stratification readiness")

    # ---- group 1: structure -------------------------------------------------
    meta = load_frozen_meta(out_dir)
    g = meta["global"]
    records = meta["episodes"]

    n_max, n_q, K = g["n_max"], g["n_q"], g["mc_cate"]
    T = n_max + n_q
    # Realistic replicate sets (synthetic/) share this schema so the evaluator
    # needs no branch, but they carry hand-written mechanisms rather than
    # sampled prior hyperparameters -- group 6 differs, nothing else does.
    set_kind = g.get("set_kind")
    is_realistic = set_kind in {"realistic", "inspect"}
    is_linear_inspection = set_kind == "inspect_linear"

    batch_files = sorted(p for p in val_dir.iterdir() if p.name.startswith("batch_"))
    tmp_files = [p for p in val_dir.iterdir() if p.name.endswith(".tmp")]
    if tmp_files:
        structure.fail(
            f"{len(tmp_files)} .tmp file(s) left in val/ -- the build was "
            "interrupted or the copy is incomplete"
        )
    if not (val_dir / "metadata.json").exists():
        structure.fail("val/metadata.json missing; LoadPriorDataset will not open it")
    if len(records) != g["n_datasets"]:
        structure.fail(
            f"sidecar holds {len(records)} episodes but claims n_datasets="
            f"{g['n_datasets']}"
        )
    expected_files = math.ceil(g["n_datasets"] / g["datasets_per_file"])
    if len(batch_files) != expected_files:
        structure.fail(
            f"{len(batch_files)} batch files on disk, expected {expected_files} "
            f"({g['n_datasets']} episodes / {g['datasets_per_file']} per file). "
            "A short copy loses whole episodes silently."
        )
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        structure.fail(
            f"{len(seeds) - len(set(seeds))} duplicate episode seed(s): episodes "
            "are not independent draws"
        )
    if [r["index"] for r in records] != list(range(len(records))):
        structure.fail("episode indices are not 0..N-1 in order")

    # The digest is what names this set in a wandb config; if the sidecar's
    # stored value disagrees with what its own config hashes to, the identity
    # every logged number is pinned to is wrong, and comparisons across runs
    # are silently meaningless. Cheap to check, so check it.
    stored_digest = g.get("content_digest")
    recomputed = _recompute_from_config(g)
    if stored_digest is None and recomputed is None:
        structure.fail(
            "content_digest is neither stored nor recomputable: this set has no "
            "stable identity, so nothing that scores it can be pinned to it."
        )
    elif stored_digest is None:
        structure.note(
            f"content_digest={recomputed} (recomputed; set predates the field)"
        )
    elif recomputed is not None and recomputed != stored_digest:
        structure.fail(
            f"content_digest mismatch: sidecar says {stored_digest}, its own "
            f"config hashes to {recomputed}. The sidecar was edited, or "
            "FROZEN_META_VERSION moved -- either way this set's identity does "
            "not match what runs logged against it."
        )
    else:
        structure.note(f"content_digest={stored_digest}")

    structure.note(f"built on gen_device={g.get('gen_device', 'cpu')}")
    structure.note(f"git_rev={g.get('git_rev')}")
    structure.note(f"n_grid={g.get('n_grid')}")

    # ---- groups 2-6: one pass over the episodes -----------------------------
    tau_all: List[torch.Tensor] = []
    tau_std_all: List[torch.Tensor] = []
    s2_std_all: List[torch.Tensor] = []
    per_ep_s: List[float] = []
    dims: Dict[str, set] = {k: set() for k in ("d_U", "d_X", "d_W", "d_Z")}
    hp_keys: Dict[str, set] = {}
    n_seen = 0
    max_recon_err = 0.0
    a_degenerate = 0

    limit = min(len(records), max_episodes) if max_episodes else len(records)

    for i, batch in enumerate(iter_episodes(val_dir, max_episodes=limit)):
        rec = records[i]
        n_seen += 1
        (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            MU0,
            MU1,
            d_U,
            d_X,
            d_W,
            d_Z,
            seq_lens,
            train_sizes,
            S2,
            K_eff,
        ) = batch

        tag = f"episode {i}"

        # -- group 2 ----------------------------------------------------------
        if int(seq_lens[0]) != T:
            invariants.fail(f"{tag}: seq_len={int(seq_lens[0])} != n_max+n_q={T}")
        if int(train_sizes[0]) != n_max:
            invariants.fail(f"{tag}: train_size={int(train_sizes[0])} != n_max={n_max}")
        if int(K_eff[0]) != K:
            invariants.fail(
                f"{tag}: K_eff={int(K_eff[0])} != mc_cate={K} -- this episode "
                "carries more target noise than the sidecar claims"
            )

        mu0 = MU0.reshape(-1)
        mu1 = MU1.reshape(-1)
        s2 = S2.reshape(-1)
        if not (
            torch.isnan(mu0[:n_max]).all()
            and torch.isnan(mu1[:n_max]).all()
            and torch.isnan(s2[:n_max]).all()
        ):
            invariants.fail(
                f"{tag}: MU0/MU1/S2 are not all-NaN on the context prefix; the "
                "set was built with mc_targets_on_context=True"
            )
        q = slice(n_max, T)
        if not (
            torch.isfinite(mu0[q]).all()
            and torch.isfinite(mu1[q]).all()
            and torch.isfinite(s2[q]).all()
        ):
            invariants.fail(f"{tag}: non-finite MU0/MU1/S2 on the query rows")
        if (s2[q] < 0).any():
            invariants.fail(f"{tag}: negative S2")

        for name, t in (("X", X), ("W", W), ("Z", Z), ("Y_obs", Y_obs), ("A", A)):
            if not torch.isfinite(t).all():
                invariants.fail(f"{tag}: non-finite values in {name}")

        a_ctx = A.reshape(-1)[:n_max]
        if not torch.isin(A.reshape(-1), torch.tensor([0.0, 1.0])).all():
            invariants.fail(f"{tag}: A is not binary")
        frac1 = float(a_ctx.float().mean())
        if frac1 <= 0.0 or frac1 >= 1.0:
            a_degenerate += 1

        # -- group 3 ----------------------------------------------------------
        tau_disk = (mu1[q] - mu0[q]).float()
        tau_side = rec["tau_star"].float().reshape(-1)
        if tau_disk.shape != tau_side.shape:
            agreement.fail(
                f"{tag}: tau_star has {tau_side.numel()} rows in the sidecar but "
                f"{tau_disk.numel()} on disk"
            )
        else:
            err = float((tau_disk - tau_side).abs().max())
            max_recon_err = max(max_recon_err, err)
            if not torch.allclose(tau_disk, tau_side, atol=RECON_ATOL, rtol=RECON_RTOL):
                agreement.fail(
                    f"{tag}: tau_star recomputed from val/ differs from the "
                    f"sidecar by up to {err:.3g}. The sidecar and the val/ "
                    "directory are not from the same build."
                )

        ctx_y = Y_obs.reshape(-1)[:n_max].float()
        y_mean = ctx_y.mean()
        y_std = ctx_y.std(unbiased=False).clamp(min=Y_STD_FLOOR)
        for name, got, want in (
            ("y_context_mean", y_mean, rec["y_context_mean"]),
            ("y_context_std", y_std, rec["y_context_std"]),
        ):
            if not torch.allclose(got, want.float(), atol=RECON_ATOL, rtol=RECON_RTOL):
                agreement.fail(
                    f"{tag}: {name} recomputed = {float(got):.6g}, sidecar = "
                    f"{float(want):.6g}"
                )

        # -- groups 4-6 -------------------------------------------------------
        tau_all.append(tau_disk)
        tau_std_all.append(rec["tau_star_std"].float().reshape(-1))
        s2_std_all.append(rec["s2_std"].float().reshape(-1))

        # Per-episode s in head units. Deliberately NOT divided by the episode's
        # own sd(tau*): that measures effect *heterogeneity* across query rows,
        # not posterior width, and a near-homogeneous episode drives it to zero
        # while its posterior stays perfectly well-scaled. The pooled prior
        # spread below is the only denominator available without a model.
        per_ep_s.append(float(rec["s2_std"].float().mean().sqrt()))

        for k in dims:
            dims[k].add(int(rec[k]))
        for k, v in (rec.get("sampled_hp") or {}).items():
            hp_keys.setdefault(k, set())
            if isinstance(v, (int, float, bool)):
                hp_keys[k].add(float(v))

    if n_seen != limit:
        structure.fail(
            f"iterated {n_seen} episodes from val/ but the sidecar has {limit}"
        )

    agreement.note(f"max |tau_star(disk) - tau_star(sidecar)| = {max_recon_err:.3g}")

    tau = torch.cat(tau_all) if tau_all else torch.zeros(0)
    tau_std = torch.cat(tau_std_all) if tau_std_all else torch.zeros(0)
    s2_std = torch.cat(s2_std_all) if s2_std_all else torch.zeros(0)

    # ---- group 4 ------------------------------------------------------------
    stats: Dict[str, Any] = {}
    if tau.numel():
        s_pooled = float(s2_std.mean().sqrt())
        sd_tau_pooled = float(tau_std.std(unbiased=False))
        ratio_pooled = s_pooled / sd_tau_pooled if sd_tau_pooled > 0 else float("nan")
        s_worst = max(per_ep_s) if per_ep_s else float("nan")
        ratio_worst = s_worst / sd_tau_pooled if sd_tau_pooled > 0 else float("nan")
        stats.update(
            s_pooled=s_pooled,
            sd_tau_pooled=sd_tau_pooled,
            s_over_sd_tau_pooled=ratio_pooled,
            s_worst_episode=s_worst,
            s_over_sd_tau_worst_episode=ratio_worst,
        )
        noise.note(
            f"s = {s_pooled:.5g}, sd(tau*) = {sd_tau_pooled:.5g}, "
            f"ratio = {ratio_pooled:.4f} (head units)"
        )
        noise.note(
            f"noisiest episode: s = {s_worst:.5g} = {ratio_worst:.4f} of the "
            "pooled spread"
        )
        noise.note(
            "This is s against the *prior spread* of tau*, an upper bound on "
            "sigma_post. The ratio that decides whether the predictive-side "
            "noise convolution is needed is s/sigma_post, which is "
            "larger and needs a checkpoint -- run eval_datasets.prior_eval.report."
        )
        if ratio_pooled > 0.2:
            noise.fail(
                f"s/sd(tau*) = {ratio_pooled:.3f} > 0.2 already against the "
                "prior spread; against an actual posterior it will be far "
                f"worse. K={K} is too small for this set."
            )

        # ---- group 5 --------------------------------------------------------
        below = float((tau_std < cate_vmin).float().mean())
        above = float((tau_std > cate_vmax).float().mean())
        oog = below + above
        lo, q01, q50, q99, hi = _quantiles(tau_std, [0.0, 0.01, 0.5, 0.99, 1.0])
        stats.update(out_of_grid_frac=oog, tau_std_min=lo, tau_std_max=hi)
        grid.note(
            f"tau_star_std: min={lo:.3f} p1={q01:.3f} p50={q50:.3f} "
            f"p99={q99:.3f} max={hi:.3f}"
        )
        grid.note(
            f"grid [{cate_vmin}, {cate_vmax}]: {below:.4%} below, {above:.4%} above"
        )
        if oog > 1e-3:
            grid.fail(
                f"out_of_grid_frac = {oog:.4f} > 1e-3: these targets sit outside "
                "the head's tau grid, so no forecast can cover them and every "
                "distributional number on this set is scored against the "
                "impossible. Widen --cate_vmin/--cate_vmax."
            )

    # ---- group 6 ------------------------------------------------------------
    for k, vals in dims.items():
        strata.note(f"{k}: {len(vals)} distinct values, range {min(vals)}..{max(vals)}")
    varying = [k for k, v in hp_keys.items() if len(v) > 1]
    if is_linear_inspection:
        expected_levels = {float(v) for v in g.get("proxy_noise_stds", [])}
        observed_levels = {
            float(r["proxy_noise_std"]) for r in records if "proxy_noise_std" in r
        }
        missing_reliability = [
            r["index"] for r in records if "proxy_reliability" not in r
        ]
        if not expected_levels:
            strata.fail("global proxy_noise_stds is absent or empty")
        elif observed_levels != expected_levels:
            strata.fail(
                "per-episode proxy noise coverage does not match the global grid: "
                f"observed={sorted(observed_levels)}, expected={sorted(expected_levels)}"
            )
        else:
            strata.note(f"{len(observed_levels)} proxy-noise levels represented")
        if missing_reliability:
            strata.fail(f"{len(missing_reliability)} episode(s) omit proxy_reliability")
        elif len({float(r["proxy_reliability"]) for r in records}) < 2:
            strata.fail("proxy reliability does not vary across episodes")
        else:
            strata.note("proxy reliability is recorded and varies across episodes")
    elif is_realistic:
        # Realistic sets have no sampled prior hyperparameters -- their
        # mechanisms are hand-written, and the natural cut is mechanism_id.
        mech_counts: Dict[int, int] = {}
        for r in records:
            m = r.get("mechanism_id")
            if m is not None:
                mech_counts[int(m)] = mech_counts.get(int(m), 0) + 1
        if not mech_counts:
            strata.fail(
                "no per-episode mechanism_id in the sidecar: there is nothing to "
                "stratify a realistic set on"
            )
        else:
            spread = f"{min(mech_counts.values())}..{max(mech_counts.values())}"
            strata.note(
                f"{len(mech_counts)} mechanisms represented, " f"{spread} episodes each"
            )
    elif not hp_keys:
        strata.fail(
            "no per-episode sampled_hp in the sidecar: the stratified cuts have "
            "nothing to cut on (was collect_mechanism_diagnostics set?)"
        )
    else:
        strata.note(
            f"{len(hp_keys)} scalar hyperparameters recorded, {len(varying)} of "
            "them vary across episodes"
        )
    if a_degenerate:
        strata.note(
            f"{a_degenerate} episode(s) have a single-arm context -- expected to "
            "be rare; many of them means the propensity prior is degenerate"
        )

    checks = [structure, invariants, agreement, noise, grid, strata]
    stats["episodes_checked"] = n_seen
    stats["ok"] = all(c.ok for c in checks)
    stats["_checks"] = checks
    return stats


def _print_report(out_dir: Path, stats: Dict[str, Any]) -> None:
    print(f"\nFrozen set: {out_dir}")
    print(f"Episodes checked: {stats['episodes_checked']}\n")
    for check in stats["_checks"]:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}")
        for msg in check.failures:
            print(f"    ! {msg}")
        for msg in check.notes:
            print(f"    - {msg}")
    print()
    if stats["ok"]:
        print(
            "All checks passed. Next: eval_datasets.prior_eval.report --ckpt <ckpt> for the"
        )
        print("s/sigma_post read-off, which is the one thing this cannot answer.")
    else:
        print("FAILURES above. Read the first one: a later group's numbers are")
        print("only meaningful once the earlier groups pass.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval_datasets.prior_eval.verify",
        description="Integrity and sanity check for a frozen evaluation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="frozen set ROOT (the directory holding val/ and frozen_meta.pt)",
    )
    p.add_argument("--cate_vmin", type=float, default=-3.0)
    p.add_argument("--cate_vmax", type=float, default=3.0)
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="check only the first N episodes (a quick pass; the full run is "
        "the one that catches a short copy)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stats = verify(
        args.dir,
        cate_vmin=args.cate_vmin,
        cate_vmax=args.cate_vmax,
        max_episodes=args.max_episodes,
    )
    _print_report(args.dir, stats)
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
