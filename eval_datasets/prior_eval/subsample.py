"""Derive a smaller frozen set by taking a context prefix of an existing one.

    uv run python -m eval_datasets.prior_eval.subsample \
        --src data/prior_eval/prior_val_v2_smalldim_20000rows \
        --name prior_val_v2_smalldim_2000rows_subsample --n_max 2000

Most estimators in ``benchmarks/`` are kernel methods whose cost is quadratic
or cubic in the row count, so a 20k-row set is not runnable across the method
grid.  This module writes a set with a smaller ``n_max`` **without
regenerating**, which matters for more than speed: the episodes are then the
*same SCMs with the same draws*, so a comparison across sizes varies ``n`` and
nothing else.

What "subsample" means here, precisely
--------------------------------------
Not the first ``K`` rows of the file.  An episode is laid out as ``n_max``
context rows followed by ``n_q`` query rows, and only the query rows carry the
Monte-Carlo targets (``MU0``/``MU1``/``S2``) that any CATE or ATE score is
computed against.  Slicing ``[0:K]`` off the front would keep only context and
produce a set with no estimand at all.  The operation is therefore

    rows = [0 : K]  ++  [n_max : n_max + n_q]

i.e. a **context prefix with the query block carried over whole**.  That is the
same object ``BuildConfig.n_grid`` already names -- "prefix ``n`` is the first
``n`` context rows of an already-generated episode" -- so the estimand is
untouched by construction and ``K`` trades against nothing.

The sidecar is *not* copied verbatim
------------------------------------
``y_context_mean``/``y_context_std`` are context-only statistics, so they move
with ``K`` and are recomputed here through the same ``_frozen_scale`` the
builder uses (the trainer's biased-``std`` formula, not ``predict``'s).
``tau_star`` and ``s2`` are properties of the query rows and are carried over
unchanged; their ``*_std`` companions are re-derived against the new scale.

``content_digest`` is deliberately **not** a native digest.  Recomputing one
from a ``BuildConfig`` with ``n_max=K`` would produce the hash of a set
*generated* at that size -- a different set, with different RNG draws -- and
``verify.py`` would then happily certify this one as that.  A derived marker is
written instead, so the mismatch is loud rather than silent, and the parent's
digest is recorded beside it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from proximalfm.prior._genload import sparse2dense, write_batch_file
from eval_datasets.prior_eval.build_prior_frozen_set import (
    DEFAULT_N_GRID,
    FROZEN_META_VERSION,
    _frozen_scale,
    _git_rev,
)

#: Row-major fields stored as ``[B, T]`` -- sliced along the row axis.
_ROW_FIELDS = ("A", "A_probs", "Y_obs", "Y0", "Y1", "MU0", "MU1", "S2")
#: Feature blocks stored sparsely, reconstructed to ``[B, T, d]`` before slicing.
_FEATURE_FIELDS = ("U", "X", "W", "Z")


def _densify(payload: Dict[str, Any], field: str, T: int) -> torch.Tensor:
    dims = payload[f"d_{field}"]
    b = dims.shape[0]
    max_d = int(dims.max().item())
    flat = sparse2dense(payload[field], dims.repeat_interleave(T), max_len=max_d)
    return flat.view(b, T, max_d)


def _row_index(n_max: int, n_q: int, k: int) -> torch.Tensor:
    """``[0:k] ++ [n_max : n_max+n_q]`` -- prefix plus the whole query block."""
    return torch.cat(
        [
            torch.arange(0, k, dtype=torch.long),
            torch.arange(n_max, n_max + n_q, dtype=torch.long),
        ]
    )


def subsample(
    src: str | Path,
    out: str | Path,
    *,
    n_max: int,
    progress: bool = True,
    cli_command: str = "",
) -> Path:
    """Write a context-prefix set of ``src`` at ``out``; return its ``val/``."""
    src, out = Path(src), Path(out)
    src_val = src / "val"
    files = sorted(src_val.glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files in {src_val}")

    src_meta = torch.load(
        src / "frozen_meta.pt", map_location="cpu", weights_only=False
    )
    g = src_meta["global"]
    src_n_max, n_q = int(g["n_max"]), int(g["n_q"])
    if not 0 < n_max <= src_n_max:
        raise ValueError(
            f"--n_max must be in (0, {src_n_max}] (the source's n_max); got {n_max}."
        )
    if n_max == src_n_max:
        raise ValueError(
            f"--n_max {n_max} equals the source's; that is a copy, not a subsample."
        )

    val_dir = out / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    idx = _row_index(src_n_max, n_q, n_max)
    new_T = len(idx)

    y_scales: List[tuple] = []
    for batch_idx, path in enumerate(tqdm(files, desc="batches", disable=not progress)):
        p = torch.load(path, map_location="cpu", weights_only=True)
        T = int(p["seq_lens"][0].item())
        if T != src_n_max + n_q:
            raise ValueError(
                f"{path}: seq_len {T} does not match the sidecar's "
                f"n_max + n_q = {src_n_max + n_q}."
            )
        b = p["d_X"].shape[0]

        feats = {f: _densify(p, f, T)[:, idx, :] for f in _FEATURE_FIELDS}
        rows = {f: p[f][:, idx] for f in _ROW_FIELDS}
        for i in range(b):
            y_scales.append(_frozen_scale(rows["Y_obs"][i], n_max))

        write_batch_file(
            val_dir,
            batch_idx,
            **feats,
            **rows,
            d_U=p["d_U"],
            d_X=p["d_X"],
            d_W=p["d_W"],
            d_Z=p["d_Z"],
            K_eff=p["K_eff"],
            seq_lens=torch.full((b,), new_T, dtype=torch.long),
            train_sizes=torch.full((b,), n_max, dtype=torch.long),
            seq_len_per_gp=False,
        )

    # ── sidecar ──────────────────────────────────────────────────────────────
    src_digest = g.get("content_digest")
    new_global = {
        **g,
        "n_max": n_max,
        "seq_len": new_T,
        "n_grid": [n for n in DEFAULT_N_GRID if n <= n_max] + [n_max],
        "content_digest": f"derived:prefix{n_max}:{src_digest}",
        "derived_from": {
            "path": str(src),
            "content_digest": src_digest,
            "n_max": src_n_max,
        },
        "derived_op": "context_prefix",
        "cli_command": cli_command,
        "git_rev": _git_rev(),
    }
    new_global["n_grid"] = sorted(set(new_global["n_grid"]))

    episodes = []
    for rec, (y_mean, y_std) in zip(src_meta["episodes"], y_scales):
        rec = dict(rec)
        rec["y_context_mean"] = y_mean
        rec["y_context_std"] = y_std
        rec["query_index"] = torch.arange(n_max, new_T, dtype=torch.long)
        rec["tau_star_std"] = rec["tau_star"] / y_std
        rec["s2_std"] = rec["s2"] / (y_std**2)
        episodes.append(rec)

    meta = {
        "format_version": FROZEN_META_VERSION,
        "global": new_global,
        "episodes": episodes,
    }
    tmp = out / "frozen_meta.pt.tmp"
    torch.save(meta, tmp)
    tmp.replace(out / "frozen_meta.pt")
    return val_dir


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="source set ROOT (holds val/)")
    p.add_argument("--out", default=None, help="explicit destination root")
    p.add_argument("--out_root", default="data/prior_eval/", help="with --name")
    p.add_argument("--name", default=None, help="destination name under --out_root")
    p.add_argument(
        "--n_max", type=int, required=True, help="context rows to keep per episode"
    )
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    args = p.parse_args(argv)

    if not (args.out or args.name):
        p.error("give --out or --name")
    out = Path(args.out) if args.out else Path(args.out_root) / args.name

    if (out / "val").exists() and any((out / "val").glob("batch_*.pt")):
        if not args.force:
            p.error(f"{out}/val already holds batch files; pass --force to overwrite.")
        for f in (out / "val").glob("batch_*.pt"):
            f.unlink()

    cli = shlex.join(
        [
            sys.executable,
            "-m",
            "eval_datasets.prior_eval.subsample",
            *(argv or sys.argv[1:]),
        ]
    )
    val_dir = subsample(args.src, out, n_max=args.n_max, cli_command=cli)

    meta = torch.load(out / "frozen_meta.pt", map_location="cpu", weights_only=False)
    g = meta["global"]
    print(f"\nwrote {len(list(val_dir.glob('batch_*.pt')))} batch files -> {val_dir}")
    print(f"  episodes   : {len(meta['episodes'])}")
    print(f"  n_max      : {g['derived_from']['n_max']} -> {g['n_max']}")
    print(f"  seq_len    : {g['seq_len']}  ({g['n_max']} context + {g['n_q']} query)")
    print(f"  digest     : {g['content_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
