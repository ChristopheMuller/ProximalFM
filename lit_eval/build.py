"""Build a fixed literature-dataset evaluation set.

    python -m lit_eval.build --dataset dummy_linear --name dummy_v1 \
        --n_datasets 48 --n_max 512 --n_q 64 --mc_cate 200 --seed 20260811

Output is the ONE on-disk format the rest of the project consumes -- a directory
of raw sparse ``batch_*.pt`` files plus ``metadata.json``, exactly what
``proximalfm.prior._genload.SavePriorDataset`` writes and ``LoadPriorDataset``
reads -- so a build is used exactly like a ``synt_eval`` build::

    <out_root>/<name>/
        val/                # benchmark:  data: {source: load, val_dir: <...>/val}
        frozen_meta.pt      # only when --mc_cate K > 1
        manifest.json
        config.yaml

    python -m prior_eval.score --dir <out_root>/<name> --ckpt <ckpt>
    python -m prior_eval.verify --dir <out_root>/<name>

One dataset per build. Unlike the realistic builder, which mixes twelve
mechanisms into one pool, literature datasets differ in dimension, sample size
and semantics, and mixing them would average away exactly the per-dataset
comparison the literature table is for.

Replicate mode (``--mc_cate K``, ``K > 1``)
-------------------------------------------

At ``K = 1`` the set carries realized potential outcomes: ``MU0``/``MU1`` are
single ITE draws, no sidecar is written, and the set supports point metrics
only. The trainer *refuses* such a set (``--synt_eval_dir`` requires
``frozen_meta.pt``) rather than silently scoring a posterior against sampled
ITEs, which would be minimised by an over-wide posterior.

At ``K > 1`` each query row is replicated through the latent and noise draw with
its covariate row held fixed, ``MU0``/``MU1`` become Monte Carlo estimates of
``E[Y_a | X]`` and ``S2`` their paired sampling variance, and the sidecar makes
the set scorable distributionally. Memory scales as ``N = n_max + n_q*K`` rows
for ONE episode at a time, and an unaffordable ``K`` raises rather than being
capped -- a silent cap would make the target noise vary across episodes and
machines, invisibly.

Because ``n_q`` is what trades against ``K``, the row split may be given as
``--n_max C --n_q Q`` (the spelling ``prior_eval`` uses) or as
``--seq_len``/``--prop_train``. They are two spellings of the same pair, resolve
identically, and produce the same ``content_digest``; passing both is refused.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

from proximalfm.prior._genload import write_batch_file

# `python lit_eval/build.py` and `python -m lit_eval.build` put different
# directories on sys.path; put the repo root on it so the package-absolute
# imports below resolve under either spelling, and there is exactly one module
# object for each (two would give two distinct classes and break isinstance).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lit_eval.base import (  # noqa: E402
    POOL_FIELDS,
    EpisodeSpec,
    ProximalDataset,
    stack_episodes,
)
from lit_eval.registry import available_datasets, build_dataset  # noqa: E402
from lit_eval.sidecar import (  # noqa: E402
    LitBuildConfig,
    check_no_silent_mutation,
    episode_records,
    write_sidecar,
)

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def generate_pool(
    dataset: ProximalDataset,
    n_datasets: int,
    seq_len: int,
    train_size: int,
    mc_cate: int,
    seed: int,
    mem_budget: int,
) -> Tuple[Tuple[Tensor, ...], List[Dict[str, Any]], List[int]]:
    """Draw ``n_datasets`` episodes and stack them into the pool.

    Episode ``i`` is seeded from ``seed`` and ``i`` alone, so episodes are
    independent of how many were requested and of the order they are drawn in.
    No shuffle is applied (and none is needed): with one dataset per build there
    is no mechanism ordering to break up, and keeping build order equal to disk
    order is what makes the sidecar's positional alignment checkable.
    """
    episodes: List[Tuple[Tensor, ...]] = []
    infos: List[Dict[str, Any]] = []
    seeds: List[int] = []
    for i in range(n_datasets):
        episode_seed = int(seed) + 1_000 * i
        spec = EpisodeSpec(
            index=i,
            seed=episode_seed,
            seq_len=seq_len,
            train_size=train_size,
            mc_cate=mc_cate,
        )
        tensors, info = dataset.generate(spec, mem_budget=mem_budget)
        episodes.append(tensors)
        infos.append(info)
        seeds.append(episode_seed)

    pool = stack_episodes(episodes, seq_len=seq_len, train_size=train_size)
    return pool, infos, seeds


# -----------------------------------------------------------------------------
# val/ -> batch_*.pt (raw, sparse) + metadata.json
# -----------------------------------------------------------------------------


def _save_val_batch(
    pool: Tuple[Tensor, ...], idx: range, out_dir: Path, file_idx: int
) -> None:
    """Write episodes ``idx`` as one ``batch_*.pt`` through the canonical writer.

    ``proximalfm.prior._genload.write_batch_file`` is the single writer of this
    format. It stores whatever shape it is handed and does not move tensors off
    the accelerator, so the trailing-singleton squeeze and the ``.cpu()`` happen
    here.
    """
    p = dict(zip(POOL_FIELDS, pool))
    sl = slice(idx.start, idx.stop)

    def block(name: str) -> Tensor:
        # write_batch_file uses .view(), which refuses a non-contiguous input.
        return p[name][sl].cpu().contiguous()

    def col(name: str) -> Tensor:
        t = p[name][sl].cpu()
        if t.dim() == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        return t.contiguous()

    fields = {
        **{k: block(k) for k in ("U", "X", "W", "Z")},
        **{
            k: col(k) for k in ("A", "A_probs", "Y_obs", "Y0", "Y1", "MU0", "MU1", "S2")
        },
        **{
            k: p[k][sl].cpu()
            for k in ("d_U", "d_X", "d_W", "d_Z", "seq_lens", "train_sizes", "K_eff")
        },
    }
    write_batch_file(out_dir, file_idx, seq_len_per_gp=False, **fields)


def save_validation(
    pool: Tuple[Tensor, ...],
    out_dir: Path,
    dataset: ProximalDataset,
    mc_cate: int,
    datasets_per_file: Optional[int],
    cli_command: str,
) -> Tuple[int, int]:
    """Write ``val/``: sparse ``batch_*.pt`` files + ``metadata.json``.

    ``LoadPriorDataset`` reconstructs each file to *that file's* max active
    feature dim, so files are only safe to consume one at a time: if the loader
    ever concatenates two (because one file holds fewer episodes than the
    trainer's ``batch_size``) the feature widths can mismatch. Default is
    therefore a SINGLE file, which is safe at any batch size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    B = pool[1].shape[0]
    T = int(pool[POOL_FIELDS.index("seq_lens")][0].item())
    train_n = int(pool[POOL_FIELDS.index("train_sizes")][0].item())
    dims = dataset.max_dims

    if datasets_per_file is None or datasets_per_file <= 0:
        datasets_per_file = B

    n_files = 0
    for file_idx, start in enumerate(range(0, B, datasets_per_file)):
        stop = min(start + datasets_per_file, B)
        _save_val_batch(pool, range(start, stop), out_dir, file_idx)
        n_files += 1

    # Mirrors SavePriorDataset.save_metadata, which LoadPriorDataset reads.
    metadata = {
        "prior_type": f"lit_eval:{dataset.name}",
        "batch_size": datasets_per_file,
        "batch_size_per_gp": 1,
        "min_seq_len": T,
        "max_seq_len": T,
        "log_seq_len": False,
        "seq_len_per_gp": False,
        "min_dim_U": 1,
        "max_dim_U": dims.d_U,
        "min_dim_W": 1,
        "max_dim_W": dims.d_W,
        "min_dim_Z": 1,
        "max_dim_Z": dims.d_Z,
        "min_dim_X": 1,
        "max_dim_X": dims.d_X,
        "min_train_size": train_n,
        "max_train_size": train_n,
        "replay_small": False,
        "mc_cate": int(mc_cate),
        "mc_targets_on_context": False,
        "cli_command": cli_command,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return n_files, datasets_per_file


# -----------------------------------------------------------------------------
# Manifest / config
# -----------------------------------------------------------------------------


def write_manifest(
    root: Path,
    args: argparse.Namespace,
    dataset: ProximalDataset,
    n_val_files: int,
    datasets_per_file: int,
    cli_command: str,
) -> None:
    manifest = {
        "name": args.name,
        "dataset": dataset.name,
        "citation": dataset.citation,
        "n_datasets": args.n_datasets,
        "seq_len": args.seq_len,
        "prop_train": args.prop_train,
        # The same split, spelled the way prior_eval spells it.
        "n_max": args.n_max,
        "n_q": args.n_q,
        "mc_cate": args.mc_cate,
        "seed": args.seed,
        "options": dataset.options,
        "cli_command": cli_command,
        "datasets_per_val_file": datasets_per_file,
        "n_val_files": n_val_files,
        "val_dir": str((root / "val").resolve()),
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def write_config(
    root: Path,
    args: argparse.Namespace,
    dataset: ProximalDataset,
    cli_command: str,
) -> None:
    """Dump every resolved setting (defaults filled in) to ``config.yaml``.

    Unlike the CLI string, this captures the *effective* value of each argument
    even when it was left at its default, so the build stays reproducible if the
    defaults change later. Re-run with ``--config config.yaml``.
    """
    config = dict(vars(args))
    config.pop("config", None)
    # Persist the resolved option dict, not the raw ``k=v`` strings.
    config["option"] = []
    config["options"] = dict(dataset.options)
    config["cli_command"] = cli_command
    with open(root / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build(args: argparse.Namespace) -> Path:
    dataset = build_dataset(args.dataset, **parse_options(args.option))
    root = Path(args.out_root) / args.name
    val_dir = root / "val"
    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])

    # Refuse to rewrite a differently-configured set in place *before* spending
    # the generation time. Only replicate sets carry a sidecar, so only they
    # have an identity to conflict on.
    sidecar_cfg = None
    if args.mc_cate > 1:
        sidecar_cfg = LitBuildConfig(
            dataset=dataset.name,
            n_datasets=args.n_datasets,
            seq_len=args.seq_len,
            prop_train=args.prop_train,
            mc_cate=args.mc_cate,
            seed=args.seed,
            options=dict(dataset.options),
        )
        if root.exists():
            check_no_silent_mutation(root, sidecar_cfg, force=args.force)

    train_n = args.n_max
    print(
        f"Building '{args.name}' from {dataset.describe()}\n"
        f"  {args.n_datasets} episodes x T={args.seq_len} "
        f"(n_max={args.n_max} + n_q={args.n_q}), K={args.mc_cate}, "
        f"seed={args.seed}"
    )

    pool, infos, seeds = generate_pool(
        dataset,
        n_datasets=args.n_datasets,
        seq_len=args.seq_len,
        train_size=train_n,
        mc_cate=args.mc_cate,
        seed=args.seed,
        mem_budget=args.mc_mem_budget,
    )

    n_val_files, dpf = save_validation(
        pool, val_dir, dataset, args.mc_cate, args.datasets_per_file, cli_command
    )
    print(
        f"  [datasets] {n_val_files} file(s) in {val_dir}\n"
        f"             benchmark: data: {{source: load, val_dir: {val_dir}}}"
    )

    if sidecar_cfg is not None:
        records = episode_records(
            pool, POOL_FIELDS, infos, seeds, dataset.name, train_n
        )
        digest = write_sidecar(
            root,
            sidecar_cfg,
            records,
            train_n,
            dpf,
            citation=dataset.citation,
            cli_command=cli_command,
        )
        print(
            f"  [sidecar]   {root / 'frozen_meta.pt'}  "
            f"(content_digest={digest}, n_max={train_n}, K={args.mc_cate})\n"
            f"             score it:  python -m prior_eval.score "
            f"--dir {root} --ckpt <ckpt>\n"
            f"             verify it: python -m prior_eval.verify --dir {root}"
        )
    else:
        print(
            "             no sidecar at --mc_cate 1: point metrics only, and "
            "the trainer will refuse this set. Build with K > 1 for a "
            "distributionally scorable one."
        )

    write_manifest(root, args, dataset, n_val_files, dpf, cli_command)
    write_config(root, args, dataset, cli_command)
    print(
        f"  [manifest]  {root / 'manifest.json'}\n  [config]    {root / 'config.yaml'}"
    )
    return root


def parse_options(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``--option k=v`` -> a typed dict (YAML scalar rules for ``v``)."""
    out: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--option expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lit_eval.build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file to seed defaults from (e.g. a config.yaml from a "
        "previous build). Explicit CLI flags still win.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the registered datasets and exit.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Registered dataset adapter to build from (see --list).",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Subfolder / identifier for this build (e.g. demand_val_v1). Use a "
        "different name -- never a rebuild in place -- for a different config.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="data/lit_eval/",
        help="Root under which <name>/{val,frozen_meta.pt,...} is written.",
    )
    p.add_argument(
        "--option",
        type=str,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Dataset-specific option, repeatable. Part of the set's "
        "content_digest, so a knob that changes the data cannot move silently.",
    )
    p.add_argument("--n_datasets", type=int, default=48, help="Episodes to build.")
    p.add_argument("--seq_len", type=int, default=2048, help="Rows per episode.")
    p.add_argument(
        "--prop_train",
        type=float,
        default=0.75,
        help="Fraction of rows used as context (the rest are query rows).",
    )
    p.add_argument(
        "--n_max",
        type=int,
        default=None,
        help="Context rows per episode. Alternative spelling of the row split "
        "(prior_eval's vocabulary); requires --n_q and is mutually exclusive "
        "with --seq_len/--prop_train.",
    )
    p.add_argument(
        "--n_q",
        type=int,
        default=None,
        help="Query rows per episode -- the axis that trades against --mc_cate "
        "for memory (N = n_max + n_q*K). Requires --n_max.",
    )
    p.add_argument(
        "--mc_cate",
        type=int,
        default=1,
        help="Monte Carlo replicates per query row. 1 (default) = no "
        "replication: MU0/MU1 are the realized potential outcomes, no sidecar "
        "is written, and the set is point-only. K > 1 turns on replicate mode.",
    )
    p.add_argument("--seed", type=int, default=20260811, help="Master seed.")
    p.add_argument(
        "--datasets_per_file",
        type=int,
        default=None,
        help="Episodes per val batch_*.pt. Default: all in one file (safest).",
    )
    p.add_argument(
        "--mc_mem_budget",
        type=int,
        default=2_000_000_000,
        help="Element budget for the transient N-row replicate block of ONE "
        "episode. Exceeding it raises rather than silently capping K.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Discard an existing set at the same path instead of refusing.",
    )
    return p


def resolve_rows(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str],
) -> argparse.Namespace:
    """Fold ``--n_max``/``--n_q`` into the ``seq_len``/``prop_train`` pair.

    A front-end spelling, not a second parameterisation: both resolve to the
    same ``(seq_len, prop_train)`` and hence the same ``content_digest``. Mixing
    the two is refused -- a split that came from two sources silently is exactly
    what the digest cannot warn about, since the resulting set is
    self-consistent.
    """
    if (args.n_max is None) != (args.n_q is None):
        parser.error("--n_max and --n_q must be given together")

    if args.n_max is None:
        n_ctx = max(1, min(args.seq_len - 1, round(args.prop_train * args.seq_len)))
        args.n_max, args.n_q = n_ctx, args.seq_len - n_ctx
        return args

    conflicting = [f for f in ("--seq_len", "--prop_train") if f in argv]
    if conflicting:
        parser.error(
            f"--n_max/--n_q replace {'/'.join(conflicting)}; pass one spelling "
            "of the row split, not both"
        )
    if args.n_max < 1 or args.n_q < 1:
        parser.error("--n_max and --n_q must both be >= 1")

    args.seq_len = args.n_max + args.n_q
    args.prop_train = args.n_max / args.seq_len
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args, seeding defaults from ``--config`` YAML if given.

    Precedence (low -> high): argparse defaults < config file < explicit flags.
    """
    parser = build_parser()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        with open(known.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in file_cfg.items() if k in valid})
        # `options:` in a config.yaml is the resolved dict; feed it back in as
        # --option pairs so a replayed build gets the same knobs.
        opts = file_cfg.get("options") or {}
        if opts:
            parser.set_defaults(
                option=[f"{k}={json.dumps(v)}" for k, v in opts.items()]
            )

    cli = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.list:
        return args
    for required in ("dataset", "name"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required (or supply it via --config)")
    return resolve_rows(parser, args, cli)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.list:
        print("Registered datasets:")
        for name in available_datasets():
            print(f"  {name:24s} {build_dataset(name).citation}")
        return
    build(args)


if __name__ == "__main__":
    main()
