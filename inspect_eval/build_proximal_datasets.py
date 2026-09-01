"""Build fixed datasets from the minimal linear-Gaussian proximal SCM.

The output mirrors ``synt_eval``: ``<out_root>/<name>/val`` contains canonical
``batch_*.pt`` files plus ``metadata.json``; the root also contains a resolved
``config.yaml`` and ``manifest.json``.  With ``--mc_cate K`` for K > 1, a
``frozen_meta.pt`` sidecar makes the set usable by the trainer's
``--synt_eval_dir`` and by ``prior_eval.score``.

Example::

    python -m inspect_eval.build_proximal_datasets \
        --name linear_smoke --n_max 512 --n_q 128 --mc_cate 1000
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import yaml

from proximalfm.prior._genload import write_batch_file
from synt_eval.build_proximal_datasets import POOL_FIELDS, resolve_rows

from inspect_eval.linear_gaussian_mechanism import (
    GeneratorConfig,
    LinearGaussianGenerator,
)
from inspect_eval.sidecar import (
    InspectBuildConfig,
    check_no_silent_mutation,
    episode_records,
    write_sidecar,
)

Tensor = torch.Tensor


def generate_pool(
    gen: LinearGaussianGenerator,
    n_datasets: int,
    seq_len: int,
    train_size: int | float,
    shuffle: bool = True,
) -> Tuple[Tuple[Tensor, ...], Tensor, Tensor]:
    pool = gen.get_batch(
        mechanism_id=1,
        batch_size=n_datasets,
        seq_len=seq_len,
        train_size=train_size,
    )
    mechanism_ids = torch.ones(n_datasets, dtype=torch.long)
    seeds = int(gen.cfg.seed) + 100_000 + torch.arange(n_datasets, dtype=torch.long)
    if shuffle:
        g = torch.Generator().manual_seed(int(gen.cfg.seed) + 7)
        order = torch.randperm(n_datasets, generator=g)
        pool = tuple(t[order] for t in pool)
        mechanism_ids, seeds = mechanism_ids[order], seeds[order]
    return pool, mechanism_ids, seeds


def _save_val_batch(pool: Tuple[Tensor, ...], sl: slice, out: Path, index: int) -> None:
    p = dict(zip(POOL_FIELDS, pool))

    def block(name: str) -> Tensor:
        return p[name][sl].cpu().contiguous()

    def col(name: str) -> Tensor:
        value = p[name][sl].cpu()
        if value.dim() == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value.contiguous()

    fields = {
        **{name: block(name) for name in ("U", "X", "W", "Z")},
        **{
            name: col(name)
            for name in ("A", "A_probs", "Y_obs", "Y0", "Y1", "MU0", "MU1", "S2")
        },
        **{
            name: p[name][sl].cpu()
            for name in ("d_U", "d_X", "d_W", "d_Z", "seq_lens", "train_sizes", "K_eff")
        },
    }
    write_batch_file(out, index, seq_len_per_gp=False, **fields)


def save_validation(
    pool: Tuple[Tensor, ...],
    out: Path,
    cfg: GeneratorConfig,
    datasets_per_file: Optional[int],
    cli_command: str,
) -> Tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    n_datasets = pool[1].shape[0]
    datasets_per_file = datasets_per_file or n_datasets
    if datasets_per_file < 1:
        datasets_per_file = n_datasets

    n_files = 0
    for index, start in enumerate(range(0, n_datasets, datasets_per_file)):
        _save_val_batch(
            pool, slice(start, min(start + datasets_per_file, n_datasets)), out, index
        )
        n_files += 1

    seq_len = int(pool[POOL_FIELDS.index("seq_lens")][0])
    metadata = {
        "prior_type": "inspect_linear_gaussian",
        "batch_size": datasets_per_file,
        "batch_size_per_gp": 1,
        "min_seq_len": seq_len,
        "max_seq_len": seq_len,
        "log_seq_len": False,
        "seq_len_per_gp": False,
        "min_dim_U": 1,
        "max_dim_U": 1,
        "min_dim_X": 1,
        "max_dim_X": 1,
        "min_dim_W": 1,
        "max_dim_W": 1,
        "min_dim_Z": 1,
        "max_dim_Z": 1,
        "min_train_size": cfg.train_size,
        "max_train_size": cfg.train_size,
        "replay_small": False,
        "mc_cate": cfg.mc_cate,
        "mc_targets_on_context": False,
        "cli_command": cli_command,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return n_files, datasets_per_file


def build(args: argparse.Namespace) -> None:
    root = Path(args.out_root) / args.name
    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
    sidecar_cfg = None
    if args.mc_cate > 1:
        sidecar_cfg = InspectBuildConfig(
            n_datasets=args.n_datasets,
            seq_len=args.seq_len,
            prop_train=args.prop_train,
            mc_cate=args.mc_cate,
            seed=args.seed,
            overlap_eps=args.overlap_eps,
            extra_observation_noise=args.extra_observation_noise,
            shuffle=not args.no_shuffle,
        )
        if root.exists():
            check_no_silent_mutation(root, sidecar_cfg, force=args.force)

    with tempfile.TemporaryDirectory(prefix="inspect_gen_") as scratch:
        cfg = GeneratorConfig(
            save_dir=scratch,
            K=args.n_datasets,
            seq_len=args.seq_len,
            train_size=args.prop_train,
            min_dim_U=1,
            max_dim_U=1,
            min_dim_X=1,
            max_dim_X=1,
            min_dim_W=1,
            max_dim_W=1,
            min_dim_Z=1,
            max_dim_Z=1,
            overlap_eps=args.overlap_eps,
            extra_observation_noise=args.extra_observation_noise,
            mc_cate=args.mc_cate,
            mc_mem_budget=args.mc_mem_budget,
            seed=args.seed,
            device="cpu",
        )
        pool, mechanism_ids, seeds = generate_pool(
            LinearGaussianGenerator(cfg),
            args.n_datasets,
            args.seq_len,
            args.prop_train,
            shuffle=not args.no_shuffle,
        )

    val_dir = root / "val"
    n_files, dpf = save_validation(
        pool, val_dir, cfg, args.datasets_per_file, cli_command
    )
    digest = None
    if sidecar_cfg is not None:
        n_max = int(pool[POOL_FIELDS.index("train_sizes")][0])
        records = episode_records(pool, POOL_FIELDS, mechanism_ids, seeds, n_max)
        digest = write_sidecar(root, sidecar_cfg, records, n_max, dpf, cli_command)

    config = {k: v for k, v in vars(args).items() if k != "config"}
    config.update(
        mechanism="linear_gaussian",
        mechanism_version=1,
        cli_command=cli_command,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    manifest = {
        "name": args.name,
        "mechanism": "linear_gaussian",
        "n_datasets": args.n_datasets,
        "seq_len": args.seq_len,
        "prop_train": args.prop_train,
        "n_max": args.n_max,
        "n_q": args.n_q,
        "mc_cate": args.mc_cate,
        "seed": args.seed,
        "dimensions": {"U": 1, "X": 1, "W": 1, "Z": 1},
        "n_val_files": n_files,
        "datasets_per_val_file": dpf,
        "val_dir": str(val_dir.resolve()),
        "content_digest": digest,
        "cli_command": cli_command,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Built {args.n_datasets} scalar linear-Gaussian datasets in {root}")
    print(f"  benchmark: data: {{source: load, val_dir: {val_dir}}}")
    if digest:
        print(f"  trainer: --synt_eval_dir {root} (content_digest={digest})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out_root", default="data/inspect_eval/")
    parser.add_argument("--n_datasets", type=int, default=240)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--prop_train", type=float, default=0.75)
    parser.add_argument("--n_max", type=int)
    parser.add_argument("--n_q", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--datasets_per_file", type=int)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--mc_cate", type=int, default=1)
    parser.add_argument("--mc_mem_budget", type=int, default=2_000_000_000)
    parser.add_argument("--overlap_eps", type=float, default=0.02)
    parser.add_argument("--extra_observation_noise", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        with open(known.config) as handle:
            values = yaml.safe_load(handle) or {}
        destinations = {action.dest for action in parser._actions}
        overrides = {k: v for k, v in values.items() if k in destinations}
        parser.set_defaults(**overrides)
        for action in parser._actions:
            if action.dest in overrides:
                action.required = False
    cli = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.n_datasets < 1:
        parser.error("--n_datasets must be >= 1")
    if args.mc_cate < 1:
        parser.error("--mc_cate must be >= 1")
    return resolve_rows(parser, args, cli)


if __name__ == "__main__":
    build(parse_args())
