"""Build benchmark-compatible semi-synthetic sets from arbitrary CSV files.

Examples::

    uv run python -m semi_synt_eval.build --list
    uv run python -m semi_synt_eval.build --csv bank --name bank_linear_v1 \
        --n_max 1024 --n_q 256

The package has no dependency on ``lit_eval`` and performs no Monte Carlo
replication. Bundled mechanisms write their analytic CATE explicitly; a future
ITE-only mechanism may omit it and the benchmark will use ``Y1 - Y0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import yaml

from proximalfm.prior._genload import write_batch_file
from semi_synt_eval.generator import CSVProximalDataset, Episode
from semi_synt_eval.mechanisms import available_mechanisms
from semi_synt_eval.source import DEFAULT_DATA_DIR, discover_csvs, load_csv

FORMAT_VERSION = 1


def make_dataset(args: argparse.Namespace) -> CSVProximalDataset:
    source = load_csv(
        args.csv,
        data_dir=args.data_dir,
        x_columns=args.x_columns,
        u_columns=args.u_columns,
        drop_columns=args.drop_columns,
        max_features=args.max_features,
        u_fraction=args.u_fraction,
        split_seed=args.split_seed,
    )
    return CSVProximalDataset(
        source,
        mechanism=args.mechanism,
        proxy_noise=args.proxy_noise,
        treatment_scale=args.treatment_scale,
        outcome_noise=args.outcome_noise,
        effect_scale=args.effect_scale,
        overlap_eps=args.overlap_eps,
        row_sampling=args.row_sampling,
    )


def generate_episodes(
    dataset: CSVProximalDataset,
    *,
    n_datasets: int,
    seq_len: int,
    seed: int,
) -> tuple[list[Episode], list[int]]:
    episodes: list[Episode] = []
    seeds: list[int] = []
    for index in range(n_datasets):
        episode_seed = int(seed) + 1_000 * index
        episodes.append(dataset.generate(seed=episode_seed, seq_len=seq_len))
        seeds.append(episode_seed)
    return episodes, seeds


def _stack(episodes: Sequence[Episode], field: str) -> torch.Tensor:
    return torch.stack([getattr(episode, field) for episode in episodes])


def save_validation(
    episodes: Sequence[Episode],
    *,
    out_dir: Path,
    train_size: int,
    datasets_per_file: Optional[int],
    dataset: CSVProximalDataset,
    cli_command: str,
) -> tuple[int, int]:
    """Write the canonical sparse ``val/`` format without replicate targets."""
    if not episodes:
        raise ValueError("no episodes to save")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(episodes)
    dpf = (
        total if not datasets_per_file or datasets_per_file <= 0 else datasets_per_file
    )
    n_files = 0
    for file_idx, start in enumerate(range(0, total, dpf)):
        block = episodes[start : min(start + dpf, total)]
        batch_size = len(block)
        seq_len = block[0].X.shape[0]
        dims = {
            name: torch.tensor([getattr(e, name) for e in block], dtype=torch.long)
            for name in ("d_U", "d_X", "d_W", "d_Z")
        }
        cate = _stack(block, "CATE") if block[0].CATE is not None else None
        if any((e.CATE is None) != (cate is None) for e in block):
            raise ValueError("episodes cannot mix exact-CATE and ITE-only targets")
        y0, y1 = _stack(block, "Y0"), _stack(block, "Y1")
        write_batch_file(
            out_dir,
            file_idx,
            U=_stack(block, "U"),
            X=_stack(block, "X"),
            W=_stack(block, "W"),
            Z=_stack(block, "Z"),
            A=_stack(block, "A").squeeze(-1),
            A_probs=_stack(block, "A_probs").squeeze(-1),
            Y_obs=_stack(block, "Y_obs").squeeze(-1),
            Y0=y0.squeeze(-1),
            Y1=y1.squeeze(-1),
            # Required legacy fields remain honest single draws. The explicit
            # CATE field, not these values, activates exact-CATE benchmarking.
            MU0=y0.squeeze(-1),
            MU1=y1.squeeze(-1),
            S2=torch.zeros(batch_size, seq_len),
            K_eff=torch.ones(batch_size, dtype=torch.long),
            CATE=None if cate is None else cate.squeeze(-1),
            seq_lens=torch.full((batch_size,), seq_len, dtype=torch.long),
            train_sizes=torch.full((batch_size,), train_size, dtype=torch.long),
            seq_len_per_gp=False,
            **dims,
        )
        n_files += 1

    metadata = {
        "prior_type": dataset.name,
        "batch_size": dpf,
        "batch_size_per_gp": 1,
        "min_seq_len": episodes[0].X.shape[0],
        "max_seq_len": episodes[0].X.shape[0],
        "log_seq_len": False,
        "seq_len_per_gp": False,
        "min_dim_U": dataset.d_U,
        "max_dim_U": dataset.d_U,
        "min_dim_X": dataset.d_X,
        "max_dim_X": dataset.d_X,
        "min_dim_W": dataset.d_W,
        "max_dim_W": dataset.d_W,
        "min_dim_Z": dataset.d_Z,
        "max_dim_Z": dataset.d_Z,
        "min_train_size": train_size,
        "max_train_size": train_size,
        "replay_small": False,
        "target_kind": dataset.target_capability,
        "cli_command": cli_command,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return n_files, dpf


def _owned_outputs(root: Path) -> list[Path]:
    paths = [
        root / "manifest.json",
        root / "config.yaml",
        root / "val" / "metadata.json",
    ]
    paths.extend((root / "val").glob("batch_*.pt"))
    paths.extend((root / "val").glob("batch_*.pt.tmp"))
    # Remove sidecars produced by the superseded replicate-based implementation
    # only under an explicit --force rebuild.
    paths.extend([root / "frozen_meta.pt", root / "frozen_meta.pt.tmp"])
    return [path for path in paths if path.exists()]


def _clear_owned_outputs(root: Path) -> None:
    for path in _owned_outputs(root):
        path.unlink()


def _content_digest(args: argparse.Namespace, dataset: CSVProximalDataset) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset.options,
        "n_datasets": args.n_datasets,
        "seq_len": args.seq_len,
        "train_size": args.n_max,
        "seed": args.seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_metadata_files(
    root: Path,
    args: argparse.Namespace,
    dataset: CSVProximalDataset,
    seeds: Sequence[int],
    *,
    n_files: int,
    datasets_per_file: int,
    cli_command: str,
) -> None:
    source = dataset.source
    manifest = {
        "format_version": FORMAT_VERSION,
        "content_digest": _content_digest(args, dataset),
        "name": args.name,
        "source_file": source.path.name,
        "source_path": str(source.path),
        "source_sha256": source.digest,
        "source_rows": int(source.values.shape[0]),
        "encoded_columns": list(source.columns),
        "x_columns": list(source.x_columns),
        "u_columns": list(source.u_columns),
        "mechanism": args.mechanism,
        "target_kind": dataset.target_capability,
        "n_datasets": args.n_datasets,
        "episode_seeds": list(seeds),
        "n_max": args.n_max,
        "n_q": args.n_q,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "datasets_per_val_file": datasets_per_file,
        "n_val_files": n_files,
        "val_dir": str((root / "val").resolve()),
        "cli_command": cli_command,
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    config = dict(vars(args))
    config.pop("config", None)
    for key in ("data_dir", "out_root"):
        config[key] = str(config[key])
    config["cli_command"] = cli_command
    with open(root / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=True, default_flow_style=False)


def build(args: argparse.Namespace) -> Path:
    dataset = make_dataset(args)
    root = Path(args.out_root) / args.name
    val_dir = root / "val"
    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
    existing = _owned_outputs(root)
    if existing and not args.force:
        raise FileExistsError(
            f"{root} already contains a generated set. Choose a new --name, "
            "or pass --force to replace builder-owned outputs."
        )

    source = dataset.source
    print(
        f"Building '{args.name}' from {source.path.name} with {args.mechanism}\n"
        f"  rows={source.values.shape[0]}, X={list(source.x_columns)}, "
        f"U={list(source.u_columns)}\n"
        f"  {args.n_datasets} episodes x T={args.seq_len} "
        f"(n_max={args.n_max} + n_q={args.n_q}), seed={args.seed}\n"
        f"  target={dataset.target_capability} (no Monte Carlo replication)"
    )
    episodes, seeds = generate_episodes(
        dataset,
        n_datasets=args.n_datasets,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    if existing and args.force:
        _clear_owned_outputs(root)
    n_files, dpf = save_validation(
        episodes,
        out_dir=val_dir,
        train_size=args.n_max,
        datasets_per_file=args.datasets_per_file,
        dataset=dataset,
        cli_command=cli_command,
    )
    _write_metadata_files(
        root,
        args,
        dataset,
        seeds,
        n_files=n_files,
        datasets_per_file=dpf,
        cli_command=cli_command,
    )
    print(f"  benchmark: data: {{source: load, val_dir: {val_dir}}}")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semi_synt_eval.build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Resolved YAML build config.")
    parser.add_argument(
        "--list", action="store_true", help="List available CSVs and mechanisms."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--csv", default=None, help="CSV path, filename, or stem under --data_dir."
    )
    parser.add_argument("--name", default=None, help="Build name under --out_root.")
    parser.add_argument("--out_root", type=Path, default=Path("data/real_eval"))
    parser.add_argument("--mechanism", choices=available_mechanisms(), default="linear")
    parser.add_argument(
        "--x_columns", default=None, help="Comma-separated explicit X columns."
    )
    parser.add_argument(
        "--u_columns", default=None, help="Comma-separated explicit U columns."
    )
    parser.add_argument(
        "--drop_columns", default=None, help="Comma-separated columns to ignore."
    )
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--u_fraction", type=float, default=0.4)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument(
        "--row_sampling",
        choices=("auto", "without_replacement", "bootstrap"),
        default="auto",
    )
    parser.add_argument("--proxy_noise", type=float, default=0.75)
    parser.add_argument("--treatment_scale", type=float, default=1.0)
    parser.add_argument("--outcome_noise", type=float, default=0.5)
    parser.add_argument("--effect_scale", type=float, default=1.0)
    parser.add_argument("--overlap_eps", type=float, default=0.02)
    parser.add_argument("--n_datasets", type=int, default=24)
    parser.add_argument("--seq_len", type=int, default=1280)
    parser.add_argument("--prop_train", type=float, default=0.8)
    parser.add_argument("--n_max", type=int, default=None)
    parser.add_argument("--n_q", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--datasets_per_file", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def resolve_rows(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str],
) -> argparse.Namespace:
    if (args.n_max is None) != (args.n_q is None):
        parser.error("--n_max and --n_q must be given together")
    if args.n_max is None:
        n_ctx = max(1, min(args.seq_len - 1, round(args.prop_train * args.seq_len)))
        args.n_max, args.n_q = n_ctx, args.seq_len - n_ctx
        return args
    conflicting = [flag for flag in ("--seq_len", "--prop_train") if flag in argv]
    if conflicting:
        parser.error(
            f"--n_max/--n_q replace {'/'.join(conflicting)}; pass one row-split spelling"
        )
    if args.n_max < 1 or args.n_q < 1:
        parser.error("--n_max and --n_q must both be >= 1")
    args.seq_len = args.n_max + args.n_q
    args.prop_train = args.n_max / args.seq_len
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        with open(known.config) as f:
            file_config = yaml.safe_load(f) or {}
        valid = {action.dest for action in parser._actions}
        parser.set_defaults(
            **{key: value for key, value in file_config.items() if key in valid}
        )
    cli = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.list:
        return args
    for field in ("csv", "name"):
        if getattr(args, field) is None:
            parser.error(f"--{field} is required (or supply it via --config)")
    if args.n_datasets < 1:
        parser.error("--n_datasets must be >= 1")
    return resolve_rows(parser, args, cli)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.list:
        print("CSV sources:")
        for path in discover_csvs(args.data_dir):
            print(f"  {path.stem:28s} {path.name}")
        print("Mechanisms:")
        for name in available_mechanisms():
            print(f"  {name}")
        return
    build(args)


if __name__ == "__main__":
    main()
