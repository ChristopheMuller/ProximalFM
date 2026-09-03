"""Build benchmark-compatible semi-synthetic sets from arbitrary CSV files.

Examples::

    uv run python -m semisynt_eval.build --list
    uv run python -m semisynt_eval.build --name all_mechanisms_v1

The package has no dependency on ``lit_eval`` and performs no row-level Monte
Carlo replication. Bundled mechanisms write their analytic CATE explicitly; a
future ITE-only mechanism may omit it and the benchmark will use ``Y1 - Y0``.
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
from semisynt_eval._config import load_mechanism_configs
from semisynt_eval.generator import CSVProximalDataset, Episode
from semisynt_eval.mechanisms import available_mechanisms
from semisynt_eval.source import discover_csvs, load_csv

FORMAT_VERSION = 3
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BUILD_CONFIG_FIELDS = (
    "data_dir",
    "out_root",
    "csv",
    "name",
    "mechanism",
    "x_columns",
    "u_columns",
    "drop_columns",
    "n_features_x",
    "n_features_u",
    "split_seed",
    "row_sampling",
    "replicates_per_csv",
    "seq_len",
    "prop_train",
    "n_max",
    "n_q",
    "seed",
    "datasets_per_file",
)


def make_datasets(args: argparse.Namespace) -> list[CSVProximalDataset]:
    selectors = discover_csvs(args.data_dir) if args.csv == "all" else [args.csv]
    if not selectors:
        raise ValueError(f"no CSV files found directly under {args.data_dir}")

    sources = []
    for selector in selectors:
        source = load_csv(
            selector,
            data_dir=args.data_dir,
            x_columns=args.x_columns,
            u_columns=args.u_columns,
            drop_columns=args.drop_columns,
            n_features_x=args.n_features_x,
            n_features_u=args.n_features_u,
            split_seed=args.split_seed,
        )
        expected_digest = args._expected_source_sha256.get(source.path.name)
        if expected_digest not in (None, source.digest):
            raise ValueError(
                f"source SHA-256 mismatch for {source.path}: expected "
                f"{expected_digest}, found {source.digest}"
            )
        sources.append(source)

    actual_files = {source.path.name for source in sources}
    expected_files = set(args._expected_source_sha256)
    if expected_files and actual_files != expected_files:
        raise ValueError(
            "source set differs from the saved recipe: "
            f"missing={sorted(expected_files-actual_files)}, "
            f"unexpected={sorted(actual_files-expected_files)}"
        )
    mechanism_names = (
        available_mechanisms() if args.mechanism == "all" else [args.mechanism]
    )
    return [
        CSVProximalDataset(
            source,
            mechanism=mechanism_name,
            mechanism_config=args._mechanism_configs[mechanism_name],
            row_sampling=args.row_sampling,
        )
        for source in sources
        for mechanism_name in mechanism_names
    ]


def generate_episodes(
    datasets: Sequence[CSVProximalDataset],
    *,
    replicates_per_csv: int,
    seq_len: int,
    seed: int,
) -> tuple[list[Episode], list[dict]]:
    episodes: list[Episode] = []
    records: list[dict] = []
    episode_index = 0
    for dataset in datasets:
        for replicate in range(replicates_per_csv):
            episode_seed = int(seed) + 1_000 * episode_index
            episodes.append(dataset.generate(seed=episode_seed, seq_len=seq_len))
            records.append(
                {
                    "source_file": dataset.source.path.name,
                    "mechanism": dataset.mechanism_name,
                    "replicate": replicate,
                    "seed": episode_seed,
                }
            )
            episode_index += 1
    return episodes, records


def _stack(episodes: Sequence[Episode], field: str) -> torch.Tensor:
    return torch.stack([getattr(episode, field) for episode in episodes])


def save_validation(
    episodes: Sequence[Episode],
    *,
    out_dir: Path,
    train_size: int,
    datasets_per_file: Optional[int],
    dataset: CSVProximalDataset | Sequence[CSVProximalDataset],
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

    datasets = [dataset] if isinstance(dataset, CSVProximalDataset) else list(dataset)
    target_kinds = {item.target_capability for item in datasets}
    if len(target_kinds) != 1:
        raise ValueError("all CSV sources must use the same target capability")
    metadata = {
        "prior_type": "semisynt_eval",
        "batch_size": dpf,
        "batch_size_per_gp": 1,
        "min_seq_len": episodes[0].X.shape[0],
        "max_seq_len": episodes[0].X.shape[0],
        "log_seq_len": False,
        "seq_len_per_gp": False,
        "min_dim_U": min(e.d_U for e in episodes),
        "max_dim_U": max(e.d_U for e in episodes),
        "min_dim_X": min(e.d_X for e in episodes),
        "max_dim_X": max(e.d_X for e in episodes),
        "min_dim_W": min(e.d_W for e in episodes),
        "max_dim_W": max(e.d_W for e in episodes),
        "min_dim_Z": min(e.d_Z for e in episodes),
        "max_dim_Z": max(e.d_Z for e in episodes),
        "min_train_size": train_size,
        "max_train_size": train_size,
        "replay_small": False,
        "target_kind": next(iter(target_kinds)),
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


def _content_digest(
    args: argparse.Namespace, datasets: Sequence[CSVProximalDataset]
) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "datasets": [dataset.options for dataset in datasets],
        "replicates_per_csv": args.replicates_per_csv,
        "seq_len": args.seq_len,
        "train_size": args.n_max,
        "seed": args.seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _write_metadata_files(
    root: Path,
    args: argparse.Namespace,
    datasets: Sequence[CSVProximalDataset],
    episode_records: Sequence[dict],
    *,
    n_files: int,
    datasets_per_file: int,
    cli_command: str,
) -> None:
    sources_by_file = {}
    for dataset in datasets:
        source = dataset.source
        sources_by_file.setdefault(
            source.path.name,
            {
                "file": source.path.name,
                "path": str(source.path),
                "sha256": source.digest,
                "rows": int(source.values.shape[0]),
                "encoded_columns": list(source.columns),
                "x_columns": list(source.x_columns),
                "u_columns": list(source.u_columns),
            },
        )
    sources = list(sources_by_file.values())
    mechanisms_used = list(
        dict.fromkeys(dataset.mechanism_name for dataset in datasets)
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "content_digest": _content_digest(args, datasets),
        "name": args.name,
        "sources": sources,
        "mechanism": args.mechanism,
        "mechanisms_used": mechanisms_used,
        "mechanism_configs": {
            name: args._raw_mechanisms[name] for name in mechanisms_used
        },
        "target_kind": datasets[0].target_capability,
        "replicates_per_csv": args.replicates_per_csv,
        "n_episodes": len(episode_records),
        "episodes": list(episode_records),
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

    resolved_build = {key: getattr(args, key) for key in BUILD_CONFIG_FIELDS}
    for key in ("data_dir", "out_root"):
        resolved_build[key] = str(resolved_build[key])
    config = {
        "format_version": FORMAT_VERSION,
        "build": resolved_build,
        "mechanisms": args._raw_mechanisms,
        "sources": [
            {"file": source["file"], "sha256": source["sha256"]} for source in sources
        ],
    }
    with open(root / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def build(args: argparse.Namespace) -> Path:
    datasets = make_datasets(args)
    root = Path(args.out_root) / args.name
    val_dir = root / "val"
    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
    existing = _owned_outputs(root)
    if existing and not args.force:
        raise FileExistsError(
            f"{root} already contains a generated set. Choose a new --name, "
            "or pass --force to replace builder-owned outputs."
        )

    source_names = list(dict.fromkeys(dataset.source.path.name for dataset in datasets))
    mechanism_names = list(
        dict.fromkeys(dataset.mechanism_name for dataset in datasets)
    )
    print(
        f"Building '{args.name}' from {len(source_names)} CSV source(s) "
        f"and {len(mechanism_names)} mechanism(s)\n"
        f"  sources={source_names}\n"
        f"  mechanisms={mechanism_names}\n"
        f"  {args.replicates_per_csv} replicates/source/mechanism, "
        f"{len(datasets) * args.replicates_per_csv} episodes x T={args.seq_len} "
        f"(n_max={args.n_max} + n_q={args.n_q}), seed={args.seed}\n"
        f"  target={datasets[0].target_capability} (no Monte Carlo replication)"
    )
    episodes, episode_records = generate_episodes(
        datasets,
        replicates_per_csv=args.replicates_per_csv,
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
        dataset=datasets,
        cli_command=cli_command,
    )
    _write_metadata_files(
        root,
        args,
        datasets,
        episode_records,
        n_files=n_files,
        datasets_per_file=dpf,
        cli_command=cli_command,
    )
    print(f"  benchmark: data: {{source: load, val_dir: {val_dir}}}")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semisynt_eval.build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Complete YAML recipe (defaults to semisynt_eval/config.yaml).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available CSVs and mechanisms."
    )
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument(
        "--csv",
        help=(
            "CSV path/name/stem, or 'all' for every CSV directly under "
            "--data_dir (default: all)."
        ),
    )
    parser.add_argument("--name", default=None, help="Build name under --out_root.")
    parser.add_argument("--out_root", type=Path)
    parser.add_argument("--mechanism", choices=["all", *available_mechanisms()])
    parser.add_argument(
        "--x_columns", default=None, help="Comma-separated explicit X columns."
    )
    parser.add_argument(
        "--u_columns", default=None, help="Comma-separated explicit U columns."
    )
    parser.add_argument(
        "--drop_columns", default=None, help="Comma-separated columns to ignore."
    )
    parser.add_argument("--n_features_x", type=int)
    parser.add_argument("--n_features_u", type=int)
    parser.add_argument("--split_seed", type=int)
    parser.add_argument(
        "--row_sampling",
        choices=("auto", "without_replacement", "bootstrap"),
    )
    parser.add_argument("--replicates_per_csv", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--prop_train", type=float)
    parser.add_argument("--n_max", type=int, default=None)
    parser.add_argument("--n_q", type=int, default=None)
    parser.add_argument("--seed", type=int)
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


def _load_recipe(path: Path) -> dict:
    with open(path) as f:
        recipe = yaml.safe_load(f) or {}
    allowed_top_level = {"format_version", "build", "mechanisms", "sources"}
    unknown = set(recipe) - allowed_top_level
    missing = {"format_version", "build", "mechanisms"} - set(recipe)
    if missing or unknown:
        raise ValueError(
            f"invalid config {path}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if recipe["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported config format_version={recipe['format_version']}; "
            f"expected {FORMAT_VERSION}"
        )
    build_keys = set(recipe["build"])
    expected_build_keys = set(BUILD_CONFIG_FIELDS)
    if build_keys != expected_build_keys:
        raise ValueError(
            f"invalid config build keys: "
            f"missing={sorted(expected_build_keys-build_keys)}, "
            f"unknown={sorted(build_keys-expected_build_keys)}"
        )
    sources = recipe.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("config sources must be a list")
    source_files = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {"file", "sha256"}:
            raise ValueError(
                f"config sources[{index}] must contain exactly file and sha256"
            )
        source_files.append(source["file"])
    if len(source_files) != len(set(source_files)):
        raise ValueError("config sources contains duplicate filenames")
    return recipe


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = pre.parse_known_args(argv)
    recipe = _load_recipe(known.config)
    mechanism_configs = load_mechanism_configs(
        recipe["mechanisms"], available=set(available_mechanisms())
    )
    parser = build_parser()
    parser.set_defaults(**recipe["build"])
    cli = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    args.data_dir = Path(args.data_dir)
    args.out_root = Path(args.out_root)
    args._mechanism_configs = mechanism_configs
    args._raw_mechanisms = recipe["mechanisms"]
    args._expected_source_sha256 = {
        source["file"]: source["sha256"] for source in recipe.get("sources", [])
    }
    if args.list:
        return args
    for field in ("name",):
        if getattr(args, field) is None:
            parser.error(f"--{field} is required (or supply it via --config)")
    if args.replicates_per_csv < 1:
        parser.error("--replicates_per_csv must be >= 1")
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
