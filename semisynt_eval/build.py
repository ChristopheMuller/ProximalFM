"""Build the semi-synthetic mechanism matrix from empirical CSVs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import yaml

from proximalfm.prior._genload import write_batch_file
from semisynt_eval._config import CaseConfig, load_case_configs
from semisynt_eval.generator import CSVProximalDataset, Episode
from semisynt_eval.mechanisms import available_mechanisms
from semisynt_eval.source import PreparedCSV, discover_csvs, load_csv

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BUILD_CONFIG_FIELDS = (
    "data_dir",
    "out_root",
    "csv",
    "name",
    "x_columns",
    "u_columns",
    "drop_columns",
    "n_features_x",
    "n_features_u",
    "n_calibration",
    "split_seed",
    "row_sampling",
    "replicates_per_csv",
    "n_context",
    "n_query",
    "seed",
    "datasets_per_file",
)


def _load_sources(args: argparse.Namespace) -> list[PreparedCSV]:
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
            n_calibration=args.n_calibration,
            split_seed=args.split_seed,
        )
        n_episode = args.n_context + args.n_query
        if len(source.evaluation_row_indices) < n_episode:
            raise ValueError(
                f"{source.path.name}: {len(source.calibration_row_indices)} "
                f"calibration rows plus an episode of {n_episode} rows require "
                f"at least {len(source.calibration_row_indices) + n_episode} "
                f"source rows; found "
                f"{len(source.calibration_row_indices) + len(source.evaluation_row_indices)}"
            )
        expected = args._expected_source_sha256.get(source.path.name)
        if expected not in (None, source.digest):
            raise ValueError(
                f"source SHA-256 mismatch for {source.path}: expected {expected}, "
                f"found {source.digest}"
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
    return sources


def make_datasets(
    sources: Sequence[PreparedCSV], case: CaseConfig, *, row_sampling: str
) -> list[CSVProximalDataset]:
    return [
        CSVProximalDataset(
            source,
            mechanism=case.mechanism,
            mechanism_config=case,
            row_sampling=row_sampling,
        )
        for source in sources
    ]


def generate_episodes(
    datasets: Sequence[CSVProximalDataset],
    *,
    replicates_per_csv: int,
    n_rows: int,
    seed: int,
) -> tuple[list[Episode], list[dict]]:
    """Generate episodes with seeds paired by source and replicate across cases."""
    episodes, records = [], []
    for source_index, dataset in enumerate(datasets):
        for replicate in range(replicates_per_csv):
            pair_index = source_index * replicates_per_csv + replicate
            episode_seed = int(seed) + 1_000 * pair_index
            episode = dataset.generate(seed=episode_seed, n_rows=n_rows)
            episodes.append(episode)
            row_bytes = episode.source_row_indices.numpy().astype("<i8").tobytes()
            records.append(
                {
                    "pair_id": f"{dataset.source.path.name}:{replicate}",
                    "source_file": dataset.source.path.name,
                    "case": dataset.mechanism_config.name,
                    "mechanism": dataset.mechanism_name,
                    "family": dataset.mechanism_config.family,
                    "replicate": replicate,
                    "seed": episode_seed,
                    "source_row_indices_sha256": hashlib.sha256(row_bytes).hexdigest(),
                }
            )
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
    """Write one case in the canonical sparse ``val/`` format."""
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
        batch_size, seq_len = len(block), block[0].X.shape[0]
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
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return n_files, dpf


def _case_payload(case: CaseConfig) -> dict:
    # Round-trip through JSON so tuples become plain lists and the same payload
    # compares identically after JSON/YAML serialization and reload.
    return json.loads(json.dumps(asdict(case), sort_keys=True))


def _row_indices_sha256(indices: tuple[int, ...]) -> str:
    values = torch.tensor(indices, dtype=torch.int64).numpy().astype("<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _source_payload(
    sources: Sequence[PreparedCSV], *, include_row_indices: bool = False
) -> list[dict]:
    payload = []
    for source in sources:
        item = {
            "file": source.path.name,
            "path": str(source.path),
            "sha256": source.digest,
            "rows": len(source.calibration_row_indices)
            + len(source.evaluation_row_indices),
            "n_calibration": len(source.calibration_row_indices),
            "n_evaluation": len(source.evaluation_row_indices),
            "partition_seed": source.partition_seed,
            "calibration_row_indices_sha256": _row_indices_sha256(
                source.calibration_row_indices
            ),
            "evaluation_row_indices_sha256": _row_indices_sha256(
                source.evaluation_row_indices
            ),
            "encoded_columns": list(source.columns),
            "x_columns": list(source.x_columns),
            "u_columns": list(source.u_columns),
            "preprocessing": source.preprocessing,
        }
        if include_row_indices:
            item["calibration_row_indices"] = list(source.calibration_row_indices)
            item["evaluation_row_indices"] = list(source.evaluation_row_indices)
        payload.append(item)
    return payload


def _content_digest(
    args: argparse.Namespace,
    case: CaseConfig,
    datasets: Sequence[CSVProximalDataset],
) -> str:
    payload = {
        "mechanism": case.mechanism,
        "case": _case_payload(case),
        "datasets": [dataset.options for dataset in datasets],
        "replicates_per_csv": args.replicates_per_csv,
        "n_context": args.n_context,
        "n_query": args.n_query,
        "seed": args.seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _build_values(args: argparse.Namespace) -> dict:
    values = {key: getattr(args, key) for key in BUILD_CONFIG_FIELDS}
    for key in ("data_dir", "out_root"):
        values[key] = str(values[key])
    return values


def _resolved_recipe(args: argparse.Namespace, sources: Sequence[PreparedCSV]) -> dict:
    return {
        "build": _build_values(args),
        "common": args._raw_common,
        "cases": args._raw_cases,
        "resolved_cases": {
            name: _case_payload(case) for name, case in args._case_configs.items()
        },
        "sources": [
            {"file": source.path.name, "sha256": source.digest} for source in sources
        ],
    }


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, default_flow_style=False))


def _owned_outputs(root: Path, case_names: Sequence[str]) -> list[Path]:
    if not root.exists():
        return []
    paths = [root / "manifest.json", root / "config.yaml"]
    for name in case_names:
        case_root = root / name
        paths.extend(
            [
                case_root / "manifest.json",
                case_root / "config.yaml",
                case_root / "val" / "metadata.json",
                case_root / "frozen_meta.pt",
                case_root / "frozen_meta.pt.tmp",
            ]
        )
        paths.extend((case_root / "val").glob("batch_*.pt"))
        paths.extend((case_root / "val").glob("batch_*.pt.tmp"))
    return [path for path in paths if path.exists()]


def build(args: argparse.Namespace) -> Path:
    unavailable = {case.mechanism for case in args._case_configs.values()} - set(
        available_mechanisms()
    )
    if unavailable:
        raise NotImplementedError(
            "configured mechanism is not implemented yet: "
            f"{', '.join(sorted(unavailable))}"
        )
    root = Path(args.out_root) / args.name
    existing = _owned_outputs(root, list(args._case_configs))
    if existing and not args.force:
        raise FileExistsError(
            f"{root} already contains a generated set. Choose a new --name, "
            "or pass --force to replace builder-owned outputs."
        )
    sources = _load_sources(args)
    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
    generated = []
    pending = []
    for case in args._case_configs.values():
        datasets = make_datasets(sources, case, row_sampling=args.row_sampling)
        episodes, records = generate_episodes(
            datasets,
            replicates_per_csv=args.replicates_per_csv,
            n_rows=args.n_context + args.n_query,
            seed=args.seed,
        )
        pending.append((case, datasets, episodes, records))
    if existing and args.force:
        for path in existing:
            path.unlink()

    root.mkdir(parents=True, exist_ok=True)
    for case, datasets, episodes, records in pending:
        case_root = root / case.name
        digest = _content_digest(args, case, datasets)
        n_files, dpf = save_validation(
            episodes,
            out_dir=case_root / "val",
            train_size=args.n_context,
            datasets_per_file=args.datasets_per_file,
            dataset=datasets,
            cli_command=cli_command,
        )
        manifest = {
            "content_digest": digest,
            "matrix_name": args.name,
            "case": case.name,
            "mechanism": case.mechanism,
            "case_config": _case_payload(case),
            "sources": _source_payload(sources),
            "target_kind": datasets[0].target_capability,
            "replicates_per_csv": args.replicates_per_csv,
            "n_episodes": len(records),
            "episodes": records,
            "n_context": args.n_context,
            "n_query": args.n_query,
            "seed": args.seed,
            "datasets_per_val_file": dpf,
            "n_val_files": n_files,
            "val_dir": str((case_root / "val").resolve()),
            "cli_command": cli_command,
        }
        (case_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _write_yaml(case_root / "config.yaml", _resolved_recipe(args, sources))
        generated.append(
            {
                "case": case.name,
                "mechanism": case.mechanism,
                "family": case.family,
                "case_config": _case_payload(case),
                "content_digest": digest,
                "val_dir": str((case_root / "val").resolve()),
                "pair_ids": [record["pair_id"] for record in records],
            }
        )

    matrix_identity = [
        {
            "case": item["case"],
            "mechanism": item["mechanism"],
            "family": item["family"],
            "case_config": item["case_config"],
            "content_digest": item["content_digest"],
            "pair_ids": item["pair_ids"],
        }
        for item in generated
    ]
    matrix_payload = {
        "name": args.name,
        "mechanisms": sorted({item["mechanism"] for item in generated}),
        "content_digest": hashlib.sha256(
            json.dumps(matrix_identity, sort_keys=True).encode()
        ).hexdigest()[:16],
        "sources": _source_payload(sources, include_row_indices=True),
        "cases": generated,
        "cli_command": cli_command,
    }
    (root / "manifest.json").write_text(json.dumps(matrix_payload, indent=2) + "\n")
    root_recipe = {
        "build": _build_values(args),
        "common": args._raw_common,
        "cases": args._raw_cases,
        "resolved_cases": {
            name: _case_payload(case) for name, case in args._case_configs.items()
        },
        "sources": [
            {"file": source.path.name, "sha256": source.digest} for source in sources
        ],
    }
    _write_yaml(root / "config.yaml", root_recipe)
    print(
        f"Built '{args.name}': {len(generated)} cases x {len(sources)} source(s) "
        f"x {args.replicates_per_csv} replicates under {root}"
    )
    for item in generated:
        print(f"  {item['case']}: data: {{source: load, val_dir: {item['val_dir']}}}")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semisynt_eval.build", description=__doc__
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--csv")
    parser.add_argument("--name")
    parser.add_argument("--out_root", type=Path)
    parser.add_argument("--x_columns")
    parser.add_argument("--u_columns")
    parser.add_argument("--drop_columns")
    parser.add_argument("--n_features_x", type=int)
    parser.add_argument("--n_features_u", type=int)
    parser.add_argument("--n_calibration", type=int)
    parser.add_argument("--split_seed", type=int)
    parser.add_argument("--row_sampling", choices=("without_replacement",))
    parser.add_argument("--replicates_per_csv", type=int)
    parser.add_argument("--n_context", type=int)
    parser.add_argument("--n_query", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--datasets_per_file", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def _load_recipe(path: Path) -> dict:
    recipe = yaml.safe_load(path.read_text()) or {}
    allowed = {
        "build",
        "common",
        "cases",
        "resolved_cases",
        "sources",
    }
    missing, unknown = {"build", "common", "cases"} - set(recipe), set(recipe) - allowed
    if missing or unknown:
        raise ValueError(
            f"invalid config {path}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    build_keys = set(recipe["build"])
    expected = set(BUILD_CONFIG_FIELDS)
    if build_keys != expected:
        raise ValueError(
            f"invalid config build keys: missing={sorted(expected-build_keys)}, "
            f"unknown={sorted(build_keys-expected)}"
        )
    sources = recipe.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("config sources must be a list")
    names = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {"file", "sha256"}:
            raise ValueError(
                f"config sources[{index}] must contain exactly file and sha256"
            )
        names.append(source["file"])
    if len(names) != len(set(names)):
        raise ValueError("config sources contains duplicate filenames")
    return recipe


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = pre.parse_known_args(argv)
    recipe = _load_recipe(known.config)
    cases = load_case_configs(recipe["common"], recipe["cases"])
    resolved_cases = recipe.get("resolved_cases")
    if resolved_cases is not None:
        if not isinstance(resolved_cases, dict):
            raise ValueError("config resolved_cases must be a mapping")
        expected = {name: _case_payload(case) for name, case in cases.items()}
        if resolved_cases != expected:
            raise ValueError(
                "config resolved_cases does not match common/cases; regenerate "
                "the recipe instead of editing its resolved snapshot"
            )
    parser = build_parser()
    parser.set_defaults(**recipe["build"])
    args = parser.parse_args(argv)
    args.data_dir, args.out_root = Path(args.data_dir), Path(args.out_root)
    args._case_configs = cases
    args._raw_common, args._raw_cases = recipe["common"], recipe["cases"]
    args._expected_source_sha256 = {
        source["file"]: source["sha256"] for source in recipe.get("sources", [])
    }
    if args.list:
        return args
    if args.name is None:
        parser.error("--name is required (or supply it via --config)")
    if args.replicates_per_csv < 1:
        parser.error("--replicates_per_csv must be >= 1")
    if args.n_context < 1 or args.n_query < 1:
        parser.error("--n_context and --n_query must both be >= 1")
    if args.n_calibration < 1:
        parser.error("--n_calibration must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.list:
        print("CSV sources:")
        for path in discover_csvs(args.data_dir):
            print(f"  {path.stem:28s} {path.name}")
        print("Cases:")
        for name, case in args._case_configs.items():
            print(f"  {name:38s} {case.mechanism}/{case.family}")
        return
    build(args)


if __name__ == "__main__":
    main()
