"""Build fixed linear-Gaussian proxy-informativeness inspection sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
from typing import Sequence

import torch
import yaml

from proximalfm.prior._genload import write_batch_file

from .generator import LinearProximalGenerator
from .sidecar import (
    LinearBuildConfig,
    check_no_silent_mutation,
    episode_records,
    write_sidecar,
)

POOL_FIELDS = LinearProximalGenerator.POOL_FIELDS
DEFAULT_PROXY_NOISE_STDS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def episode_seed(base_seed: int, index: int) -> int:
    payload = f"inspect-linear:{base_seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def allocate_noise_levels(n_datasets: int, levels: Sequence[float]) -> list[float]:
    if not levels:
        raise ValueError("proxy_noise_stds must not be empty")
    if len(set(float(v) for v in levels)) != len(levels):
        raise ValueError("proxy_noise_stds must contain unique values")
    if any(float(v) < 0 for v in levels):
        raise ValueError("proxy_noise_stds must be >= 0")
    if n_datasets < len(levels):
        raise ValueError("n_datasets must be at least the number of proxy noise levels")
    base, remainder = divmod(n_datasets, len(levels))
    return [
        float(level)
        for i, level in enumerate(levels)
        for _ in range(base + (i < remainder))
    ]


def generate_pool(
    generator: LinearProximalGenerator,
    *,
    n_datasets: int,
    seq_len: int,
    train_size: int,
    proxy_noise_stds: Sequence[float],
    seed: int,
    shuffle: bool,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    assigned = allocate_noise_levels(n_datasets, proxy_noise_stds)
    seeds = torch.tensor(
        [episode_seed(seed, i) for i in range(n_datasets)], dtype=torch.long
    )
    episodes = [
        generator.generate_dataset(
            seq_len=seq_len,
            train_size=train_size,
            proxy_noise_std=noise,
            seed=int(seeds[i]),
        )
        for i, noise in enumerate(assigned)
    ]
    pool = tuple(torch.stack(field) for field in zip(*episodes))
    noise = torch.tensor(assigned, dtype=torch.float64)
    if shuffle:
        rng = torch.Generator().manual_seed(int(seed) + 7)
        order = torch.randperm(n_datasets, generator=rng)
        pool = tuple(field[order] for field in pool)
        noise = noise[order]
        seeds = seeds[order]
    return pool, noise, seeds


def _save_batch(
    pool: tuple[torch.Tensor, ...], start: int, stop: int, val_dir: Path, index: int
) -> None:
    p = dict(zip(POOL_FIELDS, pool))
    sl = slice(start, stop)

    def block(name: str) -> torch.Tensor:
        return p[name][sl].cpu().contiguous()

    def column(name: str) -> torch.Tensor:
        value = p[name][sl].cpu()
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value.contiguous()

    write_batch_file(
        val_dir,
        index,
        U=block("U"),
        X=block("X"),
        W=block("W"),
        Z=block("Z"),
        **{
            name: column(name)
            for name in (
                "A",
                "A_probs",
                "Y_obs",
                "Y0",
                "Y1",
                "MU0",
                "MU1",
                "S2",
            )
        },
        **{
            name: p[name][sl].cpu().contiguous()
            for name in (
                "d_U",
                "d_X",
                "d_W",
                "d_Z",
                "seq_lens",
                "train_sizes",
                "K_eff",
            )
        },
        seq_len_per_gp=False,
    )


def save_validation(
    pool: tuple[torch.Tensor, ...],
    val_dir: Path,
    cfg: LinearBuildConfig,
    datasets_per_file: int | None,
    cli_command: str,
) -> tuple[int, int]:
    val_dir.mkdir(parents=True, exist_ok=True)
    size = (
        cfg.n_datasets
        if not datasets_per_file or datasets_per_file <= 0
        else datasets_per_file
    )
    n_files = 0
    for index, start in enumerate(range(0, cfg.n_datasets, size)):
        _save_batch(pool, start, min(start + size, cfg.n_datasets), val_dir, index)
        n_files += 1
    metadata = {
        "prior_type": "inspection_linear_gaussian",
        "batch_size": size,
        "batch_size_per_gp": 1,
        "min_seq_len": cfg.seq_len,
        "max_seq_len": cfg.seq_len,
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
        "min_train_size": cfg.prop_train,
        "max_train_size": cfg.prop_train,
        "replay_small": False,
        "mc_cate": cfg.mc_cate,
        "mc_targets_on_context": False,
        "cli_command": cli_command,
    }
    (val_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return n_files, size


def build_config(args: argparse.Namespace) -> LinearBuildConfig:
    return LinearBuildConfig(
        n_datasets=args.n_datasets,
        seq_len=args.seq_len,
        prop_train=args.prop_train,
        mc_cate=args.mc_cate,
        seed=args.seed,
        proxy_noise_stds=list(args.proxy_noise_stds),
        shuffle=not args.no_shuffle,
        beta_ux=args.beta_ux,
        sigma_u=args.sigma_u,
        proxy_loading=args.proxy_loading,
        alpha_0=args.alpha_0,
        alpha_x=args.alpha_x,
        alpha_u=args.alpha_u,
        sigma_a=args.sigma_a,
        beta_0=args.beta_0,
        beta_x=args.beta_x,
        beta_u=args.beta_u,
        sigma_y=args.sigma_y,
        tau_0=args.tau_0,
        tau_x=args.tau_x,
    )


def build(args: argparse.Namespace) -> Path:
    cfg = build_config(args)
    generator = LinearProximalGenerator(cfg.generator_config(args.mc_mem_budget))
    root = Path(args.out_root) / args.name
    if root.exists() and not (root / "frozen_meta.pt").exists() and not args.force:
        raise FileExistsError(
            f"{root} exists without a sidecar; choose a new name or pass --force"
        )
    if root.exists():
        check_no_silent_mutation(root, cfg, force=args.force)
    if args.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    cli_command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
    pool, noise_stds, seeds = generate_pool(
        generator,
        n_datasets=cfg.n_datasets,
        seq_len=cfg.seq_len,
        train_size=args.n_max,
        proxy_noise_stds=cfg.proxy_noise_stds,
        seed=cfg.seed,
        shuffle=cfg.shuffle,
    )
    n_files, per_file = save_validation(
        pool, root / "val", cfg, args.datasets_per_file, cli_command
    )
    records = episode_records(pool, POOL_FIELDS, noise_stds, seeds, args.n_max, cfg)
    digest = write_sidecar(
        root,
        cfg,
        records,
        args.n_max,
        per_file,
        cli_command=cli_command,
    )
    counts = {
        str(level): int((noise_stds == level).sum()) for level in cfg.proxy_noise_stds
    }
    manifest = {
        "name": args.name,
        "set_kind": "inspect_linear",
        "content_digest": digest,
        "n_datasets": cfg.n_datasets,
        "seq_len": cfg.seq_len,
        "n_max": args.n_max,
        "n_q": args.n_q,
        "mc_cate": cfg.mc_cate,
        "proxy_noise_counts": counts,
        "datasets_per_val_file": per_file,
        "n_val_files": n_files,
        "val_dir": str((root / "val").resolve()),
        "cli_command": cli_command,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config_values = dict(vars(args))
    config_values.pop("config", None)
    config_values["out_root"] = str(config_values["out_root"])
    # Replaying a recorded recipe must never inherit destructive permission.
    config_values["force"] = False
    config_values["cli_command"] = cli_command
    (root / "config.yaml").write_text(yaml.safe_dump(config_values, sort_keys=True))
    print(
        f"Built {cfg.n_datasets} linear-Gaussian episodes at {root} "
        f"(digest={digest}, noise={cfg.proxy_noise_stds})"
    )
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument(
        "--out_root", type=Path, default=Path("data/inspect_eval_linear")
    )
    parser.add_argument("--n_datasets", type=int, default=120)
    parser.add_argument("--seq_len", type=int, default=640)
    parser.add_argument("--prop_train", type=float, default=0.8)
    parser.add_argument("--n_max", type=int)
    parser.add_argument("--n_q", type=int)
    parser.add_argument("--mc_cate", type=int, default=1000)
    parser.add_argument("--mc_mem_budget", type=int, default=2_000_000_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--proxy_noise_stds", type=float, nargs="+", default=DEFAULT_PROXY_NOISE_STDS
    )
    parser.add_argument("--datasets_per_file", type=int)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    for name, default in (
        ("beta_ux", 0.5),
        ("sigma_u", 1.0),
        ("proxy_loading", 1.0),
        ("alpha_0", 0.0),
        ("alpha_x", 0.25),
        ("alpha_u", 1.0),
        ("sigma_a", 1.0),
        ("beta_0", 0.0),
        ("beta_x", 0.5),
        ("beta_u", 1.0),
        ("sigma_y", 1.0),
        ("tau_0", 1.0),
        ("tau_x", 0.5),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)
    return parser


def _explicit(argv: Sequence[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in argv)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    config_path = pre.parse_known_args(argv)[0].config
    parser = build_parser()
    if config_path:
        loaded = yaml.safe_load(config_path.read_text()) or {}
        loaded.pop("cli_command", None)
        loaded.pop("config", None)
        parser.set_defaults(**loaded)
    args = parser.parse_args(argv)
    if not args.name:
        parser.error("--name is required (or must be present in --config)")

    direct_split = args.n_max is not None or args.n_q is not None
    if direct_split:
        if args.n_max is None or args.n_q is None:
            parser.error("--n_max and --n_q must be supplied together")
        if _explicit(argv, "--seq_len") or _explicit(argv, "--prop_train"):
            parser.error("do not mix --n_max/--n_q with --seq_len/--prop_train")
        resolved_len = args.n_max + args.n_q
        resolved_prop = args.n_max / resolved_len
        # A resolved config records both spellings; accept it only if consistent.
        if config_path and (
            args.seq_len != resolved_len
            or round(args.prop_train * resolved_len) != args.n_max
        ):
            parser.error("row split in --config is internally inconsistent")
        args.seq_len = resolved_len
        args.prop_train = resolved_prop
    else:
        if args.seq_len < 2 or not 0 < args.prop_train < 1:
            parser.error("seq_len must be >= 2 and prop_train must be in (0, 1)")
        args.n_max = round(args.seq_len * args.prop_train)
        args.n_max = max(1, min(args.n_max, args.seq_len - 1))
        args.n_q = args.seq_len - args.n_max
    if args.n_datasets <= 0:
        parser.error("n_datasets must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
