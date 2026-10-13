"""Plot posterior-width contraction across training and context size.

Example
-------
    uv run python -m analysis.analysis_training.posterior_contraction \
        --ckpt-dir whole_1 \
        --data-dir data/synt_eval/synt_val_v1 \
        --n-datasets 12

The heatmap shows the row-weighted mean predictive posterior standard
deviation in the frozen comparison units.  A single logarithmic colour scale is
used for the entire grid, so colours are comparable across checkpoints and
context sizes.  The posterior standard deviation uses the project's
piecewise-uniform histogram convention, including the within-bin ``w^2 / 12``
term.  Episode results are cached under
``analysis/analysis_training/cache/posterior_contraction/`` by checkpoint, dataset
folder, and dataset index; later runs reuse any context sizes already present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from matplotlib.colors import LogNorm

from analysis.analysis_training.design import (
    apply_style,
    figsize,
    plt,
    save_figure,
)
from eval_datasets.prior_eval.build_prior_frozen_set import (
    frozen_content_digest,
    load_frozen_meta,
)
from eval_datasets.prior_eval.report import load_model
from proximalfm.train._dist_eval import episode_predictive, iter_frozen_episodes
from proximalfm.train._dist_metrics import predictive_mean_var

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "analysis" / "analysis_training" / "results"
DEFAULT_CKPT_DIR = REPO_ROOT / "whole_1"
NAME = "posterior_contraction"
DEFAULT_CACHE_DIR = REPO_ROOT / "analysis" / "analysis_training" / "cache" / NAME

CACHE_VERSION = 1

DEFAULT_CONTEXTS = (16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_STEPS = (1, 500, 1000, 5000, 10000, 75000)


def parse_int_list(value: str, *, name: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one integer")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def resolve_checkpoints(ckpt_dir: Path, steps: Sequence[int]) -> list[Path]:
    """Resolve requested training steps to the repository checkpoint names."""
    ckpt_dir = Path(ckpt_dir)
    paths = []
    missing = []
    for step in steps:
        candidates = (
            ckpt_dir / f"step-{step:06d}.ckpt",
            ckpt_dir / f"step-{step}.ckpt",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            missing.append(f"step {step} ({candidates[0]})")
        else:
            paths.append(path)
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    return paths


def _cache_identity(
    ckpt_path: Path,
    data_dir: Path,
    dataset_index: int,
    *,
    dataset_digest: str | None,
    n_max: int,
) -> dict:
    """Return the inputs that identify one checkpoint/episode prediction."""
    ckpt_path = Path(ckpt_path).resolve()
    stat = ckpt_path.stat()
    return {
        "version": CACHE_VERSION,
        "checkpoint": str(ckpt_path),
        # Invalidate an entry when a checkpoint is replaced in place.
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        # Keep the folder in the identity even when two folders have the same
        # content digest: the requested cache granularity is per dataset folder.
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_digest": dataset_digest,
        "dataset_index": int(dataset_index),
        "n_max": int(n_max),
    }


def _cache_path(identity: dict, cache_dir: Path) -> Path:
    """Return a readable, collision-resistant path for one episode cache."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    checkpoint = Path(identity["checkpoint"]).stem
    dataset_folder = Path(identity["data_dir"]).name
    filename = (
        f"{checkpoint}__{dataset_folder}__dataset-"
        f"{identity['dataset_index']:05d}__{digest}.npz"
    )
    return Path(cache_dir) / filename


def _load_cache(path: Path, identity: dict) -> dict[int, tuple[float, int]]:
    """Load cached ``n_ctx -> (sum SD, row count)`` values, if valid."""
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as cached:
            stored_identity = json.loads(str(cached["identity"].item()))
            contexts = cached["contexts"].astype(np.int64, copy=False)
            sums = cached["sums"].astype(np.float64, copy=False)
            counts = cached["counts"].astype(np.int64, copy=False)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return {}
    if stored_identity != identity or not (
        contexts.ndim == sums.ndim == counts.ndim == 1
        and len(contexts) == len(sums) == len(counts)
        and np.all(contexts > 0)
        and np.all(counts > 0)
        and np.all(np.isfinite(sums))
    ):
        return {}
    return {
        int(context): (float(total), int(count))
        for context, total, count in zip(contexts, sums, counts)
    }


def _save_cache(
    path: Path, identity: dict, values: dict[int, tuple[float, int]]
) -> None:
    """Atomically save one episode's cached context-size results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    contexts = np.array(sorted(values), dtype=np.int64)
    sums = np.array([values[int(n)][0] for n in contexts], dtype=np.float64)
    counts = np.array([values[int(n)][1] for n in contexts], dtype=np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identity=json.dumps(identity, sort_keys=True),
            contexts=contexts,
            sums=sums,
            counts=counts,
        )
    temporary.replace(path)


def posterior_widths(
    model,
    data_dir: Path,
    records: Sequence[dict],
    *,
    n_ctx_list: Sequence[int],
    n_max: int,
    n_datasets: int,
    device: str = "cpu",
    progress: bool = True,
    ckpt_path: Path | None = None,
    cache_dir: Path | None = None,
    dataset_digest: str | None = None,
    force_cache: bool = False,
) -> np.ndarray:
    """Compute mean posterior SD, optionally caching each checkpoint/episode.

    A cache entry holds the row sum and count separately so aggregation remains
    row weighted even for sets whose episodes have different query sizes. New
    context sizes are appended to an existing entry rather than invalidating it.
    """
    if (ckpt_path is None) != (cache_dir is None):
        raise ValueError("ckpt_path and cache_dir must be provided together")
    sums = np.zeros(len(n_ctx_list), dtype=np.float64)
    counts = np.zeros(len(n_ctx_list), dtype=np.int64)
    episodes = iter_frozen_episodes(Path(data_dir) / "val", max_episodes=n_datasets)

    for episode_index, batch in enumerate(episodes):
        record = records[episode_index]
        identity = None
        path = None
        cached: dict[int, tuple[float, int]] = {}
        cache_changed = False
        if cache_dir is not None:
            identity = _cache_identity(
                ckpt_path,
                data_dir,
                episode_index,
                dataset_digest=dataset_digest,
                n_max=n_max,
            )
            path = _cache_path(identity, cache_dir)
            cached = _load_cache(path, identity)
        if progress:
            print(
                f"  episode {episode_index + 1}/{n_datasets}",
                end="\r",
                flush=True,
            )
        with torch.inference_mode():
            for column, n_ctx in enumerate(n_ctx_list):
                if n_ctx in cached and not force_cache:
                    total, count = cached[n_ctx]
                    sums[column] += total
                    counts[column] += count
                    continue
                probs, edges = episode_predictive(
                    model,
                    batch,
                    record,
                    n_ctx=n_ctx,
                    n_max=n_max,
                    device=device,
                )
                _, variance = predictive_mean_var(probs, edges)
                sd = variance.clamp_min(0.0).sqrt()
                total = float(sd.detach().double().sum().cpu())
                count = sd.numel()
                sums[column] += total
                counts[column] += count
                cached[n_ctx] = (total, count)
                cache_changed = True
        if path is not None and identity is not None and cache_changed:
            _save_cache(path, identity, cached)
    if progress:
        print(" " * 40, end="\r")
    if not np.all(counts):
        raise RuntimeError("No validation rows were available for one context size")
    return sums / counts


def build_figure(
    widths: np.ndarray,
    n_ctx_list: Sequence[int],
    ckpt_steps: Sequence[int],
) -> "plt.Figure":
    """Build the shared-scale posterior-width heatmap."""
    widths = np.asarray(widths, dtype=float)
    n_ctx_list = list(n_ctx_list)
    ckpt_steps = list(ckpt_steps)
    expected_shape = (len(n_ctx_list), len(ckpt_steps))
    if widths.shape != expected_shape:
        raise ValueError(f"widths has shape {widths.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("widths must be finite and strictly positive")

    apply_style()
    fig, ax = plt.subplots(figsize=figsize())
    norm = LogNorm(vmin=float(widths.min()), vmax=float(widths.max()))
    checkpoint_positions = np.arange(len(ckpt_steps), dtype=float)
    checkpoint_edges = np.arange(len(ckpt_steps) + 1, dtype=float) - 0.5
    context_positions = np.arange(len(n_ctx_list), dtype=float)
    context_edges = np.arange(len(n_ctx_list) + 1, dtype=float) - 0.5
    mesh = ax.pcolormesh(
        checkpoint_edges,
        context_edges,
        widths,
        shading="flat",
        cmap="viridis",
        norm=norm,
    )
    ax.set_xticks(checkpoint_positions)
    ax.set_xticklabels([f"{step:g}" for step in ckpt_steps])
    ax.set_yticks(context_positions)
    ax.set_yticklabels([f"{n:g}" for n in n_ctx_list])
    ax.set_xlabel("training step")
    ax.set_ylabel(r"context size $n_{\mathrm{ctx}}$")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    colorbar.set_label("mean predictive SD")

    log_min, log_max = np.log10(widths.min()), np.log10(widths.max())
    for row, n_ctx in enumerate(n_ctx_list):
        for column, step in enumerate(ckpt_steps):
            value = widths[row, column]
            relative = (np.log10(value) - log_min) / max(log_max - log_min, 1e-12)
            text_color = "white" if relative < 0.55 else "black"
            ax.text(
                checkpoint_positions[column],
                context_positions[row],
                f"{value:.2g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="frozen validation set root containing val/ and frozen_meta.pt",
    )
    parser.add_argument(
        "--n-ctx",
        default=None,
        help="comma-separated context sizes (default: 16,64,128,512,1024,2048)",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="maximum context size to evaluate (default: dataset n_max, usually 2048)",
    )
    parser.add_argument(
        "--ckpt-steps",
        default=None,
        help="comma-separated checkpoint steps (default: 1,50,200,1000,10000,75000)",
    )
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=None,
        help="number of episodes from index 0 to evaluate (default: all)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="per-checkpoint, per-dataset cache directory",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="recompute predictions and replace matching cache entries",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    meta = load_frozen_meta(args.data_dir)
    dataset_digest = frozen_content_digest(meta)
    dataset_n_max = int(meta["global"]["n_max"])
    max_context = dataset_n_max if args.n_max is None else args.n_max
    if not 1 <= max_context <= dataset_n_max:
        raise ValueError(f"n_max must be in [1, {dataset_n_max}], got {max_context}")

    if args.n_ctx is None:
        n_ctx_list = [n for n in DEFAULT_CONTEXTS if n <= max_context]
    else:
        n_ctx_list = parse_int_list(args.n_ctx, name="n-ctx")
        if any(n > max_context for n in n_ctx_list):
            raise ValueError(
                f"n-ctx values must be no larger than n-max={max_context}: {n_ctx_list}"
            )
    if not n_ctx_list:
        raise ValueError("no default context sizes are at most n-max")
    n_ctx_list = sorted(n_ctx_list)

    ckpt_steps = (
        DEFAULT_STEPS
        if args.ckpt_steps is None
        else tuple(parse_int_list(args.ckpt_steps, name="ckpt-steps"))
    )
    ckpt_steps = sorted(ckpt_steps)
    ckpt_paths = resolve_checkpoints(args.ckpt_dir, ckpt_steps)

    total_datasets = len(meta["episodes"])
    n_datasets = total_datasets if args.n_datasets is None else args.n_datasets
    if not 1 <= n_datasets <= total_datasets:
        raise ValueError(
            f"n-datasets must be in [1, {total_datasets}], got {n_datasets}"
        )

    widths = np.empty((len(n_ctx_list), len(ckpt_steps)), dtype=float)
    for column, (step, ckpt_path) in enumerate(zip(ckpt_steps, ckpt_paths)):
        print(f"checkpoint step {step}: {ckpt_path}")
        model, loaded_step = load_model(ckpt_path, device=args.device)
        widths[:, column] = posterior_widths(
            model,
            args.data_dir,
            meta["episodes"],
            n_ctx_list=n_ctx_list,
            n_max=dataset_n_max,
            n_datasets=n_datasets,
            device=args.device,
            progress=not args.no_progress,
            ckpt_path=ckpt_path,
            cache_dir=args.cache_dir,
            dataset_digest=dataset_digest,
            force_cache=args.force_cache,
        )
        loaded_text = "unknown" if loaded_step is None else str(int(loaded_step))
        print(f"  loaded checkpoint step={loaded_text}; widths={widths[:, column]}")
        del model

    fig = build_figure(widths, n_ctx_list, ckpt_steps)
    paths = save_figure(fig, args.outdir, args.name)
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
