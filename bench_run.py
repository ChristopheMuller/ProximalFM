"""
bench_run.py — Single entry-point for the full benchmark pipeline.

Usage
─────
    python bench_run.py                          # uses bench_config.yaml
    python bench_run.py --config my_config.yaml
    python bench_run.py --force                  # rerun even if preds exist
    python bench_run.py --no-parallel            # single-process (debug)

Pipeline
────────
    1. Load or generate datasets            (Step 1)
    2. For each method:
       a. Skip if predictions already exist (unless --force)
       b. Fit + predict across all batches in parallel
       c. Save predictions to disk immediately on completion
       d. Print running ATE metrics after each method finishes
    3. (Optional) Run neural model checkpoints
    4. Print final ATE table + CATE table

Design notes
────────────
* Incremental saving
    Each method's predictions are saved to {output_dir}/preds/{name}_preds.npz
    as soon as they complete.  If the process is interrupted and restarted,
    already-saved methods are skipped automatically.  Pass --force to redo.

* Parallel execution
    ProcessPoolExecutor dispatches batches to worker processes.  Each worker
    calls build_method() to reconstruct the estimator from the config dict
    (plain-data serialisable) rather than pickling the estimator object itself.
    This is safer across OS (fork/spawn) and avoids sklearn pickling quirks.

* Live display
    Rich Progress shows a bar per method with batch count + elapsed time.
    After each method finishes, a partial results table is printed so you
    can see which methods are already clearly worse without waiting for the
    full run.

* No global mutable state
    Worker functions receive all their inputs as arguments and return plain
    numpy arrays — no globals, no shared memory, no locks needed.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table

from bench_methods import build_method
from bench_metrics import (
    compute_all_metrics,
    build_summary_tables,
    make_ate_rich_table,
    make_cate_rich_table,
)

# ── Optional: neural model support (may not be installed) ─────────────────────
try:
    from proximalfm.Structure import ProximalPriorConfig, ModelConfig
    from proximalfm.PriorDataLoader import create_causal_data_loaders
    from proximalfm.Model import ProximalNanoTabICLv2

    _PROXFM_AVAILABLE = True
except ImportError:
    _PROXFM_AVAILABLE = False

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Validate required top-level keys
    for key in ("data", "run"):
        if key not in cfg:
            raise KeyError(f"bench_config.yaml is missing required section '{key}'.")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────


def _generate_data(data_cfg: dict, run_cfg: dict, output_dir: Path) -> dict:
    """Generate synthetic datasets from the prior and save to disk."""
    if not _PROXFM_AVAILABLE:
        raise RuntimeError(
            "proximalfm is not installed. Set data.source: load to use an "
            "existing data file, or install proximalfm to generate new data."
        )

    n_train = data_cfg["n_train"]
    n_test = data_cfg["n_test"]
    n_batch = data_cfg["n_batch"]
    prior_kw = data_cfg["prior"].copy()
    seed = run_cfg.get("seed", 42)

    prior_kw["dtype"] = torch.float32
    prior_config = ProximalPriorConfig(**prior_kw)

    console.print(
        f"    Generating [bold]{n_batch}[/] datasets "
        f"(n_train={n_train}, n_test={n_test}, seed={seed})…"
    )
    loaders = create_causal_data_loaders(
        prior_config=prior_config,
        batch_size=n_batch,
        samples_per_episode=n_train + n_test,
        train_episodes=1,
        val_episodes=n_batch,
        num_workers=0,
        seed=seed,
    )
    batch = next(iter(loaders["val"]))

    data = {
        "X": batch["X"].transpose(0, 1).cpu(),
        "Z": batch["Z"].transpose(0, 1).cpu(),
        "W": batch["W"].transpose(0, 1).cpu(),
        "A": batch["A"].transpose(0, 1).cpu(),
        "Y": batch["Y"].transpose(0, 1).cpu(),
        # true_ITEs: (n_batch, n_test) — test-set ground truth
        "true_ITEs": batch["ITE"][n_train:, :, 0].transpose(0, 1).cpu().numpy(),
        "N_TRAIN": n_train,
        "N_TEST": n_test,
        "N_BATCH": n_batch,
    }
    torch.save(data, output_dir / "data.pt")
    with open(output_dir / "prior_config.yaml", "w") as f:
        yaml.dump(data_cfg["prior"], f)

    console.print(f"    [green]✓[/] Saved → {output_dir / 'data.pt'}")
    return data


def get_data(cfg: dict, output_dir: Path, force_regen: bool = False) -> dict:
    """Return dataset dict — generate or load depending on config."""
    data_path = output_dir / "data.pt"
    source = cfg["data"].get("source", "generate")

    if source == "generate" and (force_regen or not data_path.exists()):
        return _generate_data(cfg["data"], cfg["run"], output_dir)

    if data_path.exists():
        console.print(f"    Loading existing data from [cyan]{data_path}[/]…")
        return torch.load(data_path, weights_only=False)

    raise FileNotFoundError(
        f"Data file not found at {data_path}. "
        "Set data.source: generate in bench_config.yaml."
    )


def build_batches(data: dict) -> list[dict]:
    """Slice the full data tensor into a list of per-batch numpy dicts.

    WHY pre-slice here?
        Workers receive only the arrays they need — not the full (n_batch, ...)
        tensor.  Smaller pickle payloads = faster inter-process transfers.
    """
    N_TRAIN = data["N_TRAIN"]
    X_np = data["X"].numpy()
    Z_np = data["Z"].numpy()
    W_np = data["W"].numpy()
    A_np = data["A"].numpy()
    Y_np = data["Y"].numpy()

    return [
        {
            "b": b,
            # Training context (seen by the estimator during fit)
            "X_ctx": X_np[b, :N_TRAIN],
            "W_ctx": W_np[b, :N_TRAIN],
            "Z_ctx": Z_np[b, :N_TRAIN],
            "A_ctx": A_np[b, :N_TRAIN, 0],  # squeeze feature dim; A is scalar
            "Y_ctx": Y_np[b, :N_TRAIN, 0],
            # Query points (held-out test set)
            "X_q": X_np[b, N_TRAIN:],
            "W_q": W_np[b, N_TRAIN:],
            "Z_q": Z_np[b, N_TRAIN:],
        }
        for b in range(data["N_BATCH"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Parallel method runner
# ─────────────────────────────────────────────────────────────────────────────


def _worker(args: tuple) -> tuple[int, np.ndarray | None, str | None]:
    """Worker function: fit one method on one batch and return predictions.

    Receives plain numpy arrays + a config dict.
    Reconstructs the estimator from the config dict (avoids pickling sklearn
    objects, which behaves differently across fork/spawn start methods).

    Returns:
        (batch_index, predictions, error_message_or_None)
    """
    b, method_cfg, X_ctx, A_ctx, Y_ctx, W_ctx, Z_ctx, X_q, W_q, Z_q = args
    try:
        _, method = build_method(method_cfg)
        method.fit(X_ctx, A_ctx, Y_ctx, W=W_ctx, Z=Z_ctx)
        preds = method.predict(X_q, W=W_q, Z=Z_q)
        return b, preds.astype(np.float32), None
    except Exception as exc:
        return b, None, str(exc)


def run_method_parallel(
    method_cfg: dict,
    batches: list[dict],
    n_test: int,
    n_workers: int,
    progress: Progress,
) -> np.ndarray:
    """Fit + predict one method across all batches using a process pool.

    Args:
        method_cfg : one entry from config["methods"].
        batches    : list of per-batch dicts from build_batches().
        n_test     : number of test points per batch.
        n_workers  : max parallel worker processes.
        progress   : Rich Progress instance (must already be started).

    Returns:
        preds array of shape (n_batch, n_test).  NaN where a batch failed.
    """
    n_batch = len(batches)
    preds = np.full((n_batch, n_test), np.nan, dtype=np.float32)
    n_errors = 0

    task = progress.add_task(
        f"  [cyan]{method_cfg['name']:<28}[/]",
        total=n_batch,
    )

    args_list = [
        (
            b["b"],
            method_cfg,
            b["X_ctx"],
            b["A_ctx"],
            b["Y_ctx"],
            b["W_ctx"],
            b["Z_ctx"],
            b["X_q"],
            b["W_q"],
            b["Z_q"],
        )
        for b in batches
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            b_idx, result, err = future.result()
            if result is not None:
                preds[b_idx] = result
            else:
                n_errors += 1
            progress.advance(task)

    if n_errors:
        console.print(
            f"    [yellow]⚠[/] {method_cfg['name']}: "
            f"{n_errors}/{n_batch} batches failed (NaN filled)."
        )

    return preds


def run_method_sequential(
    method_cfg: dict,
    batches: list[dict],
    n_test: int,
    progress: Progress,
) -> np.ndarray:
    """Single-process fallback — useful for debugging."""
    n_batch = len(batches)
    preds = np.full((n_batch, n_test), np.nan, dtype=np.float32)

    task = progress.add_task(
        f"  [cyan]{method_cfg['name']:<28}[/] [dim](sequential)[/]",
        total=n_batch,
    )

    for b_dict in batches:
        args = (
            b_dict["b"],
            method_cfg,
            b_dict["X_ctx"],
            b_dict["A_ctx"],
            b_dict["Y_ctx"],
            b_dict["W_ctx"],
            b_dict["Z_ctx"],
            b_dict["X_q"],
            b_dict["W_q"],
            b_dict["Z_q"],
        )
        b_idx, result, err = _worker(args)
        if result is not None:
            preds[b_idx] = result
        if err:
            console.print(f"    [red]✗[/] batch {b_idx}: {err}")
        progress.advance(task)

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 (optional) — Neural model inference
# ─────────────────────────────────────────────────────────────────────────────


def run_neural_model(
    model_cfg: dict,
    data: dict,
    prior_cfg: dict,
    device_str: str,
    preds_dir: Path,
    force: bool,
) -> tuple[str, np.ndarray] | None:
    """Load a ProximalNanoTabICLv2 checkpoint and run inference.

    Returns:
        (name, preds) or None if skipped / failed.
    """
    if not _PROXFM_AVAILABLE:
        console.print(
            "    [yellow]⚠[/] proximalfm not installed — skipping neural models."
        )
        return None

    name = model_cfg["name"]
    ckpt_dir = Path(model_cfg["checkpoint"])
    pred_file = preds_dir / f"{name}_preds.npz"

    if pred_file.exists() and not force:
        console.print(f"    [dim]Skip {name!r} — predictions already saved.[/]")
        return name, np.load(pred_file)["preds"]

    config_path = ckpt_dir / "model_config.yaml"
    best_path = ckpt_dir / "best_model.pth"

    if not best_path.exists():
        console.print(f"    [red]✗[/] Checkpoint not found: {best_path}")
        return None

    with open(config_path) as f:
        mcfg = yaml.safe_load(f)

    device = torch.device(device_str)
    dtype = getattr(torch, mcfg.get("dtype", "float32"))

    model_config = ModelConfig(
        embed_dim=mcfg.get("embed_dim", 64),
        col_num_blocks=mcfg.get("col_num_blocks", 3),
        row_num_blocks=mcfg.get("row_num_blocks", 3),
        icl_num_blocks=mcfg.get("icl_num_blocks", 12),
        col_nhead=mcfg.get("col_nhead", 8),
        row_nhead=mcfg.get("row_nhead", 8),
        icl_nhead=mcfg.get("icl_nhead", 8),
        feature_group_size_x=prior_cfg["d_x"],
        feature_group_size_w=prior_cfg["d_w"],
        feature_group_size_z=prior_cfg["d_z"],
    )
    model = ProximalNanoTabICLv2(config=model_config).to(device, dtype=dtype)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    console.print(f"    [green]✓[/] Loaded {name!r} (epoch {ckpt.get('epoch', '?')})")

    N_TRAIN = data["N_TRAIN"]
    N_TEST = data["N_TEST"]
    N_BATCH = data["N_BATCH"]

    x_t = data["X"].to(device, dtype=dtype)
    z_t = data["Z"].to(device, dtype=dtype)
    w_t = data["W"].to(device, dtype=dtype)
    a_t = data["A"].to(device, dtype=dtype)
    y_t = data["Y"].to(device, dtype=dtype)

    preds = np.zeros((N_BATCH, N_TEST), dtype=np.float32)
    chunk = 10

    with torch.no_grad():
        for start in range(0, N_BATCH, chunk):
            end = min(start + chunk, N_BATCH)
            preds_ = model.estimate_CATE(
                x = x_t[start:end],
                a_train = a_t[start:end, :N_TRAIN, 0],
                w = w_t[start:end],
                z = z_t[start:end],
                y_train = y_t[start:end, :N_TRAIN, 0],
                return_log_variance=False
            ).cpu().numpy()[:,:]
            if len(preds_.shape) == 3:
                preds[start:end] = preds_[:,:,0]
            else:
                preds[start:end] = preds_

    np.savez(pred_file, preds=preds)
    console.print(f"    [green]✓[/] {name} → {pred_file}")
    return name, preds


# ─────────────────────────────────────────────────────────────────────────────
# Partial results printer (called after each method completes)
# ─────────────────────────────────────────────────────────────────────────────


def print_partial_results(
    accumulated: dict[str, np.ndarray],
    true_ites: np.ndarray,
) -> None:
    """Print a quick ATE summary for all methods completed so far.

    WHY print partial results?
        Long benchmarks can run for hours.  Seeing intermediate results lets
        you spot clearly-failing methods early and decide whether to abort.
    """
    if not accumulated:
        return

    frames = [
        compute_all_metrics(preds, true_ites, name)
        for name, preds in accumulated.items()
    ]
    all_df = pd.concat(frames, ignore_index=True)
    ate_df, _ = build_summary_tables(all_df)

    console.print(
        Padding(make_ate_rich_table(ate_df, title="ATE — results so far"), (1, 4))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CATE benchmark — generate data, run methods, evaluate."
    )
    parser.add_argument(
        "--config",
        default="bench_config.yaml",
        help="Path to the YAML config file (default: bench_config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions even if they are already saved.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Run sequentially (useful for debugging / profiling).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["data"]["output_dir"])
    preds_dir = output_dir / "preds"
    results_dir = output_dir / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    force = args.force or cfg["run"].get("force_rerun", False)
    n_workers = cfg["run"].get("n_workers", max(1, (os.cpu_count() or 4) - 2))

    console.print(Rule("[bold blue]CATE Benchmark[/]"))
    console.print(f"  Config   : [cyan]{args.config}[/]")
    console.print(f"  Output   : [cyan]{output_dir}[/]")
    console.print(f"  Workers  : [cyan]{n_workers}[/]")
    console.print(f"  Force    : [cyan]{force}[/]")
    console.print()

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1  Data[/]"))
    data = get_data(cfg, output_dir, force_regen=force)

    N_TEST = data["N_TEST"]
    N_BATCH = data["N_BATCH"]
    true_ITEs = data["true_ITEs"]  # (N_BATCH, N_TEST)

    console.print(
        f"    [green]✓[/] Loaded: "
        f"{N_BATCH} episodes × "
        f"(n_train={data['N_TRAIN']}, n_test={N_TEST})"
    )

    # Warn if no ITE variance across any episode
    global_ite_var = true_ITEs.var(axis=1)  # variance per episode
    frac_homogeneous = (global_ite_var < 1e-6).mean()
    if frac_homogeneous > 0.5:
        console.print(
            f"    [yellow]⚠[/] {frac_homogeneous:.0%} of episodes have near-zero "
            "ITE variance — CATE R² will be skipped for those episodes."
        )

    batches = build_batches(data)  # make N_BATCH dicts

    # ── Step 2: Baseline methods ──────────────────────────────────────────────
    method_cfgs: list[dict] = cfg.get("methods", [])

    if not method_cfgs:
        console.print("    [yellow]No methods configured.[/]")
    else:
        console.print(Rule(f"[bold]Step 2  Methods ({len(method_cfgs)})[/]"))
        console.print()

    accumulated: dict[str, np.ndarray] = {}

    # Progress bars rendered via Rich; one task per method
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,  # keep finished bars visible
    )

    with progress:
        for mcfg in method_cfgs:
            name = mcfg["name"]
            pred_file = preds_dir / f"{name}_preds.npz"

            if pred_file.exists() and not force:
                preds = np.load(pred_file)["preds"]
                console.print(f"  [dim]↩  {name:<28} loaded from cache[/]")
                accumulated[name] = preds
                continue

            t0 = time.perf_counter()
            if args.no_parallel:
                preds = run_method_sequential(mcfg, batches, N_TEST, progress)
            else:
                preds = run_method_parallel(mcfg, batches, N_TEST, n_workers, progress)

            elapsed = time.perf_counter() - t0

            # Incremental save
            np.savez(pred_file, preds=preds)
            console.print(f"  [green]✓[/] {name:<28} saved  [dim]({elapsed:.1f}s)[/]")
            accumulated[name] = preds

            # Show partial results after each method completes
            print_partial_results(accumulated, true_ITEs)

    # ── Step 3 (optional): Neural model checkpoints ───────────────────────────
    model_cfgs = cfg.get("models", [])
    if model_cfgs:
        console.print(Rule(f"[bold]Step 3  Neural Models ({len(model_cfgs)})[/]"))
        prior_cfg = cfg["data"]["prior"]
        device = cfg["run"].get("device", "cpu")

        for mcfg in model_cfgs:
            result = run_neural_model(mcfg, data, prior_cfg, device, preds_dir, force)
            if result is not None:
                name, preds = result
                accumulated[name] = preds

    # ── Step 4: Final evaluation ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 4  Evaluation[/]"))
    console.print()

    if not accumulated:
        console.print("  [red]No method predictions found. Exiting.[/]")
        return

    all_results = pd.concat(
        [
            compute_all_metrics(preds, true_ITEs, name)
            for name, preds in accumulated.items()
        ],
        ignore_index=True,
    )

    ate_summary, cate_summary = build_summary_tables(all_results)

    console.print(Padding(make_ate_rich_table(ate_summary), (1, 4)))
    console.print(Padding(make_cate_rich_table(cate_summary), (1, 4)))

    # Save full episode-level results for further analysis
    results_csv = results_dir / "results.csv"
    all_results.to_csv(results_csv, index=False)
    ate_summary.to_csv(results_dir / "ate_summary.csv")
    cate_summary.to_csv(results_dir / "cate_summary.csv")

    console.print(f"  [green]✓[/] Full results → [cyan]{results_csv}[/]")
    console.print(Rule("[bold green]Done[/]"))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if sys.platform != "win32":
        multiprocessing.set_start_method("spawn", force=True)

    main()
