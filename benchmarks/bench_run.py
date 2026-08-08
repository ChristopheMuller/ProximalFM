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
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import re

from rich.console import Console
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.bench_data import build_benchmark_dict, load_benchmark_from_val_dir
from benchmarks.bench_data_csv import load_benchmark_from_csv
from benchmarks.bench_methods import build_method
from benchmarks.bench_metrics import (
    compute_all_metrics,
    build_summary_table,
    make_summary_rich_table,
)

# ── Optional: neural model support ───────────────────────────────────────────
try:
    from proximalfm._model.proximalfm import ProximalFM
    from proximalfm.prior._dataset import PriorDataset
    from proximalfm.prior._linear_dataset import LinearPriorDataset

    _PROXFM_AVAILABLE = True
except ImportError:
    _PROXFM_AVAILABLE = False

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Utils & Config
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "run"):
        if key not in cfg:
            raise KeyError(f"bench_config.yaml is missing required section '{key}'.")
    return cfg


def _empty_accelerator_cache(device: torch.device) -> None:
    """Free cached accelerator memory (CUDA or MPS); no-op on CPU."""
    import gc

    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────


def _generate_data(data_cfg: dict, run_cfg: dict, output_dir: Path) -> dict:
    """Generate synthetic datasets from the prior, standardize, and save to disk."""
    if not _PROXFM_AVAILABLE:
        raise RuntimeError(
            "proximalfm is not installed. Set data.source: load to use an "
            "existing data file, or install proximalfm to generate new data."
        )

    n_rows = data_cfg["n_rows"]
    n_batch = data_cfg["n_batch"]
    prior_kw = data_cfg.get("prior", {}).copy()
    prop_train = data_cfg.get("prop_train", 0.75)
    seed = run_cfg.get("seed", 42)

    # Force dimensions for the benchmark chunk
    prior_kw.update(
        {
            "batch_size": n_batch,
            "min_seq_len": None,
            "max_seq_len": n_rows,
            "min_train_size": prop_train,
            "max_train_size": prop_train + 0.001,
            "device": "cpu",
        }
    )

    console.print(
        f"    Generating [bold]{n_batch}[/] datasets "
        f"(n_rows={n_rows}, seed={seed})…"
    )

    # Generate reproducible batch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = PriorDataset(**prior_kw)
    # dataset = LinearPriorDataset(**prior_kw)
    batch = next(iter(dataset))

    # Prior dataset yields 14 elements
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
        seq_len,
        train_size,
    ) = batch

    # Standardize (context-only stats, no test leakage) and package into the
    # benchmark dict; the shared builder is also used when loading val-format
    # directories, so both data paths produce an identical schema.
    data = build_benchmark_dict(
        U=U,
        X=X,
        W=W,
        Z=Z,
        A=A,
        Y_obs=Y_obs,
        Y0=Y0,
        Y1=Y1,
        d_X=d_X,
        d_W=d_W,
        d_Z=d_Z,
        train_size=int(train_size[0].item()),
    )

    torch.save(data, output_dir / "data.pt")
    with open(output_dir / "prior_config.yaml", "w") as f:
        yaml.dump(data_cfg.get("prior", {}), f)

    console.print(f"    [green]✓[/] Saved (Normalized) → {output_dir / 'data.pt'}")
    return data


def get_data(cfg: dict, output_dir: Path, force_regen: bool = False) -> dict:
    """Return dataset dict — generate or load depending on config.

    Loading resolves in this order:
      1. ``data.val_dir`` set and no cached ``data.pt`` (or --force): build the
         standardized dict from the raw val-format directory (``batch_*.pt``,
         as written by synt_eval/build_proximal_datasets.py) and cache it as
         ``data.pt`` in the run's output dir (bench_plots.py reads it there).
      2. ``source: csv`` and no cached ``data.pt`` (or --force): build the
         standardized dict from a single real-world CSV (e.g. a Causal
         Chambers export) via ``data.csv_path`` + ``data.columns``, a
         column-name to variable-block mapping. Always yields ``N_BATCH=1``.
      3. Otherwise fall back to an existing ``data.pt`` in the output dir.
    """
    data_path = output_dir / "data.pt"
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "generate")
    val_dir = data_cfg.get("val_dir")

    if source == "generate" and (force_regen or not data_path.exists()):
        return _generate_data(data_cfg, cfg["run"], output_dir)

    if source == "csv" and (force_regen or not data_path.exists()):
        csv_path = data_cfg["csv_path"]
        console.print(f"    Building benchmark data from CSV [cyan]{csv_path}[/]…")
        data = load_benchmark_from_csv(
            csv_path,
            data_cfg["columns"],
            prop_train=data_cfg.get("prop_train", 0.75),
            shuffle=data_cfg.get("shuffle", True),
            seed=cfg["run"].get("seed", 42),
            normalize_treatment=data_cfg.get("normalize_treatment", True),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, data_path)
        console.print(f"    [green]✓[/] Saved (Normalized) → {data_path}")
        return data

    if val_dir and (force_regen or not data_path.exists()):
        console.print(f"    Building benchmark data from val dir [cyan]{val_dir}[/]…")
        data = load_benchmark_from_val_dir(val_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, data_path)
        console.print(f"    [green]✓[/] Saved (Normalized) → {data_path}")
        return data

    if data_path.exists():
        console.print(f"    Loading existing data from [cyan]{data_path}[/]…")
        return torch.load(data_path, weights_only=False)

    raise FileNotFoundError(
        f"Data file not found at {data_path}. "
        "Set data.source: generate, point data.val_dir at a directory of "
        "batch_*.pt files, or set data.source: csv with data.csv_path + "
        "data.columns in bench_config.yaml."
    )


def build_batches(data: dict) -> list[dict]:
    """Slice the full data tensor into a list of per-batch numpy dicts."""
    N_TRAIN = data["N_TRAIN"]
    U_np = data["U"].numpy()
    X_np = data["X"].numpy()
    Z_np = data["Z"].numpy()
    W_np = data["W"].numpy()
    A_np = data["A"].numpy()
    Y_np = data["Y"].numpy()
    true_ITEs_np = data["true_ITEs"]

    batches = []
    for b in range(data["N_BATCH"]):
        A_ctx = A_np[b, :N_TRAIN]
        Y_ctx = Y_np[b, :N_TRAIN]

        # Squeeze trailing dimension if prior yielded (B, T, 1)
        if A_ctx.ndim > 1:
            A_ctx = A_ctx.squeeze(-1)
        if Y_ctx.ndim > 1:
            Y_ctx = Y_ctx.squeeze(-1)

        batches.append(
            {
                "b": b,
                "U_ctx": U_np[b, :N_TRAIN],
                "X_ctx": X_np[b, :N_TRAIN],
                "W_ctx": W_np[b, :N_TRAIN],
                "Z_ctx": Z_np[b, :N_TRAIN],
                "A_ctx": A_ctx,
                "Y_ctx": Y_ctx,
                "U_q": U_np[b, N_TRAIN:],
                "X_q": X_np[b, N_TRAIN:],
                "W_q": W_np[b, N_TRAIN:],
                "Z_q": Z_np[b, N_TRAIN:],
                "true_ITE_std": true_ITEs_np[b],
            }
        )

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Parallel method runner
# ─────────────────────────────────────────────────────────────────────────────


def method_output_names(method_cfg: dict) -> list[str]:
    """Return the list of prediction-set names a single method config yields.

    A plain config produces one output named after ``cfg['name']``. A config
    with an ``outputs`` list (multi-output methods, e.g. NMMR emitting several
    CATE variants from one trained model) produces one output per entry.
    """
    outputs = method_cfg.get("outputs")
    if outputs:
        return [o["name"] for o in outputs]
    return [method_cfg["name"]]


def _worker(args: tuple) -> tuple[int, dict[str, np.ndarray] | None, float, str | None]:
    b, method_cfg, U_ctx, X_ctx, A_ctx, Y_ctx, W_ctx, Z_ctx, U_q, X_q, W_q, Z_q = args
    try:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        t0 = time.perf_counter()
        _, method = build_method(method_cfg)

        # --- ORACLE INJECTION ---
        if method_cfg.get("is_oracle", False):
            X_ctx = np.concatenate([X_ctx, U_ctx], axis=-1)
            X_q = np.concatenate([X_q, U_q], axis=-1)
        # ------------------------

        method.fit(X_ctx, A_ctx, Y_ctx, W=W_ctx, Z=Z_ctx)

        # One fit → one or more named prediction sets. The default estimator
        # returns a single output keyed by None, which we map to the config name.
        variants = method.predict_variants(X_q, W=W_q, Z=Z_q)
        name = method_cfg["name"]
        result = {
            (name if k is None else k): np.asarray(v).astype(np.float32)
            for k, v in variants.items()
        }
        t1 = time.perf_counter()
        return b, result, t1 - t0, None
    except Exception as exc:
        return b, None, 0.0, str(exc)


def run_method_parallel(
    method_cfg: dict,
    batches: list[dict],
    n_test: int,
    n_workers: int,
    progress: Progress,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_batch = len(batches)
    output_names = method_output_names(method_cfg)
    preds = {
        name: np.full((n_batch, n_test), np.nan, dtype=np.float32)
        for name in output_names
    }
    times = {name: np.zeros(n_batch, dtype=np.float32) for name in output_names}
    n_errors = 0

    task = progress.add_task(f"  [cyan]{method_cfg['name']:<28}[/]", total=n_batch)
    args_list = [
        (
            b["b"],
            method_cfg,
            b["U_ctx"],
            b["X_ctx"],
            b["A_ctx"],
            b["Y_ctx"],
            b["W_ctx"],
            b["Z_ctx"],
            b["U_q"],
            b["X_q"],
            b["W_q"],
            b["Z_q"],
        )
        for b in batches
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            b_idx, result, elapsed, err = future.result()
            if result is not None:
                # All variants share the single fit, so record the same elapsed.
                for name, arr in result.items():
                    preds[name][b_idx] = arr
                    times[name][b_idx] = elapsed
            else:
                n_errors += 1
            progress.advance(task)

    if n_errors:
        console.print(
            f"    [yellow]⚠[/] {method_cfg['name']}: {n_errors}/{n_batch} batches failed (NaN filled)."
        )
    return preds, times


def run_method_sequential(
    method_cfg: dict,
    batches: list[dict],
    n_test: int,
    progress: Progress,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_batch = len(batches)
    output_names = method_output_names(method_cfg)
    preds = {
        name: np.full((n_batch, n_test), np.nan, dtype=np.float32)
        for name in output_names
    }
    times = {name: np.zeros(n_batch, dtype=np.float32) for name in output_names}

    task = progress.add_task(
        f"  [cyan]{method_cfg['name']:<28}[/] [dim](sequential)[/]", total=n_batch
    )

    for b_dict in batches:
        args = (
            b_dict["b"],
            method_cfg,
            b_dict["U_ctx"],
            b_dict["X_ctx"],
            b_dict["A_ctx"],
            b_dict["Y_ctx"],
            b_dict["W_ctx"],
            b_dict["Z_ctx"],
            b_dict["U_q"],
            b_dict["X_q"],
            b_dict["W_q"],
            b_dict["Z_q"],
        )
        b_idx, result, elapsed, err = _worker(args)
        if result is not None:
            for name, arr in result.items():
                preds[name][b_idx] = arr
                times[name][b_idx] = elapsed
        if err:
            console.print(f"    [red]✗[/] batch {b_idx}: {err}")
        progress.advance(task)

    return preds, times


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 (optional) — Neural model inference
# ─────────────────────────────────────────────────────────────────────────────


def run_neural_model(
    model_cfg: dict,
    data: dict,
    device_str: str,
    preds_dir: Path,
    force: bool,
) -> list[tuple[str, np.ndarray, np.ndarray]] | None:
    """Load ProximalFM checkpoint(s) and run inference.

    Returns a list of tuples containing (output_name, preds, times).
    """
    if not _PROXFM_AVAILABLE:
        console.print(
            "    [yellow]⚠[/] proximalfm not installed — skipping neural models."
        )
        return None

    name = model_cfg["name"]
    ckpt_dir = Path(model_cfg["checkpoint"])
    run_all = model_cfg.get("all", False)
    # Number of feature-transformed views to average over at inference.
    # <= 1 (or unset) means a single plain predict; > 1 enables ensemble prediction.
    n_ensemble = int(model_cfg.get("ensemble", 1) or 1)
    # Optional bijective value transforms cycled across ensemble views (e.g.
    # [sign, signed_log, arcsinh, quantile]). Unset/empty => permutation-only.
    ensemble_transforms = model_cfg.get("transforms", None) or None

    # 1. Identify which checkpoints we want to process
    targets = []  # Elements: (display_suffix, target_ckpt_path)

    if ckpt_dir.is_file():
        targets.append(("", ckpt_dir))
    elif run_all:
        # Collect ALL checkpoints inside the directory
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        valid_ckpts = []
        for f in ckpt_files:
            match = re.search(r"(\d+)", f.name)
            step_val = int(match.group(1)) if match else -1
            valid_ckpts.append((step_val, f))

        # Sort chronologically/numerically by step count
        valid_ckpts.sort(key=lambda x: x[0])

        if not valid_ckpts:
            console.print(
                f"    [red]✗[/] No checkpoints found in directory: {ckpt_dir}"
            )
            return None

        for step, path in valid_ckpts:
            suffix = f"_step_{step}" if step != -1 else f"_{path.stem}"
            targets.append((suffix, path))
        console.print(
            f"    [dim]Found 'all: true'. Queueing {len(targets)} checkpoints from {ckpt_dir}...[/]"
        )
    else:
        # Default behavior: look for best_model.ckpt or fall back to highest numerical step
        best_path = ckpt_dir / "best_model.ckpt"
        if best_path.exists():
            targets.append(("", best_path))
        else:
            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
            valid_ckpts = []
            for f in ckpt_files:
                match = re.search(r"(\d+)", f.name)
                if match:
                    valid_ckpts.append((int(match.group(1)), f))

            if valid_ckpts:
                highest_step, fallback_path = max(valid_ckpts, key=lambda x: x[0])
                targets.append((f"_{highest_step}", fallback_path))
                console.print(
                    f"    [dim]best_model.ckpt not found. Selected highest step: {fallback_path.name}[/]"
                )
            else:
                console.print(f"    [red]✗[/] No checkpoints found in: {ckpt_dir}")
                return None

    # 2. Iterate through determined targets sequentially
    completed_runs = []
    device = torch.device(device_str)

    N_TRAIN = data["N_TRAIN"]
    N_TEST = data["N_TEST"]
    N_BATCH = data["N_BATCH"]

    x_t = data["X"].to(device)
    z_t = data["Z"].to(device)
    w_t = data["W"].to(device)
    a_t = data["A"].to(device)
    y_t = data["Y"].to(device)
    d_x = data["d_X"].to(device, dtype=torch.long)
    d_z = data["d_Z"].to(device, dtype=torch.long)
    d_w = data["d_W"].to(device, dtype=torch.long)

    if a_t.ndim == 3:
        a_t = a_t.squeeze(-1)
    if y_t.ndim == 3:
        y_t = y_t.squeeze(-1)

    for suffix, ckpt_path in targets:
        output_name = f"{name}{suffix}"
        pred_file = preds_dir / f"{output_name}_preds.npz"

        if pred_file.exists() and not force:
            console.print(
                f"    [dim]Skip {output_name!r} — predictions already saved.[/]"
            )
            loaded = np.load(pred_file)
            preds = loaded["preds"]
            times = loaded.get("times", np.zeros(len(preds), dtype=np.float32))
            completed_runs.append((output_name, preds, times))
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Initialize model architecture matching checkpoint parameters
        mcfg = ckpt.get("config", {})
        model = ProximalFM(**mcfg).to(device)

        # Safely map state weights (handling explicit/implicit module boundaries)
        state_dict = ckpt.get("state_dict", ckpt)
        clean_state_dict = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()

        if n_ensemble > 1:
            tf_note = (
                f" [{'+'.join(ensemble_transforms)}]" if ensemble_transforms else ""
            )
            ensemble_note = f", ensemble×{n_ensemble}{tf_note}"
        else:
            ensemble_note = ""
        console.print(
            f"    [green]✓[/] Processing {output_name!r} (step {ckpt.get('curr_step', '?')}{ensemble_note})"
        )

        preds = np.zeros((N_BATCH, N_TEST), dtype=np.float32)
        times = np.zeros(N_BATCH, dtype=np.float32)
        chunk = 3

        neural_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("·"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

        with torch.no_grad(), neural_progress:
            task = neural_progress.add_task(
                f"    [magenta]{output_name:<28}[/]", total=N_BATCH
            )
            for start in range(0, N_BATCH, chunk):
                end = min(start + chunk, N_BATCH)
                t0 = time.perf_counter()

                micro_X = x_t[start:end]
                micro_Z = z_t[start:end]
                micro_W = w_t[start:end]
                micro_A = a_t[start:end]
                micro_Y_obs = y_t[start:end]
                micro_d_X = d_x[start:end]
                micro_d_Z = d_z[start:end]
                micro_d_W = d_w[start:end]

                # Truncate active dimensions for execution efficiency
                max_X = micro_d_X.max().item()
                if micro_X.shape[-1] > max_X:
                    micro_X = micro_X[..., :max_X]

                max_W = micro_d_W.max().item()
                if micro_W.shape[-1] > max_W:
                    micro_W = micro_W[..., :max_W]

                max_Z = micro_d_Z.max().item()
                if micro_Z.shape[-1] > max_Z:
                    micro_Z = micro_Z[..., :max_Z]

                y_ctx = micro_Y_obs[:, :N_TRAIN]

                if n_ensemble > 1:
                    # Average CATE over `n_ensemble` feature-transformed views.
                    pred_cate = model.predict_ensemble(
                        X=micro_X,
                        Z=micro_Z,
                        W=micro_W,
                        A=micro_A,
                        y_train=y_ctx,
                        d_X=micro_d_X,
                        d_Z=micro_d_Z,
                        d_W=micro_d_W,
                        target="cate",
                        output_type="mean",
                        n_estimators=n_ensemble,
                        value_transforms=ensemble_transforms,
                    )
                else:
                    pred_cate = model.predict(
                        X=micro_X,
                        Z=micro_Z,
                        W=micro_W,
                        A=micro_A,
                        y_train=y_ctx,
                        d_X=micro_d_X,
                        d_Z=micro_d_Z,
                        d_W=micro_d_W,
                        target="cate",
                        output_type="mean",
                    )

                t1 = time.perf_counter()

                chunk_size = end - start
                avg_time = (t1 - t0) / chunk_size
                times[start:end] = avg_time
                preds[start:end] = pred_cate.cpu().numpy()

                neural_progress.advance(task, chunk_size)

        np.savez(pred_file, preds=preds, times=times)
        console.print(f"    [green]✓[/] {output_name} saved → {pred_file}")
        completed_runs.append((output_name, preds, times))

        # Release the model and free accelerator memory before the next
        # checkpoint/model. Without this, sequentially loaded checkpoints (and
        # the extra forward passes done by ensemble runs) accumulate on the
        # device; on MPS this can surface as spurious "index out of bounds"
        # errors once memory is exhausted.
        del model
        _empty_accelerator_cache(device)

    return completed_runs


# ─────────────────────────────────────────────────────────────────────────────
# Results presentation
# ─────────────────────────────────────────────────────────────────────────────


def print_partial_results(
    accumulated: dict[str, tuple[np.ndarray, np.ndarray]], true_ites: np.ndarray
) -> None:
    if not accumulated:
        return
    frames = [
        compute_all_metrics(preds, true_ites, times, name)
        for name, (preds, times) in accumulated.items()
    ]
    all_df = pd.concat(frames, ignore_index=True)
    summary = build_summary_table(all_df)
    console.print(
        Padding(
            make_summary_rich_table(summary, title="Live Results So Far"),
            (1, 4),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CATE benchmark — generate data, run methods, evaluate."
    )
    default_config = Path(__file__).resolve().parent / "bench_config.yaml"
    parser.add_argument(
        "--config", default=str(default_config), help="Path to the YAML config file."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions even if they are already saved.",
    )
    parser.add_argument("--no-parallel", action="store_true", help="Run sequentially.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["data"]["output_dir"])
    preds_dir = output_dir / "preds"
    results_dir = output_dir / "results"

    for d in [output_dir, preds_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    force = args.force or cfg["run"].get("force_rerun", False)
    n_workers = cfg["run"].get("n_workers", max(1, (os.cpu_count() or 4) - 2))

    console.print(Rule("[bold blue]CATE Benchmark[/]"))
    console.print(f"  Config   : [cyan]{args.config}[/]")
    console.print(f"  Output   : [cyan]{output_dir}[/]")
    console.print(f"  Workers  : [cyan]{n_workers}[/]")
    console.print()

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1  Data[/]"))
    data = get_data(cfg, output_dir, force_regen=force)

    N_TEST = data["N_TEST"]
    N_BATCH = data["N_BATCH"]

    console.print(
        f"    [green]✓[/] Loaded: {N_BATCH} episodes × (n_train={data['N_TRAIN']}, n_test={N_TEST})"
    )
    batches = build_batches(data)
    true_ITEs = np.stack(
        [np.asarray(b["true_ITE_std"]).squeeze() for b in batches]
    ).astype(np.float32)

    # ── Step 2: Baseline methods ─────────────────────────────────────────────
    method_cfgs: list[dict] = cfg.get("methods", [])
    if not method_cfgs:
        console.print("    [yellow]No methods configured.[/]")
    else:
        console.print(Rule(f"[bold]Step 2  Methods ({len(method_cfgs)})[/]"))
        console.print()

    accumulated: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        for mcfg in method_cfgs:
            name = mcfg["name"]
            output_names = method_output_names(mcfg)
            pred_files = {n: preds_dir / f"{n}_preds.npz" for n in output_names}

            # Skip only when every output of this config is already cached.
            if not force and all(f.exists() for f in pred_files.values()):
                for out_name, f in pred_files.items():
                    loaded = np.load(f)
                    preds = loaded["preds"]
                    times = loaded.get("times", np.zeros(len(preds), dtype=np.float32))
                    console.print(f"  [dim]↩  {out_name:<28} loaded from cache[/]")
                    accumulated[out_name] = (preds, times)
                continue

            t0 = time.perf_counter()
            if args.no_parallel:
                preds_map, times_map = run_method_sequential(
                    mcfg, batches, N_TEST, progress
                )
            else:
                preds_map, times_map = run_method_parallel(
                    mcfg, batches, N_TEST, n_workers, progress
                )

            elapsed = time.perf_counter() - t0
            for out_name in output_names:
                preds, times = preds_map[out_name], times_map[out_name]
                np.savez(pred_files[out_name], preds=preds, times=times)
                console.print(
                    f"  [green]✓[/] {out_name:<28} saved  [dim]({elapsed:.1f}s)[/]"
                )
                accumulated[out_name] = (preds, times)

            print_partial_results(accumulated, true_ITEs)

    # ── Step 3 (optional): Neural model checkpoints ──────────────────────────
    model_cfgs = cfg.get("models", [])
    if model_cfgs:
        console.print(Rule(f"[bold]Step 3  Neural Models ({len(model_cfgs)})[/]"))
        device = cfg["run"].get("device", "cpu")

        for mcfg in model_cfgs:
            results = run_neural_model(mcfg, data, device, preds_dir, force)
            if results is not None:
                for name, preds, times in results:
                    accumulated[name] = (preds, times)
                    print_partial_results(accumulated, true_ITEs)

    # ── Step 4: Final evaluation ─────────────────────────────────────────────
    console.print(Rule("[bold]Step 4  Evaluation[/]"))
    console.print()

    if not accumulated:
        console.print("  [red]No method predictions found. Exiting.[/]")
        return

    all_results = pd.concat(
        [
            compute_all_metrics(preds, true_ITEs, times, name)
            for name, (preds, times) in accumulated.items()
        ],
        ignore_index=True,
    )

    summary = build_summary_table(all_results)
    console.print(
        Padding(
            make_summary_rich_table(summary, title="Final Evaluation"),
            (1, 4),
        )
    )

    results_csv = results_dir / "results.csv"
    all_results.to_csv(results_csv, index=False)
    summary.to_csv(results_dir / "summary.csv")

    console.print(f"  [green]✓[/] Full results → [cyan]{results_csv}[/]")
    console.print(Rule("[bold green]Done[/]"))


if __name__ == "__main__":
    if sys.platform != "win32":
        multiprocessing.set_start_method("spawn", force=True)
    main()
