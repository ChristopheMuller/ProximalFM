"""
run.py — Single entry-point for the full benchmark pipeline.

Usage
─────
    python run.py                          # uses config_val.yaml
    python run.py --config my_config.yaml
    python run.py --force                  # rerun even if preds exist
    python run.py --no-parallel            # single-process (debug)

Compute
───────
    Device and thread counts belong to `benchmarks.compute`, not to this script and
    not to the estimators. `run.compute` in the YAML is the run-wide default; each
    method (and each checkpoint under `models:`) may overlay a `compute:` block on
    it. A method is handed its resolved budget and decides for itself whether to fan
    its episodes out through the shared WorkerManager or hold a device in-process.

Pipeline
────────
    1. Load the datasets                    (Step 1)
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
import sys
import time
from dataclasses import replace
from typing import Optional
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

from benchmarks.compute import (
    INLINE,
    ComputeConfig,
    WorkerManager,
    apply_compute,
    parse_global_compute,
    resolve_method_compute,
)
from benchmarks.data import load_benchmark_from_val_dir
from benchmarks.data_csv import load_benchmark_from_csv
from benchmarks.methods import method_estimand
from benchmarks.runners import method_output_names, run_method
from benchmarks.metrics import (
    ATE_RICH_COLUMNS,
    DEFAULT_RICH_COLUMNS,
    compute_all_metrics,
    build_summary_table,
    make_summary_rich_table,
)

# Above this share of the CATE target's variance being Monte-Carlo sampling
# noise, the set is too coarsely built to rank methods on and we say so.
_S2_WARN_THRESHOLD = 0.05

# ── Optional: neural model support ───────────────────────────────────────────
try:
    from proximalfm._model.proximalfm import ProximalFM

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
            raise KeyError(f"config_*.yaml is missing required section '{key}'.")
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


def get_data(cfg: dict, output_dir: Path, force_regen: bool = False) -> dict:
    """Return the standardized dataset dict named by the config.

    Every benchmark run scores a *pre-built* set — one of the three builders
    (`prior_eval`, `synt_eval`, `lit_eval`), or a single real-world CSV. There
    is no on-the-fly prior sampling: a set that only exists inside one run is
    one nobody else can score against.

    Loading resolves in this order:
      1. ``data.val_dir`` set and no cached ``data.pt`` (or --force): build the
         standardized dict from the raw val-format directory (``batch_*.pt``,
         as written by ``proximalfm.prior._genload.write_batch_file``) and cache
         it as ``data.pt`` in the run's output dir (plots.py reads it there).
      2. ``source: csv`` and no cached ``data.pt`` (or --force): build the
         standardized dict from a single real-world CSV (e.g. a Causal
         Chambers export) via ``data.csv_path`` + ``data.columns``, a
         column-name to variable-block mapping. Always yields ``N_BATCH=1``.
      3. Otherwise fall back to an existing ``data.pt`` in the output dir.

    A ``data.pt`` cached before the CATE target existed carries no
    ``true_CATEs``, so the run falls back to the ITE track and says so on the
    console. Delete it (or pass ``--force``) to rebuild with the CATE target.
    """
    data_path = output_dir / "data.pt"
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "load")
    val_dir = data_cfg.get("val_dir")

    # Refused rather than ignored: a config still saying `generate` asks for
    # datasets nobody else can score against, and falling through to "no data
    # file found" would read as a missing path rather than a removed feature.
    if source == "generate":
        raise ValueError(
            "data.source: generate has been removed — the benchmark scores "
            "pre-built sets only. Build one first (prior_eval.build, "
            "synt_eval/build_proximal_datasets.py or lit_eval.build) and point "
            "data.val_dir at its val/ directory with data.source: load."
        )

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
        "Point data.val_dir at a directory of batch_*.pt files, or set "
        "data.source: csv with data.csv_path + data.columns in config_*.yaml."
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
    true_CATEs_np = data.get("true_CATEs")

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
                "true_CATE_std": (None if true_CATEs_np is None else true_CATEs_np[b]),
            }
        )

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 (optional) — Neural model inference
# ─────────────────────────────────────────────────────────────────────────────


def run_neural_model(
    model_cfg: dict,
    data: dict,
    compute: ComputeConfig,
    preds_dir: Path,
    force: bool,
) -> list[tuple[str, np.ndarray, np.ndarray]] | None:
    """Load ProximalFM checkpoint(s) and run inference.

    Always device-resident and single-process: it batches episodes into the model's
    own batch dimension, which is a better use of a GPU than any episode fan-out, so
    only `compute.device` and `compute.torch_threads` matter here. It reads the same
    resolved `ComputeConfig` as the `methods:` entries, so a checkpoint can pin its
    own device with a `compute:` block.

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
    apply_compute(compute)
    device = torch.device(compute.device)

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


ORACLE_CATE = "Oracle-CATE(X)"
ORACLE_ATE = "Oracle-ATE"
ORACLE_ITE = "Oracle-ITE"

# The reference rows answer no config, so they declare their own tags. Oracle-ITE
# is the realized Y1-Y0, which carries that row's own U and noise — hence the u.
REFERENCE_ESTIMANDS = {
    ORACLE_CATE: "cate(x)",
    ORACLE_ATE: "ate(x)",
    ORACLE_ITE: "cate(x,u)",
}

TRACK_TITLES = {
    "cate": "CATE  —  scored against tau*(x) = E[Y1-Y0 | X]",
    "ite": "ITE  —  scored against the realized Y1-Y0",
    "ate": "ATE  —  level only",
}


def build_targets(data: dict, batches: list[dict]) -> tuple[dict, Optional[np.ndarray]]:
    """The per-row ground truths to score against, keyed by target kind.

    Always carries ``"ite"``; carries ``"cate"`` only when the set was built
    with Monte-Carlo replicates (see ``data.build_benchmark_dict``). The
    second return value is the per-row sampling variance of the CATE target, or
    ``None`` — reported as ``s2_over_var``, never subtracted.
    """
    targets = {
        "ite": np.stack(
            [np.asarray(b["true_ITE_std"]).squeeze() for b in batches]
        ).astype(np.float32)
    }
    if data.get("true_CATEs") is not None:
        targets["cate"] = np.stack(
            [np.asarray(b["true_CATE_std"]).squeeze() for b in batches]
        ).astype(np.float32)
    s2 = data.get("cate_s2")
    return targets, None if s2 is None else np.asarray(s2, dtype=np.float32)


def describe_targets(targets: dict, s2: Optional[np.ndarray], data: dict) -> None:
    """Say which estimand is being scored, and how trustworthy the target is."""
    if "cate" not in targets:
        console.print(
            "    [yellow]⚠[/] No Monte-Carlo CATE target in this set — scoring "
            "the realized ITE only. Every method is then capped by "
            "E[Var(Y1-Y0 | X)], which is usually most of Var(ITE). Rebuild with "
            "[cyan]--mc_cate K[/] (K > 1) for the CATE track."
        )
        return

    K = data.get("cate_K_eff")
    k_txt = f", K={int(K.min())}" if K is not None else ""
    console.print(
        f"    [green]✓[/] Scoring tau*(x) = E[Y1-Y0 | X] (MU1-MU0{k_txt}), "
        "with the realized ITE as a secondary track."
    )
    if s2 is None:
        return
    var = targets["cate"].var(axis=1)
    share = float(np.median(s2.mean(axis=1)[var > 0] / var[var > 0]))
    if share > _S2_WARN_THRESHOLD:
        console.print(
            f"    [yellow]⚠[/] {share:.1%} of the CATE target's variance is "
            "Monte-Carlo noise in the target itself (median over episodes). "
            "It is reported as s2_over_var and never subtracted, so every "
            "method scores worse than it is; rebuild with a larger --mc_cate."
        )


def method_output_estimands(method_cfg: dict) -> dict[str, str]:
    """Estimand of each prediction set a single method config yields."""
    outputs = method_cfg.get("outputs")
    if outputs:
        return {o["name"]: method_estimand(method_cfg, o) for o in outputs}
    return {method_cfg["name"]: method_estimand(method_cfg)}


def primary_track(targets: dict) -> str:
    """The track the live table and the default plots speak in."""
    return "cate" if "cate" in targets else "ite"


def reference_predictions(targets: dict) -> dict[str, np.ndarray]:
    """Oracle rows: predictions that need no fitting.

    ``Oracle-CATE(X)`` is tau*(x) itself. On the CATE track it must score ~0 —
    a self-check on the target plumbing — and on the ITE track it is the
    irreducible noise floor, i.e. the best any method conditioning on X alone
    can do, which is what makes the tau(x, w) methods' advantage readable.
    ``Oracle-ATE`` is the best per-episode constant, the nPEHE = 1 anchor.

    ``Oracle-ITE`` is the realized Y1-Y0 itself — the mirror image of
    ``Oracle-CATE(X)``, and only emitted when the two targets differ (without a
    Monte-Carlo target it would duplicate that row exactly). It is the perfect
    score on the ITE track and, on the CATE track, the *price of the noise*: it
    predicts tau*(x) + that row's own U and noise, so its nPEHE is
    SD(ITE - tau*)/SD(tau*), typically well above 1, with a strongly negative
    Shape R². Reading it beside ``Oracle-ATE`` says how much of Var(ITE) no
    tau(x) method should even try to explain. At the ATE level it is nearly
    unbiased — only the row-average of the noise survives — so it is also the
    per-episode sampling floor for the ATE columns.
    """
    base = targets.get("cate", targets["ite"])
    refs = {
        ORACLE_CATE: base,
        ORACLE_ATE: np.repeat(base.mean(axis=1, keepdims=True), base.shape[1], axis=1),
    }
    if "cate" in targets:
        refs[ORACLE_ITE] = targets["ite"]
    return refs


def score_everything(
    accumulated: dict[str, tuple[np.ndarray, np.ndarray]],
    estimands: dict[str, str],
    targets: dict[str, np.ndarray],
    s2: Optional[np.ndarray] = None,
    references: Optional[dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Score every method against every available target, into one long frame.

    One row per (method, episode, target). Scoring is pure numpy over arrays
    already in memory, so there is no reason to be selective here — the tables
    decide which rows they can rank fairly.
    """
    frames = []
    for kind, target in targets.items():
        rows = dict(accumulated)
        if references:
            zeros = np.zeros(len(target), dtype=np.float32)
            rows.update({n: (p, zeros) for n, p in references.items()})
        for name, (preds, times) in rows.items():
            frames.append(
                compute_all_metrics(
                    preds,
                    target,
                    times,
                    name,
                    target_kind=kind,
                    estimand=estimands.get(
                        name, REFERENCE_ESTIMANDS.get(name, "cate(x)")
                    ),
                    is_reference=bool(references and name in references),
                    s2=s2 if kind == "cate" else None,
                )
            )
    return pd.concat(frames, ignore_index=True)


def print_partial_results(
    accumulated: dict[str, tuple[np.ndarray, np.ndarray]],
    estimands: dict[str, str],
    targets: dict[str, np.ndarray],
) -> None:
    if not accumulated:
        return
    track = primary_track(targets)
    all_df = score_everything(accumulated, estimands, {track: targets[track]})
    summary = build_summary_table(all_df, track)
    if summary.empty:
        return
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
        description="CATE benchmark — load data, run methods, evaluate."
    )
    default_config = Path(__file__).resolve().parent / "config_val.yaml"
    parser.add_argument(
        "--config", default=str(default_config), help="Path to the YAML config file."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions even if they are already saved.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Debug: force every method to run in this process (compute.n_workers=1).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["data"]["output_dir"])
    preds_dir = output_dir / "preds"
    results_dir = output_dir / "results"

    for d in [output_dir, preds_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    force = args.force or cfg["run"].get("force_rerun", False)

    # The run-wide default. Each method overlays its own `compute:` block on this;
    # `--no-parallel` is now just an override of it, not a second code path.
    global_compute = parse_global_compute(cfg["run"])
    if args.no_parallel:
        global_compute = replace(global_compute, n_workers=1, execution=INLINE)

    console.print(Rule("[bold blue]CATE Benchmark[/]"))
    console.print(f"  Config   : [cyan]{args.config}[/]")
    console.print(f"  Output   : [cyan]{output_dir}[/]")
    console.print(f"  Compute  : [cyan]{global_compute.describe()}[/]")
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
    targets, cate_s2 = build_targets(data, batches)
    describe_targets(targets, cate_s2, data)

    # output name -> estimand, filled in as methods are configured; the tables
    # use it to decide which track can rank a given output.
    estimands: dict[str, str] = {}

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

    # One manager for the whole session. It hands out pools lazily and keeps them,
    # so consecutive methods that agree on a budget reuse warm workers instead of
    # paying respawn + re-import + checkpoint load each.
    with WorkerManager() as manager, progress:
        for mcfg in method_cfgs:
            name = mcfg["name"]
            output_names = method_output_names(mcfg)
            estimands.update(method_output_estimands(mcfg))
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

            method_compute = resolve_method_compute(global_compute, mcfg)
            task = progress.add_task(
                f"  [cyan]{name:<28}[/] [dim]{method_compute.describe()}[/]",
                total=len(batches),
            )

            t0 = time.perf_counter()
            preds_map, times_map, n_errors = run_method(
                mcfg,
                batches,
                N_TEST,
                method_compute,
                manager,
                on_progress=lambda: progress.advance(task),
            )
            if n_errors:
                console.print(
                    f"    [yellow]⚠[/] {name}: {n_errors}/{len(batches)} "
                    "batches failed (NaN filled)."
                )

            elapsed = time.perf_counter() - t0
            for out_name in output_names:
                preds, times = preds_map[out_name], times_map[out_name]
                np.savez(pred_files[out_name], preds=preds, times=times)
                console.print(
                    f"  [green]✓[/] {out_name:<28} saved  [dim]({elapsed:.1f}s)[/]"
                )
                accumulated[out_name] = (preds, times)

            print_partial_results(accumulated, estimands, targets)

    # ── Step 3 (optional): Neural model checkpoints ──────────────────────────
    model_cfgs = cfg.get("models", [])
    if model_cfgs:
        console.print(Rule(f"[bold]Step 3  Neural Models ({len(model_cfgs)})[/]"))

        for mcfg in model_cfgs:
            # Same resolution as a method, so a checkpoint carries its own
            # `compute: {device: mps}` rather than depending on a global default.
            model_compute = resolve_method_compute(global_compute, mcfg)
            results = run_neural_model(mcfg, data, model_compute, preds_dir, force)
            if results is not None:
                for name, preds, times in results:
                    accumulated[name] = (preds, times)
                    print_partial_results(accumulated, estimands, targets)

    # ── Step 4: Final evaluation ─────────────────────────────────────────────
    console.print(Rule("[bold]Step 4  Evaluation[/]"))
    console.print()

    if not accumulated:
        console.print("  [red]No method predictions found. Exiting.[/]")
        return

    eval_cfg = cfg.get("eval", {})
    references = (
        reference_predictions(targets) if eval_cfg.get("reference_rows", True) else None
    )
    all_results = score_everything(
        accumulated, estimands, targets, s2=cate_s2, references=references
    )

    # One table per track. `cate` is the headline; `ite` is where the methods
    # that read the query rows' proxies are allowed their advantage; `ate` is
    # the level-only view, and the only place a constant-prediction method says
    # anything.
    tracks = [t for t in ("cate", "ite", "ate") if t in targets or t == "ate"]
    for track in tracks:
        summary = build_summary_table(all_results, track)
        if summary.empty:
            continue
        columns = ATE_RICH_COLUMNS if track == "ate" else DEFAULT_RICH_COLUMNS
        console.print(
            Padding(
                make_summary_rich_table(
                    summary, title=TRACK_TITLES[track], columns=columns
                ),
                (1, 4),
            )
        )
        summary.to_csv(results_dir / f"summary_{track}.csv")

    # summary.csv keeps its name and its meaning "the headline table", so
    # anything downstream that globs for it still finds the primary track.
    build_summary_table(all_results, primary_track(targets)).to_csv(
        results_dir / "summary.csv"
    )

    results_csv = results_dir / "results.csv"
    all_results.to_csv(results_csv, index=False)

    console.print(f"  [green]✓[/] Full results → [cyan]{results_csv}[/]")
    console.print(Rule("[bold green]Done[/]"))


if __name__ == "__main__":
    if sys.platform != "win32":
        multiprocessing.set_start_method("spawn", force=True)
    main()
