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
    method may overlay a `compute:` block on it. A method is handed its resolved
    budget and decides for itself whether to fan its episodes out through the shared
    WorkerManager or hold a device in-process.

Pipeline
────────
    1. Load the datasets                    (Step 1)
    2. For each method:
       a. Skip if predictions already exist (unless --force)
       b. Fit + predict across all batches in parallel
       c. Save predictions to disk immediately on completion
       d. Print running ATE metrics after each method finishes
    3. Print final ATE table + CATE table

ProximalFM is a method like any other — `cls: ProximalFM` under `methods:`,
implemented in `benchmarks/methods/proximalfm.py`. The `models:` block and the
`run_neural_model` step that used to run it through a parallel pipeline of its
own are gone; a config still carrying `models:` is refused with the translation.
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
import yaml

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
    WorkerManager,
    parse_global_compute,
    resolve_method_compute,
)
from benchmarks.data import load_benchmark_from_val_dir
from benchmarks.data_csv import load_benchmark_from_csvs, resolve_csv_paths
from benchmarks.methods import method_estimand, normalize_method_cfg
from benchmarks.runners import method_output_names, run_method
from benchmarks.metrics import TASKS, TaskReport, primary_task, score_method

# Above this share of the CATE target's variance being Monte-Carlo sampling
# noise, the set is too coarsely built to rank methods on and we say so.
_S2_WARN_THRESHOLD = 0.05

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

    # Refused rather than aliased: `models:` entries ran through a separate
    # pipeline with its own device handling and its own prediction loop, and
    # ProximalFM is now an ordinary method. Translating silently would leave the
    # config claiming a distinction that no longer exists.
    if cfg.get("models"):
        raise ValueError(
            "config `models:` is gone — ProximalFM is a method like any other. "
            "Move each entry into `methods:` and give it `cls: ProximalFM`; "
            "`checkpoint:`, `all:`, `ensemble:`, `transforms:` and `compute:` "
            "all keep their meaning."
        )
    cfg["methods"] = [normalize_method_cfg(m) for m in cfg.get("methods") or []]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────────────────────────────────────


def get_data(cfg: dict, output_dir: Path, force_regen: bool = False) -> dict:
    """Return the standardized dataset dict named by the config.

    Every benchmark run scores a *pre-built* set — one of the three builders
    (`prior_eval`, `synt_eval`, `lit_eval`), or real-world CSV files. There
    is no on-the-fly prior sampling: a set that only exists inside one run is
    one nobody else can score against.

    Loading resolves in this order:
      1. ``data.val_dir`` set and no cached ``data.pt`` (or --force): build the
         standardized dict from the raw val-format directory (``batch_*.pt``,
         as written by ``proximalfm.prior._genload.write_batch_file``) and cache
         it as ``data.pt`` in the run's output dir (plots.py reads it there).
      2. ``source: csv`` and no cached ``data.pt`` (or --force): build the
         standardized dict from one or more real-world CSVs. Each file becomes
         one independently standardized episode, selected by ``csv_path``,
         ``csv_paths``, or ``csv_dir``.
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
        csv_paths = resolve_csv_paths(data_cfg)
        console.print(
            f"    Building benchmark data from [cyan]{len(csv_paths)} CSV file(s)[/]…"
        )
        data = load_benchmark_from_csvs(
            csv_paths,
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
        "data.source: csv with a CSV input and data.columns in config_*.yaml."
    )


def build_batches(data: dict) -> list[dict]:
    """Slice the full data tensor into a list of per-batch numpy dicts.

    ``X``/``W``/``Z`` are stored zero-padded to the pool's widest episode, so
    each episode's arrays are trimmed back to its own ``d_X``/``d_W``/``d_Z``
    here. Episodes are fit independently, so nothing downstream wants the
    padding: a method sees exactly the columns its episode has, and no estimator
    needs to be told a dimension count separately from the array it describes.
    """
    N_TRAIN = data["N_TRAIN"]
    U_np = data["U"].numpy()
    X_np = data["X"].numpy()
    Z_np = data["Z"].numpy()
    W_np = data["W"].numpy()
    A_np = data["A"].numpy()
    Y_np = data["Y"].numpy()
    d_X = data["d_X"].tolist()
    d_W = data["d_W"].tolist()
    d_Z = data["d_Z"].tolist()
    true_ITEs_np = data["true_ITEs"]
    true_CATEs_np = data.get("true_CATEs")

    batches = []
    for b in range(data["N_BATCH"]):
        X_b = X_np[b, :, : d_X[b]]
        W_b = W_np[b, :, : d_W[b]]
        Z_b = Z_np[b, :, : d_Z[b]]
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
                "X_ctx": X_b[:N_TRAIN],
                "W_ctx": W_b[:N_TRAIN],
                "Z_ctx": Z_b[:N_TRAIN],
                "A_ctx": A_ctx,
                "Y_ctx": Y_ctx,
                "U_q": U_np[b, N_TRAIN:],
                "X_q": X_b[N_TRAIN:],
                "W_q": W_b[N_TRAIN:],
                "Z_q": Z_b[N_TRAIN:],
                "true_ITE_std": true_ITEs_np[b],
                "true_CATE_std": (None if true_CATEs_np is None else true_CATEs_np[b]),
            }
        )

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Results presentation
# ─────────────────────────────────────────────────────────────────────────────


ORACLE_CATE = "Oracle-CATE(X)"
ORACLE_ATE = "Oracle-ATE"
ORACLE_ITE = "Oracle-ITE"

# The reference rows answer no config, so they declare their own tags. Oracle-ITE
# is the realized Y1-Y0, which carries that row's own U, W and noise — hence the u.
REFERENCE_ESTIMANDS = {
    ORACLE_CATE: "cate(x)",
    ORACLE_ATE: "ate(x)",
    ORACLE_ITE: "cate(x,w,u)",
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


def reference_predictions(targets: dict) -> dict[str, np.ndarray]:
    """Oracle rows: predictions that need no fitting.

    All three are scored on all three tasks — they are never ranked, so the
    admissibility rules that keep the *methods* honest do not apply to them, and
    an anchor is worth having in every table.

    ``Oracle-CATE(X)`` is tau*(x) itself. On the CATE track it must score ~0 —
    a self-check on the target plumbing — and on the ITE track it is the
    irreducible noise floor, i.e. the best any method conditioning on X alone
    can do, which is what makes the tau(x, w) methods' advantage readable.
    ``Oracle-ATE`` is the best per-episode constant, the nPEHE = 1 / Shape R² = 0
    anchor on both row-level tables and the unbiased-by-construction row on the
    ATE one.

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
    """Score every method for every available task, into one long frame.

    One row per (method, episode, task) — and only for the tasks whose table can
    rank that method fairly (``Task.admits``), so the frame the CSV carries and
    the rows each table shows are the same thing. Which per-row target a task
    scores is the task's own business (``Task.target_kind``): ``ate`` reads the
    Monte-Carlo CATE when the set has one and the realized ITE otherwise.

    The reference rows are the exception: ``admits`` is a fairness rule for
    *competitors*, and an oracle is never ranked, so all three appear in all
    three tables. That is the informative part — ``Oracle-ITE`` on the CATE
    track prices the noise no tau(x) method should try to explain, and
    ``Oracle-ATE`` puts the nPEHE = 1 anchor on the row-level tables.
    """
    frames = []
    for task in TASKS.values():
        kind = task.target_kind(targets)
        if kind is None:
            continue
        target = targets[kind]
        rows = dict(accumulated)
        if references:
            zeros = np.zeros(len(target), dtype=np.float32)
            rows.update({n: (p, zeros) for n, p in references.items()})
        for name, (preds, times) in rows.items():
            is_reference = bool(references and name in references)
            tag = estimands.get(name, REFERENCE_ESTIMANDS.get(name, "cate(x)"))
            if not is_reference and not task.admits(tag):
                continue
            frames.append(
                score_method(
                    preds,
                    target,
                    times,
                    name,
                    task,
                    estimand=tag,
                    is_reference=is_reference,
                    s2=s2 if kind == "cate" else None,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_diagnostics_yaml(path: Path, diagnostics: dict[str, np.ndarray]) -> None:
    """Diagnostics as YAML, saved next to a method's predictions.

    Diagnostics are read, not scored, so text wins over npz here: a NaN-aware
    mean-per-key ``summary`` up top for a human to glance at, and the full
    per-episode values under ``episodes`` for anyone who needs to correlate a
    diagnostic with a specific episode. ``preds``/``times`` stay npz — they are
    the numbers everything downstream computes on, where size and dtype
    fidelity matter and nobody reads the file directly.
    """
    # Rounded to float32's ~7 significant digits: the values are float32 already
    # (see `runners.run_method`), so this drops noise from the float64 widening
    # YAML's dumper would otherwise print (`0.800000011920929` for `0.8`).
    summary: dict[str, Optional[float]] = {}
    episodes: dict[str, list] = {}
    for key, arr in sorted(diagnostics.items()):
        finite = arr[~np.isnan(arr)]
        summary[key] = round(float(finite.mean()), 6) if finite.size else None
        episodes[key] = [None if np.isnan(v) else round(float(v), 6) for v in arr]
    with open(path, "w") as f:
        yaml.safe_dump({"summary": summary, "episodes": episodes}, f, sort_keys=False)


def load_diagnostics_yaml(path: Path) -> dict[str, np.ndarray]:
    """Inverse of `save_diagnostics_yaml`, for reading a finished run back in.

    Not called by the run loop — diagnostics are written and never re-read
    during a run — but kept as the round-trip partner of the writer, so
    analysis code has one place to parse the format rather than reaching into
    the YAML itself.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {
        key: np.array([np.nan if v is None else v for v in vals], dtype=np.float32)
        for key, vals in data.get("episodes", {}).items()
    }


def print_partial_results(
    accumulated: dict[str, tuple[np.ndarray, np.ndarray]],
    estimands: dict[str, str],
    targets: dict[str, np.ndarray],
    *,
    episode_aggregation: Optional[str] = None,
) -> None:
    if not accumulated:
        return
    # The ITE view contains every per-row predictor, so it is the useful live
    # comparison: its input blocks make the information advantage explicit.
    # Fall back to the ordinary primary task only for unusual target sets that
    # do not carry realized ITEs.
    task = TASKS["ite" if "ite" in targets else primary_task(targets)]
    all_df = score_everything(accumulated, estimands, {task.name: targets[task.name]})
    report = TaskReport.from_results(
        all_df, task, episode_aggregation=episode_aggregation
    )
    if report.empty:
        return
    console.print(
        Padding(
            report.rich_table(title=f"Live {task.name.upper()} Results So Far"), (1, 4)
        ),
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
    if cfg["data"].get("source", "load") == "csv":
        widths = {
            "X": int(data["d_X"][0]),
            "W": int(data["d_W"][0]),
            "Z": int(data["d_Z"][0]),
            "U": int(data["U"].shape[-1]),
        }
        detail = ", ".join(f"{key}={value}" for key, value in widths.items())
        console.print(
            f"    [green]✓[/] Feature columns: {sum(widths.values())} total ({detail})"
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
    eval_cfg = cfg.get("eval", {})
    episode_aggregation = eval_cfg.get("episode_aggregation")
    if episode_aggregation not in (None, "mean"):
        raise ValueError(
            "eval.episode_aggregation must be omitted or 'mean', got "
            f"{episode_aggregation!r}."
        )

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
            # One diagnostics file per output, named after its preds file: they
            # describe the shared fit (like `times`), so a multi-output method's
            # outputs get identical, duplicated content — the point is that a
            # diagnostic is always found next to the predictions it explains.
            diag_files = {n: preds_dir / f"{n}_preds_diag.yaml" for n in output_names}

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
            preds_map, times_map, diagnostics, n_errors = run_method(
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
                if diagnostics:
                    save_diagnostics_yaml(diag_files[out_name], diagnostics)
                console.print(
                    f"  [green]✓[/] {out_name:<28} saved  [dim]({elapsed:.1f}s)[/]"
                )
                accumulated[out_name] = (preds, times)

            print_partial_results(
                accumulated,
                estimands,
                targets,
                episode_aggregation=episode_aggregation,
            )

    # ── Step 3: Final evaluation ─────────────────────────────────────────────
    console.print(Rule("[bold]Step 3  Evaluation[/]"))
    console.print()

    if not accumulated:
        console.print("  [red]No method predictions found. Exiting.[/]")
        return

    references = (
        reference_predictions(targets) if eval_cfg.get("reference_rows", True) else None
    )
    all_results = score_everything(
        accumulated, estimands, targets, s2=cate_s2, references=references
    )
    if data.get("episode_files") is not None:
        episode_files = data["episode_files"]
        all_results.insert(
            all_results.columns.get_loc("episode") + 1,
            "episode_file",
            all_results["episode"].map(lambda episode: episode_files[int(episode)]),
        )

    # One table per task. `cate` is the headline; `ite` is where the methods
    # that read the query rows' proxies are allowed their advantage; `ate` is
    # the level-only view, and the only place a constant-prediction method says
    # anything.
    for task in TASKS.values():
        report = TaskReport.from_results(
            all_results, task, episode_aggregation=episode_aggregation
        )
        if report.empty:
            continue
        console.print(Padding(report.rich_table(), (1, 4)))
        report.summary().to_csv(results_dir / f"summary_{task.name}.csv")

    # summary.csv keeps its name and its meaning "the headline table", so
    # anything downstream that globs for it still finds the primary task.
    TaskReport.from_results(
        all_results,
        TASKS[primary_task(targets)],
        episode_aggregation=episode_aggregation,
    ).summary().to_csv(results_dir / "summary.csv")

    results_csv = results_dir / "results.csv"
    all_results.to_csv(results_csv, index=False)

    console.print(f"  [green]✓[/] Full results → [cyan]{results_csv}[/]")
    console.print(Rule("[bold green]Done[/]"))


if __name__ == "__main__":
    if sys.platform != "win32":
        multiprocessing.set_start_method("spawn", force=True)
    main()
