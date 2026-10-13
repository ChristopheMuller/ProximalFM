"""Posterior calibration of ProximalFM over nested context prefixes.

The command accepts either a sidecar-backed frozen set (prior, realistic, or
literature) or a semi-synthetic cell carrying an analytic ``CATE`` field.  It
runs the model once per episode and context prefix, retains compact row-level
diagnostics, and writes a calibration/contraction figure used by the paper.

Run from the repository root::

    uv run python -m analysis.analysis_eval.posterior_calibration \
      --data-root data/semisynt_eval/semisynt_posterior_calibration_v1/d-5/nonlinear_conf_high_proxy_low \
      --ckpt path/to/checkpoint.ckpt --device mps
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import platform
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prior_eval.build_prior_frozen_set import (
    Y_STD_FLOOR,
    frozen_content_digest,
    load_frozen_meta,
)
from prior_eval.report import load_model
from proximalfm.train._dist_eval import (
    PosteriorEvaluator,
    episode_predictive,
    iter_frozen_episodes,
    require_single_cate_head,
)
from proximalfm.train._dist_metrics import pit, predictive_mean_var
from proximalfm.train._eval import CausalEvaluator

DEFAULT_CONTEXTS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_LEVEL_PERCENTS = (1, *range(5, 100, 5), 99)
PIT_GRID = np.linspace(0.0, 1.0, 201)
OUTPUT_FILES = (
    "points.parquet",
    "metrics.csv",
    "coverage.csv",
    "pit_ecdf.csv",
    "run.json",
    "posterior_calibration.pdf",
    "posterior_calibration.png",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved dataset plus records aligned with ``iter_frozen_episodes``."""

    root: Path
    val_dir: Path
    kind: str
    content_digest: Optional[str]
    n_max: int
    records: list[dict[str, Any]]

    @property
    def n_episodes(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class CalibrationResult:
    points: pd.DataFrame
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    pit_ecdf: pd.DataFrame


def _squeeze_rows(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape (episodes, rows), got {tuple(value.shape)}"
        )
    return value


def _batch_files(val_dir: Path) -> list[Path]:
    files = sorted(Path(val_dir).glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"no batch_*.pt files found in {val_dir}")
    return files


def _disk_layout(
    val_dir: Path, *, expected_n_max: Optional[int] = None
) -> tuple[int, int]:
    """Validate a uniform split and return ``(episodes, n_max)``."""
    total = 0
    observed_n_max: Optional[int] = None
    observed_seq_len: Optional[int] = None
    for path in _batch_files(val_dir):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        train_sizes = payload.get("train_sizes")
        seq_lens = payload.get("seq_lens")
        if train_sizes is None or seq_lens is None:
            raise ValueError(f"{path}: missing train_sizes or seq_lens")
        if train_sizes.numel() != seq_lens.numel():
            raise ValueError(f"{path}: train_sizes and seq_lens have different lengths")
        if not bool((train_sizes == train_sizes[0]).all()):
            raise ValueError(f"{path}: episodes have different context sizes")
        if not bool((seq_lens == seq_lens[0]).all()):
            raise ValueError(f"{path}: episodes have different sequence lengths")
        n_max = int(train_sizes[0])
        if n_max <= 0 or int(seq_lens[0]) <= n_max:
            raise ValueError(
                f"{path}: invalid context/query split {n_max}/{int(seq_lens[0])}"
            )
        if observed_n_max is None:
            observed_n_max = n_max
        elif observed_n_max != n_max:
            raise ValueError(
                f"{val_dir}: batch files disagree on context size "
                f"({observed_n_max} versus {n_max})"
            )
        seq_len = int(seq_lens[0])
        if observed_seq_len is None:
            observed_seq_len = seq_len
        elif observed_seq_len != seq_len:
            raise ValueError(
                f"{val_dir}: batch files disagree on sequence length "
                f"({observed_seq_len} versus {seq_len})"
            )
        total += int(train_sizes.numel())
    assert observed_n_max is not None
    if expected_n_max is not None and observed_n_max != expected_n_max:
        raise ValueError(
            f"dataset sidecar says n_max={expected_n_max}, but batches use "
            f"n_max={observed_n_max}"
        )
    return total, observed_n_max


def _analytic_records(
    root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Construct frozen-space records for a set with an exact analytic CATE."""
    val_dir = root / "val"
    declared = manifest.get("episodes")
    if not isinstance(declared, list):
        raise ValueError(f"{root / 'manifest.json'}: episodes must be a list")
    n_max = manifest.get("n_context")
    if isinstance(n_max, bool) or not isinstance(n_max, int) or n_max <= 0:
        raise ValueError(f"{root / 'manifest.json'}: n_context must be positive")
    n_query = manifest.get("n_query")
    if isinstance(n_query, bool) or not isinstance(n_query, int) or n_query <= 0:
        raise ValueError(f"{root / 'manifest.json'}: n_query must be positive")
    expected_seq_len = n_max + n_query

    records: list[dict[str, Any]] = []
    offset = 0
    for path in _batch_files(val_dir):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "CATE" not in payload:
            raise ValueError(
                f"{path}: no analytic CATE; a dataset without CATE needs frozen_meta.pt"
            )
        y_obs = _squeeze_rows(payload["Y_obs"], "Y_obs").float()
        cate = _squeeze_rows(payload["CATE"], "CATE").float()
        train_sizes = payload["train_sizes"]
        seq_lens = payload["seq_lens"]
        if y_obs.shape != cate.shape or y_obs.shape[0] != train_sizes.numel():
            raise ValueError(f"{path}: CATE/Y_obs/batch dimensions do not agree")

        for row in range(y_obs.shape[0]):
            if int(train_sizes[row]) != n_max:
                raise ValueError(
                    f"{path}: episode {row} uses train_size={int(train_sizes[row])}, "
                    f"expected {n_max}"
                )
            seq_len = int(seq_lens[row])
            if seq_len != y_obs.shape[1] or seq_len != expected_seq_len:
                raise ValueError(
                    f"{path}: episode {row} has seq_len={seq_len}, expected "
                    f"n_context + n_query = {expected_seq_len}"
                )
            if offset >= len(declared):
                raise ValueError("more episodes on disk than in the cell manifest")

            ctx = y_obs[row, :n_max]
            y_mean = ctx.mean()
            y_std = ctx.std(unbiased=False).clamp(min=Y_STD_FLOOR)
            tau = cate[row, n_max:seq_len]
            info = declared[offset]
            if not isinstance(info, dict):
                raise ValueError(f"manifest episode {offset} is not an object")
            records.append(
                {
                    "index": offset,
                    "episode_id": info.get(
                        "episode_id", info.get("pair_id", str(offset))
                    ),
                    "source_file": info.get("source_file"),
                    "replicate": info.get("replicate"),
                    "query_index": torch.arange(n_max, seq_len, dtype=torch.long),
                    "y_context_mean": y_mean,
                    "y_context_std": y_std,
                    "tau_star": tau,
                    "tau_star_std": tau / y_std,
                    "s2": torch.zeros_like(tau),
                    "s2_std": torch.zeros_like(tau),
                }
            )
            offset += 1
    if offset != len(declared):
        raise ValueError(
            f"manifest declares {len(declared)} episodes but batches contain {offset}"
        )
    return records, n_max


def resolve_dataset(data_root: Path) -> DatasetSpec:
    """Resolve a sidecar-backed set or an analytic-CATE semi-synthetic cell."""
    root = Path(data_root).resolve()
    val_dir = root / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"dataset root must contain val/: {root} (pass the cell/set root, not val/)"
        )

    if (root / "frozen_meta.pt").exists():
        meta = load_frozen_meta(root)
        global_meta = meta.get("global", {})
        n_max = int(global_meta["n_max"])
        records = meta.get("episodes")
        if not isinstance(records, list):
            raise ValueError(f"{root / 'frozen_meta.pt'}: episodes must be a list")
        disk_count, _ = _disk_layout(val_dir, expected_n_max=n_max)
        if disk_count != len(records):
            raise ValueError(
                f"sidecar has {len(records)} episodes but val/ contains {disk_count}"
            )
        return DatasetSpec(
            root=root,
            val_dir=val_dir,
            kind=str(global_meta.get("set_kind", "prior")),
            content_digest=frozen_content_digest(meta),
            n_max=n_max,
            records=records,
        )

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{root} has neither frozen_meta.pt nor an analytic-CATE manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_kind") != "exact_cate":
        raise ValueError(
            f"{manifest_path}: target_kind={manifest.get('target_kind')!r}; "
            "posterior scoring requires exact_cate or frozen_meta.pt"
        )
    records, n_max = _analytic_records(root, manifest)
    return DatasetSpec(
        root=root,
        val_dir=val_dir,
        kind="semisynthetic_exact_cate",
        content_digest=manifest.get("content_digest"),
        n_max=n_max,
        records=records,
    )


def default_contexts(n_max: int) -> list[int]:
    if n_max <= 0:
        raise ValueError("n_max must be positive")
    values = [n for n in DEFAULT_CONTEXTS if n <= n_max]
    if not values or values[-1] != n_max:
        values.append(n_max)
    return values


def normalize_contexts(values: Optional[Sequence[int]], n_max: int) -> list[int]:
    if values is None:
        return default_contexts(n_max)
    contexts = sorted(set(int(value) for value in values))
    if not contexts:
        raise ValueError("contexts must not be empty")
    bad = [value for value in contexts if value <= 0 or value > n_max]
    if bad:
        raise ValueError(f"context sizes outside [1, {n_max}]: {bad}")
    if len(contexts) != len(values):
        raise ValueError("context sizes must be unique")
    return contexts


def normalize_levels(values: Optional[Sequence[float]]) -> tuple[float, ...]:
    """Convert CLI percentage levels to probabilities."""
    raw = DEFAULT_LEVEL_PERCENTS if values is None else values
    levels = tuple(sorted(float(value) / 100.0 for value in raw))
    if not levels or any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError(
            "coverage levels must be percentages strictly between 0 and 100"
        )
    if len(set(levels)) != len(levels):
        raise ValueError("coverage levels must be unique")
    tags = [round(value * 100) for value in levels]
    if len(set(tags)) != len(tags):
        raise ValueError(
            "coverage levels must remain distinct after integer-percent labelling"
        )
    return levels


def _record_identity(record: dict[str, Any], index: int) -> tuple[str, Any, Any]:
    episode_id = str(record.get("episode_id", record.get("index", index)))
    return episode_id, record.get("source_file"), record.get("replicate")


def _coverage_frame(metrics: pd.DataFrame, levels: Sequence[float]) -> pd.DataFrame:
    rows = []
    for metric_row in metrics.to_dict("records"):
        for level in levels:
            tag = round(level * 100)
            coverage = float(metric_row[f"cov{tag}_pooled"])
            rows.append(
                {
                    "n_ctx": int(metric_row["n_ctx"]),
                    "nominal": level,
                    "empirical": coverage,
                    "calibration_error": coverage - level,
                    "width": float(metric_row[f"width{tag}"]),
                    "episode_p25": float(metric_row.get(f"cov{tag}/p25", np.nan)),
                    "episode_p50": float(metric_row.get(f"cov{tag}/p50", np.nan)),
                    "episode_p75": float(metric_row.get(f"cov{tag}/p75", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _pit_ecdf_frame(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_ctx, frame in points.groupby("n_ctx", sort=True):
        values = np.sort(frame["pit"].to_numpy(dtype=float))
        ecdf = np.searchsorted(values, PIT_GRID, side="right") / len(values)
        rows.extend(
            {"n_ctx": int(n_ctx), "pit": float(x), "ecdf": float(y)}
            for x, y in zip(PIT_GRID, ecdf)
        )
    return pd.DataFrame(rows)


def run_calibration(
    model: Any,
    dataset: DatasetSpec,
    *,
    contexts: Sequence[int],
    levels: Sequence[float],
    device: str = "cpu",
    max_episodes: Optional[int] = None,
    progress: bool = True,
) -> CalibrationResult:
    """Run one checkpoint over every episode and requested nested prefix."""
    require_single_cate_head(model)
    contexts = normalize_contexts(contexts, dataset.n_max)
    levels = tuple(levels)
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(
            "internal coverage levels must lie strictly between zero and one"
        )
    total = dataset.n_episodes
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        total = min(total, max_episodes)

    dist = {n: PosteriorEvaluator(levels=levels) for n in contexts}
    point = {n: CausalEvaluator() for n in contexts}
    mse_sums = {n: 0.0 for n in contexts}
    scale_sums = {n: 0.0 for n in contexts}
    row_counts = {n: 0 for n in contexts}
    chunks: list[pd.DataFrame] = []

    iterator: Iterable[tuple[int, Sequence[torch.Tensor]]] = enumerate(
        iter_frozen_episodes(dataset.val_dir, max_episodes)
    )
    iterator = tqdm(
        iterator,
        total=total,
        desc="episodes",
        unit="ep",
        disable=not progress,
    )
    head_edges = model.cate_bin_edges.detach().float()
    head_span = float(head_edges[-1] - head_edges[0])

    with torch.inference_mode():
        seen = 0
        for index, batch in iterator:
            if index >= total:
                break
            record = dataset.records[index]
            target = record["tau_star_std"].float().reshape(-1)
            target_s2 = record["s2_std"].float().reshape(-1)
            if target.shape != target_s2.shape:
                raise ValueError(
                    f"episode {index}: tau_star_std and s2_std differ in shape"
                )
            query_index = torch.as_tensor(
                record.get("query_index", torch.arange(target.numel())),
                dtype=torch.long,
            ).reshape(-1)
            if query_index.numel() != target.numel():
                raise ValueError(
                    f"episode {index}: query_index and target lengths differ"
                )
            episode_id, source_file, replicate = _record_identity(record, index)

            for n_ctx in contexts:
                probs, edges = episode_predictive(
                    model,
                    batch,
                    record,
                    n_ctx=n_ctx,
                    n_max=dataset.n_max,
                    device=device,
                )
                target_device = target.to(probs.device)
                s2_device = target_s2.to(probs.device)
                dist[n_ctx].update(
                    probs=probs,
                    bin_edges=edges,
                    y=target_device,
                    s2=s2_device,
                )
                mean, variance = predictive_mean_var(probs, edges)
                point[n_ctx].update(
                    pred_cate=mean.reshape(1, -1),
                    true_cate=target_device.reshape(1, -1),
                )
                pit_values = pit(probs, edges, target_device)
                squared_error = (mean - target_device) ** 2
                n_rows = int(target.numel())
                grid_scale = float((edges[-1] - edges[0]) / max(head_span, 1e-12))
                mse_sums[n_ctx] += float(squared_error.detach().cpu().double().sum())
                scale_sums[n_ctx] += grid_scale * n_rows
                row_counts[n_ctx] += n_rows

                edge_lo, edge_hi = float(edges[0]), float(edges[-1])
                target_np = target.numpy()
                chunks.append(
                    pd.DataFrame(
                        {
                            "episode": index,
                            "episode_id": episode_id,
                            "source_file": source_file,
                            "replicate": replicate,
                            "query_index": query_index.numpy(),
                            "n_ctx": n_ctx,
                            "target": target_np,
                            "target_s2": target_s2.numpy(),
                            "posterior_mean": mean.detach().cpu().numpy(),
                            "posterior_sd": variance.detach().cpu().sqrt().numpy(),
                            "pit": pit_values.detach().cpu().numpy(),
                            "out_of_grid": (target_np < edge_lo)
                            | (target_np > edge_hi),
                        }
                    )
                )
            seen += 1
    if seen != total:
        raise ValueError(f"expected {total} episodes but loader yielded {seen}")

    points = pd.concat(chunks, ignore_index=True)
    metric_rows = []
    for n_ctx in contexts:
        metrics = dist[n_ctx].compute(prefix="", include_count=True)
        metrics.update(point[n_ctx].compute(prefix="point/", include_count=True))
        rows = row_counts[n_ctx]
        metrics["point/cate/rmse_pooled"] = math.sqrt(mse_sums[n_ctx] / rows)
        metrics["grid_scale"] = scale_sums[n_ctx] / rows
        metrics["n_ctx"] = n_ctx
        metric_rows.append(metrics)
    metrics = pd.DataFrame(metric_rows).sort_values("n_ctx").reset_index(drop=True)
    coverage = _coverage_frame(metrics, levels)
    pit_ecdf = _pit_ecdf_frame(points)
    return CalibrationResult(points, metrics, coverage, pit_ecdf)


def _plot_calibration_panels(
    coverage: pd.DataFrame,
    points: pd.DataFrame,
    axes: Sequence[Any],
    *,
    titles: Sequence[str],
    row_label: Optional[str] = None,
) -> None:
    """Render one three-panel calibration row onto supplied axes.

    Coverage is pooled over all query rows. Posterior width is its row-level
    arithmetic mean in frozen target units; its IQR is taken across
    episode-level mean widths, making the band a description of heterogeneity
    rather than a standard error for the pooled mean.
    """
    from analysis.analysis_eval.plotting import plt

    if len(axes) != 3 or len(titles) != 3:
        raise ValueError("calibration rows need exactly three axes and titles")
    ax_cov, ax_reliability, ax_width = axes
    contexts = sorted(int(value) for value in coverage["n_ctx"].unique())
    if contexts != sorted(int(value) for value in points["n_ctx"].unique()):
        raise ValueError("coverage and point tables contain different context sizes")
    if "posterior_sd" not in points:
        raise ValueError("point table must contain row-level posterior_sd")
    if "episode" not in points:
        raise ValueError("point table must identify each episode")
    if (
        not np.isfinite(points["posterior_sd"]).all()
        or (points["posterior_sd"] <= 0).any()
    ):
        raise ValueError(
            "posterior_sd must be strictly positive for log-scale plotting"
        )

    levels = sorted(float(value) for value in coverage["nominal"].unique())
    level_colours = plt.cm.viridis(np.linspace(0.06, 0.94, len(levels)))
    context_colours = plt.cm.plasma(np.linspace(0.1, 0.9, len(contexts)))
    for colour, level in zip(level_colours, levels):
        cov = coverage[coverage["nominal"].eq(level)].sort_values("n_ctx")
        ax_cov.plot(
            cov["n_ctx"],
            cov["empirical"],
            color=colour,
            marker="o",
            markersize=2.4,
            linewidth=1.35,
            label=f"{level:.0%}",
        )
        ax_cov.axhline(level, color=colour, linestyle=":", linewidth=0.8, alpha=0.45)

    ax_reliability.plot([0, 1], [0, 1], color="0.45", linestyle=":", linewidth=1)
    for colour, n_ctx in zip(context_colours, contexts):
        cov = coverage[coverage["n_ctx"].eq(n_ctx)].sort_values("nominal")
        ax_reliability.plot(
            cov["nominal"],
            cov["empirical"],
            color=colour,
            marker="o",
            markersize=2.4,
            linewidth=1.35,
            label=f"{n_ctx:,}",
        )

    episode_widths = points.groupby(["n_ctx", "episode"], sort=True)[
        "posterior_sd"
    ].mean()
    width_summary = episode_widths.groupby("n_ctx").agg(
        mean="mean",
        p25=lambda value: value.quantile(0.25),
        p75=lambda value: value.quantile(0.75),
    )
    width_summary = width_summary.reindex(contexts)
    ax_width.fill_between(
        width_summary.index,
        width_summary["p25"],
        width_summary["p75"],
        color="0.55",
        alpha=0.25,
        linewidth=0,
        label="Episode IQR",
    )
    ax_width.plot(
        width_summary.index,
        width_summary["mean"],
        color="black",
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="Row mean",
    )

    for ax in (ax_cov, ax_width):
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts)
        ax.set_xticklabels([f"{n:,}" for n in contexts])
    ax_cov.set_ylim(0, 1)
    ax_cov.set_xlabel("Context size")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_title(titles[0], loc="left")
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.set_aspect("equal", adjustable="box")
    ax_reliability.set_xlabel("Nominal central credible level")
    ax_reliability.set_ylabel("Empirical coverage")
    ax_reliability.set_title(titles[1], loc="left")
    ax_width.set_yscale("log")
    ax_width.set_xlabel("Context size")
    ax_width.set_ylabel("Mean posterior SD")
    ax_width.set_title(titles[2], loc="left")
    ax_reliability.legend(title="Context size", frameon=False, fontsize="small")
    ax_width.legend(frameon=False, fontsize="small")
    # Give every panel the same square box. The calibration panel therefore
    # keeps a geometrically meaningful diagonal, while the x-axes and titles
    # line up across a row instead of its square axis floating vertically.
    for ax in (ax_cov, ax_reliability, ax_width):
        ax.set_box_aspect(1)
        ax.set_anchor("S")
    if row_label is not None:
        ax_cov.annotate(
            row_label,
            xy=(-0.34, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize="medium",
            fontweight="bold",
        )


def make_calibration_figure(coverage: pd.DataFrame, points: pd.DataFrame) -> Any:
    """Plot coverage calibration and posterior-width contraction by context."""
    # Keep plotting imports lazy: argument parsing, dataset validation, and
    # ``--help`` must not initialize Matplotlib or scan the system font cache.
    from analysis.analysis_eval.design import FULL_PAGE_WIDTH_IN, apply_style
    from analysis.analysis_eval.plotting import plt

    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_PAGE_WIDTH_IN, 3.35), constrained_layout=True
    )
    _plot_calibration_panels(
        coverage,
        points,
        axes,
        titles=(
            "(a) Coverage by context",
            "(b) Calibration",
            "(c) Posterior contraction",
        ),
    )
    return fig


def make_comparison_figure(
    prior_coverage: pd.DataFrame,
    prior_points: pd.DataFrame,
    semisynt_coverage: pd.DataFrame,
    semisynt_points: pd.DataFrame,
) -> Any:
    """Compare cached prior and semi-synthetic calibration diagnostics."""
    from analysis.analysis_eval.design import FULL_PAGE_WIDTH_IN, apply_style
    from analysis.analysis_eval.plotting import plt

    apply_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_PAGE_WIDTH_IN, 6.4),
        constrained_layout=True,
    )
    _plot_calibration_panels(
        prior_coverage,
        prior_points,
        axes[0],
        titles=(
            "(a) Coverage by context",
            "(b) Calibration",
            "(c) Posterior contraction",
        ),
        row_label="Prior",
    )
    _plot_calibration_panels(
        semisynt_coverage,
        semisynt_points,
        axes[1],
        titles=(
            "(d) Coverage by context",
            "(e) Calibration",
            "(f) Posterior contraction",
        ),
        row_label="Semi-synthetic",
    )
    return fig


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def _git_revision() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_checkpoint(path: Path) -> Path:
    """Resolve file/directory exactly as the benchmark ProximalFM method does."""
    from benchmarks.methods.proximalfm import resolve_checkpoints

    resolved = resolve_checkpoints({"checkpoint": str(Path(path)), "all": False})
    return resolved[0][1].resolve()


def default_output_dir(dataset: DatasetSpec, checkpoint: Path) -> Path:
    dataset_name = _slug(dataset.root.name)
    checkpoint_name = _slug(f"{checkpoint.parent.name}-{checkpoint.stem}")
    return (
        Path(__file__).with_name("results")
        / "posterior_calibration"
        / (f"{dataset_name}__{checkpoint_name}")
    )


def _check_outputs(out_dir: Path, force: bool) -> None:
    existing = [out_dir / name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing calibration outputs in {out_dir}: {names}; "
            "pass --force to replace them"
        )


def write_outputs(
    result: CalibrationResult,
    out_dir: Path,
    *,
    provenance: dict[str, Any],
    force: bool = False,
) -> list[Path]:
    _check_outputs(out_dir, force)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.points.to_parquet(out_dir / "points.parquet", index=False)
    result.metrics.to_csv(out_dir / "metrics.csv", index=False)
    result.coverage.to_csv(out_dir / "coverage.csv", index=False)
    result.pit_ecdf.to_csv(out_dir / "pit_ecdf.csv", index=False)
    (out_dir / "run.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure = make_calibration_figure(result.coverage, result.points)
    from analysis.analysis_eval.plotting import save_figure

    save_figure(figure, out_dir, "posterior_calibration")
    return [out_dir / name for name in OUTPUT_FILES]


def rerender_cached_figure(out_dir: Path) -> list[Path]:
    """Regenerate the figure from cached row diagnostics without inference."""
    cache_paths = {
        "points": out_dir / "points.parquet",
        "coverage": out_dir / "coverage.csv",
    }
    missing = [name for name, path in cache_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot render cached figure in {out_dir}: missing "
            f"{', '.join(cache_paths[name].name for name in missing)}"
        )
    figure = make_calibration_figure(
        pd.read_csv(cache_paths["coverage"]), pd.read_parquet(cache_paths["points"])
    )
    from analysis.analysis_eval.plotting import save_figure

    return save_figure(figure, out_dir, "posterior_calibration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.analysis_eval.posterior_calibration",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contexts", type=int, nargs="+", default=None)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=None,
        help="central credible levels as percentages",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the PDF/PNG from cached points.parquet and coverage.csv",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    dataset = resolve_dataset(args.data_root)
    contexts = normalize_contexts(args.contexts, dataset.n_max)
    levels = normalize_levels(args.levels)
    checkpoint = resolve_checkpoint(args.ckpt)
    out_dir = (
        default_output_dir(dataset, checkpoint)
        if args.out_dir is None
        else args.out_dir.resolve()
    )
    if args.plot_only:
        paths = rerender_cached_figure(out_dir)
        print(f"Regenerated cached figure in: {out_dir}")
        for path in paths:
            print(f"  {path.name}")
        return 0
    _check_outputs(out_dir, args.force)

    model, step = load_model(checkpoint, device=args.device)
    result = run_calibration(
        model,
        dataset,
        contexts=contexts,
        levels=levels,
        device=args.device,
        max_episodes=args.max_episodes,
        progress=not args.no_progress,
    )
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "checkpoint_loss_type": getattr(model, "loss_type", None),
        "dataset_root": str(dataset.root),
        "dataset_kind": dataset.kind,
        "dataset_content_digest": dataset.content_digest,
        "n_max": dataset.n_max,
        "n_episodes": int(result.points["episode"].nunique()),
        "n_query": int(
            result.points[result.points["n_ctx"].eq(contexts[0])].shape[0]
            / result.points["episode"].nunique()
        ),
        "contexts": contexts,
        "coverage_levels": list(levels),
        "inference": "plain_single_forward",
        "device": args.device,
        "python": platform.python_version(),
        "git_revision": _git_revision(),
        "elapsed_seconds": time.time() - started,
    }
    paths = write_outputs(result, out_dir, provenance=provenance, force=args.force)

    print(f"Dataset: {dataset.root} [{dataset.kind}, digest={dataset.content_digest}]")
    print(f"Checkpoint: {checkpoint}" + (f" (step {step})" if step is not None else ""))
    print(f"Episodes: {provenance['n_episodes']}; contexts: {contexts}")
    print(f"Outputs: {out_dir}")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
