"""Plot one fixed query's posterior evolution across training and context size.

Example
-------
    uv run python -m analysis.analysis_training.posterior_evolution \
        --data-dir data/synt_eval/synt_val_v1 \
        --dataset-index 0 --query-index 3

The two panels hold the query row fixed and vary only the context prefix:
``n_ctx=16`` on the left and ``n_ctx=2048`` on the right.  Each curve is the
PDF obtained by differentiating the lightly smoothed posterior CDF, using the
same convention as :mod:`analysis.analysis_training.posterior_snapshot`.  Both panels
have identical axis limits so changes in concentration are directly visible.
Smoothing is for visualization only; it is not used by any reported metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from analysis.analysis_training.design import (
    ROLES,
    apply_style,
    figsize_grid,
    panel_tag,
    plt,
    save_figure,
)
from analysis.analysis_training.posterior_contraction import (
    DEFAULT_CKPT_DIR,
    parse_int_list,
    resolve_checkpoints,
)
from analysis.analysis_training.posterior_snapshot import (
    _load_episode,
    posterior_view_limits,
    smooth_cdf,
)
from eval_datasets.prior_eval.report import load_model
from proximalfm.train._dist_eval import episode_predictive

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "analysis" / "analysis_training" / "results"
NAME = "posterior_evolution"
DEFAULT_CONTEXTS = (64, 1024)
DEFAULT_STEPS = (1, 500, 5000, 75000)


def _validate_posteriors(
    posteriors: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    n_contexts: int,
    n_steps: int,
) -> None:
    if len(posteriors) != n_contexts:
        raise ValueError(
            f"posteriors has {len(posteriors)} context rows, expected {n_contexts}"
        )
    for row in posteriors:
        if len(row) != n_steps:
            raise ValueError(
                f"posterior row has {len(row)} checkpoints, expected {n_steps}"
            )
        for edges, probs in row:
            edges = np.asarray(edges, dtype=float)
            probs = np.asarray(probs, dtype=float)
            if edges.ndim != 1 or probs.ndim != 1 or len(edges) != len(probs) + 1:
                raise ValueError("each edge array must have one more value than probs")
            if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
                raise ValueError(
                    "posterior edges must be finite and strictly increasing"
                )
            if (
                not np.all(np.isfinite(probs))
                or np.any(probs < 0.0)
                or probs.sum() <= 0.0
            ):
                raise ValueError(
                    "posterior probabilities must be finite and non-negative"
                )


def build_figure(
    posteriors: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    n_ctx_list: Sequence[int],
    ckpt_steps: Sequence[int],
    true_value: float,
    *,
    cdf_window: int = 101,
    cdf_polyorder: int = 3,
    zoom_sd: float = 3.0,
) -> "plt.Figure":
    """Build the shared-axis two-panel posterior-evolution figure."""
    n_ctx_list = list(n_ctx_list)
    ckpt_steps = list(ckpt_steps)
    if len(n_ctx_list) != 2:
        raise ValueError("the posterior-evolution figure requires exactly two contexts")
    if not ckpt_steps:
        raise ValueError("at least one checkpoint is required")
    if not np.isfinite(true_value):
        raise ValueError("true_value must be finite")
    _validate_posteriors(posteriors, len(n_ctx_list), len(ckpt_steps))

    smoothed: list[list[tuple[np.ndarray, np.ndarray]]] = []
    limits: list[tuple[float, float]] = []
    max_density = 0.0
    for row in posteriors:
        smooth_row = []
        for edges, probs in row:
            edges = np.asarray(edges, dtype=float)
            probs = np.asarray(probs, dtype=float)
            _, _, density = smooth_cdf(
                edges,
                probs,
                window=cdf_window,
                polyorder=cdf_polyorder,
            )
            smooth_row.append((edges, density))
            limits.append(posterior_view_limits(edges, probs, true_value, n_sd=zoom_sd))
            max_density = max(max_density, float(density.max()))
        smoothed.append(smooth_row)

    x_limits = (min(pair[0] for pair in limits), max(pair[1] for pair in limits))
    y_limits = (0.0, max_density * 1.06)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ckpt_steps)))

    apply_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize_grid(ncols=2, nrows=1, panel_aspect=0.78),
        sharex=True,
        sharey=True,
    )
    for ax, n_ctx, row in zip(axes, n_ctx_list, smoothed):
        for step, color, (edges, density) in zip(ckpt_steps, colors, row):
            ax.plot(edges, density, color=color, lw=1.35, label=f"{step:,}")
        ax.axvline(
            true_value,
            **{**ROLES["reference"], "label": "true CATE", "zorder": 1},
        )
        ax.set_title(rf"$n_{{\mathrm{{ctx}}}}={n_ctx:,}$")
        ax.set_xlabel(r"CATE $Q(x^\star)$")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    axes[0].set_ylabel("density")
    for tag, ax in zip("ab", axes):
        panel_tag(ax, tag)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="training step",
        loc="outside upper center",
        ncols=len(labels),
        frameon=False,
    )
    return fig


def posterior_grid(
    ckpt_paths: Sequence[Path],
    batch,
    record: dict,
    *,
    n_ctx_list: Sequence[int],
    n_max: int,
    query_index: int,
    device: str = "cpu",
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], list[int | None]]:
    """Evaluate one query for every checkpoint and context prefix."""
    grid: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in n_ctx_list]
    loaded_steps: list[int | None] = []
    for ckpt_path in ckpt_paths:
        model, loaded_step = load_model(ckpt_path, device=device)
        loaded_steps.append(None if loaded_step is None else int(loaded_step))
        with torch.inference_mode():
            for row, n_ctx in zip(grid, n_ctx_list):
                probs, edges = episode_predictive(
                    model,
                    batch,
                    record,
                    n_ctx=n_ctx,
                    n_max=n_max,
                    device=device,
                )
                row.append(
                    (
                        edges.float().cpu().numpy(),
                        probs[query_index].float().cpu().numpy(),
                    )
                )
        del model
    return grid, loaded_steps


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--ckpt-steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
        help="comma-separated training steps (default: 1,500,1000,5000,10000,75000)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="frozen validation set root containing val/ and frozen_meta.pt",
    )
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=3)
    parser.add_argument(
        "--n-ctx",
        default=",".join(str(n) for n in DEFAULT_CONTEXTS),
        help="exactly two comma-separated context sizes (default: 16,2048)",
    )
    parser.add_argument("--cdf-window", type=int, default=101)
    parser.add_argument("--cdf-polyorder", type=int, default=3)
    parser.add_argument("--zoom-sd", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args(argv)

    n_ctx_list = parse_int_list(args.n_ctx, name="n-ctx")
    if len(n_ctx_list) != 2:
        raise ValueError("n-ctx must contain exactly two context sizes")
    ckpt_steps = parse_int_list(args.ckpt_steps, name="ckpt-steps")
    ckpt_paths = resolve_checkpoints(args.ckpt_dir, ckpt_steps)

    batch, record, n_max = _load_episode(args.data_dir, args.dataset_index)
    if any(n_ctx > n_max for n_ctx in n_ctx_list):
        raise ValueError(f"n-ctx values must be no larger than n-max={n_max}")
    n_query = int(record["tau_star_std"].reshape(-1).numel())
    if not 0 <= args.query_index < n_query:
        raise IndexError(f"query_index={args.query_index} is outside [0, {n_query})")

    posteriors, loaded_steps = posterior_grid(
        ckpt_paths,
        batch,
        record,
        n_ctx_list=n_ctx_list,
        n_max=n_max,
        query_index=args.query_index,
        device=args.device,
    )
    true_value = float(record["tau_star_std"].reshape(-1)[args.query_index])
    fig = build_figure(
        posteriors,
        n_ctx_list,
        ckpt_steps,
        true_value,
        cdf_window=args.cdf_window,
        cdf_polyorder=args.cdf_polyorder,
        zoom_sd=args.zoom_sd,
    )
    paths = save_figure(fig, args.outdir, args.name)
    loaded_text = ", ".join(
        "unknown" if step is None else f"{step:,}" for step in loaded_steps
    )
    print(
        f"loaded steps={loaded_text}; dataset={args.dataset_index}; "
        f"query={args.query_index}; true CATE={true_value:+.6f}"
    )
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
