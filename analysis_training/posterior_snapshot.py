"""Inspect one CATE predictive distribution from a frozen validation episode.

Example
-------
    uv run python -m analysis_training.posterior_snapshot \
        --ckpt path/to/step-075000.ckpt \
        --data-dir data/synt_eval/realistic_v1 \
        --dataset-index 0 --n-ctx 512 --query-index 3

The four panels show the raw posterior density, its CDF, a lightly smoothed CDF,
and the density obtained by differentiating that smoothed CDF. The red vertical
line marks the frozen Monte Carlo CATE target for the selected query row.
Smoothing is intended for visual inspection only; no validation metric is
recomputed from the smoothed curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import savgol_filter

from analysis_training.design import (
    OKABE_ITO,
    ROLES,
    apply_style,
    figsize_grid,
    panel_tag,
    plt,
    save_figure,
)
from prior_eval.build_prior_frozen_set import load_frozen_meta
from prior_eval.report import load_model
from proximalfm.train._dist_eval import episode_predictive, iter_frozen_episodes
from proximalfm.train._dist_metrics import predictive_mean_var

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "analysis_training" / "results"
NAME = "posterior_snapshot"


def smooth_cdf(
    edges: np.ndarray,
    probs: np.ndarray,
    *,
    window: int = 101,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the raw CDF, smoothed CDF, and its non-negative derivative."""
    edges = np.asarray(edges, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if edges.ndim != 1 or probs.ndim != 1 or len(edges) != len(probs) + 1:
        raise ValueError("edges must have one more element than probs")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")

    probs = probs / probs.sum()
    cdf = np.concatenate(([0.0], np.cumsum(probs)))
    n = len(cdf)
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    window = min(window, n if n % 2 else n - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        raise ValueError(f"window={window} must be larger than polyorder={polyorder}")

    cdf_smooth = savgol_filter(cdf, window, polyorder, mode="interp")
    cdf_smooth = np.maximum.accumulate(np.clip(cdf_smooth, 0.0, 1.0))
    cdf_smooth[0] = 0.0
    cdf_smooth[-1] = 1.0
    pdf_smooth = np.clip(np.gradient(cdf_smooth, edges), 0.0, None)
    return cdf, cdf_smooth, pdf_smooth


def posterior_view_limits(
    edges: np.ndarray,
    probs: np.ndarray,
    true_value: float,
    *,
    n_sd: float = 2.0,
) -> tuple[float, float]:
    """Return a compact x-range centred on the posterior mean.

    The range contains the posterior mean +/- ``n_sd`` posterior standard
    deviations, clipped to the histogram support.  It is expanded when the
    frozen target lies outside that window so that the diagnostic target line
    remains visible.
    """
    edges = np.asarray(edges, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if edges.ndim != 1 or probs.ndim != 1 or len(edges) != len(probs) + 1:
        raise ValueError("edges must have one more element than probs")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probs must be finite and non-negative")
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probs must have positive total mass")
    if not np.isfinite(true_value):
        raise ValueError("true_value must be finite")
    if n_sd <= 0.0:
        raise ValueError("n_sd must be positive")

    probs = probs / total
    mean, variance = predictive_mean_var(
        torch.as_tensor(probs, dtype=torch.float32),
        torch.as_tensor(edges, dtype=torch.float32),
    )
    mean = float(mean)
    sd = float(torch.sqrt(variance))
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("posterior must have a finite, positive standard deviation")

    lo = max(edges[0], mean - n_sd * sd)
    hi = min(edges[-1], mean + n_sd * sd)
    lo = min(lo, true_value)
    hi = max(hi, true_value)
    width = hi - lo
    if width <= 0.0:
        width = max(np.diff(edges).min(), np.finfo(float).eps)
        lo -= 0.5 * width
        hi += 0.5 * width
    padding = 0.05 * width
    return float(lo - padding), float(hi + padding)


def build_figure(
    edges: np.ndarray,
    probs: np.ndarray,
    true_value: float,
    *,
    cdf_window: int = 101,
    cdf_polyorder: int = 3,
    zoom_sd: float = 2.0,
) -> "plt.Figure":
    """Build the four-panel posterior representation diagnostic.

    All panels share an x-range centred on the posterior mean and spanning
    ``zoom_sd`` posterior standard deviations on either side.
    """
    edges = np.asarray(edges, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(edges) != len(probs) + 1:
        raise ValueError("edges must have one more element than probs")
    if not np.isfinite(true_value):
        raise ValueError("true_value must be finite")
    x_min, x_max = posterior_view_limits(edges, probs, true_value, n_sd=zoom_sd)
    probs = probs / probs.sum()

    cdf, cdf_smooth, pdf_smooth = smooth_cdf(
        edges,
        probs,
        window=cdf_window,
        polyorder=cdf_polyorder,
    )
    pdf = probs / np.diff(edges)

    apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize_grid(ncols=2, nrows=2, panel_aspect=0.78),
    )
    ax_pdf, ax_cdf, ax_cdf_smooth, ax_pdf_smooth = axes.flat
    x_label = r"CATE $Q(x^\star)$"
    true_style = {**ROLES["reference"], "label": "true CATE"}

    ax_pdf.stairs(
        pdf,
        edges,
        color=OKABE_ITO["blue"],
        lw=1.2,
    )
    ax_cdf.plot(edges, cdf, color=OKABE_ITO["blue"], lw=1.2)
    ax_cdf_smooth.plot(edges, cdf_smooth, color=OKABE_ITO["blue"], lw=1.2)
    ax_pdf_smooth.plot(edges, pdf_smooth, color=OKABE_ITO["purple"], lw=1.2)

    for ax in axes.flat:
        ax.axvline(true_value, **true_style)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(x_label)

    ax_pdf.set_ylabel("density")
    ax_pdf.set_title("raw posterior PDF")
    ax_pdf.set_ylim(bottom=0.0)
    ax_pdf.legend(loc="upper left", frameon=False)

    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("raw posterior CDF")
    ax_cdf.set_ylim(0.0, 1.0)

    ax_cdf_smooth.set_ylabel("CDF")
    ax_cdf_smooth.set_title(f"smoothed CDF (window {cdf_window} bins)")
    ax_cdf_smooth.set_ylim(0.0, 1.0)

    ax_pdf_smooth.set_ylabel("density")
    ax_pdf_smooth.set_title("PDF implied by smoothed CDF")
    ax_pdf_smooth.set_ylim(bottom=0.0)
    max_pdf = max(float(pdf.max()), float(pdf_smooth.max()))
    ax_pdf.set_ylim(0.0, max_pdf * 1.08)
    ax_pdf_smooth.set_ylim(0.0, max_pdf * 1.08)

    for tag, ax in zip("abcd", axes.flat):
        panel_tag(ax, tag)
    return fig


def _load_episode(data_dir: Path, dataset_index: int):
    """Load one episode and its frozen sidecar record by validation order."""
    data_dir = Path(data_dir)
    val_dir = data_dir / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected validation shards under {val_dir}; pass the set root "
            "containing val/ and frozen_meta.pt."
        )
    meta = load_frozen_meta(data_dir)
    records = meta["episodes"]
    if not 0 <= dataset_index < len(records):
        raise IndexError(
            f"dataset_index={dataset_index} is outside [0, {len(records)})"
        )
    episodes = iter_frozen_episodes(val_dir, max_episodes=dataset_index + 1)
    for i, batch in enumerate(episodes):
        if i == dataset_index:
            return batch, records[dataset_index], int(meta["global"]["n_max"])
    raise RuntimeError(f"Could not load validation episode {dataset_index}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="frozen validation set root containing val/ and frozen_meta.pt",
    )
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--n-ctx", type=int, required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--cdf-window", type=int, default=101)
    parser.add_argument("--cdf-polyorder", type=int, default=3)
    parser.add_argument(
        "--zoom-sd",
        type=float,
        default=2.0,
        help="Number of posterior SDs to show around the mean (default: 2).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args(argv)

    batch, record, n_max = _load_episode(args.data_dir, args.dataset_index)
    n_query = int(record["tau_star_std"].reshape(-1).numel())
    if not 1 <= args.n_ctx <= n_max:
        raise ValueError(f"n_ctx must be in [1, {n_max}], got {args.n_ctx}")
    if not 0 <= args.query_index < n_query:
        raise IndexError(f"query_index={args.query_index} is outside [0, {n_query})")

    model, step = load_model(args.ckpt, device=args.device)
    with torch.inference_mode():
        probs, edges = episode_predictive(
            model,
            batch,
            record,
            n_ctx=args.n_ctx,
            n_max=n_max,
            device=args.device,
        )
    probs_np = probs[args.query_index].float().cpu().numpy()
    edges_np = edges.float().cpu().numpy()
    true_value = float(record["tau_star_std"].reshape(-1)[args.query_index])

    fig = build_figure(
        edges_np,
        probs_np,
        true_value,
        cdf_window=args.cdf_window,
        cdf_polyorder=args.cdf_polyorder,
        zoom_sd=args.zoom_sd,
    )
    paths = save_figure(fig, args.outdir, args.name)
    step_text = "unknown" if step is None else str(int(step))
    print(
        f"checkpoint step={step_text}; dataset={args.dataset_index}; "
        f"n_ctx={args.n_ctx}; query={args.query_index}; "
        f"true CATE={true_value:+.6f}"
    )
    print("wrote " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
