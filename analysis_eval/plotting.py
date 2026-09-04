"""Headless plotting and output helpers for evaluation-section artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def place_point_labels(fig: "plt.Figure", ax: "plt.Axes", labels: list) -> None:
    """Spread point annotations using display-space collision checks.

    Keeping this in the plotting utility layer makes direct-labelled figures
    deterministic without embedding data-specific offsets in each renderer.
    Labels must use ``textcoords='offset points'``.
    """
    if not labels:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    occupied = []
    candidates = (
        (5, 0),
        (5, 9),
        (5, -9),
        (5, 18),
        (5, -18),
        (-5, 9),
        (-5, -9),
        (5, 27),
        (5, -27),
    )
    for label in labels:
        chosen = None
        for offset in candidates:
            label.set_position(offset)
            fig.canvas.draw()
            box = label.get_window_extent(renderer).expanded(1.02, 1.06)
            if box.y1 > axes_box.y1 + 2 or box.y0 < axes_box.y0 - 2:
                continue
            if not any(box.overlaps(previous) for previous in occupied):
                chosen = box
                break
        occupied.append(chosen or label.get_window_extent(renderer))


def save_figure(fig: "plt.Figure", out_dir: Path, name: str) -> list[Path]:
    """Save a figure as a vector PDF and a high-resolution PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths
