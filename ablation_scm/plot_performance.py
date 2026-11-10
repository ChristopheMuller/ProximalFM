"""Render the ablation scorecard from the cached evaluation CSV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from analysis.analysis_eval.plotting import plt, save_figure
from matplotlib.patches import Rectangle

COLUMNS = (
    ("npehe", "nPEHE", "Lower is better", ".3f"),
    ("r2_shape", "Shape R²", "Higher is better", ".3f"),
    ("crps", "CRPS", "Lower is better", ".3f"),
)
INK = "#182532"
MUTED = "#536273"
BETTER = ("#E3F3E8", "#17683A")
WORSE = ("#FCE8E6", "#A02E29")
NEUTRAL = ("#EAEFF4", MUTED)


def cell_colours(metric: str, score: float, baseline: float) -> tuple[str, str]:
    if metric == "r2_shape":
        improvement = score - baseline
    else:
        improvement = baseline - score

    if math.isclose(improvement, 0.0, abs_tol=1e-12):
        return NEUTRAL
    return BETTER if improvement > 0 else WORSE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="evaluation output directory")
    root = parser.parse_args().results_dir
    table = pd.read_csv(root / "performance_table.csv")
    baseline = table.iloc[0]
    if not baseline["variant"].startswith("Default"):
        raise ValueError("performance_table.csv must start with the default variant")
    metrics = pd.read_csv(root / "metrics_by_context.csv", usecols=["variant", "group"])
    architecture = metrics.loc[metrics["group"].eq("architecture"), "variant"]
    table = table.loc[(table.index == 0) | table["variant"].isin(architecture)]

    height = 3.8 + 0.55 * max(0, len(table) - 2)
    fig = plt.figure(figsize=(10, height), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    def box(left, bottom, width, height, fill, edge="none"):
        ax.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
            )
        )

    start = 0.315
    cell_width = (0.965 - start) / len(COLUMNS)
    centers = [start + cell_width * (index + 0.5) for index in range(len(COLUMNS))]
    ax.text(0.035, 0.925, "Variant", fontsize=12, weight="bold", color=INK, va="center")
    for (_, title, note, _), x in zip(COLUMNS, centers):
        ax.text(
            x,
            0.925,
            title,
            fontsize=12,
            weight="bold",
            color=INK,
            ha="center",
            va="center",
        )
        ax.text(x, 0.89, note, fontsize=8.5, color=MUTED, ha="center", va="center")

    row_pitch = 0.78 / len(table)
    row_height = row_pitch * 0.91
    for index, row in enumerate(table.to_dict("records")):
        y = 0.855 - row_pitch * (index + 0.5)
        if index == 0:
            box(0.024, y - row_height / 2, 0.941, row_height, "white", "#D9E1E8")
        ax.text(
            0.035,
            y,
            "Default" if index == 0 else row["variant"],
            va="center",
            fontsize=11.5,
            color=INK,
            weight="bold" if index == 0 else "medium",
        )
        for (metric, _, _, spec), x in zip(COLUMNS, centers):
            score, reference = row[metric], baseline[metric]
            if index == 0:
                ax.text(
                    x,
                    y,
                    format(score, spec),
                    ha="center",
                    va="center",
                    fontsize=14,
                    weight="bold",
                    color=INK,
                )
                continue
            fill, accent = cell_colours(metric, score, reference)
            width = cell_width - 0.008
            box(x - width / 2, y - row_height / 2, width, row_height, fill)
            change = (
                f"{(score - reference) / abs(reference):+.1%}" if reference else "n/a"
            )
            ax.text(
                x,
                y + 0.015,
                format(score, spec),
                ha="center",
                va="center",
                fontsize=13,
                weight="semibold",
                color=INK,
            )
            ax.text(
                x,
                y - 0.021,
                change,
                ha="center",
                va="center",
                fontsize=10,
                weight="medium",
                color=accent,
            )

    for path in save_figure(fig, root, "performance_scorecard"):
        print(path)


if __name__ == "__main__":
    main()
