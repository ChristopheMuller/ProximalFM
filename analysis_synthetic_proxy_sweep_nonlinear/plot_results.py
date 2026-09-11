#!/usr/bin/env python3
"""Plot nonlinear proxy-strength curves for the three declared methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
METHODS = {
    "proximalfm": ("ProximalFM", "#D62728", "-"),
    "p_kernel_rf": ("P-Kernel-RF", "#1966B3", "-"),
    "s_tabicl_wz_mc64": ("S-TabICL-WZ-MC64", "#D46B16", "--"),
}


def plot_results(results_dir: Path) -> None:
    results_dir = Path(results_dir).expanduser().resolve()
    with (results_dir / "method_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("method_results.csv is empty")
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise ValueError(f"inspect {len(failed)} failed fits before plotting")
    unexpected = {row["method"] for row in rows} - set(METHODS)
    if unexpected:
        raise ValueError(f"unknown method names in results: {sorted(unexpected)}")

    dimensions = sorted({int(row["dimension"]) for row in rows})
    context_sizes = sorted({int(row["context_size"]) for row in rows})
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        len(context_sizes),
        len(dimensions),
        figsize=(13.2, max(3.4 * len(context_sizes), 4.8)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for plot_row, context_size in enumerate(context_sizes):
        for column, dimension in enumerate(dimensions):
            axis = axes[plot_row, column]
            expected_levels = None
            for method, (label, color, linestyle) in METHODS.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["dimension"]) == dimension
                        and int(row["context_size"]) == context_size
                        and row["method"] == method
                    ),
                    key=lambda row: int(row["level_id"]),
                )
                if not selected:
                    raise ValueError(
                        f"missing {method}, d={dimension}, context={context_size}"
                    )
                levels = np.array([int(row["level_id"]) for row in selected])
                strength = np.array([float(row["joint_r2_target"]) for row in selected])
                pehe = np.array([float(row["pehe"]) for row in selected])
                if not np.isfinite(pehe).all() or np.any(pehe <= 0):
                    raise ValueError("logarithmic plots require positive finite PEHE")
                if expected_levels is None:
                    expected_levels = levels
                elif not np.array_equal(levels, expected_levels):
                    raise ValueError("methods do not contain the same proxy levels")
                axis.plot(
                    strength,
                    pehe,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.1,
                )
            if plot_row == 0:
                axis.set_title(f"Dimension d = {dimension}", fontweight="bold", pad=12)
            if plot_row == len(context_sizes) - 1:
                axis.set_xlabel("Joint proxy information (population nonlinear R²)")
            if column == 0:
                axis.set_ylabel(f"CATE RMSE\ncontext = {context_size:,}")
            axis.set_yscale("log")
            axis.set_xlim(0, 1)
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1])
            axis.grid(axis="y", which="major", color="#E2E5E8", linewidth=0.8)

    fig.suptitle(
        "Nonlinear proxy informativeness across context sizes",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    settings = len({(row["dimension"], row["level_id"]) for row in rows})
    query_sizes = {int(row["n_test"]) for row in rows}
    query_label = (
        f"{next(iter(query_sizes)):,} held-out rows per setting"
        if len(query_sizes) == 1
        else "query sizes recorded in method_results.csv"
    )
    fig.text(
        0.5,
        0.015,
        f"One paired simulation · {settings} datasets · {query_label}",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=1.8)
    for extension in ("png", "pdf"):
        target = results_dir / f"proxy_informativeness_nonlinear.{extension}"
        fig.savefig(target, dpi=190, facecolor="white")
        print(target)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    plot_results(args.results_dir)


if __name__ == "__main__":
    main()
