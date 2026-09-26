"""Plot the saved ablation curves. Open in PyCharm and press Run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "proximalfm": ("ProximalFM", "#D62728", "-"),
    "proximal_2sls": ("P2SLS", "#1966B3", "-"),
    "backdoor_ols_xwz": ("OLS(X,W,Z)", "#D46B16", "-"),
    "backdoor_ols_x": ("OLS(X)", "#737373", "--"),
    "backdoor_tabicl": ("TabICL backdoor", "#9467BD", "-."),
    "backdoor_causalpfn": ("CausalPFN backdoor", "#E377C2", "-."),
    "oracle_ols_xu": ("Oracle OLS(X,U)", "#36835B", ":"),
}


def plot_results(results_dir: Path):
    with (results_dir / "method_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The results table is empty.")
    dimensions = sorted({int(row["dimension"]) for row in rows})
    context_sizes = sorted(
        {int(row.get("context_size") or row["n_train"]) for row in rows}
    )
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise ValueError(f"Inspect {len(failed)} flagged fits before plotting.")

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
        for col, dimension in enumerate(dimensions):
            ax = axes[plot_row, col]
            expected_levels = None
            for method, (label, color, style) in METHODS.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["dimension"]) == dimension
                        and row["method"] == method
                        and int(row.get("context_size") or row["n_train"])
                        == context_size
                    ),
                    key=lambda row: int(row["level_id"]),
                )
                if not selected:
                    continue
                levels = np.array([int(row["level_id"]) for row in selected])
                strength = np.array([float(row["joint_r2_target"]) for row in selected])
                pehe = np.array([float(row["pehe"]) for row in selected])
                if not np.isfinite(pehe).all() or np.any(pehe <= 0):
                    raise ValueError("Logarithmic plots require finite positive PEHE.")
                if expected_levels is None:
                    expected_levels = levels
                elif not np.array_equal(expected_levels, levels):
                    raise ValueError(
                        f"Methods have different levels for d={dimension}, "
                        f"context={context_size}."
                    )
                ax.plot(
                    strength,
                    pehe,
                    color=color,
                    linestyle=style,
                    linewidth=2.1,
                    label=label,
                )

            if expected_levels is None:
                raise ValueError(
                    f"No curves found for d={dimension}, context={context_size}."
                )
            if plot_row == 0:
                ax.set_title(f"Dimension d = {dimension}", fontweight="bold", pad=12)
            if plot_row == len(context_sizes) - 1:
                ax.set_xlabel("Joint proxy information (population R²)")
            if col == 0:
                ax.set_ylabel(f"CATE RMSE\ncontext = {context_size:,}")
            ax.set_yscale("log")
            ax.grid(axis="y", which="major", color="#E2E5E8", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    fig.suptitle(
        "Proxy informativeness across context sizes",
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
        ncol=min(len(handles), 6),
        frameon=False,
    )
    settings = len({(row["dimension"], row["level_id"]) for row in rows})
    query_sizes = {int(row["n_test"]) for row in rows}
    query_label = "Query sizes recorded in method_results.csv"
    if len(query_sizes) == 1:
        query_label = f"{next(iter(query_sizes)):,} held-out rows per setting"
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
        target = results_dir / f"proxy_informativeness.{extension}"
        fig.savefig(target, dpi=190, facecolor="white")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args()
    plot_results(args.results_dir.expanduser().resolve())
