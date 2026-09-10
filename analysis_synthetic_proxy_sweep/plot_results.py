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
    "proximal_2sls": ("Proximal 2SLS", "#1966B3", "-"),
    "backdoor_ols_xwz": ("OLS: X, W, Z", "#D46B16", "-"),
    "backdoor_ols_x": ("OLS: X", "#737373", "--"),
    "oracle_ols_xu": ("Oracle OLS: X, U", "#36835B", ":"),
}


def plot_results(results_dir: Path):
    with (results_dir / "method_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The results table is empty.")
    dimensions = sorted({int(row["dimension"]) for row in rows})
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
        2,
        len(dimensions),
        figsize=(13.2, 7.5),
        sharex=True,
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": [1.55, 1]},
    )
    for col, dimension in enumerate(dimensions):
        curves = {}
        for method, (label, color, style) in METHODS.items():
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["dimension"]) == dimension and row["method"] == method
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
            curves[method] = (levels, strength, pehe)
            axes[0, col].plot(
                strength, pehe, color=color, linestyle=style, linewidth=2.1, label=label
            )

        proximal = curves["proximal_2sls"]
        backdoor = curves["backdoor_ols_xwz"]
        if not np.array_equal(proximal[0], backdoor[0]):
            raise ValueError("Proximal and backdoor curves have different levels.")
        axes[0, col].set_title(f"Dimension d = {dimension}", fontweight="bold", pad=12)
        axes[0, col].set_yscale("log")
        axes[1, col].plot(
            proximal[1],
            proximal[2] / backdoor[2],
            color=METHODS["proximal_2sls"][1],
            linewidth=2.1,
        )
        axes[1, col].axhline(1, color="#444444", linestyle="--", linewidth=1)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Joint proxy information (population R²)")
        for ax in axes[:, col]:
            ax.grid(axis="y", which="major", color="#E2E5E8", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    axes[0, 0].set_ylabel("CATE RMSE (PEHE)")
    axes[1, 0].set_ylabel("Proximal / OLS(X,W,Z) error")
    axes[1, 0].text(
        0.97,
        0.95,
        "Below 1: proximal has lower error",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.suptitle(
        "Proxy informativeness: initial method comparison",
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
        ncol=4,
        frameon=False,
    )
    settings = len({(row["dimension"], row["level_id"]) for row in rows})
    sizes = {(int(row["n_train"]), int(row["n_test"])) for row in rows}
    sample_label = "Sample sizes recorded in method_results.csv"
    if len(sizes) == 1:
        train, test = next(iter(sizes))
        sample_label = f"{train:,} training + {test:,} test rows per setting"
    fig.text(
        0.5,
        0.015,
        f"One paired simulation · {settings} datasets · {sample_label}",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93), h_pad=2)
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
