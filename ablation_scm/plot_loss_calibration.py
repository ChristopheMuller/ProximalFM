"""Plot the five loss arms from the CSVs written by evaluate_ablation.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis.analysis_eval.plotting import plt, save_figure

COLOURS = {250: "#0072B2", 16: "#D55E00", 1: "#79529B"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="evaluation output directory")
    args = parser.parse_args()
    root = args.results_dir

    coverage = pd.read_csv(root / "coverage.csv")
    coverage = coverage.loc[coverage["group"].eq("loss")]
    provenance = json.loads((root / "run.json").read_text(encoding="utf-8"))
    contexts = provenance["contexts"]

    fig, axes = plt.subplots(2, 2, figsize=(10.1, 7.2))
    fig.subplots_adjust(
        left=0.11, right=0.96, bottom=0.15, top=0.95, wspace=0.29, hspace=0.31
    )

    reliability_axes = axes.ravel()[:3]
    width_ax = axes[1, 1]
    for n_ctx, ax in zip(contexts, reliability_axes, strict=True):
        ax.set_title(f"{n_ctx} context rows", loc="left", fontsize=11)
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.4, color="0.45")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Nominal coverage",
            ylabel="Empirical coverage",
        )
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    for _, arm in coverage.groupby("arm", sort=False):
        meta = arm.iloc[0]
        style = dict(
            color=COLOURS[meta["k_train"]],
            linestyle="-" if meta["correction"] else "--",
            linewidth=1.9,
            label=meta["variant"],
        )
        for n_ctx, ax in zip(contexts, reliability_axes, strict=True):
            frame = arm.loc[arm["n_ctx"].eq(n_ctx)].sort_values("nominal")
            ax.plot(frame["nominal"], frame["empirical"], **style)

        widths = arm.loc[arm["nominal"].eq(0.90)].sort_values("n_ctx")
        width_ax.plot(
            widths["n_ctx"], widths["width"], marker="o", markersize=4, **style
        )

    width_ax.set_title("90% interval width", loc="left", fontsize=11)
    width_ax.set(
        xlabel="Context rows", ylabel="Mean width (standardized units)", ylim=(0, None)
    )
    width_ax.set_xticks(contexts)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.92", linewidth=0.7)
        ax.tick_params(labelsize=9)

    handles, labels = width_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=3,
    )
    for path in save_figure(fig, root, "loss_calibration"):
        print(path)


if __name__ == "__main__":
    main()
