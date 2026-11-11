"""Reproduce the rebuttal figure with python -m rebuttal.plot_error_decomposition."""

from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from analysis.analysis_eval.design import (
    CONTEXT_MARKER_AREAS,
    apply_style,
    decomposition_figsize,
    method_style,
    ordered_methods,
)
from analysis.analysis_eval.error_decomposition import draw_error_decomposition_panel
from analysis.analysis_eval.learning_curves import CONF_PROXY_ORDER, DEFAULT_MATRIX_ROOT
from analysis.analysis_eval.learning_curves_error_decomposition import DEFAULT_METHODS
from analysis.analysis_eval.plotting import plt, save_figure
from analysis.analysis_eval.semisynt import (
    load_cate_results,
    summarize_error_decomposition,
)

CONTEXTS = [128, 512, 1024, 2048]
METHODS = ordered_methods(
    method for method in DEFAULT_METHODS if method != "ProximalFM-FT"
)
FAMILIES = ("linear", "nonlinear")
X_MAX = {"low": 0.08, "high": 1.8}
Y_MAX = 1.4
OUTPUT_DIR = Path(__file__).resolve().parent / "assets" / "results"
OUTPUT_NAME = "semisynt_error_decomposition"


def main() -> None:
    frame = load_cate_results(DEFAULT_MATRIX_ROOT, dimension=5)
    frame = frame[frame["n_ctx"].isin(CONTEXTS) & frame["method"].isin(METHODS)]
    summary = summarize_error_decomposition(frame, METHODS, aggregation="median")
    expected_points = (
        len(FAMILIES) * len(CONF_PROXY_ORDER) * len(METHODS) * len(CONTEXTS)
    )
    if len(summary) != expected_points or not summary["n"].eq(36).all():
        raise ValueError("Expected 36 episodes per setting, method, and context size")

    apply_style()
    fig = plt.figure(figsize=decomposition_figsize(), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.42))
    for row, family in enumerate(FAMILIES):
        for col, (confounding, proxy) in enumerate(CONF_PROXY_ORDER):
            ax = fig.add_subplot(grid[row, col])
            cell = summary[
                summary["family"].eq(family)
                & summary["confounding"].eq(confounding)
                & summary["proxy"].eq(proxy)
            ]
            x_max = X_MAX[confounding]
            if cell["shape_error"].max() > Y_MAX or (
                confounding == "high" and cell["level_error"].max() > x_max
            ):
                raise ValueError(
                    f"A trajectory exceeds the axes for {family}, {confounding}, {proxy}"
                )
            draw_error_decomposition_panel(
                ax,
                cell,
                methods=METHODS,
                contexts=CONTEXTS,
                family=family,
                confounding=confounding,
                proxy=proxy,
                xlim=(0, x_max),
                ylim=(0, Y_MAX),
                iso_levels=(0.5, 0.75, 1.0),
            )
            ax.set_title(ax.get_title() if row == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(
                f"{family.capitalize()}\n"
                + r"Heterogeneity error ($1-R^2_{\mathrm{shape}}$)"
                if col == 0
                else ""
            )
            ax.tick_params(labelbottom=row == 1, labelleft=col == 0)
            x_tick_step = 0.02 if confounding == "low" else 0.5
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
            ax.yaxis.set_major_locator(MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))

    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.set_axis_off()
    legend_ax.text(
        0.5,
        1,
        "Level error (squared normalized ATE bias)",
        ha="center",
        va="top",
        fontsize=9,
        transform=legend_ax.transAxes,
    )
    method_handles = [
        Line2D(
            [],
            [],
            label=method_style(method).label,
            **method_style(method).line_kwargs(),
        )
        for method in METHODS
    ]
    method_legend = legend_ax.legend(
        handles=method_handles,
        loc="center",
        ncol=4,
        columnspacing=1.5,
        handlelength=2.2,
    )
    legend_ax.add_artist(method_legend)
    context_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="0.35",
            markersize=area**0.5,
            label=f"{context:,}",
        )
        for context, area in zip(CONTEXTS, CONTEXT_MARKER_AREAS, strict=True)
    ]
    context_label = Line2D([], [], linestyle="none", label="Context size")
    legend_ax.legend(
        handles=[context_label, *context_handles],
        loc="lower center",
        ncol=5,
        columnspacing=1,
        handletextpad=0.4,
        borderaxespad=0,
    )
    for path in save_figure(fig, OUTPUT_DIR, OUTPUT_NAME):
        print(path)


if __name__ == "__main__":
    main()
