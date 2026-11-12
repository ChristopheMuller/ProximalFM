"""Refresh the shape R² table and figure with python -m rebuttal.proximalfm_r2shape."""

import csv
import math
import re
from pathlib import Path

from analysis.analysis_eval.design import COLORS, FULL_PAGE_WIDTH_IN, apply_style
from analysis.analysis_eval.plotting import plt

REBUTTAL_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = REBUTTAL_DIR / "results" / "semisynt_proximalfm_shape_summary.csv"
APPENDIX_PATH = REBUTTAL_DIR / "new_appendix.tex"
FIGURE_PATH = REBUTTAL_DIR / "assets" / "rebuttal" / "proximalfm_r2shape.png"
CONTEXTS = (128, 512, 1024, 2048)
SETTINGS = [
    (
        f"{family}_conf_{confounding}_proxy_{proxy}",
        family.capitalize(),
        f"{confounding}/{strength}",
    )
    for family in ("linear", "nonlinear")
    for confounding in ("low", "high")
    for proxy, strength in (("low", "weak"), ("high", "strong"))
]
TABLE_LABEL = r"\label{tab:shape_paired}"
TABLE_PATTERN = re.compile(
    r"^\\begin\{table\}.*?^\\end\{table\}", re.MULTILINE | re.DOTALL
)


def load_summary() -> dict[tuple[str, int], dict[str, str]]:
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        (row["case"], int(row["n_ctx"])): row
        for row in rows
        if row["method"] == "ProximalFM"
    }
    cases = [case for case, _, _ in SETTINGS] + ["overall"]
    expected = {(case, context) for case in cases for context in CONTEXTS}
    if len(rows) != len(expected) or scores.keys() != expected:
        raise ValueError("Expected one ProximalFM summary per setting and context")
    return scores


def render_table(scores: dict[tuple[str, int], dict[str, str]]) -> str:
    """Render the paired ProximalFM metrics at the paper's four context sizes."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Paired nPEHE and shape $R^2$ ($R_s^2$) for the fixed pre-trained "
        r"\modelname on the original semi-synthetic benchmark. Each setting averages "
        r"three replicates per source across twelve sources. Conf./proxy denotes "
        r"confounding strength and proxy reliability.}",
        TABLE_LABEL,
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        " & ".join(["", "", *(rf"\multicolumn{{2}}{{c}}{{$N={n}$}}" for n in CONTEXTS)])
        + r"\\",
        " & ".join(
            ["Mechanism", "Conf./proxy", *(["nPEHE", "$R_s^2$"] * len(CONTEXTS))]
        )
        + r"\\",
        r"\midrule",
    ]
    for case, mechanism, conf_proxy in [*SETTINGS, ("overall", "Overall", "")]:
        if case == "overall":
            lines.append(r"\midrule")
        values = [
            f"{float(scores[case, context][metric]):.3f}"
            for context in CONTEXTS
            for metric in ("mean_npehe", "mean_shape_r2")
        ]
        lines.append(" & ".join([mechanism, conf_proxy, *values]) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_figure(scores: dict[tuple[str, int], dict[str, str]]) -> plt.Figure:
    shape = [
        [float(scores[case, context]["mean_shape_r2"]) for context in CONTEXTS]
        for case, _, _ in SETTINGS
    ]
    overall = [scores["overall", context] for context in CONTEXTS]
    heterogeneity = [float(row["mean_heterogeneity_error"]) for row in overall]
    level = [float(row["mean_level_error"]) for row in overall]
    totals = [float(row["mean_squared_npehe"]) for row in overall]
    if not all(
        math.isclose(shape_error + level_error, total, rel_tol=1e-12)
        for shape_error, level_error, total in zip(
            heterogeneity, level, totals, strict=True
        )
    ):
        raise ValueError("Error components do not sum to mean squared nPEHE")

    apply_style()
    fig, (heatmap, bars) = plt.subplots(
        1,
        2,
        figsize=(FULL_PAGE_WIDTH_IN, 3.45),
        width_ratios=(1.4, 1),
        layout="constrained",
    )
    x = range(len(CONTEXTS))
    labels = [f"{context:,}" for context in CONTEXTS]

    heatmap.grid(False)
    heatmap.imshow(shape, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    heatmap.set_yticks(
        range(len(SETTINGS)),
        [
            f"{mechanism} | {conf_proxy.replace('/', ' / ')}"
            for _, mechanism, conf_proxy in SETTINGS
        ],
    )
    heatmap.spines[:].set_visible(False)
    heatmap.axhline(3.5, color="white", linewidth=1.5)
    for row, values in enumerate(shape):
        for col, value in enumerate(values):
            heatmap.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value >= 0.58 else COLORS["charcoal"],
            )

    bars.grid(False, axis="x")
    bars.bar(
        x, heterogeneity, width=0.62, color=COLORS["blue"], label="Heterogeneity error"
    )
    top = bars.bar(
        x,
        level,
        width=0.62,
        bottom=heterogeneity,
        color=COLORS["orange"],
        label="Level error",
    )
    bars.bar_label(top, labels=[f"{total:.3f}" for total in totals], padding=3)
    bars.set_ylabel("Mean squared nPEHE")
    bars.set_ylim(0, max(totals) * 1.2)
    bars.legend(loc="upper right")
    for ax in (heatmap, bars):
        ax.set_xticks(x, labels)
        ax.set_xlabel("Context observations")
    return fig


def main() -> None:
    scores = load_summary()
    appendix = APPENDIX_PATH.read_text(encoding="utf-8")
    matches = [
        match for match in TABLE_PATTERN.finditer(appendix) if TABLE_LABEL in match[0]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {TABLE_LABEL} in {APPENDIX_PATH}")
    table = matches[0]
    updated = appendix[: table.start()] + render_table(scores) + appendix[table.end() :]
    fig = make_figure(scores)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)
    if updated != appendix:
        APPENDIX_PATH.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
