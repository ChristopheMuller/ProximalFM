# %%
"""bench_tables.py — Export the benchmark summary as a LaTeX table.

Reads the single `summary.csv` produced by bench_run (one row per method with
nPEHE, Shape R², ATE Bias, Mean Rank, Failure Rate, Time) and emits a LaTeX
table with the top-3 scores bolded (direction-aware per column).
"""
import pandas as pd
from pathlib import Path

# %% Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Resolve paths relative to the project root so the script works regardless of
# the current working directory (e.g. run as a `# %%` cell or from benchmarks/).
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # __file__ is undefined in some interactive runners
    PROJECT_ROOT = Path.cwd()
    while (
        not (PROJECT_ROOT / "pyproject.toml").exists()
        and PROJECT_ROOT != PROJECT_ROOT.parent
    ):
        PROJECT_ROOT = PROJECT_ROOT.parent

experiment_dir = PROJECT_ROOT / "results/test_tbd/results/"
methods_to_include: list[str] = []  # empty → all methods

# (column, decimals, direction) — direction picks which 3 cells get bolded.
#   "min"  : lower is better        (nPEHE, Mean Rank, Failure Rate, Time)
#   "max"  : higher is better       (Shape R²)
#   "zero" : closest to 0 is better (ATE Bias)
COLUMN_SPEC = [
    ("nPEHE", 3, "min"),
    ("Shape R²", 3, "max"),
    ("ATE Bias", 3, "zero"),
    ("Mean Rank", 2, "min"),
    ("Failure Rate", 2, "min"),
    ("Time (s)", 2, None),  # informational — never bolded
]

# %% Load and filter
# ─────────────────────────────────────────────────────────────────────────────
summary_csv = experiment_dir / "summary.csv"
if not summary_csv.exists():
    raise FileNotFoundError(
        f"Could not find {summary_csv}. Run bench_run.py first, or fix "
        "experiment_dir above."
    )

df = pd.read_csv(summary_csv, index_col="method")

if methods_to_include:
    df = df.loc[[m for m in methods_to_include if m in df.index]]

spec = [(c, d, dr) for (c, d, dr) in COLUMN_SPEC if c in df.columns]
df = df[[c for c, _, _ in spec]]


# %% Direction-aware top-3 bold formatter
# ─────────────────────────────────────────────────────────────────────────────
def create_top3_formatter(series: pd.Series, decimals: int, direction: str | None):
    if direction == "max":
        top3 = series.nlargest(3).values
    elif direction == "zero":
        top3 = series.loc[series.abs().nsmallest(3).index].values
    elif direction == "min":
        top3 = series.nsmallest(3).values
    else:
        top3 = []

    fmt_str = f"{{:.{decimals}f}}"

    def formatter(val):
        if pd.isna(val):
            return "-"
        v_str = fmt_str.format(val)
        return f"\\textbf{{{v_str}}}" if val in top3 else v_str

    return formatter


formatters = {
    col: create_top3_formatter(df[col], dec, direction) for col, dec, direction in spec
}


def to_latex(label_cols: pd.DataFrame, fmts: dict) -> str:
    styled = label_cols.copy()
    styled.columns = [c.replace("R²", "$R^2$") for c in styled.columns]
    fmts = {k.replace("R²", "$R^2$"): v for k, v in fmts.items()}
    return (
        styled.style.format(fmts)
        .to_latex(
            caption="Benchmark Summary (nPEHE, Shape $R^2$, ATE Bias).",
            label="tab:bench_results",
            hrules=True,
        )
        .replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    )


# %% Print
# ─────────────────────────────────────────────────────────────────────────────
print(to_latex(df, formatters))
