# %%
"""tables.py — Export the benchmark summaries as LaTeX tables.

run writes one summary per *track* — the three views of the same run:

    summary_cate.csv   tau*(x) = E[Y1-Y0 | X], the Monte-Carlo CATE target.
                       The headline table: methods that answer tau(x), ranked
                       on the estimand ProximalFM is trained on.
    summary_ite.csv    the realized Y1-Y0. Adds the methods that read the query
                       row's own proxies (estimand "xw"), which is a genuine
                       advantage here and no help at all on the CATE target;
                       read them against the Oracle-CATE(X) row, the floor no
                       tau(x) method can go below.
    summary_ate.csv    the level only, and the one table where a method that
                       returns a single scalar per episode (P2SLS, the
                       `marginal` strategies) says anything.

Each becomes its own LaTeX table with the top-3 scores bolded (direction-aware
per column). Reference rows (the oracles) carry no Mean Rank and are printed
last, exactly as they are ranked: they are anchors, not competitors.
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
    ("Estimand", 0, None),
    ("nPEHE", 3, "min"),
    ("Shape R²", 3, "max"),
    ("ATE Bias", 3, "zero"),
    ("Mean Rank", 2, "min"),
    ("Failure Rate", 2, "min"),
    ("Time (s)", 2, None),  # informational — never bolded
]

# The ATE track ranks on the level alone; nPEHE/Shape there would only repeat
# the CATE table for every method that appears in both.
ATE_COLUMN_SPEC = [
    ("Estimand", 0, None),
    ("ATE Bias", 3, "zero"),
    ("ATE RMSE", 3, "min"),
    ("Mean Rank", 2, "min"),
    ("Failure Rate", 2, "min"),
    ("Time (s)", 2, None),
]

TRACKS = {
    "cate": (
        COLUMN_SPEC,
        "Benchmark summary on the CATE target " "$\\tau^*(x) = E[Y_1 - Y_0 \\mid X]$.",
        "tab:bench_cate",
    ),
    "ite": (
        COLUMN_SPEC,
        "Benchmark summary on the realized ITE $Y_1 - Y_0$. Methods with "
        "estimand \\texttt{xw} condition on the query row's proxies.",
        "tab:bench_ite",
    ),
    "ate": (
        ATE_COLUMN_SPEC,
        "Benchmark summary at the ATE level.",
        "tab:bench_ate",
    ),
}


# %% Direction-aware top-3 bold formatter
# ─────────────────────────────────────────────────────────────────────────────
def create_top3_formatter(
    series: pd.Series,
    decimals: int,
    direction: str | None,
    ranked: pd.Series | None = None,
):
    """Format a column, bolding its best three *competing* values.

    ``ranked`` restricts which rows may win the bold: the oracle rows would
    otherwise take every slot on the metrics they define.
    """
    pool = series if ranked is None else series[ranked]
    if direction == "max":
        top3 = pool.nlargest(3).values
    elif direction == "zero":
        top3 = pool.loc[pool.abs().nsmallest(3).index].values
    elif direction == "min":
        top3 = pool.nsmallest(3).values
    else:
        top3 = []

    fmt_str = f"{{:.{decimals}f}}"

    def formatter(val):
        if pd.isna(val):
            return "-"
        if isinstance(val, str):  # the Estimand tag
            return f"\\texttt{{{val}}}"
        v_str = fmt_str.format(val)
        return f"\\textbf{{{v_str}}}" if val in top3 else v_str

    return formatter


def to_latex(df: pd.DataFrame, fmts: dict, caption: str, label: str) -> str:
    styled = df.copy()
    styled.columns = [c.replace("R²", "$R^2$") for c in styled.columns]
    fmts = {k.replace("R²", "$R^2$"): v for k, v in fmts.items()}
    return (
        styled.style.format(fmts)
        .to_latex(caption=caption, label=label, hrules=True)
        .replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    )


def track_table(path: Path, spec: list, caption: str, label: str) -> str:
    df = pd.read_csv(path, index_col="method")
    reference = (
        df["Reference"].astype(bool)
        if "Reference" in df.columns
        else pd.Series(False, index=df.index)
    )
    if methods_to_include:
        keep = [m for m in methods_to_include if m in df.index]
        # Reference rows are anchors for reading the table, so a method filter
        # must not silently drop them.
        keep += [m for m in df.index[reference] if m not in keep]
        df, reference = df.loc[keep], reference.loc[keep]

    spec = [(c, d, dr) for (c, d, dr) in spec if c in df.columns]
    df = df[[c for c, _, _ in spec]]
    fmts = {
        col: create_top3_formatter(df[col], dec, dr, ranked=~reference)
        for col, dec, dr in spec
    }
    return to_latex(df, fmts, caption, label)


# %% Emit one table per available track
# ─────────────────────────────────────────────────────────────────────────────
found = False
for track, (spec, caption, label) in TRACKS.items():
    path = experiment_dir / f"summary_{track}.csv"
    if not path.exists():
        continue
    found = True
    print(f"% ── {track} ──")
    print(track_table(path, spec, caption, label))

if not found:
    # Runs from before the tracks existed wrote only summary.csv.
    legacy = experiment_dir / "summary.csv"
    if not legacy.exists():
        raise FileNotFoundError(
            f"No summary_*.csv (nor {legacy}) in {experiment_dir}. Run "
            "run.py first, or fix experiment_dir above."
        )
    print(track_table(legacy, COLUMN_SPEC, "Benchmark Summary.", "tab:bench_results"))
