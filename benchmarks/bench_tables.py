# %%
import pandas as pd
from pathlib import Path

# %% Configuration
# ─────────────────────────────────────────────────────────────────────────────
experiment_dir = Path("results/bnn_1st/results/")
methods_to_include = []

ate_metrics_to_include = ["NormBias", "NormRMSE", "NormRMedSE", "ATE R²"]
ate_decimals = [4, 2, 2, 2]

cate_metrics_to_include = [
    "NormRMSE",
    "NormRMedSE",
    "Mean R²",
    "Median R²",
    "Mean R²(shape)",
]
cate_decimals = [2, 2, 2, 2, 2]

# %% Load and Filter
# ─────────────────────────────────────────────────────────────────────────────
ate_csv = experiment_dir / "ate_summary.csv"
cate_csv = experiment_dir / "cate_summary.csv"

if not ate_csv.exists() or not cate_csv.exists():
    raise FileNotFoundError(f"Could not find summary CSVs in {experiment_dir}")

df_ate = pd.read_csv(ate_csv, index_col="method")
df_cate = pd.read_csv(cate_csv, index_col="method")

if methods_to_include:
    valid_methods = [m for m in methods_to_include if m in df_ate.index]
    df_ate = df_ate.loc[valid_methods]
    df_cate = df_cate.loc[valid_methods]

df_ate = df_ate[[m for m in ate_metrics_to_include if m in df_ate.columns]]
df_cate = df_cate[[m for m in cate_metrics_to_include if m in df_cate.columns]]


def format_columns_for_latex(cols):
    return [c.replace("R²", "$R^2$").replace("²", "$^2$") for c in cols]


df_ate.columns = format_columns_for_latex(df_ate.columns)
df_cate.columns = format_columns_for_latex(df_cate.columns)


# %% Dynamic Top-3 Formatter Logic
# ─────────────────────────────────────────────────────────────────────────────
def create_top3_formatter(series, decimals):
    """
    Creates a formatting function that automatically bolds the best 3 scores.
    Handles directions automatically: maximizes R^2, minimizes absolute Bias, minimizes others.
    """
    # Extract name (handles both normal columns and MultiIndex tuples)
    c_name = series.name[-1] if isinstance(series.name, tuple) else series.name

    # 1. Determine the top 3 values based on metric type
    if "R^2" in c_name:
        # Higher is better
        top3_vals = series.nlargest(3).values
    elif "Bias" in c_name:
        # Closer to zero is better (absolute minimum)
        top3_vals = series.loc[series.abs().nsmallest(3).index].values
    else:
        # Lower is better (RMSE, MAE, etc.)
        top3_vals = series.nsmallest(3).values

    fmt_str = f"{{:.{decimals}f}}"

    # 2. Return a formatting function to apply to each cell
    def formatter(val):
        if pd.isna(val):
            return "-"

        v_str = fmt_str.format(val)
        # Check if this specific value is in the top 3
        if val in top3_vals:
            return f"\\textbf{{{v_str}}}"
        return v_str

    return formatter


# Build dictionary of formatters for each dataframe
ate_formatters = {
    col: create_top3_formatter(df_ate[col], d)
    for col, d in zip(df_ate.columns, ate_decimals)
}
cate_formatters = {
    col: create_top3_formatter(df_cate[col], d)
    for col, d in zip(df_cate.columns, cate_decimals)
}

# Generate LaTeX with centered environments
latex_ate = (
    df_ate.style.format(ate_formatters)
    .to_latex(
        caption="Average Treatment Effect (ATE) Results",
        label="tab:ate_results",
        hrules=True,
    )
    .replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
)

latex_cate = (
    df_cate.style.format(cate_formatters)
    .to_latex(
        caption="Conditional Average Treatment Effect (CATE) Results",
        label="tab:cate_results",
        hrules=True,
    )
    .replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
)

# %% Generate Combined Best Metrics Table
# ─────────────────────────────────────────────────────────────────────────────
ate_comb_cols = ["ATE $R^2$", "NormRMSE"]
cate_comb_cols = ["Mean $R^2$", "Mean $R^2$(shape)", "NormRMSE"]

df_ate_comb = df_ate[[c for c in ate_comb_cols if c in df_ate.columns]].copy()
df_cate_comb = df_cate[[c for c in cate_comb_cols if c in df_cate.columns]].copy()

# Rename to MultiIndex
df_ate_comb.columns = pd.MultiIndex.from_tuples(
    [
        ("ATE", col.replace("ATE ", "").replace("Norm", "N"))
        for col in df_ate_comb.columns
    ]
)
df_cate_comb.columns = pd.MultiIndex.from_tuples(
    [
        ("CATE", col.replace("Mean ", "").replace("Norm", "N"))
        for col in df_cate_comb.columns
    ]
)

# Concatenate side-by-side
df_combined = pd.concat([df_ate_comb, df_cate_comb], axis=1)

# Create formatters for the combined table (defaulting to 2 decimals)
combined_formatters = {
    col: create_top3_formatter(df_combined[col], 2) for col in df_combined.columns
}

# Generate LaTeX with Multi-column alignment and centering
latex_combined = (
    df_combined.style.format(combined_formatters)
    .to_latex(
        caption="Combined Top Performance Metrics for ATE and CATE",
        label="tab:combined_results",
        hrules=True,
        multicol_align="c",
    )
    .replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
)

# %% Print Results
# ─────────────────────────────────────────────────────────────────────────────
print("%%% ATE TABLE %%%\n")
print(latex_ate)
print("\n%%% CATE TABLE %%%\n")
print(latex_cate)
print("\n%%% COMBINED TABLE %%%\n")
print(latex_combined)
