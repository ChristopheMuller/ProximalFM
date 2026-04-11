# %%
import pandas as pd
from pathlib import Path

# %% Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Define what you want to make a table of here.
# ─────────────────────────────────────────────────────────────────────────────

# 1. Which experiment (path to the directory containing the CSVs)
experiment_dir = Path("results/linear_cate_conf4_218/results/")

# 2. Which methods to include (leave empty to include all)
# Example: methods_to_include = ["Method A", "Method B", "Method C"]
methods_to_include = [] 

# 3. Which ATE metrics to include in the table
ate_metrics_to_include = [
    'NormBias', 'NormRMSE', 'NormRMedSE', 'ATE R²'
]
ate_decimals = [4, 2, 2, 2]

# 4. Which CATE metrics to include in the table
cate_metrics_to_include = [
    "NormRMSE", "NormRMedSE",
    "Mean R²", "Median R²", 
    "Mean R²(shape)"
]
cate_decimals = [2, 2, 2, 2, 2]

# %% Load, Filter, and Generate LaTeX
# ─────────────────────────────────────────────────────────────────────────────

ate_csv = experiment_dir / "ate_summary.csv"
cate_csv = experiment_dir / "cate_summary.csv"

if not ate_csv.exists() or not cate_csv.exists():
    raise FileNotFoundError(f"Could not find summary CSVs in {experiment_dir}")

df_ate = pd.read_csv(ate_csv, index_col="method")
df_cate = pd.read_csv(cate_csv, index_col="method")

# Filter by methods if specified
if methods_to_include:
    # Keep only methods that actually exist in the dataframe
    valid_methods = [m for m in methods_to_include if m in df_ate.index]
    df_ate = df_ate.loc[valid_methods]
    df_cate = df_cate.loc[valid_methods]

# Filter by metrics
print(df_ate.columns)
df_ate = df_ate[[m for m in ate_metrics_to_include if m in df_ate.columns]]
df_cate = df_cate[[m for m in cate_metrics_to_include if m in df_cate.columns]]

# Format column names for LaTeX (e.g., converting 'R²' to math mode '$R^2$')
def format_columns_for_latex(cols):
    return [c.replace("R²", "$R^2$").replace("²", "$^2$") for c in cols]

df_ate.columns = format_columns_for_latex(df_ate.columns)
df_cate.columns = format_columns_for_latex(df_cate.columns)

# Create format dictionaries mapping the LaTeX-formatted column names to format strings
# This turns an integer like `2` into a format string like `"{:.2f}"`
ate_format_dict = {col: f"{{:.{d}f}}" for col, d in zip(df_ate.columns, ate_decimals)}
cate_format_dict = {col: f"{{:.{d}f}}" for col, d in zip(df_cate.columns, cate_decimals)}

# Generate LaTeX using pandas styling
latex_ate = df_ate.style.format(ate_format_dict).to_latex(
    caption="Average Treatment Effect (ATE) Results",
    label="tab:ate_results",
    hrules=True,
)

latex_cate = df_cate.style.format(cate_format_dict).to_latex(
    caption="Conditional Average Treatment Effect (CATE) Results",
    label="tab:cate_results",
    hrules=True,
)

print("%%% ATE TABLE %%%\n")
print(latex_ate)
print("\n%%% CATE TABLE %%%\n")
print(latex_cate)