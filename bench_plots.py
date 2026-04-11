import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Method Styling Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Grouping: 
# - OLS : Blue
# - RF : Green
# - TabICL : Purple
# - P2SLS : Orange
# - Tabular / Neural Causal Models : Red
METHOD_STYLES = {
    # ── OLS baselines (Blue) ──
    "S-OLS":      {"color": "#1f77b4", "marker": "o"},  # circle
    "S-OLS+AX":   {"color": "#1f77b4", "marker": "s"},  # square
    "T-OLS":      {"color": "#1f77b4", "marker": "^"},  # triangle up

    # ── RF baselines (Green) ──
    "S-RF":       {"color": "#2ca02c", "marker": "o"},
    "T-RF":       {"color": "#2ca02c", "marker": "^"},

    # ── TabICL baselines (Purple) ──
    "S-TabICL":   {"color": "#9467bd", "marker": "o"},
    "T-TabICL":   {"color": "#9467bd", "marker": "^"},

    # ── Proximal baselines (Orange) ──
    "P2SLS":      {"color": "#ff7f0e", "marker": "o"},
    "P2SLS+AX":   {"color": "#ff7f0e", "marker": "s"},

    # ── Tabular & Neural Models (Red) ──
    "CausalFM":   {"color": "#d62728", "marker": "X"},  # cross
    "DoPFN":      {"color": "#d62728", "marker": "D"},  # diamond
    "CausalPFN":  {"color": "#d62728", "marker": "P"},  # plus
    "ProximalFM": {"color": "#d62728", "marker": "*", "s": 200},  # big star
}


def plot_ate_vs_cate_r2(results_dir: Path):
    """
    Plots a unit square comparing Cross-Episode ATE R² vs Average CATE R².
    """
    ate_csv = results_dir / "ate_summary.csv"
    cate_csv = results_dir / "cate_summary.csv"

    if not ate_csv.exists() or not cate_csv.exists():
        warnings.warn(f"Summary CSVs not found in {results_dir}. Skipping ATE vs CATE R² plot.")
        return

    df_ate = pd.read_csv(ate_csv, index_col="method")
    df_cate = pd.read_csv(cate_csv, index_col="method")

    # We use standard "Mean R²" for CATE. You could change this to "Mean R²(shape)" if preferred.
    df = pd.DataFrame({
        "ATE_R2": df_ate["ATE R²"],
        "CATE_R2": df_cate["Mean R²"]
    }).dropna()

    plt.figure(figsize=(7, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Add reference quadrants & target lines
    plt.axhline(0, color='black', linestyle='-', alpha=0.2, zorder=0)
    plt.axvline(0, color='black', linestyle='-', alpha=0.2, zorder=0)
    plt.plot([-1, 1], [-1, 1], ls="--", color="gray", alpha=0.4, zorder=0, label="y = x")
    
    # Unit square boundaries
    plt.axhline(1, color='green', linestyle=':', alpha=0.5, zorder=0)
    plt.axvline(1, color='green', linestyle=':', alpha=0.5, zorder=0)

    # Plot each method
    for method, row in df.iterrows():
        style = METHOD_STYLES.get(method, {"color": "gray", "marker": "o", "s": 100})
        
        plt.scatter(
            row["ATE_R2"], 
            row["CATE_R2"],
            color=style.get("color"),
            marker=style.get("marker"),
            s=style.get("s", 120),  # default size 120
            label=method,
            edgecolors='white',
            linewidths=0.8,
            zorder=5
        )

    # Formatting the Unit Square
    # We clip the lower bound slightly below 0 so bad models are visible, but the focus is [0, 1]
    plt.xlim(-0.2, 1.05)
    plt.ylim(-0.2, 1.05)
    
    plt.title("ATE vs CATE Estimation Performance ($R^2$)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("ATE $R^2$ (Cross-Episode Tracking)")
    plt.ylabel("CATE $R^2$ (Average per Episode)")
    
    # Legend formatting (put it outside the unit square if it gets crowded)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Methods", frameon=False)

    output_file = results_dir / "plot_ate_vs_cate_r2.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()

def plot_ate_vs_cate_normrmse(results_dir: Path):
    """
    Plots a scatter comparing Cross-Episode ATE NormRMSE vs Average CATE NormRMSE.
    Ideal performance is at the origin (0, 0).
    """
    ate_csv = results_dir / "ate_summary.csv"
    cate_csv = results_dir / "cate_summary.csv"

    if not ate_csv.exists() or not cate_csv.exists():
        warnings.warn(f"Summary CSVs not found in {results_dir}. Skipping NormRMSE plot.")
        return

    df_ate = pd.read_csv(ate_csv, index_col="method")
    df_cate = pd.read_csv(cate_csv, index_col="method")

    df = pd.DataFrame({
        "ATE_NormRMSE": df_ate["NormRMSE"],
        "CATE_NormRMSE": df_cate["NormRMSE"]
    }).dropna()

    plt.figure(figsize=(7, 7))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Plot each method
    for method, row in df.iterrows():
        style = METHOD_STYLES.get(method, {"color": "gray", "marker": "o", "s": 100})
        
        plt.scatter(
            row["ATE_NormRMSE"], 
            row["CATE_NormRMSE"],
            color=style.get("color"),
            marker=style.get("marker"),
            s=style.get("s", 120),  # default size 120
            label=method,
            edgecolors='white',
            linewidths=0.8,
            zorder=5
        )

    # Calculate dynamic limits to frame the data properly
    max_val = max(df["ATE_NormRMSE"].max(), df["CATE_NormRMSE"].max())
    pad = max_val * 0.1
    
    # Add reference line y=x (Where ATE error = CATE error)
    plt.plot([-pad, max_val + pad], [-pad, max_val + pad], ls="--", color="gray", alpha=0.4, zorder=0, label="y = x")
    
    # Emphasize the axes/origin (0 is perfect)
    plt.axhline(0, color='black', linestyle='-', alpha=0.4, zorder=0)
    plt.axvline(0, color='black', linestyle='-', alpha=0.4, zorder=0)

    # Set limits starting slightly below 0 so the origin is clearly visible
    min_limit = min(df["ATE_NormRMSE"].min(), df["CATE_NormRMSE"].min()) - 0.03
    plt.xlim(min_limit, max_val + pad)
    plt.ylim(min_limit, max_val + pad)

    # use log scale for x and y
    plt.xscale("log")
    plt.yscale("log")
    
    plt.title("ATE vs CATE Estimation Error (NormRMSE)\n[Closer to Origin is Better]", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("ATE NormRMSE (Cross-Episode)")
    plt.ylabel("CATE NormRMSE (Average per Episode)")
    
    # Legend formatting
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Methods", frameon=False)

    output_file = results_dir / "plot_ate_vs_cate_normrmse.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()

def plot_true_vs_pred_ate(csv_path: str):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    min_val = min(
        df["ate_true"].min(), max(df["ate_pred"].min(), 2 * df["ate_true"].min())
    )
    max_val = max(
        df["ate_true"].max(), min(df["ate_pred"].max(), 2 * df["ate_true"].max())
    )
    pad = (max_val - min_val) * 0.05  # 5% padding
    axis_limits = (min_val - pad, max_val + pad)

    g = sns.FacetGrid(df, col="method", col_wrap=4, height=3.5, aspect=1.0)

    def plot_scatter_with_ref(x, y, **kwargs):
        sns.scatterplot(x=x, y=y, alpha=0.6, s=20, edgecolor=None, **kwargs)

        ax = plt.gca()
        ax.plot(
            [axis_limits[0], axis_limits[1]],
            [axis_limits[0], axis_limits[1]],
            ls="--",
            c="black",
            alpha=0.6,
            zorder=0,
        )

    g.map(plot_scatter_with_ref, "ate_true", "ate_pred", color="#2c7fb8")

    g.set_axis_labels("True ATE", "Predicted ATE")
    g.set_titles(col_template="{col_name}", fontweight="bold")
    g.set(xlim=axis_limits, ylim=axis_limits)

    plt.suptitle(
        "True vs. Predicted ATE across Episodes", y=1.02, fontsize=16, fontweight="bold"
    )

    output_file = Path(csv_path).parent / "plot_ate_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()

def plot_cate_r2_boxplot(csv_path: str):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    df_valid = df.dropna(subset=["cate_r2_shape"]).copy()

    if df_valid.empty:
        warnings.warn("No heterogeneity in treatment effect. Skipping CATE R2 boxplot.")
        return

    plt.figure(figsize=(10, 6))

    # Create the horizontal boxplot
    ax = sns.boxplot(
        data=df_valid,
        x="cate_r2_shape",
        y="method",
        palette="viridis",
    )

    # Add reference lines for R^2 = 0 (predicting the mean) and R^2 = 1 (perfect)
    ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="R² = 0 (Baseline)")
    ax.axvline(1, color="green", linestyle="--", alpha=0.6, label="R² = 1 (Perfect)")

    # Clip the X-axis so massive negative outliers don't squash the plot
    ax.set_xlim(-2.05, 1.05)

    plt.title("CATE Estimation: R² (Shape) Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("CATE R² (pure shape recovery, ATE bias removed)")
    plt.ylabel("")
    plt.legend(loc="lower left")

    output_file = Path(csv_path).parent / "plot_cate_r2_box.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()

def plot_single_episode_cate(exp_dir: Path, episode_idx: int = 0):
    # 1. Load the ground truth CATE for this specific episode
    data_path = exp_dir / "data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data.pt at {data_path}")
        
    data = torch.load(data_path, weights_only=False)
    true_cate = data["true_ITEs"][episode_idx]

    true_cate_eval = np.array(true_cate.cpu() if hasattr(true_cate, "cpu") else true_cate)
    if np.allclose(true_cate_eval, true_cate_eval[0]):
        warnings.warn(f"No heterogeneity in treatment effect for episode {episode_idx}. Skipping plot.")
        return

    preds_dir = exp_dir / "preds"
    records = []
    
    for npz_file in preds_dir.glob("*_preds.npz"):
        method_name = npz_file.stem.replace("_preds", "")
        method_preds = np.load(npz_file)["preds"]
        
        # Check if the episode failed (NaNs)
        if np.isnan(method_preds[episode_idx]).all():
            print(f"Skipping {method_name} - Episode {episode_idx} failed (NaN).")
            continue
            
        pred_cate = method_preds[episode_idx]
        
        for t, p in zip(true_cate, pred_cate):
            records.append({"method": method_name, "true_cate": t, "pred_cate": p})
            
    if not records:
        print("No valid predictions found to plot.")
        return
        
    df = pd.DataFrame(records)
    
    # 3. Set up the FacetGrid
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    min_val = min(df["true_cate"].min(), df["pred_cate"].min())
    max_val = max(df["true_cate"].max(), df["pred_cate"].max())
    pad = (max_val - min_val) * 0.1
    axis_limits = (min_val - pad, max_val + pad)
    
    g = sns.FacetGrid(df, col="method", col_wrap=4, height=3.5, aspect=1.0)
    
    def plot_scatter_with_ref(x, y, **kwargs):
        sns.scatterplot(x=x, y=y, alpha=0.8, s=40, edgecolor="w", **kwargs)
        # Reference line y=x
        ax = plt.gca()
        ax.plot([axis_limits[0], axis_limits[1]], 
                [axis_limits[0], axis_limits[1]], 
                ls="--", c="black", alpha=0.6, zorder=0)

    g.map(plot_scatter_with_ref, "true_cate", "pred_cate", color="#e6550d")
    
    g.set_axis_labels(f"True CATE (Ep {episode_idx})", "Predicted CATE")
    g.set_titles(col_template="{col_name}", fontweight="bold")
    g.set(xlim=axis_limits, ylim=axis_limits)
    
    plt.suptitle(f"CATE Estimation: Single Episode View (Index: {episode_idx})", 
                 y=1.02, fontsize=16, fontweight="bold")
    
    # 4. Save
    output_file = exp_dir / "results" / f"plot_cate_scatter_ep{episode_idx}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot True vs Predicted ATE")

    parser.add_argument("--exp", default="first_test", help="Experiment name")
    args = parser.parse_args()

    results_dir = Path("results") / args.exp / "results"
    full_csv_path = results_dir / "results.csv"
    
    # Existing Plots
    plot_true_vs_pred_ate(full_csv_path)
    plot_cate_r2_boxplot(full_csv_path)
    plot_ate_vs_cate_r2(results_dir)
    plot_ate_vs_cate_normrmse(results_dir)

    for i in range(5):
        plot_single_episode_cate(full_csv_path.parent.parent, episode_idx=i)