import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


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

    # Filter out homogeneous episodes where CATE R2 is undefined/NaN
    df_valid = df.dropna(subset=["cate_r2_shape"]).copy()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot True vs Predicted ATE")

    parser.add_argument("--exp", default="first_test", help="Experiment name")
    args = parser.parse_args()

    full_csv_path = Path("results") / args.exp / "results" / "results.csv"
    plot_true_vs_pred_ate(full_csv_path)
    plot_cate_r2_boxplot(full_csv_path)
