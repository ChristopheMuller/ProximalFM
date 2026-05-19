#%%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Optional: Add root to path so you can run this script directly from the test_prior folder
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

from test_prior.utils import generate_eval_datasets

#%%
def plot_individual_treatment_distributions(datasets, k=5):
    """
    Plots the distribution of A_probs for the first k datasets,
    annotated with the empirical average of A=1.
    """
    print(f"Plotting individual treatment distributions for the first {k} datasets...")
    k = min(k, len(datasets))

    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]

    for i in range(k):
        df = datasets[i]
        a_probs = df["A_probs"]
        empirical_a_mean = df["A"].mean()

        sns.histplot(
            a_probs,
            bins=np.linspace(0, 1, 21),
            ax=axes[i],
            color="coral",
            kde=False,
            edgecolor="black",
            alpha=0.7,
        )

        axes[i].set_title(f"Dataset {i}")
        axes[i].set_xlabel(r"$P(A=1 \mid U, X, Z)$")
        axes[i].set_xlim(0, 1)

        # Annotate the empirical mean of A
        axes[i].text(
            0.5,
            0.9,
            f"Avg A=1: {empirical_a_mean:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i].transAxes,
            fontsize=11,
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.3"
            ),
        )

    axes[0].set_ylabel("Count")
    plt.suptitle(
        "Propensity Score Distributions (First $k$ Datasets)", y=1.05, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def plot_global_treatment_statistics(datasets):
    """
    Computes and plots the distribution of the mean, std, and skewness
    of the treatment probabilities and assignments across all datasets.
    """
    print("Computing global treatment statistics across all datasets...")

    records = []
    for i, df in enumerate(datasets):
        a_probs = df["A_probs"]
        a_assign = df["A"]

        records.append(
            {
                "dataset_idx": i,
                "mean_A": a_assign.mean(),
                "mean_A_probs": a_probs.mean(),
                "std_A_probs": a_probs.std(),
                "skew_A_probs": skew(a_probs),  # SciPy skewness
            }
        )

    stats_df = pd.DataFrame(records)

    # Setup a 2x2 grid for the macro distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of the Empirical Mean of A
    sns.histplot(
        stats_df["mean_A"], bins=30, kde=True, ax=axes[0, 0], color="mediumseagreen"
    )
    axes[0, 0].set_title(r"Distribution of Average $A=1$ ($\bar{A}$)")
    axes[0, 0].set_xlabel(r"Empirical $\bar{A}$")
    axes[0, 0].set_xlim(0, 1)

    # 2. Distribution of the Mean of A_probs
    sns.histplot(
        stats_df["mean_A_probs"], bins=30, kde=True, ax=axes[0, 1], color="teal"
    )
    axes[0, 1].set_title(r"Distribution of Mean Propensity ($E[A\_probs]$)")
    axes[0, 1].set_xlabel("Mean Propensity Score")
    axes[0, 1].set_xlim(0, 1)

    # 3. Distribution of the Standard Deviation of A_probs
    sns.histplot(
        stats_df["std_A_probs"], bins=30, kde=True, ax=axes[1, 0], color="darkorange"
    )
    axes[1, 0].set_title(r"Distribution of Propensity Spread ($Std(A\_probs)$)")
    axes[1, 0].set_xlabel("Standard Deviation")

    # 4. Distribution of the Skewness of A_probs
    sns.histplot(
        stats_df["skew_A_probs"], bins=30, kde=True, ax=axes[1, 1], color="mediumpurple"
    )
    axes[1, 1].set_title(r"Distribution of Propensity Skewness")
    axes[1, 1].set_xlabel("Fisher-Pearson Skewness")
    axes[1, 1].axvline(
        0, color="black", linestyle="--", alpha=0.5
    )  # Reference line for symmetry

    # Formatting
    for ax in axes.flatten():
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Frequency")

    plt.suptitle(
        "Global Treatment Distribution Statistics",
        y=1.02,
        fontweight="bold",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # Print a summary table to the console
    print("\n=== Summary of Global Treatment Statistics ===")
    summary = (
        stats_df[["mean_A", "mean_A_probs", "std_A_probs", "skew_A_probs"]].describe().T
    )
    print(
        summary[["mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
        .to_string()
    )


#%%

datasets = generate_eval_datasets(n_batches=10, batch_size=10)

plot_individual_treatment_distributions(datasets, k=5)
plot_global_treatment_statistics(datasets)
