#%%

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabicl import TabICLRegressor

# --- IMPORT YOUR PRIORS HERE ---
from proximalfm.prior import (
    BNNPrior,
    BNNPriorComplete,
    LinearPrior,
    ConfoundingLinearPrior,
    CATELinearPrior,
    FungibleSmall,
)
from proximalfm.Structure import ProximalPriorConfig, NetworkSpec

# ---------------------------------------------------------------------------
# 1. Config & Prior Definitions
# ---------------------------------------------------------------------------

def get_priors_to_test():
    """
    Define the priors and their configurations to test.
    This makes it easy to add, remove, or modify setups.
    """

    # Example Configs
    default_config = ProximalPriorConfig()

    linear_config = ProximalPriorConfig()
    linear_config.x_spec = NetworkSpec(min_layers=1, max_layers=1)
    linear_config.z_spec = NetworkSpec(min_layers=1, max_layers=1)
    linear_config.w_spec = NetworkSpec(min_layers=1, max_layers=1)
    linear_config.outcome_spec = NetworkSpec(min_layers=1, max_layers=1)
    
    complex_config = ProximalPriorConfig()
    complex_config.x_spec = NetworkSpec(min_layers=3, max_layers=5)
    complex_config.z_spec = NetworkSpec(min_layers=3, max_layers=5)
    complex_config.w_spec = NetworkSpec(min_layers=3, max_layers=5)
    complex_config.outcome_spec = NetworkSpec(min_layers=3, max_layers=5)
    
    noisy_config = ProximalPriorConfig(noise_scale_x=0.5, noise_scale_z=0.5)

    # Return a list of tuples: (PriorClass, config, name_string)
    priors_to_evaluate = [
        # BNN Priors
        (BNNPriorComplete, linear_config, "BNNPriorComplete (Linear)"),
        (BNNPriorComplete, complex_config, "BNNPriorComplete (Deep)"),
        (BNNPrior, default_config, "BNNPrior (Default)"),

        # Linear Priors
        (LinearPrior, default_config, "LinearPrior (Default)"),
        (LinearPrior, noisy_config, "LinearPrior (High Noise)"),
        
        # Other Priors
        (ConfoundingLinearPrior, default_config, "ConfoundingLinearPrior"),
        (CATELinearPrior, default_config, "CATELinearPrior"),
        (FungibleSmall, default_config, "FungibleSmall"),
    ]

    return priors_to_evaluate

# ---------------------------------------------------------------------------
# 2. Experiment Loop
# ---------------------------------------------------------------------------
def evaluate_prior(prior_generator, prior_name: str, sample_sizes: list[int], n_iterations: int = 5) -> pd.DataFrame:
    """Evaluates how well different models predict Y across different N, averaged over iterations."""
    print(f"\n--- Evaluating Prior: {prior_name} ---")
    results = []
    
    for n in sample_sizes:
        print(f"  Sample size N={n}...")
        for iteration in range(n_iterations):
            # 1. Generate Data
            episode = prior_generator.generate_episode(batch_size=n, episode_seed=42 + n + iteration * 100)

            # We predict Y using all observable variables: X, Z, W, and Treatment A
            # (U is explicitly excluded because it is unobserved)
            features = torch.cat([episode.X, episode.Z, episode.W, episode.A.unsqueeze(1)], dim=1)
            target = episode.Y

            X_np = features.cpu().numpy()
            y_np = target.cpu().numpy()

            # 2. Split and Scale
            X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42 + iteration)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models
            models = {
                "OLS": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=10, random_state=42 + iteration), # Reduced estimators heavily for speed
                "TabICL": TabICLRegressor()
            }

            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                # Calculate metrics
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                # NRMSE (Normalized RMSE) - Normalized by range
                y_range = np.max(y_test) - np.min(y_test)
                nrmse = rmse / y_range if y_range > 0 else 0

                results.append({
                    "Prior": prior_name,
                    "Model": model_name,
                    "N": n,
                    "Iteration": iteration,
                    "MSE": mse,
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "MAE": mae,
                    "R2": r2
                })
        
    return pd.DataFrame(results)

#%%
# ---------------------------------------------------------------------------
# 3. Main Execution & Visualization
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_sizes = [250, 500, 1000] # Adjusted max size slightly for TabICL runtime
    n_iterations = 5 # Number of repeats for confidence intervals

    priors_setup = get_priors_to_test()
    all_results = []

    # Run Evaluations
    for PriorClass, config, name in priors_setup:
        prior_instance = PriorClass(config=config)
        df_result = evaluate_prior(prior_instance, name, sample_sizes, n_iterations=n_iterations)
        all_results.append(df_result)

    df_all = pd.concat(all_results, ignore_index=True)

    # Plotting
    sns.set_theme(style="whitegrid")

    metrics = ["R2", "NRMSE", "MAE", "MSE"]
    metrics_titles = ["Predictive R² Score (Higher is better)", "Normalized RMSE (Lower is better)", "Mean Absolute Error (Lower is better)", "Mean Squared Error (Lower is better)"]

    unique_priors = df_all["Prior"].unique()
    num_priors = len(unique_priors)

    # For each metric, create a plot with a subplot for each prior
    for metric, metric_title in zip(metrics, metrics_titles):
        # We'll make a grid: say 2 columns, ceil(num_priors / 2) rows
        cols = 2
        rows = int(np.ceil(num_priors / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharex=True)
        axes = np.array(axes).flatten() # Make sure it's 1D for easy indexing

        for i, prior_name in enumerate(unique_priors):
            ax = axes[i]
            data_subset = df_all[df_all["Prior"] == prior_name]

            # sns.lineplot automatically plots mean and 95% CI when multiple values exist per x
            sns.lineplot(
                data=data_subset,
                x="N", y=metric, hue="Model", marker="o", ax=ax, linewidth=2
            )
            ax.set_title(f"{prior_name}")
            ax.set_xlabel("Sample Size (N)")
            ax.set_ylabel(metric_title)
            ax.set_xscale("log")
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels(sample_sizes)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Model {metric} vs Dataset Size", fontsize=16)
        plt.tight_layout()
        filename = f"prior_predictability_{metric}.png"
        plt.savefig(filename, dpi=300)
        print(f"\nSaved {metric} plot to '{filename}'")

