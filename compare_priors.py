#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

from benchmarks.bench_methods import build_method

# ---------------------------------------------------------------------------
# 1. Config & Prior Definitions
# ---------------------------------------------------------------------------

def get_priors_to_test():
    """
    Define the priors and their configurations to test.
    This makes it easy to add, remove, or modify setups.
    """
    config1 = ProximalPriorConfig()
    config1.x_spec.min_layers, config1.x_spec.max_layers = 1, 1
    config1.z_spec.min_layers, config1.z_spec.max_layers = 1, 1
    config1.w_spec.min_layers, config1.w_spec.max_layers = 1, 1

    # config2 = ProximalPriorConfig()
    # config2.x_spec.min_layers, config2.x_spec.max_layers = 2, 2
    # config2.z_spec.min_layers, config2.z_spec.max_layers = 2, 2
    # config2.w_spec.min_layers, config2.w_spec.max_layers = 2, 2

    config3 = ProximalPriorConfig()
    config3.x_spec.min_layers, config3.x_spec.max_layers = 3, 3
    config3.z_spec.min_layers, config3.z_spec.max_layers = 3, 3
    config3.w_spec.min_layers, config3.w_spec.max_layers = 3, 3

    # config4 = ProximalPriorConfig()
    # config4.x_spec.min_layers, config4.x_spec.max_layers = 4, 4
    # config4.z_spec.min_layers, config4.z_spec.max_layers = 4, 4
    # config4.w_spec.min_layers, config4.w_spec.max_layers = 4, 4

    config10 = ProximalPriorConfig()
    config10.x_spec.min_layers, config10.x_spec.max_layers = 10, 10
    config10.z_spec.min_layers, config10.z_spec.max_layers = 10, 10
    config10.w_spec.min_layers, config10.w_spec.max_layers = 10, 10

    priors_to_evaluate = [
        (BNNPriorComplete, config1, "BNNPriorComplete (conf1)"),
        # (BNNPriorComplete, config2, "BNNPriorComplete (conf2)"),
        (BNNPriorComplete, config3, "BNNPriorComplete (conf3)"),
        # (BNNPriorComplete, config4, "BNNPriorComplete (conf4)"),
        (BNNPriorComplete, config10, "BNNPriorComplete (conf10)"),
    ]

    return priors_to_evaluate

# ---------------------------------------------------------------------------
# 2. Experiment Loop
# ---------------------------------------------------------------------------
def evaluate_prior(prior_generator, prior_name: str, sample_sizes: list[int], n_iterations: int = 5) -> pd.DataFrame:
    """Evaluates how well different models estimate ATE across different N, recording per-episode errors."""
    print(f"\n--- Evaluating Prior: {prior_name} ---")
    results = []
    
    # Define method configurations to be fed into build_method
    method_configs = [
        {"name": "S-Learner OLS (Interact)", "cls": "S_Learner", "base": "linear", "interactions": True, "use_W": True, "use_Z": True},
        {"name": "S-Learner RF", "cls": "S_Learner", "base": "rf", "interactions": False, "use_W": True, "use_Z": True},
        {"name": "T-Learner OLS", "cls": "T_Learner", "base": "linear", "use_W": True, "use_Z": True},
        {"name": "T-Learner RF", "cls": "T_Learner", "base": "rf", "use_W": True, "use_Z": True}
    ]

    for n in sample_sizes:
        print(f"  Sample size N={n}...")
        
        true_ates = []
        pred_ates = {cfg["name"]: [] for cfg in method_configs}
        
        for iteration in range(n_iterations):
            # 1. Generate Data
            n_train = n
            n_test = int(n * 0.25)
            episode = prior_generator.generate_episode(batch_size=n_train + n_test, episode_seed=42 + n + iteration * 100)

            # Extract arrays
            X = episode.X.cpu().numpy()
            W = episode.W.cpu().numpy()
            Z = episode.Z.cpu().numpy()
            A = episode.A.cpu().numpy().ravel()
            Y = episode.Y.cpu().numpy().ravel()
            Y1 = episode.Y1.cpu().numpy().ravel()
            Y0 = episode.Y0.cpu().numpy().ravel()

            # 2. Split Data
            indices = np.arange(len(X))
            idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42 + iteration)

            A_train, A_test = A[idx_train], A[idx_test]
            Y_train, Y_test = Y[idx_train], Y[idx_test]
            Y1_test, Y0_test = Y1[idx_test], Y0[idx_test]

            # Scale covariates individually
            scaler_X = StandardScaler()
            X_train_s = scaler_X.fit_transform(X[idx_train])
            X_test_s = scaler_X.transform(X[idx_test])

            scaler_W = StandardScaler()
            W_train_s = scaler_W.fit_transform(W[idx_train])
            W_test_s = scaler_W.transform(W[idx_test])

            scaler_Z = StandardScaler()
            Z_train_s = scaler_Z.fit_transform(Z[idx_train])
            Z_test_s = scaler_Z.transform(Z[idx_test])

            # Compute True ATE on test set for this episode
            true_ate = np.mean(Y1_test - Y0_test)
            true_ates.append(true_ate)

            # 3. Fit Models & Predict ATE
            for config in method_configs:
                model_name, estimator = build_method(config)
                
                estimator.fit(X_train_s, A_train, Y_train, W=W_train_s, Z=Z_train_s)
                
                pred_cate = estimator.predict(X_test_s, W=W_test_s, Z=Z_test_s)
                pred_ate = np.mean(pred_cate)
                
                pred_ates[model_name].append(pred_ate)

        # 4. Calculate metrics per iteration
        true_ates_arr = np.array(true_ates)
        y_range = np.max(true_ates_arr) - np.min(true_ates_arr)
        if y_range == 0: y_range = 1e-9 # Prevent division by zero

        for model_name, preds in pred_ates.items():
            preds_arr = np.array(preds)
            
            # R2 requires the whole variance array, calculated once per N
            if len(np.unique(true_ates_arr)) > 1:
                r2 = r2_score(true_ates_arr, preds_arr)
            else:
                r2 = np.nan
                
            # Episode-level errors
            abs_errors = np.abs(true_ates_arr - preds_arr)
            norm_abs_errors = abs_errors / y_range

            for i in range(n_iterations):
                results.append({
                    "Prior": prior_name,
                    "Model": model_name,
                    "N": n,
                    "Iteration": i,
                    "Absolute Error": abs_errors[i],
                    "Normalized Absolute Error": norm_abs_errors[i],
                    "R2": r2
                })
        
    return pd.DataFrame(results)

#%%
# ---------------------------------------------------------------------------
# 3. Main Execution & Visualization
# ---------------------------------------------------------------------------

sample_sizes = [250, 500, 1000]
n_iterations = 30

priors_setup = get_priors_to_test()
all_results = []

# Run Evaluations
for PriorClass, config, name in priors_setup:
    prior_instance = PriorClass(config=config)
    df_result = evaluate_prior(prior_instance, name, sample_sizes, n_iterations=n_iterations)
    all_results.append(df_result)

df_all = pd.concat(all_results, ignore_index=True)

#%%
# Plotting
sns.set_theme(style="whitegrid")

# Define metrics and their plot types
metrics_info = {
    "R2": {"title": "Predictive R² Score (Higher is better)", "type": "line"},
    "Normalized Absolute Error": {"title": "Normalized Absolute Error per Episode (Lower is better)", "type": "box"},
    "Absolute Error": {"title": "Absolute Error per Episode (Lower is better)", "type": "box"}
}

unique_priors = df_all["Prior"].unique()
num_priors = len(unique_priors)

for metric, info in metrics_info.items():
    metric_title = info["title"]
    plot_type = info["type"]

    cols = 4
    rows = int(np.ceil(num_priors / cols))

    min_y = df_all[metric].dropna().min()
    max_y = df_all[metric].dropna().max()
    y_margin = (max_y - min_y) * 0.1

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharex=True)
    axes = np.array(axes).flatten()

    for i, prior_name in enumerate(unique_priors):
        ax = axes[i]
        data_subset = df_all[df_all["Prior"] == prior_name]

        if plot_type == "line":
            sns.lineplot(
                data=data_subset,
                x="N", y=metric, hue="Model", marker="o", ax=ax, linewidth=2
            )
            ax.set_xscale("log")
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels(sample_sizes)
        elif plot_type == "box":
            sns.boxplot(
                data=data_subset,
                x="N", y=metric, hue="Model", ax=ax,
                fliersize=3, linewidth=1.5
            )
            # Boxplots treat `x` categorically, so log scaling x-axis breaks them

        ax.set_title(f"{prior_name}")
        ax.set_xlabel("Sample Size (N)")
        ax.set_ylabel(metric_title)
        ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"ATE Estimation {metric} vs Dataset Size", fontsize=16)
    plt.tight_layout()
    
    filename = f"ate_prior_predictability_{metric.replace(' ', '_')}.png"
    # plt.savefig(filename, dpi=300)
    print(f"\nSaved {metric} plot to '{filename}'")

#%%