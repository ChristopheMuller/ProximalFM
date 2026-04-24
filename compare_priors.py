#%%

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- IMPORT YOUR PRIORS HERE ---
from proximalfm.prior import BNNPriorComplete
from proximalfm.Structure import ProximalPriorConfig, NetworkSpec

# ---------------------------------------------------------------------------
# 1. PyTorch Baseline (Stand-in for TabICL)
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """A standard MLP to serve as a deep learning baseline."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_pytorch_model(X_train, y_train, X_test, epochs=200, lr=0.01):
    """Trains the MLP and returns predictions on the test set."""
    # Convert to tensors
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    
    model = SimpleMLP(input_dim=X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).squeeze(1).numpy()
    return preds

# ---------------------------------------------------------------------------
# 2. Experiment Loop
# ---------------------------------------------------------------------------
def evaluate_prior(prior_generator, prior_name: str, sample_sizes: list[int]) -> pd.DataFrame:
    """Evaluates how well different models predict Y across different N."""
    print(f"\n--- Evaluating Prior: {prior_name} ---")
    results = []
    
    for n in sample_sizes:
        print(f"  Generating dataset with N={n}...")
        
        # 1. Generate Data
        episode = prior_generator.generate_episode(batch_size=n, episode_seed=42 + n)
        
        # We predict Y using all observable variables: X, Z, W, and Treatment A
        # (U is explicitly excluded because it is unobserved)
        features = torch.cat([episode.X, episode.Z, episode.W, episode.A.unsqueeze(1)], dim=1)
        target = episode.Y
        
        X_np = features.cpu().numpy()
        y_np = target.cpu().numpy()
        
        # 2. Split and Scale
        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Evaluate Models
        
        # Model A: Ordinary Least Squares (Linear Baseline)
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        ols_preds = ols.predict(X_test_scaled)
        ols_r2 = r2_score(y_test, ols_preds)
        results.append({"Prior": prior_name, "Model": "OLS", "N": n, "R2": ols_r2})
        
        # Model B: Random Forest (Non-linear Baseline)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_preds = rf.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_preds)
        results.append({"Prior": prior_name, "Model": "Random Forest", "N": n, "R2": rf_r2})
        
        # Model C: PyTorch MLP (Deep Learning / TabICL placeholder)
        mlp_preds = train_pytorch_model(X_train_scaled, y_train, X_test_scaled)
        mlp_r2 = r2_score(y_test, mlp_preds)
        results.append({"Prior": prior_name, "Model": "MLP (TabICL)", "N": n, "R2": mlp_r2})
        
        print(f"    OLS R2: {ols_r2:.3f} | RF R2: {rf_r2:.3f} | MLP R2: {mlp_r2:.3f}")
        
    return pd.DataFrame(results)


#%%
# ---------------------------------------------------------------------------
# 3. Main Execution & Visualization
# ---------------------------------------------------------------------------

sample_sizes = [250, 500, 1000, 2000, 5000]

# Setup 1: "Simple" Prior (Mostly Linear)
# We restrict the BNN layers to 1, meaning data generation is just a linear projection + noise
simple_config = ProximalPriorConfig()
simple_config.x_spec = NetworkSpec(min_layers=1, max_layers=1)
simple_config.z_spec = NetworkSpec(min_layers=1, max_layers=1)
simple_config.w_spec = NetworkSpec(min_layers=1, max_layers=1)
simple_config.outcome_spec = NetworkSpec(min_layers=1, max_layers=1)
simple_prior = BNNPriorComplete(config=simple_config, seed=100)

# Setup 2: "Complex" Prior (Highly Non-Linear)
# We force deep networks (3-5 layers) making it very hard for OLS
complex_config = ProximalPriorConfig()
complex_config.x_spec = NetworkSpec(min_layers=3, max_layers=5)
complex_config.z_spec = NetworkSpec(min_layers=3, max_layers=5)
complex_config.w_spec = NetworkSpec(min_layers=3, max_layers=5)
complex_config.outcome_spec = NetworkSpec(min_layers=3, max_layers=5)
complex_prior = BNNPriorComplete(config=complex_config, seed=200)

# Run Evaluations
df_simple = evaluate_prior(simple_prior, "Linear BNN Prior", sample_sizes)
df_complex = evaluate_prior(complex_prior, "Deep BNN Prior", sample_sizes)

df_all = pd.concat([df_simple, df_complex], ignore_index=True)

# Plotting
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot Simple Prior
sns.lineplot(
    data=df_all[df_all["Prior"] == "Linear BNN Prior"], 
    x="N", y="R2", hue="Model", marker="o", ax=axes[0], linewidth=2
)
axes[0].set_title("Data Generated by Linear Prior")
axes[0].set_xlabel("Sample Size (N)")
axes[0].set_ylabel("Predictive R² Score (Higher is better)")
axes[0].set_xscale("log") # Log scale helps visualize the data scaling
axes[0].set_xticks(sample_sizes)
axes[0].set_xticklabels(sample_sizes)

# Plot Complex Prior
sns.lineplot(
    data=df_all[df_all["Prior"] == "Deep BNN Prior"], 
    x="N", y="R2", hue="Model", marker="o", ax=axes[1], linewidth=2
)
axes[1].set_title("Data Generated by Deep BNN Prior")
axes[1].set_xlabel("Sample Size (N)")
axes[1].set_xscale("log")
axes[1].set_xticks(sample_sizes)
axes[1].set_xticklabels(sample_sizes)

plt.suptitle("Model Predictability vs Dataset Size", fontsize=16)
plt.tight_layout()
plt.savefig("prior_predictability_scaling.png", dpi=300)
print("\nSaved plot to 'prior_predictability_scaling.png'")
