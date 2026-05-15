#%%

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score

from proximalfm.prior import PriorDataset


#%% ---------------------------------------------------------------------
# 1. Instantiate Prior Dataset
# ----------------------------------------------------------------------

dataset = PriorDataset(
    batch_size=5,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=1,
    min_dim_Z=1,
    max_dim_Z=1,
    min_dim_X=2,
    max_dim_X=2,
    min_seq_len=None,
    max_seq_len=512,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.5,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)


#%% ---------------------------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------------------------


def as_2d(v):
    v = np.asarray(v)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    return v


def safe_target(y):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    return y


def safe_standardize_np(x, eps=1e-5):
    x = as_2d(x)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mu) / std


def standardize_targets_from_context(y_obs, y0, y1, train_size, eps=1e-5):
    y_obs = as_2d(y_obs)
    y0 = as_2d(y0)
    y1 = as_2d(y1)

    train_size = max(1, min(int(train_size), len(y_obs)))

    mu = y_obs[:train_size].mean(axis=0, keepdims=True)
    std = y_obs[:train_size].std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    return (y_obs - mu) / std, (y0 - mu) / std, (y1 - mu) / std


def safe_corr(x, y, eps=1e-8):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        return np.nan
    if np.std(x) < eps or np.std(y) < eps:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def print_summary_stats(name, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    print(f"\n--- {name} ---")

    if len(values) == 0:
        print("  No finite values.")
        return

    print(f"  Mean : {np.mean(values): .4f}")
    print(f"  Min  : {np.min(values): .4f}")
    print(f"  25%  : {np.percentile(values, 25): .4f}")
    print(f"  50%  : {np.median(values): .4f}")
    print(f"  75%  : {np.percentile(values, 75): .4f}")
    print(f"  Max  : {np.max(values): .4f}")


def rf_r2(features, target, random_state=42):
    features = as_2d(features)
    target = safe_target(target)

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        random_state=random_state,
    )

    rf.fit(features, target)
    pred = rf.predict(features)

    return r2_score(target, pred), pred


def delta_r2(target, base_features, extra_features, random_state=42):
    base_features = as_2d(base_features)
    extra_features = as_2d(extra_features)

    full_features = np.concatenate([base_features, extra_features], axis=1)

    r2_base, pred_base = rf_r2(base_features, target, random_state=random_state)
    r2_full, pred_full = rf_r2(full_features, target, random_state=random_state)

    return r2_base, r2_full, r2_full - r2_base, pred_base, pred_full


def rf_auc(features, a, random_state=42):
    features = as_2d(features)
    a = np.asarray(a).ravel().astype(int)

    if len(np.unique(a)) < 2:
        return np.nan, np.full(len(a), np.nan)

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        random_state=random_state,
    )

    clf.fit(features, a.ravel())

    if clf.predict_proba(features).shape[1] < 2:
        return np.nan, np.full(len(a), np.nan)

    p = clf.predict_proba(features)[:, 1]
    auc = roc_auc_score(a, p)

    return auc, p


def delta_auc(a, base_features, extra_features, random_state=42):
    base_features = as_2d(base_features)
    extra_features = as_2d(extra_features)

    full_features = np.concatenate([base_features, extra_features], axis=1)

    auc_base, p_base = rf_auc(base_features, a, random_state=random_state)
    auc_full, p_full = rf_auc(full_features, a, random_state=random_state)

    return auc_base, auc_full, auc_full - auc_base, p_base, p_full


def smd_by_treatment(values, a, eps=1e-8):
    """
    Standardized mean difference of values between A=1 and A=0.
    For multidimensional values, returns RMS SMD across dimensions.
    """
    values = as_2d(values)
    a = np.asarray(a).ravel().astype(int)

    v1 = values[a == 1]
    v0 = values[a == 0]

    if len(v1) < 2 or len(v0) < 2:
        return np.nan

    pooled_sd = np.sqrt(0.5 * (v1.var(axis=0, ddof=1) + v0.var(axis=0, ddof=1)) + eps)

    smd = (v1.mean(axis=0) - v0.mean(axis=0)) / pooled_sd

    return float(np.sqrt(np.mean(smd**2)))


def residualize(target, features, random_state=42):
    target = as_2d(target)
    features = as_2d(features)

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        random_state=random_state,
    )

    rf.fit(features, target.ravel())
    pred = as_2d(rf.predict(features))

    return target - pred


def estimate_backdoor_ate(y_obs, a_1d, x_2d, u_2d=None, random_state=42):
    """
    Estimate ATE by fitting E[Y | A, X] or E[Y | A, X, U],
    then predicting under A=1 and A=0.
    """
    y_obs = safe_target(y_obs)
    a_2d = a_1d.reshape(-1, 1)

    if u_2d is None:
        features = np.concatenate([a_2d, x_2d], axis=1)
        features_1 = np.concatenate([np.ones_like(a_2d), x_2d], axis=1)
        features_0 = np.concatenate([np.zeros_like(a_2d), x_2d], axis=1)
    else:
        features = np.concatenate([a_2d, x_2d, u_2d], axis=1)
        features_1 = np.concatenate([np.ones_like(a_2d), x_2d, u_2d], axis=1)
        features_0 = np.concatenate([np.zeros_like(a_2d), x_2d, u_2d], axis=1)

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        n_jobs=-1,
        random_state=random_state,
    )

    rf.fit(features, y_obs)

    mu1 = rf.predict(features_1)
    mu0 = rf.predict(features_0)

    return float(np.mean(mu1 - mu0))


def safe_kdeplot(values, ax, label=None, color=None, linestyle="-", fill=True):
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]

    if len(values) < 3 or np.std(values) < 1e-8:
        if len(values) > 0:
            ax.axvline(np.mean(values), label=label, color=color, linestyle=linestyle)
        return

    sns.kdeplot(
        values,
        ax=ax,
        label=label,
        color=color,
        linestyle=linestyle,
        fill=fill,
    )


#%% ---------------------------------------------------------------------
# 3. Generate and Normalize Datasets
# ----------------------------------------------------------------------

n_batches = 10
iterator = iter(dataset)

print(f"Generating {n_batches * dataset.batch_size} datasets...")

all_datasets = []

for _ in tqdm(range(n_batches), desc="Sampling Batches"):
    batch = next(iterator)

    U, X, W, Z, A, Y_obs, Y0, Y1 = batch[:8]
    seq_lens, train_sizes = batch[-2:]

    for i in range(dataset.batch_size):
        seq_len = int(seq_lens[i].item())
        train_size = int(train_sizes[i].item())
        train_size = max(1, min(train_size, seq_len))

        u_raw = U[i, :seq_len].cpu().numpy()
        x_raw = X[i, :seq_len].cpu().numpy()
        w_raw = W[i, :seq_len].cpu().numpy()
        z_raw = Z[i, :seq_len].cpu().numpy()
        a_raw = A[i, :seq_len].cpu().numpy()
        y_obs_raw = Y_obs[i, :seq_len].cpu().numpy()
        y0_raw = Y0[i, :seq_len].cpu().numpy()
        y1_raw = Y1[i, :seq_len].cpu().numpy()

        u = safe_standardize_np(u_raw)
        x = safe_standardize_np(x_raw)
        w = safe_standardize_np(w_raw)
        z = safe_standardize_np(z_raw)

        y_obs_std, y0_std, y1_std = standardize_targets_from_context(
            y_obs_raw,
            y0_raw,
            y1_raw,
            train_size=train_size,
        )

        ds = (
            u,
            x,
            w,
            z,
            a_raw,
            y_obs_std,
            y0_std,
            y1_std,
            seq_len,
            train_size,
        )

        all_datasets.append(ds)


#%% ---------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------

metrics = {
    # Structural signals
    "Signal: U -> Z (R2)": [],
    "Signal: U -> W (R2)": [],
    "Signal: U -> A (AUC)": [],
    "Marginal Signal: A -> Y (R2)": [],
    # Proxy leakage / exclusion checks
    "Leakage Delta: Z _|_ W | (X, U)": [],
    "Leakage Delta: Y _|_ Z | (A, X, U)": [],
    "Leakage Delta: W _|_ (A, Z) | (X, U)": [],
    # ATE diagnostics
    "True ATE": [],
    "Naive ATE": [],
    "Oracle ATE": [],
    "Std of Y_obs": [],
    "Confounding Bias (Naive ATE - True ATE)": [],
    "Oracle Bias (Oracle ATE - True ATE)": [],
    "Confounding Bias (Naive ATE - True ATE) Normalized": [],
    "Oracle Bias (Oracle ATE - True ATE) Normalized": [],
    "Abs Bias Reduction from U": [],
    "Relative Abs Bias Reduction from U": [],
    # Main weak-confounding diagnostics
    "Outcome Variance Gap (Delta R2_Y)": [],
    "Treatment Selection Gap (Delta AUC_A)": [],
    "Treatment U Gap over X (Delta AUC)": [],
    "Treatment U Gap over X,Z (Delta AUC)": [],
    "Raw U Imbalance by A (SMD)": [],
    "Residual U Imbalance by A after X (SMD)": [],
    "Outcome U Gap over A,X (Delta R2)": [],
    # Potential-outcome diagnostics
    "Y0 U Gap over X (Delta R2)": [],
    "Y1 U Gap over X (Delta R2)": [],
    "Tau U Gap over X (Delta R2)": [],
    # Ground-truth decomposition
    "GT Observed Contrast Bias Normalized": [],
    "GT Baseline Imbalance Bias Normalized": [],
    "GT Effect-Selection Bias Normalized": [],
    # Is U really hidden?
    "U Predictable from X (R2)": [],
    "U Predictable from X,Z,W (R2)": [],
    "U Added Predictability from Z,W over X (Delta R2)": [],
    # Alignment diagnostics
    "Confounding Alignment Corr": [],
    "Confounding Alignment Product": [],
}


#%% ---------------------------------------------------------------------
# 5. Evaluate Per Dataset
# ----------------------------------------------------------------------

print(f"\nEvaluating {len(all_datasets)} stored datasets...")

for ds in tqdm(all_datasets, desc="Evaluating Datasets"):
    u, x, w, z, a, y_obs, y0, y1, seq_len, train_size = ds

    u_2d = as_2d(u)
    x_2d = as_2d(x)
    w_2d = as_2d(w)
    z_2d = as_2d(z)

    a_1d = np.asarray(a).ravel().astype(int)
    a_2d = a_1d.reshape(-1, 1)

    y_obs_flat = safe_target(y_obs).ravel()
    y0_flat = safe_target(y0).ravel()
    y1_flat = safe_target(y1).ravel()
    tau_flat = y1_flat - y0_flat

    features_X = x_2d
    features_XU = np.concatenate([x_2d, u_2d], axis=1)
    features_AX = np.concatenate([a_2d, x_2d], axis=1)
    features_AXU = np.concatenate([a_2d, x_2d, u_2d], axis=1)
    features_XZ = np.concatenate([x_2d, z_2d], axis=1)
    features_XZU = np.concatenate([x_2d, z_2d, u_2d], axis=1)
    features_XZW = np.concatenate([x_2d, z_2d, w_2d], axis=1)
    features_XZUW = np.concatenate([x_2d, z_2d, u_2d, w_2d], axis=1)

    # ------------------------------------------------------------
    # 5.1 Structural signal strength
    # ------------------------------------------------------------

    r2_u_z, _ = rf_r2(u_2d, z_2d)
    r2_u_w, _ = rf_r2(u_2d, w_2d)
    auc_u_a, _ = rf_auc(u_2d, a_1d)
    r2_a_y, _ = rf_r2(a_2d, y_obs_flat)

    metrics["Signal: U -> Z (R2)"].append(r2_u_z)
    metrics["Signal: U -> W (R2)"].append(r2_u_w)
    metrics["Signal: U -> A (AUC)"].append(auc_u_a)
    metrics["Marginal Signal: A -> Y (R2)"].append(r2_a_y)

    # ------------------------------------------------------------
    # 5.2 Proxy leakage / exclusion diagnostics
    # ------------------------------------------------------------

    base_XU = np.concatenate([x_2d, u_2d], axis=1)
    base_AXU = np.concatenate([a_2d, x_2d, u_2d], axis=1)

    _, _, delta_zw, _, _ = delta_r2(
        target=w_2d,
        base_features=base_XU,
        extra_features=z_2d,
    )
    metrics["Leakage Delta: Z _|_ W | (X, U)"].append(delta_zw)

    _, _, delta_yz, _, _ = delta_r2(
        target=y_obs_flat,
        base_features=base_AXU,
        extra_features=z_2d,
    )
    metrics["Leakage Delta: Y _|_ Z | (A, X, U)"].append(delta_yz)

    az_concat = np.concatenate([a_2d, z_2d], axis=1)
    _, _, delta_waz, _, _ = delta_r2(
        target=w_2d,
        base_features=base_XU,
        extra_features=az_concat,
    )
    metrics["Leakage Delta: W _|_ (A, Z) | (X, U)"].append(delta_waz)

    # ------------------------------------------------------------
    # 5.3 ATE estimation
    # ------------------------------------------------------------

    true_ate = float(np.mean(tau_flat))

    naive_ate = estimate_backdoor_ate(
        y_obs=y_obs_flat,
        a_1d=a_1d,
        x_2d=x_2d,
        u_2d=None,
    )

    oracle_ate = estimate_backdoor_ate(
        y_obs=y_obs_flat,
        a_1d=a_1d,
        x_2d=x_2d,
        u_2d=u_2d,
    )

    y_std = float(np.std(y_obs_flat) + 1e-8)

    naive_bias = naive_ate - true_ate
    oracle_bias = oracle_ate - true_ate

    naive_bias_norm = naive_bias / y_std
    oracle_bias_norm = oracle_bias / y_std

    abs_bias_reduction = np.abs(naive_bias_norm) - np.abs(oracle_bias_norm)
    relative_abs_bias_reduction = abs_bias_reduction / (np.abs(naive_bias_norm) + 1e-8)

    metrics["True ATE"].append(true_ate)
    metrics["Naive ATE"].append(naive_ate)
    metrics["Oracle ATE"].append(oracle_ate)
    metrics["Std of Y_obs"].append(y_std)

    metrics["Confounding Bias (Naive ATE - True ATE)"].append(naive_bias)
    metrics["Oracle Bias (Oracle ATE - True ATE)"].append(oracle_bias)
    metrics["Confounding Bias (Naive ATE - True ATE) Normalized"].append(
        naive_bias_norm
    )
    metrics["Oracle Bias (Oracle ATE - True ATE) Normalized"].append(oracle_bias_norm)
    metrics["Abs Bias Reduction from U"].append(abs_bias_reduction)
    metrics["Relative Abs Bias Reduction from U"].append(relative_abs_bias_reduction)

    # ------------------------------------------------------------
    # 5.4 Treatment-side confounding strength
    # ------------------------------------------------------------

    auc_x, auc_xu, delta_auc_xu, p_x, p_xu = delta_auc(
        a=a_1d,
        base_features=x_2d,
        extra_features=u_2d,
    )

    auc_xz, auc_xzu, delta_auc_xzu, p_xz, p_xzu = delta_auc(
        a=a_1d,
        base_features=features_XZ,
        extra_features=u_2d,
    )

    metrics["Treatment U Gap over X (Delta AUC)"].append(delta_auc_xu)
    metrics["Treatment U Gap over X,Z (Delta AUC)"].append(delta_auc_xzu)
    metrics["Treatment Selection Gap (Delta AUC_A)"].append(delta_auc_xzu)

    metrics["Raw U Imbalance by A (SMD)"].append(smd_by_treatment(u_2d, a_1d))

    u_resid_x = residualize(u_2d, x_2d)

    metrics["Residual U Imbalance by A after X (SMD)"].append(
        smd_by_treatment(u_resid_x, a_1d)
    )

    # ------------------------------------------------------------
    # 5.5 Outcome-side confounding strength
    # ------------------------------------------------------------

    r2_y_ax, r2_y_axu, delta_r2_y_axu, yhat_ax, yhat_axu = delta_r2(
        target=y_obs_flat,
        base_features=features_AX,
        extra_features=u_2d,
    )

    metrics["Outcome U Gap over A,X (Delta R2)"].append(delta_r2_y_axu)
    metrics["Outcome Variance Gap (Delta R2_Y)"].append(delta_r2_y_axu)

    _, _, delta_r2_y0_xu, _, _ = delta_r2(
        target=y0_flat,
        base_features=x_2d,
        extra_features=u_2d,
    )

    _, _, delta_r2_y1_xu, _, _ = delta_r2(
        target=y1_flat,
        base_features=x_2d,
        extra_features=u_2d,
    )

    _, _, delta_r2_tau_xu, _, _ = delta_r2(
        target=tau_flat,
        base_features=x_2d,
        extra_features=u_2d,
    )

    metrics["Y0 U Gap over X (Delta R2)"].append(delta_r2_y0_xu)
    metrics["Y1 U Gap over X (Delta R2)"].append(delta_r2_y1_xu)
    metrics["Tau U Gap over X (Delta R2)"].append(delta_r2_tau_xu)

    # ------------------------------------------------------------
    # 5.6 Ground-truth confounding decomposition
    # ------------------------------------------------------------

    if np.sum(a_1d == 1) > 0 and np.sum(a_1d == 0) > 0:
        obs_contrast = y_obs_flat[a_1d == 1].mean() - y_obs_flat[a_1d == 0].mean()

        gt_obs_bias = obs_contrast - true_ate

        baseline_imbalance_bias = y0_flat[a_1d == 1].mean() - y0_flat[a_1d == 0].mean()

        effect_selection_bias = tau_flat[a_1d == 1].mean() - tau_flat.mean()

        metrics["GT Observed Contrast Bias Normalized"].append(gt_obs_bias / y_std)
        metrics["GT Baseline Imbalance Bias Normalized"].append(
            baseline_imbalance_bias / y_std
        )
        metrics["GT Effect-Selection Bias Normalized"].append(
            effect_selection_bias / y_std
        )
    else:
        metrics["GT Observed Contrast Bias Normalized"].append(np.nan)
        metrics["GT Baseline Imbalance Bias Normalized"].append(np.nan)
        metrics["GT Effect-Selection Bias Normalized"].append(np.nan)

    # ------------------------------------------------------------
    # 5.7 Is U really hidden?
    # ------------------------------------------------------------

    r2_u_x, _ = rf_r2(x_2d, u_2d)
    r2_u_xzw, _ = rf_r2(features_XZW, u_2d)

    metrics["U Predictable from X (R2)"].append(r2_u_x)
    metrics["U Predictable from X,Z,W (R2)"].append(r2_u_xzw)
    metrics["U Added Predictability from Z,W over X (Delta R2)"].append(
        r2_u_xzw - r2_u_x
    )

    # ------------------------------------------------------------
    # 5.8 Alignment of omitted treatment and outcome components
    # ------------------------------------------------------------

    delta_propensity_u = np.asarray(p_xu).ravel() - np.asarray(p_x).ravel()
    delta_outcome_u = np.asarray(yhat_axu).ravel() - np.asarray(yhat_ax).ravel()

    metrics["Confounding Alignment Corr"].append(
        safe_corr(delta_propensity_u, delta_outcome_u)
    )

    metrics["Confounding Alignment Product"].append(
        np.nanmean(delta_propensity_u * delta_outcome_u) / y_std
    )


#%% ---------------------------------------------------------------------
# 6. Summary Report
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("  PROXIMAL PRIOR EVALUATION REPORT")
print("  Total Datasets Evaluated:", len(all_datasets))
print("=" * 70)

for metric_name, values in metrics.items():
    print_summary_stats(metric_name, values)

df_metrics = pd.DataFrame(metrics)


#%% ---------------------------------------------------------------------
# 7. Core Diagnostic Plots
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

sns.kdeplot(
    df_metrics["Confounding Bias (Naive ATE - True ATE) Normalized"],
    ax=axes[0],
    color="crimson",
    fill=True,
    label="Naive Bias, missing U",
)

sns.kdeplot(
    df_metrics["Oracle Bias (Oracle ATE - True ATE) Normalized"],
    ax=axes[0],
    color="mediumseagreen",
    fill=True,
    label="Oracle Bias, with U",
)

axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.8)
axes[0].set_title(
    "Backdoor Adjustment Effect\nStandardized ATE Error",
    fontweight="bold",
)
axes[0].set_xlabel("Bias Magnitude, Standard Deviations")
axes[0].legend()


sns.kdeplot(
    np.abs(df_metrics["Confounding Bias (Naive ATE - True ATE) Normalized"]),
    ax=axes[1],
    color="crimson",
    fill=True,
    label="Abs Naive Bias",
)

sns.kdeplot(
    np.abs(df_metrics["Oracle Bias (Oracle ATE - True ATE) Normalized"]),
    ax=axes[1],
    color="mediumseagreen",
    fill=True,
    label="Abs Oracle Bias",
)

axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.8)
axes[1].set_title(
    "Absolute Backdoor Adjustment Effect\nStandardized ATE Error",
    fontweight="bold",
)
axes[1].set_xlabel("Absolute Bias Magnitude, Standard Deviations")
axes[1].set_xlim(0, None)
axes[1].legend()


sns.histplot(
    df_metrics["Outcome U Gap over A,X (Delta R2)"],
    ax=axes[2],
    color="darkorange",
    bins=20,
    kde=True,
)

axes[2].axvline(x=0, color="black", linestyle="--", alpha=0.8)
axes[2].set_title(
    "Outcome Variance Gap\nAdditional $R^2$ from U over A,X",
    fontweight="bold",
)
axes[2].set_xlabel("Additional $R^2$ from U")


sns.histplot(
    df_metrics["Treatment U Gap over X (Delta AUC)"],
    ax=axes[3],
    color="darkviolet",
    bins=20,
    kde=True,
)

axes[3].axvline(x=0, color="black", linestyle="--", alpha=0.8)
axes[3].set_title(
    "Treatment Selection Gap\nAdditional AUC from U over X",
    fontweight="bold",
)
axes[3].set_xlabel("Additional AUC from U")

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------
# 8. Why Is Confounding Weak? Scatter Diagnostics
# ----------------------------------------------------------------------

bias_abs = np.abs(
    df_metrics["Confounding Bias (Naive ATE - True ATE) Normalized"].to_numpy()
)

diagnostic_xs = [
    "Treatment U Gap over X (Delta AUC)",
    "Residual U Imbalance by A after X (SMD)",
    "Outcome U Gap over A,X (Delta R2)",
    "GT Baseline Imbalance Bias Normalized",
    "GT Effect-Selection Bias Normalized",
    "U Predictable from X (R2)",
    "U Predictable from X,Z,W (R2)",
    "Confounding Alignment Corr",
]

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
axes = axes.flatten()

for ax, name in zip(axes, diagnostic_xs):
    xvals = df_metrics[name].to_numpy(dtype=float)

    mask = np.isfinite(xvals) & np.isfinite(bias_abs)

    sns.scatterplot(
        x=xvals[mask],
        y=bias_abs[mask],
        ax=ax,
        alpha=0.8,
    )

    ax.set_title(name, fontweight="bold")
    ax.set_xlabel(name)
    ax.set_ylabel("|Naive ATE Error|, normalized")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------
# 9. Metric Histograms
# ----------------------------------------------------------------------

hist_metrics = [
    "Signal: U -> Z (R2)",
    "Signal: U -> W (R2)",
    "Signal: U -> A (AUC)",
    "Marginal Signal: A -> Y (R2)",
    "Treatment U Gap over X (Delta AUC)",
    "Treatment U Gap over X,Z (Delta AUC)",
    "Raw U Imbalance by A (SMD)",
    "Residual U Imbalance by A after X (SMD)",
    "Outcome U Gap over A,X (Delta R2)",
    "Y0 U Gap over X (Delta R2)",
    "Y1 U Gap over X (Delta R2)",
    "Tau U Gap over X (Delta R2)",
    "U Predictable from X (R2)",
    "U Predictable from X,Z,W (R2)",
    "Confounding Alignment Corr",
]

ncols = 3
nrows = int(np.ceil(len(hist_metrics) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
axes = axes.flatten()

for ax, name in zip(axes, hist_metrics):
    values = df_metrics[name].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    sns.histplot(values, ax=ax, bins=20, kde=True, color="royalblue")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    if "R2" in name or "AUC" in name:
        ax.axvline(x=1, color="red", linestyle="--", alpha=0.4)

    ax.set_title(name, fontweight="bold")
    ax.set_ylabel("Count")

for j in range(len(hist_metrics), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------
# 10. Distribution of True ATE
# ----------------------------------------------------------------------

ate_values = df_metrics["True ATE"].to_numpy(dtype=float) / (
    df_metrics["Std of Y_obs"].to_numpy(dtype=float) + 1e-8
)

plt.figure(figsize=(10, 6))
sns.histplot(ate_values, bins=20, kde=True, color="teal")
plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Zero ATE")
plt.title("Distribution of True ATE Across Datasets, normalized", fontweight="bold")
plt.xlabel("True ATE / SD(Y_obs)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------
# 11. Plot Individual Datasets
# ----------------------------------------------------------------------

K_to_plot = min(10, len(all_datasets))

fig, axes = plt.subplots(K_to_plot, 4, figsize=(24, 5 * K_to_plot))

if K_to_plot == 1:
    axes = np.asarray([axes])

for idx, ds in enumerate(all_datasets[:K_to_plot]):
    u, x, w, z, a_samp, y_obs_samp, y0_samp, y1_samp, seq_len, train_size = ds

    ax_scatter = axes[idx][0]
    ax_true = axes[idx][1]
    ax_obs = axes[idx][2]
    ax_cate = axes[idx][3]

    y0_flat = safe_target(y0_samp).ravel()
    y1_flat = safe_target(y1_samp).ravel()
    y_obs_flat = safe_target(y_obs_samp).ravel()
    a_flat = np.asarray(a_samp).ravel().astype(int)

    cate_flat = y1_flat - y0_flat

    # 1. Y0 vs Y1
    ax_scatter.scatter(y0_flat, y1_flat, alpha=0.5, color="purple", s=15)

    min_val = min(np.min(y0_flat), np.min(y1_flat))
    max_val = max(np.max(y0_flat), np.max(y1_flat))

    ax_scatter.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        alpha=0.7,
        label="y = x, no effect",
    )

    ax_scatter.set_title(f"DGP {idx + 1}: True Potential Outcomes")
    ax_scatter.set_xlabel("Y(0)")
    ax_scatter.set_ylabel("Y(1)")
    ax_scatter.legend()

    # 2. True causal marginals
    safe_kdeplot(y0_flat, ax=ax_true, label="Y(0)", color="blue")
    safe_kdeplot(y1_flat, ax=ax_true, label="Y(1)", color="orange")

    ax_true.set_title(f"DGP {idx + 1}: True Causal Marginals")
    ax_true.set_xlabel("Potential Outcome Value")
    ax_true.legend()

    # 3. Observed marginals by treatment
    y_obs_a0 = y_obs_flat[a_flat == 0]
    y_obs_a1 = y_obs_flat[a_flat == 1]

    safe_kdeplot(
        y_obs_a0,
        ax=ax_obs,
        label="Y_obs | A=0",
        color="blue",
        linestyle="--",
    )

    safe_kdeplot(
        y_obs_a1,
        ax=ax_obs,
        label="Y_obs | A=1",
        color="orange",
        linestyle="--",
    )

    ax_obs.set_title(f"DGP {idx + 1}: Observed Marginals")
    ax_obs.set_xlabel("Observed Outcome Value")
    ax_obs.legend()

    # 4. CATE distribution
    safe_kdeplot(cate_flat, ax=ax_cate, color="mediumseagreen")

    ate = np.mean(cate_flat)

    ax_cate.axvline(
        x=0,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Zero Effect",
    )

    ax_cate.axvline(
        x=ate,
        color="black",
        linestyle="-",
        alpha=0.8,
        label=f"ATE = {ate:.2f}",
    )

    ax_cate.set_title(f"DGP {idx + 1}: True Treatment Effect")
    ax_cate.set_xlabel("Y(1) - Y(0)")
    ax_cate.set_ylabel("Density")
    ax_cate.legend()

plt.tight_layout()
plt.show()


#%% ---------------------------------------------------------------------
# 12. Optional: Rejection Filter for Stronger Confounding
# ----------------------------------------------------------------------

strong = df_metrics[
    (df_metrics["Treatment U Gap over X (Delta AUC)"] > 0.05)
    & (df_metrics["Outcome U Gap over A,X (Delta R2)"] > 0.05)
    & (np.abs(df_metrics["GT Observed Contrast Bias Normalized"]) > 0.15)
    & (df_metrics["Residual U Imbalance by A after X (SMD)"] > 0.20)
].copy()

print("\n" + "=" * 70)
print("  STRONG CONFOUNDING FILTER")
print("=" * 70)
print(f"Kept {len(strong)} / {len(df_metrics)} DGPs")

if len(strong) > 0:
    print("\nStrong-confounding subset summary:")

    subset_metrics = [
        "Confounding Bias (Naive ATE - True ATE) Normalized",
        "Oracle Bias (Oracle ATE - True ATE) Normalized",
        "Abs Bias Reduction from U",
        "Treatment U Gap over X (Delta AUC)",
        "Outcome U Gap over A,X (Delta R2)",
        "Residual U Imbalance by A after X (SMD)",
        "GT Observed Contrast Bias Normalized",
        "GT Baseline Imbalance Bias Normalized",
        "GT Effect-Selection Bias Normalized",
        "Confounding Alignment Corr",
    ]

    for name in subset_metrics:
        print_summary_stats(name, strong[name].values)


#%% ---------------------------------------------------------------------
# 13. Quick Interpretation Flags
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("  QUICK INTERPRETATION")
print("=" * 70)

print_summary_stats(
    "Naive abs error",
    np.abs(df_metrics["Confounding Bias (Naive ATE - True ATE) Normalized"]),
)

print_summary_stats(
    "Oracle abs error",
    np.abs(df_metrics["Oracle Bias (Oracle ATE - True ATE) Normalized"]),
)

print_summary_stats(
    "Abs bias reduction from adding U",
    df_metrics["Abs Bias Reduction from U"],
)

print_summary_stats(
    "Treatment-side missing-U signal",
    df_metrics["Treatment U Gap over X (Delta AUC)"],
)

print_summary_stats(
    "Outcome-side missing-U signal",
    df_metrics["Outcome U Gap over A,X (Delta R2)"],
)

print_summary_stats(
    "Ground-truth observed contrast bias",
    df_metrics["GT Observed Contrast Bias Normalized"],
)

print_summary_stats(
    "U predictability from observed X",
    df_metrics["U Predictable from X (R2)"],
)
