import re

with open('proximalfm/Train.py', 'r') as f:
    content = f.read()

# Fix indentation and missing variables in _compute_batch_metrics
# Notice that ruff says these names are undefined on lines 343-356 but assigned on 376-386.
# This implies my regex replaced the first part of _compute_batch_metrics (creating the `metrics` dict)
# before the variables were computed!

# Let's fix _compute_batch_metrics directly
content = re.sub(r'    def _compute_batch_metrics[\s\S]*?def _load_checkpoint', '''    def _compute_batch_metrics(
        self,
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            if pred_log_var is not None and pred_log_var.numel() > 0:
                pred_std = torch.exp(0.5 * pred_log_var)
                pred_var = pred_std.pow(2)
            else:
                pred_std = None
                pred_var = None

            residuals = pred_mean - target
            abs_error = residuals.abs()
            percentage_errors = residuals / (target + 1e-8)

            # CATE R2 per episode
            ss_res_episode = residuals.pow(2).sum(dim=1)
            target_mean_episode = target.mean(dim=1, keepdim=True)
            ss_tot_episode = (target - target_mean_episode).pow(2).sum(dim=1)
            r2_episode = torch.where(
                ss_tot_episode > 1e-8,
                1.0 - (ss_res_episode / ss_tot_episode),
                torch.zeros_like(ss_tot_episode)
            )

            # ATE statistics per episode (each episode is fully contained in this batch)
            n_query = target.size(1)
            batch_size = target.size(0)
            ate_true_episode = target.mean(dim=1)          # [batch_size, 1]
            ate_pred_episode = pred_mean.mean(dim=1)       # [batch_size, 1]
            ate_residuals = ate_pred_episode - ate_true_episode  # [batch_size, 1]

            # ── Benchmark-aligned metrics ────────────────────────────────────
            # Var(τ) per episode = SS_tot / n_query
            var_tau_episode = ss_tot_episode / n_query           # [batch_size, 1]
            std_tau_episode = torch.sqrt(var_tau_episode.clamp(min=0))

            # R² decomposition: R²(shape) = R² + bias²/Var(τ)
            # ATE penalty per episode: bias²/Var(τ)
            # Identity (bench): cate_r2 = cate_r2_shape − cate_r2_ate_penalty
            cate_r2_ate_penalty = ate_residuals.pow(2) / (var_tau_episode + 1e-8)
            cate_r2_shape = r2_episode + cate_r2_ate_penalty     # [batch_size, 1]

            # CATE MAE per episode
            cate_mae_episode = residuals.abs().mean(dim=1, keepdim=True)       # [batch_size, 1]

            # Normalised CATE RMSE per episode: sqrt(MSE_ep) / SD(τ)
            mse_episode = ss_res_episode / n_query               # [batch_size, 1]
            cate_rmse_norm_episode = torch.sqrt(mse_episode) / (std_tau_episode + 1e-8)

            # Normalised ATE error per episode: (pred_ate − true_ate) / SD(τ)
            ate_error_norm_episode = ate_residuals / (std_tau_episode + 1e-8)

            metrics = {
                "sum_sq_err": residuals.pow(2).sum().item(),
                "sum_sq_pe": percentage_errors.pow(2).sum().item(),
                "sum_error": residuals.sum().item(),
                "sum_target": target.sum().item(),
                "sum_sq_target": target.pow(2).sum().item(),
                "num_samples": target.numel(),
                "sum_cate_r2": r2_episode.sum().item(),
                "num_episodes": batch_size,
                "sum_ate_true": ate_true_episode.sum().item(),
                "sum_sq_ate_true": ate_true_episode.pow(2).sum().item(),
                "sum_sq_ate_err": ate_residuals.pow(2).sum().item(),
                # R² decomposition
                "sum_cate_r2_shape":       cate_r2_shape.sum().item(),
                "sum_cate_r2_ate_penalty": cate_r2_ate_penalty.sum().item(),
                # CATE absolute
                "sum_cate_mae":            cate_mae_episode.sum().item(),
                "sum_cate_rmse_norm":      cate_rmse_norm_episode.sum().item(),
                # Normalised ATE error (for NormBias / NormMAE / NormRMSE)
                "sum_ate_error_norm":      ate_error_norm_episode.sum().item(),
                "sum_abs_ate_error_norm":  ate_error_norm_episode.abs().sum().item(),
                "sum_sq_ate_error_norm":   ate_error_norm_episode.pow(2).sum().item(),
            }

            if pred_var is not None:
                metrics["variance_mean"] = pred_var.mean().item()
                metrics["variance_sum"] = pred_var.sum().item()
                metrics["cov50"] = (abs_error <= 0.674 * pred_std).float().sum().item()
                metrics["cov95"] = (abs_error <= 1.960 * pred_std).float().sum().item()
                metrics["cov99"] = (abs_error <= 2.576 * pred_std).float().sum().item()

            return metrics

    def _load_checkpoint''', content)


# Fix validation gnll references
content = re.sub(r'        sum_gnll = 0\.0\n', '', content)
content = re.sub(r'        gnll = sum_gnll / denom\n', '', content)
content = re.sub(r'        # Global metrics\n', '', content)
content = re.sub(r'        num_batches = len\(self\.val_loader\)\n', '', content)
content = re.sub(r'import time\nfrom datetime import datetime\nfrom typing import Tuple\n\nimport numpy as np', 'import time\nfrom datetime import datetime\n\nimport numpy as np', content)

with open('proximalfm/Train.py', 'w') as f:
    f.write(content)
