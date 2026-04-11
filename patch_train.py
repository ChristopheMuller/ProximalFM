import re

with open('proximalfm/Train.py', 'r') as f:
    content = f.read()

# 1. Update _compute_batch_metrics signature and usage of log_var
content = re.sub(
    r'def _compute_batch_metrics\(\s*self,\s*pred_mean:\s*torch\.Tensor,\s*pred_log_var:\s*torch\.Tensor,\s*target:\s*torch\.Tensor,\s*loss_gnll:\s*torch\.Tensor,\s*\)\s*->\s*dict:',
    'def _compute_batch_metrics(\n        self,\n        pred_mean: torch.Tensor,\n        pred_log_var: torch.Tensor,\n        target: torch.Tensor,\n    ) -> dict:',
    content
)

content = content.replace('''            pred_std = torch.exp(0.5 * pred_log_var)
            pred_var = pred_std.pow(2)

            residuals = pred_mean - target
            abs_error = residuals.abs()''',
'''            if pred_log_var is not None and pred_log_var.numel() > 0:
                pred_std = torch.exp(0.5 * pred_log_var)
                pred_var = pred_std.pow(2)
            else:
                pred_std = None
                pred_var = None

            residuals = pred_mean - target
            abs_error = residuals.abs()''')

content = content.replace('''            # ── Benchmark-aligned metrics ────────────────────────────────────
            # Var(τ) per episode = SS_tot / n_query''',
'''
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

            # ── Benchmark-aligned metrics ────────────────────────────────────
            # Var(τ) per episode = SS_tot / n_query''')

# Remove the old return logic inside _compute_batch_metrics
content = re.sub(r'return \{\s*"sum_sq_err":[\s\S]*?"sum_sq_ate_error_norm":\s*ate_error_norm_episode\.pow\(2\)\.sum\(\)\.item\(\),\s*\}', '', content)
# We also have "sum_gnll": loss_gnll.item() * target.numel(), in the old metrics

content = content.replace('''                "sum_target": target.sum().item(),
                "sum_sq_target": target.pow(2).sum().item(),
                "sum_gnll": loss_gnll.item() * target.numel(),
                "num_samples": target.numel(),
                "variance_mean": pred_var.mean().item(),
                "variance_sum": pred_var.sum().item(),
                "cov50": (abs_error <= 0.674 * pred_std).float().sum().item(),
                "cov95": (abs_error <= 1.960 * pred_std).float().sum().item(),
                "cov99": (abs_error <= 2.576 * pred_std).float().sum().item(),
                "sum_cate_r2": r2_episode.sum().item(),''', '')


# Fix train_epoch call
content = content.replace('''                metrics = self._compute_batch_metrics(
                    pred_mean_norm,
                    pred_log_var_norm,
                    target,
                    loss_gnll,
                )''', '''                metrics = self._compute_batch_metrics(
                    pred_mean_norm,
                    pred_log_var_norm,
                    target,
                )''')

content = content.replace('''                sum_sq_err += metrics["sum_sq_err"]
                sum_sq_pe += metrics["sum_sq_pe"]
                total_gnll += metrics["sum_gnll"]
                total_samples += metrics["num_samples"]
                total_variance += metrics["variance_mean"]
                total_cov50 += metrics["cov50"]
                total_cov95 += metrics["cov95"]
                total_cov99 += metrics["cov99"]''', '''                sum_sq_err += metrics["sum_sq_err"]
                sum_sq_pe += metrics["sum_sq_pe"]
                total_gnll += loss_gnll.item() * metrics["num_samples"]
                total_samples += metrics["num_samples"]
                total_variance += metrics.get("variance_mean", 0.0)
                total_cov50 += metrics.get("cov50", 0.0)
                total_cov95 += metrics.get("cov95", 0.0)
                total_cov99 += metrics.get("cov99", 0.0)''')

with open('proximalfm/Train.py', 'w') as f:
    f.write(content)
