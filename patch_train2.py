import re

with open('proximalfm/Train.py', 'r') as f:
    content = f.read()

# Fix validate
validate_replace = '''
        self.model.eval()
        # Existing accumulators
        total_variance = 0.0
        sum_sq_err = 0.0
        sum_sq_pe = 0.0
        total_samples = 0
        total_cov50 = 0
        total_cov95 = 0
        total_cov99 = 0

        sum_error = 0.0
        sum_target = 0.0
        sum_sq_target = 0.0

        total_cate_r2 = 0.0
        total_episodes = 0
        total_ate_true = 0.0
        total_sq_ate_true = 0.0
        total_sq_ate_err = 0.0

        total_cate_r2_shape       = 0.0
        total_cate_r2_ate_penalty = 0.0
        total_cate_mae            = 0.0
        total_cate_rmse_norm      = 0.0
        total_ate_error_norm      = 0.0   # signed (for NormBias)
        total_abs_ate_error_norm  = 0.0   # absolute (for NormMAE)
        total_sq_ate_error_norm   = 0.0   # squared (for NormRMSE)

        val_start = time.time()

        with torch.no_grad():
            with tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.train_cfg['epochs']} [Val]",
            ) as pbar:
                for batch in pbar:
                    batch = self._move_batch(batch)
                    x, a, w, z, y, target, y_std = (
                        self._prepare_inputs(batch)
                    )
                    if target.dim() == 2:
                        target = target.unsqueeze(-1)

                    pred_mean, pred_log_var = self.model.estimate_CATE(
                        x=x, a_train=a, w=w, z=z, y_train=y, return_variance=True
                    )

                    metrics = self._compute_batch_metrics(
                        pred_mean,
                        pred_log_var,
                        target,
                    )

                    sum_sq_err += metrics["sum_sq_err"]
                    sum_sq_pe += metrics["sum_sq_pe"]
                    sum_error += metrics["sum_error"]
                    sum_target += metrics["sum_target"]
                    sum_sq_target += metrics["sum_sq_target"]
                    total_samples += metrics["num_samples"]
                    total_variance += metrics.get("variance_sum", 0.0)
                    total_cov50 += metrics.get("cov50", 0.0)
                    total_cov95 += metrics.get("cov95", 0.0)
                    total_cov99 += metrics.get("cov99", 0.0)

                    total_cate_r2 += metrics["sum_cate_r2"]
                    total_episodes += metrics["num_episodes"]
                    total_ate_true += metrics["sum_ate_true"]
                    total_sq_ate_true += metrics["sum_sq_ate_true"]
                    total_sq_ate_err += metrics["sum_sq_ate_err"]

                    # ── Benchmark-aligned ─────────────────────────────────────
                    total_cate_r2_shape       += metrics["sum_cate_r2_shape"]
                    total_cate_r2_ate_penalty += metrics["sum_cate_r2_ate_penalty"]
                    total_cate_mae            += metrics["sum_cate_mae"]
                    total_cate_rmse_norm      += metrics["sum_cate_rmse_norm"]
                    total_ate_error_norm      += metrics["sum_ate_error_norm"]
                    total_abs_ate_error_norm  += metrics["sum_abs_ate_error_norm"]
                    total_sq_ate_error_norm   += metrics["sum_sq_ate_error_norm"]

        val_time = time.time() - val_start
        self.validation_times.append(val_time)

        denom = max(total_samples, 1)
        num_batches = len(self.val_loader)

        mse = sum_sq_err / denom
        rmse = np.sqrt(mse)
        mspe = sum_sq_pe / denom
'''

content = re.sub(
    r'        self\.model\.eval\(\)\n        # Existing accumulators[\s\S]*?        mspe = sum_sq_pe / denom',
    validate_replace,
    content
)


validate_return = '''        avg_variance = total_variance / denom

        cov95 = total_cov95 / denom
        cov50 = total_cov50 / denom
        cov99 = total_cov99 / denom

        self.val_rmses.append(rmse)
        self.scheduler.step(rmse)

        logger.info(
            f"Epoch {epoch+1} val done: RMSE={rmse:.4f}, CATE_R2={cate_r2_mean:.4f}, ATE_R2={ate_r2:.4f}, Bias={ite_bias:.4f}"
        )

        return {
            "rmse": rmse,
            "mse": mse,
            "mspe": mspe,
            "cate_r2_mean": cate_r2_mean,
            "ate_r2": ate_r2,
            "bias": ite_bias,
            "bias_sq": ite_bias_sq,
            "error_var": ite_error_var,
            "variance": avg_variance,
            "time_s": val_time,
            "cov50": cov50,
            "cov95": cov95,
            "cov99": cov99,

            "cate_r2_shape_mean":       cate_r2_shape_mean,
            "cate_r2_ate_penalty_mean": cate_r2_ate_penalty_mean,
            "cate_mae_mean":            cate_mae_mean,
            "cate_rmse_norm_mean":      cate_rmse_norm_mean,
            "ate_normbias":             ate_normbias,
            "ate_normmae":              ate_normmae,
            "ate_normrmse":             ate_normrmse,
        }'''
content = re.sub(r'        avg_variance = total_variance / denom[\s\S]*?            "ate_normrmse":             ate_normrmse,\n        \}', validate_return, content)
content = content.replace('        self.val_losses.append(gnll)\n', '')


# Fix log_metrics
log_metrics_replace = '''        # Validation Metrics (if available)
        if val_metrics:
            # ── ATE — mirrors bench_metrics ATE summary ───────────────────────
            # ATE R²: cross-episode, how well τ̂_mean tracks τ_mean in ranking.
            # Identical to bench "ATE R²".'''

content = re.sub(
    r'        # Validation Metrics \(if available\)[\s\S]*?            # ATE R²: cross-episode, how well τ̂_mean tracks τ_mean in ranking.\n            # Identical to bench "ATE R²"\.',
    log_metrics_replace,
    content
)

content = content.replace('''            self.writer.add_scalar("Metrics/val_mspe",    val_metrics["mspe"],     epoch)
            self.writer.add_scalar("Metrics/val_gnll",    val_metrics["gnll"],     epoch)
            self.writer.add_scalar("Metrics/val_variance", val_metrics["variance"], epoch)''', '''            self.writer.add_scalar("Metrics/val_mspe",    val_metrics["mspe"],     epoch)
            self.writer.add_scalar("Metrics/val_variance", val_metrics["variance"], epoch)''')

content = content.replace('''            # ── Training diagnostics (not in benchmark) ───────────────────────
            self.writer.add_scalar("Metrics/val_mspe",    val_metrics["mspe"],     epoch)
            self.writer.add_scalar("Metrics/val_variance", val_metrics["variance"], epoch)
            self.writer.add_scalar("Time/val_s",          val_metrics["time_s"],   epoch)
            self.writer.add_scalar("Calibration/val_cov50", val_metrics["cov50"],  epoch)
            self.writer.add_scalar("Calibration/val_cov95", val_metrics["cov95"],  epoch)
            self.writer.add_scalar("Calibration/val_cov99", val_metrics["cov99"],  epoch)''', '''            # ── Training diagnostics (not in benchmark) ───────────────────────
            self.writer.add_scalar("Metrics/val_mspe",    val_metrics["mspe"],     epoch)
            if "variance" in val_metrics and val_metrics["variance"] > 0:
                self.writer.add_scalar("Metrics/val_variance", val_metrics["variance"], epoch)
            self.writer.add_scalar("Time/val_s",          val_metrics["time_s"],   epoch)
            if "cov50" in val_metrics and val_metrics["cov50"] > 0:
                self.writer.add_scalar("Calibration/val_cov50", val_metrics["cov50"],  epoch)
                self.writer.add_scalar("Calibration/val_cov95", val_metrics["cov95"],  epoch)
                self.writer.add_scalar("Calibration/val_cov99", val_metrics["cov99"],  epoch)''')

with open('proximalfm/Train.py', 'w') as f:
    f.write(content)
