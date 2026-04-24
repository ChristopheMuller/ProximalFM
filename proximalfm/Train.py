from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import wandb
from tqdm import tqdm

from proximalfm.models import build_model
from proximalfm.Structure import ProximalPriorConfig, ModelConfig
from proximalfm.PriorDataLoader import create_causal_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m {seconds % 60:.2f}s"


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class Callback:
    def on_init_end(self, trainer: "CATETrainer") -> None:
        pass

    def on_train_start(self, trainer: "CATETrainer") -> None:
        pass

    def on_train_end(
        self, trainer: "CATETrainer", exception: Exception | None = None
    ) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "CATETrainer",
        epoch: int,
        train_metrics: dict,
        val_metrics: dict | None = None,
    ) -> None:
        pass

    def on_step_end(self, trainer: "CATETrainer", step: int, logs: dict) -> None:
        pass

    def on_log_info(self, message: str) -> None:
        pass

    def on_log_error(self, message: str) -> None:
        pass


class LoggerCallback(Callback):
    def __init__(self, config: dict):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.n_params = 0

    def on_init_end(self, trainer: "CATETrainer") -> None:
        self.n_params = trainer.n_params
        exp_name = self.train_cfg.get("exp_name", "test_exp")
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        wandb.init(
            project="proximal-fm",
            name=f"{exp_name}_{run_name}",
            config=self.config,
            dir=self.train_cfg.get("log_dir", "runs"),
        )
        wandb.watch(trainer.model, log="all", log_freq=32)

    def on_train_end(
        self, trainer: "CATETrainer", exception: Exception | None = None
    ) -> None:
        wandb.finish()

    def on_epoch_end(
        self,
        trainer: "CATETrainer",
        epoch: int,
        train_metrics: dict,
        val_metrics: dict | None = None,
    ) -> None:
        logs = {}
        logs["epoch"] = epoch

        if epoch == 0:
            logs["Parameters/total_count"] = self.n_params

        # Losses
        logs["Loss/train_total"] = train_metrics.get("loss_total")
        logs["Loss/train_mse"] = train_metrics.get("loss_mse")
        logs["Loss/train_model"] = train_metrics.get("loss_model")

        # Parameters / Optimization
        logs["Parameters/alpha"] = train_metrics.get("alpha")
        logs["Parameters/grad_norm"] = train_metrics.get("grad_norm")
        if hasattr(trainer, "optimizer") and trainer.optimizer.param_groups:
            logs["Parameters/learning_rate"] = trainer.optimizer.param_groups[0]["lr"]

        # Timing
        logs["Time/train_s"] = train_metrics.get("time_s")

        # Validation Metrics
        if val_metrics:
            logs["ATE/val_r2"] = val_metrics.get("ate_r2")
            logs["ATE/val_bias"] = val_metrics.get("bias")
            logs["ATE/val_normbias"] = val_metrics.get("ate_normbias")
            logs["ATE/val_normmae"] = val_metrics.get("ate_normmae")
            logs["ATE/val_normrmse"] = val_metrics.get("ate_normrmse")

            logs["CATE/val_r2"] = val_metrics.get("cate_r2_mean")
            logs["CATE/val_r2_shape"] = val_metrics.get("cate_r2_shape_mean")
            logs["CATE/val_r2_ate_penalty"] = val_metrics.get(
                "cate_r2_ate_penalty_mean"
            )
            logs["CATE/val_rmse"] = val_metrics.get("rmse")
            logs["CATE/val_mae"] = val_metrics.get("cate_mae_mean")
            logs["CATE/val_normrmse"] = val_metrics.get("cate_rmse_norm_mean")

            logs["Metrics/val_mspe"] = val_metrics.get("mspe")
            if "variance" in val_metrics and val_metrics["variance"] > 0:
                logs["Metrics/val_variance"] = val_metrics["variance"]

            logs["Time/val_s"] = val_metrics.get("time_s")

            if "cov50" in val_metrics and val_metrics["cov50"] > 0:
                logs["Calibration/val_cov50"] = val_metrics.get("cov50")
                logs["Calibration/val_cov95"] = val_metrics.get("cov95")
                logs["Calibration/val_cov99"] = val_metrics.get("cov99")

            logs["Bias-Variance/val_bias_sq"] = val_metrics.get("bias_sq")
            logs["Bias-Variance/val_error_var"] = val_metrics.get("error_var")
            logs["Bias-Variance/mse"] = val_metrics.get("mse")

        wandb.log(logs, step=trainer.global_step)

    def on_step_end(self, trainer: "CATETrainer", step: int, logs: dict) -> None:
        if logs:
            wandb.log(logs, step=step)

    def on_log_info(self, message: str) -> None:
        logger.info(message)

    def on_log_error(self, message: str) -> None:
        logger.error(message)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CATETrainer:
    def __init__(self, config: dict, callbacks: list["Callback"] | None = None) -> None:
        self.callbacks = callbacks or []
        self.config = config
        self.train_cfg = config.get("training", {})
        self.data_cfg = config.get("data", {})
        self.prior_cfg = config.get("prior", {})
        self.model_cfg = config.get("model", {})

        # ------------------------------------------------------------------
        # Device
        # ------------------------------------------------------------------
        requested = self.train_cfg.get("device", "cpu")
        if requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.train_cfg.get('gpu', 0)}")
        elif requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self._log_info(f"Using device: {self.device}")

        # Determine dtype
        dtype_str = self.model_cfg.get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)

        # ------------------------------------------------------------------
        # Prior Config
        # ------------------------------------------------------------------
        prior_kwargs = self.prior_cfg.copy()
        prior_kwargs["dtype"] = self.dtype
        self.prior_config = ProximalPriorConfig(**prior_kwargs)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        model_kwargs = self.model_cfg.copy()

        # Remove keys that are not part of ModelConfig
        if "dtype" in model_kwargs:
            del model_kwargs["dtype"]

        self.model = build_model(ModelConfig(**model_kwargs))
        self.model.to(self.device, dtype=self.dtype)
        self.n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._log_info(f"Model parameters: {self.n_params:,}")

        # ------------------------------------------------------------------
        # Optimiser + scheduler
        # ------------------------------------------------------------------
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.get("lr", 0.001),
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.warmup_steps = self.train_cfg.get("warmup_steps", 1000)
        self.global_step = 0
        self.base_lr = self.train_cfg.get("lr", 0.001)

        self.start_epoch = 0

        # ------------------------------------------------------------------
        # Load Checkpoint
        # ------------------------------------------------------------------
        start_checkpoint = self.train_cfg.get("start_checkpoint")
        if start_checkpoint:
            self._load_checkpoint(start_checkpoint)

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        test_size = self.data_cfg.get("test_size", 56)
        min_context_size = self.data_cfg.get("min_context_size", 500)
        max_context_size = self.data_cfg.get("max_context_size", 1500)

        loaders = create_causal_data_loaders(
            prior_config=self.prior_config,
            batch_size=self.data_cfg.get("batch_size", 32),
            min_episode_size=min_context_size + test_size,
            max_episode_size=max_context_size + test_size,
            train_episodes=self.data_cfg.get("train_episodes", 400),
            val_episodes=self.data_cfg.get("val_episodes", 400),
            num_workers=self.data_cfg.get("num_workers", 4),
            seed=self.train_cfg.get("seed", 1234),
        )
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        self.test_size = test_size

        # ------------------------------------------------------------------
        # Bookkeeping
        # ------------------------------------------------------------------
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_rmses: list[float] = []
        self.best_val_loss = float("inf")
        self.epoch_times: list[float] = []
        self.validation_times: list[float] = []
        self.total_training_time = 0.0

        for cb in self.callbacks:
            cb.on_init_end(self)

        # ------------------------------------------------------------------
        # Checkpoint directories
        # ------------------------------------------------------------------
        path_dir = os.path.join(
            self.train_cfg.get("save_dir", "checkpoints"),
            self.train_cfg.get("exp_name", "test_exp"),
        )
        os.makedirs(path_dir, exist_ok=True)
        # Saving local YAMLs as backup, though W&B stores them too
        with open(os.path.join(path_dir, "train_config.yaml"), "w") as f:
            yaml.dump(self.train_cfg, f)
        with open(os.path.join(path_dir, "model_config.yaml"), "w") as f:
            yaml.dump(self.model_cfg, f)
        with open(os.path.join(path_dir, "prior_config.yaml"), "w") as f:
            yaml.dump(self.prior_cfg, f)
        with open(os.path.join(path_dir, "data_config.yaml"), "w") as f:
            yaml.dump(self.data_cfg, f)


    def _log_info(self, message: str) -> None:
        for cb in self.callbacks:
            cb.on_log_info(message)

    def _log_error(self, message: str) -> None:
        for cb in self.callbacks:
            cb.on_log_error(message)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict) -> dict:
        return {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

    def _prepare_inputs(self, batch: dict):
        x = batch["X"]
        z = batch["Z"]
        w = batch["W"]

        split_idx = x.shape[1] - self.test_size

        a = batch["A"][:, :split_idx].squeeze(-1)
        y = batch["Y"][:, :split_idx].squeeze(-1)

        target = batch["ITE"][:, split_idx:]

        y_std = y.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        return x, a, w, z, y, target, y_std

    def _forward_and_loss(self, x, a, w, z, y, target, y_std, alpha):
        out = self.model(x=x, a=a, w=w, z=z, y=y)
        loss_model = self.model.loss(out, target)

        # Let the model handle its own extraction logic instead of hardcoding GMM math
        expected_mean, overall_log_variance = self.model.estimate_CATE(
            x=x, a_train=a, w=w, z=z, y_train=y, return_log_variance=True
        )

        loss_mse_raw = F.mse_loss(expected_mean, target)

        y_std_expanded = y_std.unsqueeze(-1) if y_std.dim() == 2 else y_std
        loss_mse_normalized = F.mse_loss(
            expected_mean / y_std_expanded, target / y_std_expanded
        )

        loss = alpha * loss_mse_normalized + (1 - alpha) * loss_model
        return loss, loss_mse_raw, loss_model, expected_mean, overall_log_variance

    def _optimization_step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()
        
        clip_value = self.train_cfg.get("clip_grad", 0.0)
        max_norm = clip_value if clip_value > 0 else float("inf")
        
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, (self.global_step + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_scale * self.base_lr

        self.global_step += 1
        self.optimizer.step()

        return grad_norm

    def _compute_batch_metrics(
        self,
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            if pred_log_var is not None and pred_log_var.numel() > 0:
                pred_var = torch.exp(pred_log_var)
                pred_std = torch.sqrt(pred_var)
            else:
                pred_std = None
                pred_var = None

            residuals = pred_mean - target
            abs_error = residuals.abs()

            percentage_errors = residuals / (target + 1e-8)

            # CATE R2 per episode
            ss_res_episode = (residuals * residuals).sum(dim=1)
            target_mean_episode = target.mean(dim=1, keepdim=True)
            ss_tot_diff = target - target_mean_episode
            ss_tot_episode = (ss_tot_diff * ss_tot_diff).sum(dim=1)

            # Identify heterogeneous episodes (where R^2 is well-defined)
            valid_mask = ss_tot_episode > 1e-8
            num_valid_cate_episodes = valid_mask.sum()

            r2_episode = torch.where(
                valid_mask,
                1.0 - (ss_res_episode / ss_tot_episode),
                torch.zeros_like(ss_tot_episode),
            )

            # ATE statistics per episode
            n_query = target.size(1)
            batch_size = target.size(0)
            ate_true_episode = target.mean(dim=1)
            ate_pred_episode = pred_mean.mean(dim=1)
            ate_residuals = ate_pred_episode - ate_true_episode

            var_tau_episode = ss_tot_episode / n_query
            std_tau_episode = torch.sqrt(var_tau_episode.clamp(min=0))

            # Mask out the penalty calculation to prevent explosion
            cate_r2_ate_penalty = torch.where(
                valid_mask,
                (ate_residuals * ate_residuals) / (var_tau_episode + 1e-8),
                torch.zeros_like(var_tau_episode),
            )
            cate_r2_shape = torch.where(
                valid_mask,
                r2_episode + cate_r2_ate_penalty,
                torch.zeros_like(r2_episode),
            )

            cate_mae_episode = residuals.abs().mean(dim=1, keepdim=True)

            mse_episode = ss_res_episode / n_query
            cate_rmse_norm_episode = torch.sqrt(mse_episode) / (std_tau_episode + 1e-8)

            ate_error_norm_episode = ate_residuals / (std_tau_episode + 1e-8)

            metrics = {
                "sum_sq_err": residuals.pow(2).sum(),
                "sum_sq_pe": percentage_errors.pow(2).sum(),
                "sum_error": residuals.sum(),
                "sum_target": target.sum(),
                "sum_sq_target": target.pow(2).sum(),
                "num_samples": target.numel(),
                "sum_cate_r2": r2_episode.sum(),
                "num_episodes": batch_size,
                "sum_ate_true": ate_true_episode.sum(),
                "sum_sq_ate_true": ate_true_episode.pow(2).sum(),
                "sum_sq_ate_err": ate_residuals.pow(2).sum(),
                "sum_cate_r2_shape": cate_r2_shape.sum(),
                "sum_cate_r2_ate_penalty": cate_r2_ate_penalty.sum(),
                "num_valid_cate_episodes": num_valid_cate_episodes,
                "sum_cate_mae": cate_mae_episode.sum(),
                "sum_cate_rmse_norm": cate_rmse_norm_episode.sum(),
                "sum_ate_error_norm": ate_error_norm_episode.sum(),
                "sum_abs_ate_error_norm": ate_error_norm_episode.abs().sum(),
                "sum_sq_ate_error_norm": ate_error_norm_episode.pow(2).sum(),
            }

            if pred_var is not None:
                metrics["variance_mean"] = pred_var.mean()
                metrics["variance_sum"] = pred_var.sum()
                metrics["cov50"] = (abs_error <= 0.674 * pred_std).float().sum()
                metrics["cov95"] = (abs_error <= 1.960 * pred_std).float().sum()
                metrics["cov99"] = (abs_error <= 2.576 * pred_std).float().sum()

            return metrics

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model and optimizer state from a checkpoint file.
            - checkpoint_path: path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Start checkpoint not found: {checkpoint_path}")

        self._log_info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"]

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        if "val_loss" in checkpoint:
            self.best_val_loss = checkpoint["val_loss"]

        self._log_info(f"Resumed training from epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = torch.zeros(1, device=self.device)
        total_loss_mse = torch.zeros(1, device=self.device)
        total_loss_model = torch.zeros(1, device=self.device)
        sum_sq_err = torch.zeros(1, device=self.device)
        sum_sq_pe = torch.zeros(1, device=self.device)
        total_model = torch.zeros(1, device=self.device)
        total_samples = 0
        total_variance = torch.zeros(1, device=self.device)
        total_grad_norm = torch.zeros(1, device=self.device)
        total_cov50 = torch.zeros(1, device=self.device)
        total_cov95 = torch.zeros(1, device=self.device)
        total_cov99 = torch.zeros(1, device=self.device)

        epoch_start = time.time()

        # alpha epoch loss = 1 if epoch = 0, then linearly decay to 0 by epoch warmup_epochs
        alpha_epoch_loss = 1 - min(1, epoch / self.train_cfg.get("warmup_epochs", 300))

        with tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']} [Train]",
        ) as pbar:
            for batch in pbar:
                
                batch = self._move_batch(batch)
                x, a, w, z, y, target, y_std = self._prepare_inputs(batch)

                # Ensure target has shape (B, N_query, 1) to match predictions
                if target.dim() == 2:
                    target = target.unsqueeze(-1)

                (
                    loss,
                    loss_mse,
                    loss_model,
                    pred_mean,
                    pred_log_var,
                ) = self._forward_and_loss(
                    x, a, w, z, y, target, y_std, alpha_epoch_loss
                )

                total_grad_norm += self._optimization_step(loss)

                total_loss += loss.detach()
                total_loss_mse += loss_mse.detach()
                total_loss_model += loss_model.detach()

                metrics = self._compute_batch_metrics(
                    pred_mean,
                    pred_log_var,
                    target,
                )

                sum_sq_err += metrics["sum_sq_err"]
                sum_sq_pe += metrics["sum_sq_pe"]
                total_model += loss_model.detach() * metrics["num_samples"]
                total_samples += metrics["num_samples"]
                if "variance_mean" in metrics:
                    total_variance += metrics["variance_mean"]
                if "cov50" in metrics:
                    total_cov50 += metrics["cov50"]
                if "cov95" in metrics:
                    total_cov95 += metrics["cov95"]
                if "cov99" in metrics:
                    total_cov99 += metrics["cov99"]

        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)

        num_batches = len(self.train_loader)
        avg_loss = (total_loss / num_batches).item()
        avg_loss_mse = (total_loss_mse / num_batches).item()
        avg_loss_model = (total_loss_model / num_batches).item()
        rmse = np.sqrt(sum_sq_err.item() / max(total_samples, 1))
        model = (total_model / max(total_samples, 1)).item()
        avg_variance = (total_variance / num_batches).item()
        avg_grad_norm = (total_grad_norm / num_batches).item()
        cov50 = (total_cov50 / max(total_samples, 1)).item()
        cov95 = (total_cov95 / max(total_samples, 1)).item()
        cov99 = (total_cov99 / max(total_samples, 1)).item()

        mspe = (sum_sq_pe / max(total_samples, 1)).item()

        self.train_losses.append(avg_loss)
        self._log_info(f"Epoch {epoch + 1} train done in {format_time(epoch_time)}")

        return {
            "loss_total": avg_loss,
            "loss_mse": avg_loss_mse,
            "loss_model": avg_loss_model,
            "model": model,
            "rmse": rmse,
            "mspe": mspe,
            "variance": avg_variance,
            "grad_norm": avg_grad_norm,
            "alpha": alpha_epoch_loss,
            "time_s": epoch_time,
            "cov50": cov50,
            "cov95": cov95,
            "cov99": cov99,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, epoch: int) -> dict:
        self.model.eval()

        total_variance = torch.zeros(1, device=self.device)
        sum_sq_err = torch.zeros(1, device=self.device)
        sum_sq_pe = torch.zeros(1, device=self.device)
        total_samples = 0
        total_cov50 = torch.zeros(1, device=self.device)
        total_cov95 = torch.zeros(1, device=self.device)
        total_cov99 = torch.zeros(1, device=self.device)

        sum_error = torch.zeros(1, device=self.device)
        sum_target = torch.zeros(1, device=self.device)
        sum_sq_target = torch.zeros(1, device=self.device)

        total_cate_r2 = torch.zeros(1, device=self.device)
        total_episodes = 0
        total_valid_cate_episodes = torch.zeros(1, device=self.device)
        total_ate_true = torch.zeros(1, device=self.device)
        total_sq_ate_true = torch.zeros(1, device=self.device)
        total_sq_ate_err = torch.zeros(1, device=self.device)

        total_cate_r2_shape = torch.zeros(1, device=self.device)
        total_cate_r2_ate_penalty = torch.zeros(1, device=self.device)
        total_cate_mae = torch.zeros(1, device=self.device)
        total_cate_rmse_norm = torch.zeros(1, device=self.device)
        total_ate_error_norm = torch.zeros(1, device=self.device)
        total_abs_ate_error_norm = torch.zeros(1, device=self.device)
        total_sq_ate_error_norm = torch.zeros(1, device=self.device)

        val_start = time.time()

        with torch.no_grad():
            with tqdm(
                self.val_loader,
                desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']} [Val]",
            ) as pbar:
                for batch in pbar:
                    batch = self._move_batch(batch)
                    x, a, w, z, y, target, y_std = self._prepare_inputs(batch)
                    if target.dim() == 2:
                        target = target.unsqueeze(-1)

                    pred_mean, pred_log_var = self.model.estimate_CATE(
                        x=x, a_train=a, w=w, z=z, y_train=y, return_log_variance=True
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
                    if "variance_sum" in metrics:
                        total_variance += metrics["variance_sum"]
                    if "cov50" in metrics:
                        total_cov50 += metrics["cov50"]
                    if "cov95" in metrics:
                        total_cov95 += metrics["cov95"]
                    if "cov99" in metrics:
                        total_cov99 += metrics["cov99"]

                    total_cate_r2 += metrics["sum_cate_r2"]
                    total_episodes += metrics["num_episodes"]
                    total_valid_cate_episodes += metrics["num_valid_cate_episodes"]
                    total_ate_true += metrics["sum_ate_true"]
                    total_sq_ate_true += metrics["sum_sq_ate_true"]
                    total_sq_ate_err += metrics["sum_sq_ate_err"]

                    total_cate_r2_shape += metrics["sum_cate_r2_shape"]
                    total_cate_r2_ate_penalty += metrics["sum_cate_r2_ate_penalty"]
                    total_cate_mae += metrics["sum_cate_mae"]
                    total_cate_rmse_norm += metrics["sum_cate_rmse_norm"]
                    total_ate_error_norm += metrics["sum_ate_error_norm"]
                    total_abs_ate_error_norm += metrics["sum_abs_ate_error_norm"]
                    total_sq_ate_error_norm += metrics["sum_sq_ate_error_norm"]

        val_time = time.time() - val_start
        self.validation_times.append(val_time)

        denom = max(total_samples, 1)

        mse = (sum_sq_err / denom).item()
        rmse = np.sqrt(mse)
        mspe = (sum_sq_pe / denom).item()

        # Bias-Variance Decomposition
        ite_bias = (sum_error / denom).item()
        ite_bias_sq = ite_bias**2
        ite_error_var = mse - ite_bias_sq

        # CATE R2 Mean (per episode)
        valid_denom = max(total_valid_cate_episodes.item(), 1)
        cate_r2_mean = (total_cate_r2 / valid_denom).item()
        cate_r2_shape_mean = (total_cate_r2_shape / valid_denom).item()
        cate_r2_ate_penalty_mean = (total_cate_r2_ate_penalty / valid_denom).item()

        # ATE R2 (across episodes)
        episodes_denom = max(total_episodes, 1)
        ss_tot_ate = (total_sq_ate_true - (total_ate_true**2) / episodes_denom).item()
        total_sq_ate_err_val = total_sq_ate_err.item()
        if ss_tot_ate > 1e-8:
            ate_r2 = 1.0 - (total_sq_ate_err_val / ss_tot_ate)
        else:
            ate_r2 = 0.0

        ep_denom = max(total_episodes, 1)
        cate_mae_mean = (total_cate_mae / ep_denom).item()
        cate_rmse_norm_mean = (total_cate_rmse_norm / ep_denom).item()
        ate_normbias = (total_ate_error_norm / ep_denom).item()
        ate_normmae = (total_abs_ate_error_norm / ep_denom).item()
        ate_normrmse = np.sqrt((total_sq_ate_error_norm / ep_denom).item())

        avg_variance = (total_variance / denom).item()

        cov95 = (total_cov95 / denom).item()
        cov50 = (total_cov50 / denom).item()
        cov99 = (total_cov99 / denom).item()

        self.val_rmses.append(rmse)
        if self.global_step >= self.warmup_steps:
            self.scheduler.step(rmse)

        self._log_info(
            f"Epoch {epoch + 1} val done: RMSE={rmse:.4f}, CATE_R2={cate_r2_mean:.4f}, ATE_R2={ate_r2:.4f}, Bias={ite_bias:.4f}"
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
            "cate_r2_shape_mean": cate_r2_shape_mean,
            "cate_r2_ate_penalty_mean": cate_r2_ate_penalty_mean,
            "cate_mae_mean": cate_mae_mean,
            "cate_rmse_norm_mean": cate_rmse_norm_mean,
            "ate_normbias": ate_normbias,
            "ate_normmae": ate_normmae,
            "ate_normrmse": ate_normrmse,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_rmse: float, tag: str = "") -> None:
        path_dir = os.path.join(
            self.train_cfg.get("save_dir", "checkpoints"),
            self.train_cfg.get("exp_name", "test_exp"),
        )
        os.makedirs(path_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_rmse,
            "epoch_times": self.epoch_times,
        }
        name = f"model_epoch_{epoch}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(path_dir, name)
        torch.save(checkpoint, path)

        if val_rmse < self.best_val_loss:
            self.best_val_loss = val_rmse
            best_path = os.path.join(path_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            self._log_info(f"  → New best model (val RMSE: {val_rmse:.4f})")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        training_start = time.time()
        os.makedirs(self.train_cfg.get("save_dir", "checkpoints"), exist_ok=True)

        early_stop_count = 0

        reset_freq = self.data_cfg.get("reset_freq_train", -1)

        epoch = self.start_epoch
        last_val_rmse = float("inf")
        while epoch < self.train_cfg["epochs"]:
            if hasattr(self.train_loader.dataset, "set_reset_state") and reset_freq > 0:
                self.train_loader.dataset.set_reset_state(epoch // reset_freq)

            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if epoch % self.train_cfg.get("val_freq", 1) == 0:
                val_metrics = self.validate(epoch)
                last_val_rmse = val_metrics["rmse"]

            # Extract metrics for console logging
            train_model = train_metrics["loss_model"]
            train_mse = train_metrics["loss_mse"]
            train_total = train_metrics["loss_total"]
            val_rmse = val_metrics["rmse"] if val_metrics else float("nan")

            self._log_info(
                f"Epoch {epoch + 1}/{self.train_cfg['epochs']}  "
                f"train_model={train_model:.4f}  train_rmse={train_mse:.4f}  "
                f"train_total={train_total:.4f}"
            )
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, train_metrics, val_metrics)

            if val_metrics:
                if val_rmse < self.best_val_loss:
                    early_stop_count = 0
                    self.save_checkpoint(epoch + 1, val_rmse, tag="best")
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.train_cfg.get("early_stop", 1e9):
                        self._log_info(f"Early stopping after {epoch + 1} epochs.")
                        break

            if (epoch + 1) % self.train_cfg.get("save_freq", 10) == 0 or epoch == 0:
                self.save_checkpoint(epoch + 1, last_val_rmse)

            epoch += 1

        self.total_training_time = time.time() - training_start
        self.save_checkpoint(epoch + 1, last_val_rmse, tag="final")

        self._log_info("Training complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()
    set_seed(config["training"]["seed"])

    logger_cb = LoggerCallback(config)
    callbacks = [logger_cb]

    def log_info(message: str) -> None:
        for cb in callbacks:
            cb.on_log_info(message)

    def log_error(message: str) -> None:
        for cb in callbacks:
            cb.on_log_error(message)

    log_info(f"Training config: {config['training']}")
    trainer = CATETrainer(config, callbacks=callbacks)

    try:
        trainer.train()
    except KeyboardInterrupt:
        log_info("Training interrupted by user.")
    except Exception as e:
        log_error(f"Error during training: {e}")
        raise
    finally:
        t = trainer.total_training_time
        log_info(f"Total training time:        {format_time(t)}")
        if trainer.epoch_times:
            log_info(
                f"Mean epoch time:            {format_time(np.mean(trainer.epoch_times))}"
            )
        if trainer.validation_times:
            log_info(
                f"Mean validation time:       {format_time(np.mean(trainer.validation_times))}"
            )
        for cb in callbacks:
            cb.on_train_end(trainer)


if __name__ == "__main__":
    main()
