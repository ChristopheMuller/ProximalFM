from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from proximalfm.Model import ProximalNanoTabICLv2
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


def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    L = 0.5 * (log_var + (target - mean)^2 / var)
    """
    log_var = pred_log_var.clamp(-10.0, 4.0)
    return 0.5 * (log_var + (target - pred_mean).pow(2) / log_var.exp()).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CATETrainer:
    def __init__(self, config: dict) -> None:
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
        logger.info(f"Using device: {self.device}")

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
        # Ensure feature sizes match prior config
        model_kwargs["feature_group_size_x"] = self.prior_config.d_x
        model_kwargs["feature_group_size_w"] = self.prior_config.d_w
        model_kwargs["feature_group_size_z"] = self.prior_config.d_z

        # Remove keys that are not part of ModelConfig
        if "dtype" in model_kwargs:
            del model_kwargs["dtype"]

        self.model = ProximalNanoTabICLv2(config=ModelConfig(**model_kwargs))
        self.model.to(self.device, dtype=self.dtype)
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {self.n_params:,}")

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
        # Load Checkpoint (if requested)
        # ------------------------------------------------------------------
        start_checkpoint = self.train_cfg.get("start_checkpoint")
        if start_checkpoint:
            self._load_checkpoint(start_checkpoint)
        
        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        loaders = create_causal_data_loaders(
            prior_config=self.prior_config,
            batch_size=self.data_cfg.get("batch_size", 32),
            samples_per_episode=self.data_cfg.get("samples_per_episode", 128),
            train_episodes=self.data_cfg.get("train_episodes", 400),
            val_episodes=self.data_cfg.get("val_episodes", 400),
            num_workers=self.data_cfg.get("num_workers", 4),
            seed=self.train_cfg.get("seed", 1234),
        )
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        n = self.data_cfg.get("samples_per_episode", 128)
        self.single_eval_pos = int(n * self.data_cfg.get("context_fraction", 0.8))

        # ------------------------------------------------------------------
        # Logging + bookkeeping
        # ------------------------------------------------------------------
        base_log_dir = self.train_cfg.get("log_dir", "runs")
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        exp_name = self.train_cfg.get("exp_name", "cate_run")
        full_log_dir = os.path.join(base_log_dir, f"{exp_name}_{run_name}")

        os.makedirs(full_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=full_log_dir)

        logger.info(f"TensorBoard logging to: {full_log_dir}")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_rmses: list[float] = []
        self.best_val_loss = float("inf")
        self.epoch_times: list[float] = []
        self.validation_times: list[float] = []
        self.total_training_time = 0.0

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        path_dir = os.path.join(
            self.train_cfg["save_dir"], self.train_cfg.get("exp_name", "cate_run")
        )
        os.makedirs(path_dir, exist_ok=True)
        train_config_save_path = os.path.join(path_dir, "train_config.yaml")
        with open(train_config_save_path, "w") as f:
            yaml.dump(self.train_cfg, f)
        model_config_save_path = os.path.join(path_dir, "model_config.yaml")
        with open(model_config_save_path, "w") as f:
            yaml.dump(self.model_cfg, f)
        prior_config_save_path = os.path.join(path_dir, "prior_config.yaml")
        with open(prior_config_save_path, "w") as f:
            yaml.dump(self.prior_cfg, f)

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

    def _prepare_inputs(self, batch: dict) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        x = batch["X"].transpose(0, 1)
        z = batch["Z"].transpose(0, 1)
        w = batch["W"].transpose(0, 1)

        a = batch["A"][: self.single_eval_pos].transpose(0, 1).squeeze(-1)
        y_context = batch["Y"][: self.single_eval_pos].transpose(0, 1).squeeze(-1)
        target = batch["ITE"][self.single_eval_pos :].transpose(0, 1)

        y_mean = y_context.mean(dim=1, keepdim=True)
        y_std = y_context.std(dim=1, unbiased=False, keepdim=True) + 1e-8
        y_norm = (y_context - y_mean) / y_std        
        target_norm = (target - y_mean.unsqueeze(-1)) / y_std.unsqueeze(-1)

        def group_norm(tensor: torch.Tensor) -> torch.Tensor:
            context_slice = tensor[:, :self.single_eval_pos, :]
            
            tensor_mean = context_slice.mean(dim=1, keepdim=True)
            tensor_std = context_slice.std(dim=1, unbiased=False, keepdim=True) + 1e-8
            
            return (tensor - tensor_mean) / tensor_std

        x = group_norm(x)
        z = group_norm(z)
        w = group_norm(w)

        return x, a, w, z, y_norm, target_norm, y_mean, y_std    
    def _forward_and_loss(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y_norm: torch.Tensor,
        target_norm: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x=x, a=a, w=w, z=z, y=y_norm)
        pred_mean_norm = out[..., 0:1]
        pred_log_var_norm = out[..., 1:2]

        loss_mse = F.mse_loss(pred_mean_norm, target_norm)
        loss_gnll = gaussian_nll_loss(pred_mean_norm, pred_log_var_norm, target_norm)
        loss = alpha * loss_mse + (1 - alpha) * loss_gnll

        return loss, loss_mse, loss_gnll, pred_mean_norm, pred_log_var_norm

    def _optimization_step(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()

        if self.train_cfg.get("clip_grad", 0.0) > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_cfg["clip_grad"]
            ).item()
        else:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if len(parameters) == 0:
                grad_norm = 0.0
            else:
                device = parameters[0].grad.device
                grad_norm = torch.norm(
                    torch.stack(
                        [torch.norm(p.grad.detach(), 2).to(device) for p in parameters]
                    ),
                    2,
                ).item()

        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, (self.global_step + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_scale * self.base_lr

        self.global_step += 1
        self.optimizer.step()

        return grad_norm

    def _compute_batch_metrics(
        self,
        pred_mean_norm: torch.Tensor,
        pred_log_var_norm: torch.Tensor,
        target_norm: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        loss_gnll: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            y_mean_exp = y_mean.unsqueeze(-1) if y_mean.dim() == 2 else y_mean
            y_std_exp = y_std.unsqueeze(-1) if y_std.dim() == 2 else y_std

            pred_mean_orig = (pred_mean_norm * y_std_exp) + y_mean_exp
            target_orig = (target_norm * y_std_exp) + y_mean_exp

            pred_std_norm = torch.exp(0.5 * pred_log_var_norm)
            pred_std_orig = pred_std_norm * y_std_exp
            pred_var_orig = pred_std_orig.pow(2)

            residuals = pred_mean_orig - target_orig
            abs_error = residuals.abs()

            return {
                "sum_sq_err": residuals.pow(2).sum().item(),
                "sum_error": residuals.sum().item(),
                "sum_target": target_orig.sum().item(),
                "sum_sq_target": target_orig.pow(2).sum().item(),
                "sum_gnll": loss_gnll.item() * target_norm.numel(),
                "num_samples": target_norm.numel(),
                "variance_mean": pred_var_orig.mean().item(),
                "variance_sum": pred_var_orig.sum().item(),
                "cov50": (abs_error <= 0.674 * pred_std_orig).float().sum().item(),
                "cov95": (abs_error <= 1.960 * pred_std_orig).float().sum().item(),
                "cov99": (abs_error <= 2.576 * pred_std_orig).float().sum().item(),
            }

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model and optimizer state from a checkpoint file.
            - checkpoint_path: path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Start checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"]

        if "val_loss" in checkpoint:
            self.best_val_loss = checkpoint["val_loss"]

        logger.info(f"Resumed training from epoch {self.start_epoch}")


    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_loss_mse = 0.0
        total_loss_gnll = 0.0
        sum_sq_err = 0.0
        total_gnll = 0.0
        total_samples = 0
        total_variance = 0.0
        total_grad_norm = 0.0
        total_cov50 = 0
        total_cov95 = 0
        total_cov99 = 0

        epoch_start = time.time()

        # alpha epoch loss = 1 if epoch = 0, then linearly decay to 0 by epoch warmup_epochs
        alpha_epoch_loss = 1 - min(1, epoch / self.train_cfg.get("warmup_epochs", 300))

        with tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.train_cfg['epochs']} [Train]",
        ) as pbar:
            for batch in pbar:
                batch = self._move_batch(batch)
                x, a, w, z, y_norm, target_norm, y_mean, y_std = self._prepare_inputs(batch)

                # Ensure target_norm has shape (B, N_query, 1) to match predictions
                if target_norm.dim() == 2:
                    target_norm = target_norm.unsqueeze(-1)

                loss, loss_mse, loss_gnll, pred_mean_norm, pred_log_var_norm = (
                    self._forward_and_loss(x, a, w, z, y_norm, target_norm, alpha_epoch_loss)
                )

                total_grad_norm += self._optimization_step(loss)

                total_loss += loss.item()
                total_loss_mse += loss_mse.item()
                total_loss_gnll += loss_gnll.item()

                metrics = self._compute_batch_metrics(
                    pred_mean_norm, pred_log_var_norm, target_norm, y_mean, y_std, loss_gnll
                )

                sum_sq_err += metrics["sum_sq_err"]
                total_gnll += metrics["sum_gnll"]
                total_samples += metrics["num_samples"]
                total_variance += metrics["variance_mean"]
                total_cov50 += metrics["cov50"]
                total_cov95 += metrics["cov95"]
                total_cov99 += metrics["cov99"]

        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)

        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_loss_mse = total_loss_mse / num_batches
        avg_loss_gnll = total_loss_gnll / num_batches
        rmse = np.sqrt(sum_sq_err / max(total_samples, 1))
        gnll = total_gnll / max(total_samples, 1)
        avg_variance = total_variance / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        cov50 = total_cov50 / max(total_samples, 1)
        cov95 = total_cov95 / max(total_samples, 1)
        cov99 = total_cov99 / max(total_samples, 1)

        self.train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} train done in {format_time(epoch_time)}")

        return {
            "loss_total": avg_loss,
            "loss_mse": avg_loss_mse,
            "loss_gnll": avg_loss_gnll,
            "gnll": gnll,
            "rmse": rmse,
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
        # Existing accumulators
        total_loss_gnll = 0.0
        total_loss_mse = 0.0
        total_loss_total = 0.0
        total_variance = 0.0
        sum_sq_err = 0.0
        sum_gnll = 0.0
        total_samples = 0
        total_cov50 = 0
        total_cov95 = 0
        total_cov99 = 0
        
        sum_error = 0.0 
        sum_target = 0.0
        sum_sq_target = 0.0
        
        val_start = time.time()

        alpha_epoch_loss = 1 - min(1, epoch / self.train_cfg.get("warmup_epochs", 300))

        with torch.no_grad():
            with tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.train_cfg['epochs']} [Val]",
            ) as pbar:
                for batch in pbar:
                    batch = self._move_batch(batch)
                    x, a, w, z, y_norm, target_norm, y_mean, y_std = self._prepare_inputs(batch)
                    if target_norm.dim() == 2:
                        target_norm = target_norm.unsqueeze(-1)

                    loss_total, loss_mse, loss_gnll, pred_mean_norm, pred_log_var_norm = (
                        self._forward_and_loss(x, a, w, z, y_norm, target_norm, alpha_epoch_loss)
                    )

                    total_loss_gnll += loss_gnll.item()
                    total_loss_mse += loss_mse.item()
                    total_loss_total += loss_total.item()

                    metrics = self._compute_batch_metrics(
                        pred_mean_norm, pred_log_var_norm, target_norm, y_mean, y_std, loss_gnll
                    )

                    sum_sq_err += metrics["sum_sq_err"]
                    sum_error += metrics["sum_error"]
                    sum_target += metrics["sum_target"]
                    sum_sq_target += metrics["sum_sq_target"]
                    sum_gnll += metrics["sum_gnll"]
                    total_samples += metrics["num_samples"]
                    total_variance += metrics["variance_sum"]
                    total_cov50 += metrics["cov50"]
                    total_cov95 += metrics["cov95"]
                    total_cov99 += metrics["cov99"]

        val_time = time.time() - val_start
        self.validation_times.append(val_time)

        denom = max(total_samples, 1)
        num_batches = len(self.val_loader)
        avg_loss_gnll = total_loss_gnll / num_batches
        avg_loss_mse = total_loss_mse / num_batches
        avg_loss_total = total_loss_total / num_batches
        
        mse = sum_sq_err / denom
        rmse = np.sqrt(mse)
        
        # Bias-Variance Decomposition
        ite_bias = sum_error / denom
        ite_bias_sq = ite_bias ** 2
        ite_error_var = mse - ite_bias_sq 
        ss_tot = sum_sq_target - (sum_target ** 2) / denom
        if ss_tot > 1e-8:
            r2 = 1.0 - (sum_sq_err / ss_tot)
        else:
            r2 = 0.0
        
        # Global metrics
        gnll = sum_gnll / denom
        avg_variance = total_variance / denom

        cov95 = total_cov95 / denom
        cov50 = total_cov50 / denom
        cov99 = total_cov99 / denom

        self.val_losses.append(gnll)
        self.val_rmses.append(rmse)
        self.scheduler.step(gnll)

        logger.info(f"Epoch {epoch+1} val done: RMSE={rmse:.4f}, R2={r2:.4f}, Bias={ite_bias:.4f}")

        return {
            "loss_gnll": avg_loss_gnll,
            "loss_mse": avg_loss_mse,
            "loss_total": avg_loss_total,
            "rmse": rmse,
            "mse": mse,
            "r2": r2,
            "gnll": gnll,
            "bias": ite_bias,
            "bias_sq": ite_bias_sq,
            "error_var": ite_error_var,
            "variance": avg_variance,
            "time_s": val_time,
            "cov50": cov50,
            "cov95": cov95,
            "cov99": cov99,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        path_dir = os.path.join(
            self.train_cfg["save_dir"], self.train_cfg.get("exp_name", "cate_run")
        )
        os.makedirs(path_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch_times": self.epoch_times,
        }
        name = f"model_epoch_{epoch}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(path_dir, name)
        torch.save(checkpoint, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(path_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"  → New best model (val NLL: {val_loss:.4f})")

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log_metrics(
        self, epoch: int, train_metrics: dict, val_metrics: dict | None = None
    ) -> None:
        if epoch == 0:
            self.writer.add_scalar("Parameters/total_count", self.n_params, epoch)

        # Losses
        self.writer.add_scalar("Loss/train_total", train_metrics["loss_total"], epoch)
        self.writer.add_scalar("Loss/train_mse", train_metrics["loss_mse"], epoch)
        self.writer.add_scalar("Loss/train_gnll", train_metrics["loss_gnll"], epoch)

        # Metrics
        # self.writer.add_scalar("Metrics/train_rmse", train_metrics["rmse"], epoch)
        # self.writer.add_scalar("Metrics/train_gnll", train_metrics["gnll"], epoch)
        # self.writer.add_scalar(
        #     "Metrics/train_variance", train_metrics["variance"], epoch
        # )

        # Parameters / Optimization
        self.writer.add_scalar("Parameters/alpha", train_metrics["alpha"], epoch)
        self.writer.add_scalar(
            "Parameters/grad_norm", train_metrics["grad_norm"], epoch
        )
        self.writer.add_scalar(
            "Parameters/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
        )

        # Timing
        self.writer.add_scalar("Time/train_s", train_metrics["time_s"], epoch)

        # Calibration
        # self.writer.add_scalar("Calibration/train_cov50", train_metrics["cov50"], epoch)
        # self.writer.add_scalar("Calibration/train_cov95", train_metrics["cov95"], epoch)
        # self.writer.add_scalar("Calibration/train_cov99", train_metrics["cov99"], epoch)

        # Validation Metrics (if available)
        if val_metrics:
            self.writer.add_scalar("Loss/val_total", val_metrics["loss_total"], epoch)
            self.writer.add_scalar("Loss/val_mse", val_metrics["loss_mse"], epoch)
            self.writer.add_scalar("Loss/val_gnll", val_metrics["loss_gnll"], epoch)
            self.writer.add_scalar("Metrics/val_rmse", val_metrics["rmse"], epoch)
            self.writer.add_scalar("Metrics/val_gnll", val_metrics["gnll"], epoch)
            self.writer.add_scalar(
                "Metrics/val_variance", val_metrics["variance"], epoch
            )
            self.writer.add_scalar("Time/val_s", val_metrics["time_s"], epoch)
            self.writer.add_scalar("Calibration/val_cov50", val_metrics["cov50"], epoch)
            self.writer.add_scalar("Calibration/val_cov95", val_metrics["cov95"], epoch)
            self.writer.add_scalar("Calibration/val_cov99", val_metrics["cov99"], epoch)

            self.writer.add_scalar("Bias-Variance/val_bias_sq", val_metrics["bias_sq"], epoch)
            self.writer.add_scalar("Bias-Variance/val_error_var", val_metrics["error_var"], epoch)
            self.writer.add_scalar("Bias-Variance/mse", val_metrics["mse"], epoch)

            self.writer.add_scalar("Metrics/val_r2", val_metrics["r2"], epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        training_start = time.time()
        os.makedirs(self.train_cfg["save_dir"], exist_ok=True)

        early_stop_count = 0
        best_val_loss = getattr(self, "best_val_loss", float("inf"))

        reset_freq = self.data_cfg.get("reset_freq_train", -1)

        epoch = self.start_epoch
        last_val_nll = float('inf')
        while epoch < self.train_cfg["epochs"]:

            if hasattr(self.train_loader.dataset, 'set_reset_state') and reset_freq > 0:
                self.train_loader.dataset.set_reset_state(epoch // reset_freq)
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = None
            if epoch % self.train_cfg.get("val_freq", 1) == 0:
                val_metrics = self.validate(epoch)
                last_val_nll = val_metrics["loss_gnll"]

            # Extract metrics for console logging
            train_nll = train_metrics["loss_gnll"]
            val_nll = val_metrics["loss_gnll"] if val_metrics else float("nan")
            val_rmse = val_metrics["rmse"] if val_metrics else float("nan")

            logger.info(
                f"Epoch {epoch+1}/{self.train_cfg['epochs']}  "
                f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}  "
                f"val_rmse={val_rmse:.4f}"
            )
            self.log_metrics(epoch, train_metrics, val_metrics)

            if (epoch + 1) % self.train_cfg.get("save_freq", 10) == 0 or epoch == 0:
                # Save using the last recorded validation NLL metric
                self.save_checkpoint(epoch + 1, last_val_nll)

            if val_metrics:
                if val_nll < best_val_loss:
                    best_val_loss = val_nll
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.train_cfg["early_stop"]:
                        logger.info(f"Early stopping after {epoch+1} epochs.")
                        break
            epoch += 1

        self.total_training_time = time.time() - training_start
        # Final checkpoint uses the last recorded val_nll
        self.save_checkpoint(epoch + 1, last_val_nll, tag="final")
        self.writer.close()
        logger.info("Training complete!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()
    set_seed(config["training"]["seed"])

    logger.info(f"Training config: {config['training']}")
    trainer = CATETrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        t = trainer.total_training_time
        logger.info(f"Total training time:        {format_time(t)}")
        if trainer.epoch_times:
            logger.info(
                f"Mean epoch time:            {format_time(np.mean(trainer.epoch_times))}"
            )
        if trainer.validation_times:
            logger.info(
                f"Mean validation time:       {format_time(np.mean(trainer.validation_times))}"
            )


if __name__ == "__main__":
    main()
