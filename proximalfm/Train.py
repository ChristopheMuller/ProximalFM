from __future__ import annotations

import logging
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import your new model instead of PerFeatureTransformerCATE
from proximalfm.Model import ProximalNanoTabICLv2
from proximalfm.Structure import ProximalPriorConfig
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
        self.train_cfg = config["training"]
        self.data_cfg = config["data"]
        self.prior_cfg = config["prior"]
        self.model_cfg = config["model"]

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

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        dtype_str = self.model_cfg.get("dtype", "float32")
        self.dtype = getattr(torch, dtype_str)

        # Instantiate the new Proximal NanoTabICL model
        self.model = ProximalNanoTabICLv2(
            out_dim=2, # 2 dimensions: 1 for CATE mean, 1 for log_var
            embed_dim=self.model_cfg.get("embed_dim", 64),
            col_num_blocks=self.model_cfg.get("col_num_blocks", 3), # mapping old config to new
            row_num_blocks=self.model_cfg.get("row_num_blocks", 3),
            icl_num_blocks=self.model_cfg.get("icl_num_blocks", 12),
            col_nhead=self.model_cfg.get("col_nhead", 8),
            row_nhead=self.model_cfg.get("row_nhead", 8),
            icl_nhead=self.model_cfg.get("icl_nhead", 8),
            feature_group_size_x=self.prior_cfg["d_x"],
            feature_group_size_w=self.prior_cfg["d_w"],
            feature_group_size_z=self.prior_cfg["d_z"],
        )
        self.model.to(self.device, dtype=self.dtype)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

        # ------------------------------------------------------------------
        # Optimiser + scheduler
        # ------------------------------------------------------------------
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.warmup_steps = self.train_cfg.get("warmup_steps", 1000)
        self.global_step = 0
        self.base_lr = self.train_cfg["lr"]

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        prior_config = ProximalPriorConfig(
            d_x=self.prior_cfg["d_x"],
            d_z=self.prior_cfg["d_z"],
            d_w=self.prior_cfg["d_w"],
            d_u=self.prior_cfg["d_u"],
            dtype=self.dtype,
        )

        loaders = create_causal_data_loaders(
            prior_config=prior_config,
            batch_size=self.data_cfg["batch_size"],
            samples_per_episode=self.data_cfg["samples_per_episode"],
            train_episodes=self.data_cfg["train_episodes"],
            val_episodes=self.data_cfg["val_episodes"],
            num_workers=self.data_cfg["num_workers"],
            seed=self.train_cfg["seed"],
        )
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        n = self.data_cfg["samples_per_episode"]
        self.single_eval_pos = int(n * self.data_cfg.get("context_fraction", 0.8))

        # ------------------------------------------------------------------
        # Logging + bookkeeping
        # ------------------------------------------------------------------
        os.makedirs(self.train_cfg["log_dir"], exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.train_cfg["log_dir"])

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_rmses: list[float] = []
        self.best_val_loss = float("inf")
        self.epoch_times: list[float] = []
        self.validation_times: list[float] = []
        self.total_training_time = 0.0

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

    def _prepare_inputs(
            self, batch: dict
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Reformat a DataLoader batch for the model's forward signature.
            DataLoader returns (N, B, D). NanoTab expects (B, N, D).
            A and Y must be strictly sliced to n_train (single_eval_pos) rows and squeezed to (B, N).
            """

            x = batch["X"].transpose(0, 1)  
            z = batch["Z"].transpose(0, 1)  
            w = batch["W"].transpose(0, 1)  

            a = batch["A"][:self.single_eval_pos].transpose(0, 1).squeeze(-1)
            y = batch["Y"][:self.single_eval_pos].transpose(0, 1).squeeze(-1)

            ite_query = batch["ITE"][self.single_eval_pos:].transpose(0, 1)

            return x, a, w, z, y, ite_query

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        epoch_start = time.time()

        with tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.train_cfg['epochs']} [Train]",
        ) as pbar:
            for batch in pbar:
                batch = self._move_batch(batch)
                x, a, w, z, y, target = self._prepare_inputs(batch)

                # NanoTabICLv2 forward pass
                out = self.model(x=x, a=a, w=w, z=z, y=y)
                
                # out shape is (B, N_query, 2). Split into mean and log_var
                pred_mean = out[..., 0:1]
                pred_log_var = out[..., 1:2]

                # alpha epoch loss = 1 if epoch = 0, then linearly decay to 0 by epoch warmup_epochs
                alpha_epoch_loss = 1 - min(1, epoch / self.train_cfg.get("warmup_epochs", 300))

                loss_mse = F.mse_loss(pred_mean, target)
                loss_gnll = gaussian_nll_loss(pred_mean, pred_log_var, target)

                loss = alpha_epoch_loss * loss_mse + (1 - alpha_epoch_loss) * loss_gnll

                self.optimizer.zero_grad()
                loss.backward()
                if self.train_cfg.get("clip_grad", 0.0) > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg["clip_grad"]
                    )

                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, (self.global_step + 1) / self.warmup_steps)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_scale * self.base_lr
                
                self.global_step += 1
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"nll_loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start
        self.epoch_times.append(epoch_time)
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} train done in {format_time(epoch_time)}")
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        sum_sq_err = 0.0
        total_samples = 0
        val_start = time.time()

        with torch.no_grad():
            with tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.train_cfg['epochs']} [Val]",
            ) as pbar:
                for batch in pbar:
                    batch = self._move_batch(batch)
                    x, a, w, z, y, target = self._prepare_inputs(batch)

                    out = self.model(x=x, a=a, w=w, z=z, y=y)

                    pred_mean = out[..., 0:1]
                    pred_log_var = out[..., 1:2]

                    loss = gaussian_nll_loss(pred_mean, pred_log_var, target)
                    total_loss += loss.item()
                    sum_sq_err += (pred_mean - target).pow(2).sum().item()
                    total_samples += target.numel()

                    pbar.set_postfix({"nll_loss": f"{loss.item():.4f}"})

        val_time = time.time() - val_start
        self.validation_times.append(val_time)

        avg_loss = total_loss / len(self.val_loader)
        rmse = np.sqrt(sum_sq_err / max(total_samples, 1))
        self.val_losses.append(avg_loss)
        self.val_rmses.append(rmse)
        self.scheduler.step(avg_loss)
        logger.info(f"Epoch {epoch+1} val done in {format_time(val_time)}")
        return avg_loss, rmse

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        os.makedirs(self.train_cfg["save_dir"], exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "epoch_times": self.epoch_times,
        }
        name = f"model_epoch_{epoch}{('_' + tag) if tag else ''}.pth"
        path = os.path.join(self.train_cfg["save_dir"], name)
        torch.save(checkpoint, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.train_cfg["save_dir"], "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"  → New best model (val NLL: {val_loss:.4f})")

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log_metrics(
        self, epoch: int, train_loss: float, val_loss: float, rmse: float
    ) -> None:
        self.writer.add_scalar("Loss/train_nll", train_loss, epoch)
        self.writer.add_scalar("Loss/val_nll", val_loss, epoch)
        self.writer.add_scalar("Metrics/val_rmse", rmse, epoch)
        self.writer.add_scalar(
            "Learning_rate",
            self.optimizer.param_groups[0]["lr"],
            epoch,
        )
        if self.epoch_times:
            self.writer.add_scalar("Time/epoch_s", self.epoch_times[-1], epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        training_start = time.time()
        os.makedirs(self.train_cfg["save_dir"], exist_ok=True)

        early_stop_count = 0
        best_val_loss = float("inf")

        for epoch in range(self.train_cfg["epochs"]):
            train_loss = self.train_epoch(epoch)
            if epoch % self.train_cfg.get("val_freq", 1) == 0:
                val_loss, rmse = self.validate(epoch)

            logger.info(
                f"Epoch {epoch+1}/{self.train_cfg['epochs']}  "
                f"train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  "
                f"val_rmse={rmse:.4f}"
            )
            self.log_metrics(epoch, train_loss, val_loss, rmse)

            if (epoch + 1) % self.train_cfg.get("save_freq", 10) == 0:
                self.save_checkpoint(epoch + 1, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= self.train_cfg["early_stop"]:
                    logger.info(f"Early stopping after {epoch+1} epochs.")
                    break

        self.total_training_time = time.time() - training_start
        self.save_checkpoint(self.train_cfg["epochs"], val_loss, tag="final")
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
