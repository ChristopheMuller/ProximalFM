from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb
import yaml

from proximalfm._model.tabicl import TabICL
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._genload import LoadPriorDataset
from proximalfm.train._optim import get_scheduler
from proximalfm.train._train_config import build_parser

warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


def get_parameter_state(param_name: str, rules: dict) -> str:
    """Finds the longest matching prefix in the rules dict for a given parameter name."""
    best_match = "default"
    longest_prefix = 0
    for prefix, state in rules.items():
        if prefix == "default":
            continue
        if param_name.startswith(prefix) and len(prefix) > longest_prefix:
            best_match = prefix
            longest_prefix = len(prefix)
    return rules.get(best_match, rules.get("default", "scratch"))


with open("transfer_rules.yaml", "r") as f:
    config = yaml.safe_load(f)
TRANSFER_RULES = config.get("transfer_rules", {})


def compute_crps_loss(
    logits: torch.Tensor, y_target: torch.Tensor, bin_edges: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Ranked Probability Score (Discrete CRPS).
    Requires zero hyperparameters and naturally respects the distance between bins.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # 1. Compute predicted CDF
    pred_cdf = torch.cumsum(probs, dim=-1)

    # 2. Compute target CDF
    # It is an ideal step function: 0 for bins below the target, 1 for bins above
    y_target_expanded = y_target.unsqueeze(-1)
    # bin_edges[1:] represents the upper edge of each bin
    target_cdf = (bin_edges[1:].view(1, 1, -1) >= y_target_expanded).float()

    # 3. CRPS is the Mean Squared Error between the CDFs
    crps = torch.mean((pred_cdf - target_cdf) ** 2, dim=-1)

    return crps.mean()


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        self.configure_validation_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(
                original_batch_size / self.ddp_world_size
            )

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            current_time = timeit.default_timer()
            name_with_time = f"{self.config.wandb_name}_{current_time:.0f}"

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=name_with_time,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""

        self.model_config = {
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "col_affine": self.config.col_affine,
            "col_ssmax": self.config.col_ssmax,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "row_rope_interleaved": self.config.row_rope_interleaved,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "icl_ssmax": self.config.icl_ssmax,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "bias_free_ln": self.config.bias_free_ln,
            "recompute": self.config.recompute,
            "sigma": self.config.sigma,
        }
        model = TabICL(**self.model_config)
        model.to(device=self.config.device)

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.master_process:
                print(f"Applying Surgical Transfer Learning from {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path, map_location=self.config.device, weights_only=True
            )
            old_state_dict = checkpoint.get("state_dict", checkpoint)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            # 1. Filter weights based on TRANSFER_RULES
            for name, weight in old_state_dict.items():
                if (
                    name in model_state_dict
                    and weight.shape == model_state_dict[name].shape
                ):
                    state = get_parameter_state(name, TRANSFER_RULES)
                    if state in ["finetune", "freeze"]:
                        filtered_state_dict[name] = weight

            # 2. Load the allowed weights (strict=False leaves "scratch" weights as random)
            model.load_state_dict(filtered_state_dict, strict=False)

        # 3. Apply freezing logic
        for name, param in model.named_parameters():
            state = get_parameter_state(name, TRANSFER_RULES)
            if state == "freeze":
                param.requires_grad = False
                if self.master_process and self.config.wandb_log:
                    pass
            else:
                param.requires_grad = True

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            )
            print(
                f"Model built: {num_params:,} trainable params | {frozen_params:,} frozen params."
            )

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(
                model, device_ids=[self.ddp_local_rank], broadcast_buffers=False
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        """Set up a tabular dataset generator for proximal synthetic data."""
        if self.config.prior_dir is None:
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_dim_U=self.config.min_dim_U,
                max_dim_U=self.config.max_dim_U,
                min_dim_W=self.config.min_dim_W,
                max_dim_W=self.config.max_dim_W,
                min_dim_Z=self.config.min_dim_Z,
                max_dim_Z=self.config.max_dim_Z,
                min_dim_X=self.config.min_dim_X,
                max_dim_X=self.config.max_dim_X,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        # Create dataloader for efficient loading and prefetching
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=8,
            prefetch_factor=16,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device
            if self.config.prior_device == "cpu"
            else "",
        )

    def configure_validation_prior(self):
        """Set up the dataloader for the fixed validation set."""
        if self.config.val_dir is None:
            self.val_dataloader = None
            return

        # 1. Count exactly how many batch files exist in the folder
        all_val_files = [
            f
            for f in os.listdir(self.config.val_dir)
            if f.startswith("batch_") and f.endswith(".pt")
        ]
        total_val_files = len(all_val_files)

        if total_val_files == 0:
            raise ValueError(f"No validation files found in {self.config.val_dir}")

        self.val_batches_per_rank = len(
            [
                i
                for i in range(total_val_files)
                if i % self.ddp_world_size == self.ddp_rank
            ]
        )

        val_dataset = LoadPriorDataset(
            data_dir=self.config.val_dir,
            batch_size=self.config.batch_size,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            start_from=0,
            delete_after_load=False,
            device=self.config.prior_device,
        )

        if self.master_process:
            print(
                f"Validation dataset configured from {self.config.val_dir} ({total_val_files} total files)."
            )

        # Create dataloader
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=1,  # Keep this at 1 for static validation to avoid worker deadlock
            prefetch_factor=2,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device
            if self.config.prior_device == "cpu"
            else "",
        )

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

        self.optimizer = optim.AdamW(
            params=self.raw_model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled.")
            self.amp_ctx = torch.autocast(
                device_type="cuda",
                dtype=torch.float16
                if self.config.dtype == "float16"
                else torch.float32,
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
            )[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load optimizer and scheduler state from checkpoint (Weights are handled in build_model)."""
        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return

        # If you are doing transfer learning, you almost NEVER want to load the old optimizer states.
        if self.config.only_load_model:
            print("Transfer learning active: Starting with a fresh optimizer.")
            return

        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.device, weights_only=True
        )

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.curr_step = checkpoint["curr_step"]
        print(f"Resuming optimizer state at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps), desc="Step", leave=True
            )
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            # Get the next batch
            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})

                # Update progress bar with rounded values for cleaner display
                step_progress.set_postfix(
                    **{
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in results.items()
                    }
                )

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if (
                        is_temp_save
                        and not is_perm_save
                        and self.config.max_checkpoints > 0
                    ):
                        self.manage_checkpoint()

            # Must run on ALL processes to prevent DDP hangs during all_reduce
            if (
                self.val_dataloader is not None
                and self.curr_step % self.config.val_interval == 0
            ):
                val_results = self.validate()
                results.update(val_results)

                # Log to console only on master
                if self.master_process:
                    print(
                        f"\n[Step {self.curr_step}] Validation | "
                        + " | ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
                    )

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor
            Sequence lengths for each dataset, shape ``(micro_batch_size,)``.

        micro_train_size : Tensor
            Training sizes (split positions) for each dataset, shape
            ``(micro_batch_size,)``.

        Returns
        -------
        seq_len : int
            The common sequence length for the micro batch.

        train_size : int
            The common train size for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if len(torch.unique(micro_seq_len)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same sequence length."
            )

        if len(torch.unique(micro_train_size)) > 1:
            raise ValueError(
                "All datasets in the micro batch must have the same training size."
            )

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(
        self,
        micro_X,
        micro_W,
        micro_Z,
        micro_A,
        micro_Y_obs,
        micro_Y0,
        micro_Y1,
        micro_d_X,
        micro_d_W,
        micro_d_Z,
        seq_len,
    ):
        """Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (``micro_d.max()``) respectively. This
        optimizes memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor
            Input features per dataset of shape ``(B, T, H)``.

        micro_y : Tensor
            Target labels per dataset of shape ``(B, T)``.

        micro_d : Tensor
            Number of active features per dataset of shape ``(B,)``.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        micro_X : Tensor
            Truncated features of shape ``(B, seq_len, micro_d.max())``.

        micro_y : Tensor
            Truncated labels of shape ``(B, seq_len)``.
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]
            micro_W = micro_W[:, :seq_len]
            micro_Z = micro_Z[:, :seq_len]
            micro_A = micro_A[:, :seq_len]
            micro_Y_obs = micro_Y_obs[:, :seq_len]
            micro_Y0 = micro_Y0[:, :seq_len]
            micro_Y1 = micro_Y1[:, :seq_len]

        # Truncate feature dimensions
        max_X = micro_d_X.max().item()
        if micro_X.shape[-1] > max_X:
            micro_X = micro_X[..., :max_X]

        max_W = micro_d_W.max().item()
        if micro_W.shape[-1] > max_W:
            micro_W = micro_W[..., :max_W]

        max_Z = micro_d_Z.max().item()
        if micro_Z.shape[-1] > max_Z:
            micro_Z = micro_Z[..., :max_Z]

        return micro_X, micro_W, micro_Z, micro_A, micro_Y_obs, micro_Y0, micro_Y1

    def run_micro_batch(
        self, micro_batch, micro_batch_idx, num_micro_batches, is_train=True
    ):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors
            for the micro batch.

        micro_batch_idx : int
            Index of the current micro batch.

        num_micro_batches : int
            Total number of micro batches.

        Returns
        -------
        dict
            Result dictionary with 'ce' and 'accuracy' keys.
        """
        # Unpack the 12 items
        (
            micro_U,
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
            micro_d_U,
            micro_d_X,
            micro_d_W,
            micro_d_Z,
            micro_seq_len,
            micro_train_size,
        ) = micro_batch

        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)

        (
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
        ) = self.align_micro_batch(
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
            micro_d_X,
            micro_d_W,
            micro_d_Z,
            seq_len,
        )

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_W = micro_W.to(self.config.device)
        micro_Z = micro_Z.to(self.config.device)
        micro_A = micro_A.to(self.config.device)
        micro_Y_obs = micro_Y_obs.to(self.config.device)
        micro_Y0 = micro_Y0.to(self.config.device)
        micro_Y1 = micro_Y1.to(self.config.device)
        micro_d_X = micro_d_X.to(self.config.device)
        micro_d_W = micro_d_W.to(self.config.device)
        micro_d_Z = micro_d_Z.to(self.config.device)

        def safe_standardize(x):
            mu = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            # Prevent division by zero for zero-variance or zero-padded features
            std = torch.where(std < 1e-5, torch.ones_like(std), std)
            return (x - mu) / std

        micro_X = safe_standardize(micro_X)
        micro_W = safe_standardize(micro_W)
        micro_Z = safe_standardize(micro_Z)

        y_train_obs = micro_Y_obs[:, :train_size]

        # Squeeze A down to (B, T) to match y_train_obs
        A_train = micro_A[:, :train_size].squeeze(-1)
        n0 = (A_train == 0).sum(dim=1)  # (B,)
        n1 = (A_train == 1).sum(dim=1)  # (B,)
        if (n0 == 0).any() or (n1 == 0).any():
            # print(
            #     f"[Step {self.curr_step}] Skipping micro-batch: degenerate treatment groups"
            # )
            return (
                self.last_results
                if hasattr(self, "last_results")
                else {
                    "loss_total": 0.0,
                    "mse_cate": 0.0,
                    "loss_y0": 0.0,
                    "loss_y1": 0.0,
                }
            )

        # Test targets for loss
        y_test_0 = micro_Y0[:, train_size:]
        y_test_1 = micro_Y1[:, train_size:]
        true_cate = y_test_1 - y_test_0

        if self.ddp:
            self.model.require_backward_grad_sync = (
                micro_batch_idx == num_micro_batches - 1
            )

        with self.amp_ctx:
            # 1. Get shift/scale from the context
            (
                y0_shift,
                y0_scale,
                y1_shift,
                y1_scale,
            ) = self.raw_model._get_y0_y1_shift_scale(y_train_obs, A_train)

            y0_scale = torch.where(y0_scale < 1e-7, torch.ones_like(y0_scale), y0_scale)
            y1_scale = torch.where(y1_scale < 1e-7, torch.ones_like(y1_scale), y1_scale)

            # 2. Standardize Context Y BEFORE feeding it to the model
            y_train_obs_std = torch.where(
                A_train == 1,
                (y_train_obs - y1_shift) / y1_scale,
                (y_train_obs - y0_shift) / y0_scale,
            )

            # 3. Model predicts logits of shape (B, test_size, nbins * 2)
            pred_logits = self.model(
                micro_X,
                micro_Z,
                micro_W,
                micro_A,
                y_train_obs_std,
                micro_d_X,
                micro_d_Z,
                micro_d_W,
            )
            if pred_logits.isnan().any():
                # skip this loop,
                # print(
                #     f"\n--- NaN detected in model predictions at step {self.curr_step} ---"
                # )
                return (
                    self.last_results
                    if hasattr(self, "last_results")
                    else {
                        "loss_total": 0.0,
                        "mse_cate": 0.0,
                        "loss_y0": 0.0,
                        "loss_y1": 0.0,
                    }
                )

            nbins = self.raw_model.nbins
            logits_y0 = pred_logits[..., :nbins]
            logits_y1 = pred_logits[..., nbins:]

            # 4. Standardize Ground Truth targets using context statistics
            y_test_0_std = (y_test_0 - y0_shift) / y0_scale
            y_test_1_std = (y_test_1 - y1_shift) / y1_scale

            # 5. Compute CRPS Loss for BOTH potential outcomes
            loss_y0 = compute_crps_loss(
                logits_y0,
                y_test_0_std,
                self.raw_model.bin_edges,
            )
            loss_y1 = compute_crps_loss(
                logits_y1,
                y_test_1_std,
                self.raw_model.bin_edges,
            )

            total_loss = loss_y0 + loss_y1

        if is_train:
            scaled_loss = total_loss / num_micro_batches
            self.scaler.scale(scaled_loss).backward()
            reported_loss_total = scaled_loss.item()
        else:
            reported_loss_total = total_loss.item()

        # 6. Calculate Mean Squared Error of the CATE for WandB tracking (No Gradients)
        with torch.no_grad():
            probs_y0 = F.softmax(logits_y0, dim=-1)
            probs_y1 = F.softmax(logits_y1, dim=-1)

            # Expected value in standardized space
            mean_y0_std = torch.sum(probs_y0 * self.raw_model.bin_centers, dim=-1)
            mean_y1_std = torch.sum(probs_y1 * self.raw_model.bin_centers, dim=-1)

            # Unstandardize back to reality
            pred_y0 = (mean_y0_std * y0_scale) + y0_shift
            pred_y1 = (mean_y1_std * y1_scale) + y1_shift

            pred_cate = pred_y1 - pred_y0
            mse_cate = F.mse_loss(pred_cate, true_cate)

            micro_results = {
                "loss_total": reported_loss_total if is_train else total_loss.item(),
                "mse_cate": (mse_cate.item() / num_micro_batches)
                if is_train
                else mse_cate.item(),
                "loss_y0": (loss_y0.item() / num_micro_batches)
                if is_train
                else loss_y0.item(),
                "loss_y1": (loss_y1.item() / num_micro_batches)
                if is_train
                else loss_y1.item(),
            }
            self.last_results = micro_results

        return micro_results

    def run_batch(self, batch):
        """Train the model on a batch of datasets.

        Handles gradient accumulation by splitting the batch into micro-batches.
        Supports variable-sized datasets by padding. Skips micro-batches on CUDA
        OOM errors. Updates model parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch : tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence
            lengths).

        Returns
        -------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Pad nested tensors to the same size
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Split the batch into micro-batches along the first dimension
        num_micro_batches = math.ceil(
            self.config.batch_size / self.config.micro_batch_size
        )
        micro_batches = [
            torch.split(t, self.config.micro_batch_size, dim=0) for t in batch
        ]
        micro_batches = list(zip(*micro_batches))

        results = {"loss_total": 0.0, "mse_cate": 0.0, "loss_y0": 0.0, "loss_y1": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, idx, num_micro_batches
                )
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Clip the gradient
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results

    def run_val_batch(self, batch):
        """Evaluate the model on a batch without gradients."""
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        num_micro_batches = math.ceil(
            self.config.batch_size / self.config.micro_batch_size
        )
        micro_batches = [
            torch.split(t, self.config.micro_batch_size, dim=0) for t in batch
        ]
        micro_batches = list(zip(*micro_batches))

        results = {"loss_total": 0.0, "mse_cate": 0.0, "loss_y0": 0.0, "loss_y1": 0.0}
        valid_micro_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                with self.amp_ctx:
                    micro_results = self.run_micro_batch(
                        micro_batch, idx, num_micro_batches, is_train=False
                    )
                    for k, v in micro_results.items():
                        results[k] += v
                valid_micro_batches += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        # Average across the valid micro-batches for this specific batch
        if valid_micro_batches > 0:
            for k in results.keys():
                results[k] /= valid_micro_batches

        return results

    @torch.no_grad()
    def validate(self):
        """Run a full validation pass over the val_dir."""
        # Safety check if there are no batches for this rank
        if self.val_dataloader is None or getattr(self, "val_batches_per_rank", 0) == 0:
            return {}

        self.model.eval()
        total_metrics = {
            "loss_total": 0.0,
            "mse_cate": 0.0,
            "loss_y0": 0.0,
            "loss_y1": 0.0,
        }

        # We recreate the iterator to loop through the directory from the start
        val_iterator = iter(self.val_dataloader)

        # EXACTLY loop self.val_batches_per_rank times to prevent timeout
        for _ in range(self.val_batches_per_rank):
            try:
                batch = next(val_iterator)
                batch_results = self.run_val_batch(batch)
                for k, v in batch_results.items():
                    total_metrics[k] += v
            except StopIteration:
                # Fallback just in case the dataset does raise StopIteration
                break

        self.model.train()

        # Average the metrics across the batches processed by this rank
        for k in total_metrics.keys():
            total_metrics[k] /= self.val_batches_per_rank

        # If DDP, average the metrics across all GPUs
        if self.ddp:
            for k, v in total_metrics.items():
                tensor_v = torch.tensor(v, device=self.config.device)
                torch.distributed.all_reduce(
                    tensor_v, op=torch.distributed.ReduceOp.AVG
                )
                total_metrics[k] = tensor_v.item()

        # Prefix keys with "val_" for clean logging
        return {f"val_{k}": v for k, v in total_metrics.items()}


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
