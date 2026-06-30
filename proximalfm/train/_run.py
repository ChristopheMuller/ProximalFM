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
import datetime
import wandb
import yaml

from proximalfm._model.proximalfm import ProximalFM
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._genload import LoadPriorDataset
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP
from proximalfm.prior._linear_dataset import LinearPriorDataset
from proximalfm.train._optim import get_scheduler
from proximalfm.train._train_config import build_parser
from proximalfm.train._eval import CausalEvaluator


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


def is_oom_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is an out-of-memory error from any accelerator.

    Covers CUDA (``torch.cuda.OutOfMemoryError``, a ``RuntimeError`` subclass) as
    well as MPS / generic backends, which surface OOM as a plain ``RuntimeError``
    whose message mentions running out of memory. Note that *asynchronous* MPS
    command-buffer failures (kIOGPUCommandBufferCallbackErrorOutOfMemory) may not
    be raised at the originating op and can leave the context unusable; those are
    not reliably recoverable even when caught here.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "insufficient memory" in msg
            or "mps backend out of memory" in msg
        )
    return False


def empty_accelerator_cache(device: str) -> None:
    """Free cached allocator memory on the active accelerator (CUDA or MPS)."""
    device = str(device)
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device and hasattr(torch, "mps"):
        torch.mps.empty_cache()


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


def compute_hl_gaussian_cross_entropy_loss(
    logits: torch.Tensor,
    y_target: torch.Tensor,
    bin_edges: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Histogram Loss with Gaussian soft targets.

    For each scalar target y_i, constructs a discrete target distribution p_i
    by integrating N(y_i, sigma^2) over the histogram bins, then computes

        HL = - sum_l p_l log q_l

    where q_l = softmax(logits)_l.

    Args:
        logits:
            Tensor of shape (..., nbins).
        y_target:
            Tensor of shape (...).
        bin_edges:
            Tensor of shape (nbins + 1,).
        sigma:
            Standard deviation of the Gaussian target on the standardized target scale.
        eps:
            Numerical stability constant.

    Returns:
        Scalar tensor.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if y_target.shape != logits.shape[:-1]:
        raise ValueError(
            f"y_target shape {y_target.shape} must match logits shape "
            f"{logits.shape[:-1]} except for the last dimension."
        )

    # Compute target probabilities in fp32 for numerical stability,
    # especially under AMP.
    logits_fp32 = logits.float()
    y_target_fp32 = y_target.float()
    bin_edges_fp32 = bin_edges.to(device=logits.device, dtype=torch.float32)

    lower_edges = bin_edges_fp32[:-1]
    upper_edges = bin_edges_fp32[1:]

    # Shape for broadcasting edges against y_target:
    # y_target: (...), edges: (nbins,) -> (..., nbins)
    edge_shape = (1,) * y_target_fp32.ndim + (-1,)
    lower_edges = lower_edges.view(edge_shape)
    upper_edges = upper_edges.view(edge_shape)

    y = y_target_fp32.unsqueeze(-1)

    normalizer = math.sqrt(2.0) * sigma

    cdf_upper = 0.5 * (1.0 + torch.erf((upper_edges - y) / normalizer))
    cdf_lower = 0.5 * (1.0 + torch.erf((lower_edges - y) / normalizer))

    p = cdf_upper - cdf_lower

    # Numerical safety.
    p = p.clamp_min(0.0)

    # Truncated-and-renormalized Gaussian over the available bin support.
    p = p / (p.sum(dim=-1, keepdim=True) + eps)

    log_probs = F.log_softmax(logits_fp32, dim=-1)

    loss = -torch.sum(p * log_probs, dim=-1)

    return loss.mean()


def compute_gaussian_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood for a Gaussian distribution.
    Assumes isotropic gaussian (used here for ATE modeling).
    """
    nll = 0.5 * torch.exp(-log_var) * (target - mean) ** 2 + 0.5 * log_var
    return nll.mean()


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


class ExperimentLogger:
    """Handles communication with wandb to abstract tracking away from the main loop."""

    def __init__(self, config, is_master=True):
        self.config = config
        self.is_master = is_master
        self.use_wandb = config.wandb_log and self.is_master
        self.run = None
        self.last_logged_step = -1

        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        if hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
        if self.config.wandb_id is None:
            if os.path.exists(id_path):
                with open(id_path, "r") as f:
                    self.config.wandb_id = f.read().strip()

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_with_time = f"{self.config.wandb_name}_{current_time}"

        self.run = wandb.init(
            dir=self.config.wandb_dir,
            project=self.config.wandb_project,
            name=name_with_time,
            id=self.config.wandb_id,
            config=self.config,
            resume="allow",
            mode=self.config.wandb_mode,
        )

        with open(id_path, "w") as f:
            f.write(self.run.id)

    def log(self, metrics_dict: dict, step: int):
        """Logs metrics to wandb if initialized."""
        if self.use_wandb and self.run is not None:
            if step > self.last_logged_step:
                wandb.log(metrics_dict, step=step)
                self.last_logged_step = step

    def log_summary(self, summary_dict: dict):
        """Logs static summary metrics (e.g., parameter counts)."""
        if self.use_wandb and self.run is not None:
            for k, v in summary_dict.items():
                self.run.summary[k] = v


class Trainer:
    """This class handles the complete training lifecycle for ProximalFM."""

    def __init__(self, config):
        self.config = config

        self.loss_type = getattr(self.config, "loss_type", "bins_po")

        # HL-Gaussian sigma schedule.
        self.hl_start_sigma = getattr(self.config, "hl_start_sigma", 0.3)
        self.hl_warmup_sigma_steps = getattr(self.config, "hl_warmup_sigma_steps", 1000)
        self.hl_linear_decay_steps = getattr(self.config, "hl_linear_decay_steps", 5000)
        self.hl_final_sigma = getattr(self.config, "hl_final_sigma", 0.01)

        self.configure_ddp()

        # Initialize the dedicated logger
        self.logger = ExperimentLogger(self.config, self.master_process)

        self.build_model()
        self.configure_prior()
        self.configure_validation_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

        self.train_evaluator = CausalEvaluator()
        self.val_evaluator = CausalEvaluator()
        self.eval_buffer_episodes = getattr(self.config, "eval_buffer_episodes", 64)

        self.val_micro_batch_size = (
            getattr(self.config, "val_micro_batch_size", None)
            or self.config.micro_batch_size
        )

    def configure_ddp(self):
        """Set up distributed training and system configuration."""
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            self.config.prior_device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

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

        self.curr_step = 0

        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def build_model(self):
        """Build and initialize the ProximalFM model."""

        self.model_config = {
            "loss_type": self.loss_type,
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
        }
        model = ProximalFM(**self.model_config)
        model.to(device=self.config.device)

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        loaded_params_count = 0
        total_model_params = sum(p.numel() for p in model.parameters())

        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.master_process:
                print(f"Applying Surgical Transfer Learning from {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path, map_location=self.config.device, weights_only=True
            )
            old_state_dict = checkpoint.get("state_dict", checkpoint)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}

            clean_old_state_dict = {}
            for k, v in old_state_dict.items():
                clean_key = k
                # Remove DDP wrapper prefix if present
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                # Remove torch.compile wrapper prefix if present
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key[10:]
                # Catch edge cases where DDP wraps a compiled model (module._orig_mod.)
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key[10:]

                clean_old_state_dict[clean_key] = v

            # 1. Filter weights based on TRANSFER_RULES using the clean state dict
            for name, weight in clean_old_state_dict.items():
                if (
                    name in model_state_dict
                    and weight.shape == model_state_dict[name].shape
                ):
                    state = get_parameter_state(name, TRANSFER_RULES)
                    if state in ["finetune", "freeze"]:
                        filtered_state_dict[name] = weight
                        # Track the count of weights successfully mapped over
                        loaded_params_count += weight.numel()

            # 2. Load the allowed weights (strict=False leaves "scratch" weights as random)
            model.load_state_dict(filtered_state_dict, strict=False)
        # 3. Apply freezing logic
        for name, param in model.named_parameters():
            state = get_parameter_state(name, TRANSFER_RULES)
            if state == "freeze":
                param.requires_grad = False
            else:
                param.requires_grad = True

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            )

            # Calculate what percentage of the architecture came from the checkpoint
            pct_loaded = (
                (loaded_params_count / total_model_params * 100)
                if total_model_params > 0
                else 0.0
            )

            print(
                f"Model built: {num_params:,} trainable params | "
                f"{frozen_params:,} frozen params.\n"
                f"Checkpoint tracking: {loaded_params_count:,}/{total_model_params:,} "
                f"parameters loaded from checkpoint ({pct_loaded:.2f}%)."
            )

            # Route summary logging through the new Logger class
            self.logger.log_summary(
                {
                    "trainable_params": num_params,
                    "frozen_params": frozen_params,
                    "total_params": total_model_params,
                    "loaded_checkpoint_params": loaded_params_count,
                    "pct_params_loaded": pct_loaded,
                }
            )

        if self.config.model_compile:
            torch._dynamo.config.cache_size_limit = 64
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
        c = self.config
        if c.prior_dir is None:
            dataset = PriorDataset(
                batch_size=c.batch_size,
                batch_size_per_gp=c.batch_size_per_gp,
                min_dim_U=c.min_dim_U,
                max_dim_U=c.max_dim_U,
                min_dim_W=c.min_dim_W,
                max_dim_W=c.max_dim_W,
                min_dim_Z=c.min_dim_Z,
                max_dim_Z=c.max_dim_Z,
                min_dim_X=c.min_dim_X,
                max_dim_X=c.max_dim_X,
                min_seq_len=c.min_seq_len,
                max_seq_len=c.max_seq_len,
                log_seq_len=c.log_seq_len,
                seq_len_per_gp=c.seq_len_per_gp,
                min_train_size=c.min_train_size,
                max_train_size=c.max_train_size,
                replay_small=c.replay_small,
                prior_type=c.prior_type,
                scm_fixed_hp={
                    **DEFAULT_FIXED_HP,
                    "mcmc_cate": getattr(c, "mcmc_cate", 200),
                    "prior_mc_mem_budget": getattr(
                        c,
                        "prior_mc_mem_budget",
                        DEFAULT_FIXED_HP["prior_mc_mem_budget"],
                    ),
                },
                device=c.prior_device,
                n_jobs=1,
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=c.prior_dir,
                batch_size=c.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=c.load_prior_start,
                delete_after_load=c.delete_after_load,
                device=c.prior_device,
            )

        if self.master_process:
            print(dataset)

        # Create dataloader for efficient loading and prefetching.
        # pin_memory only accelerates CPU->CUDA host transfers. It is a no-op (and
        # warns) when the prior is already generated on an accelerator, and is
        # unsupported on MPS, so only pin when the prior comes from the CPU and the
        # training device is CUDA.
        prior_on_cpu = "cpu" in getattr(self.config, "prior_device", "cpu")
        pin_to_cuda = prior_on_cpu and "cuda" in str(self.config.device)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=c.train_dataloader_num_workers,
            prefetch_factor=c.train_dataloader_prefetch_factor,
            pin_memory=pin_to_cuda,
            pin_memory_device=self.config.device if pin_to_cuda else "",
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
            max_batches=total_val_files,
            delete_after_load=False,
            device=self.config.prior_device,
        )

        if self.master_process:
            # Peek at the first batch file to report which CATE target the
            # validation will use: the true response surfaces (MU0/MU1) or the
            # realized potential outcomes (Y0/Y1) as a surrogate, in which case
            # true_cate = Y1 - Y0 (see LoadPriorDataset._load_batch_file).
            first_file = os.path.join(self.config.val_dir, sorted(all_val_files)[0])
            try:
                peek = torch.load(first_file, map_location="cpu", weights_only=True)
                has_mu = "MU0" in peek and "MU1" in peek
                del peek
            except Exception:
                has_mu = None

            if has_mu is True:
                target_note = "using true MU0/MU1 (noise-free CATE target)"
            elif has_mu is False:
                target_note = (
                    "MU0/MU1 absent -> using Y0/Y1 as surrogate "
                    "(true_cate = Y1 - Y0, the realized ITE)"
                )
            else:
                target_note = "could not determine CATE target source"

            print(
                f"Validation dataset configured from {self.config.val_dir} "
                f"({total_val_files} total files). {target_note}."
            )

        pin_memory = self.config.prior_device == "cpu" and "cuda" in str(
            self.config.device
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
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
        dtype_str = getattr(self.config, "dtype", "bfloat16")

        if dtype_str == "bfloat16":
            amp_dtype, need_scaler = torch.bfloat16, False
        elif dtype_str == "float16":
            amp_dtype, need_scaler = torch.float16, True
        else:
            amp_dtype, need_scaler = torch.float32, False

        self.scaler = torch.GradScaler("cuda", enabled=self.amp and need_scaler)

        if self.amp:
            if self.master_process:
                print(f"AMP enabled with dtype={dtype_str}.")
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not ckpt_dir:
            return None

        os.makedirs(ckpt_dir, exist_ok=True)

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

        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "curr_step" in checkpoint:
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
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

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

    def _shuffle_batch_rows(self, batch):
        """Permute the datasets within a batch to decorrelate group ordering.

        Only applied when the entire batch shares a single ``seq_len`` and
        ``train_size`` (the ``seq_len_per_gp=False`` case). In that case any
        permutation is safe. Otherwise — nested tensors, or pre-generated
        batches loaded from disk whose rows carry heterogeneous metadata — the
        original ordering is kept untouched, since ``run_batch`` relies on
        consecutive rows being homogeneous within a micro-batch
        (``validate_micro_batch``) and a random shuffle could mix sequence
        lengths / train sizes and trigger a ValueError.
        """
        if any(t.is_nested for t in batch):
            return batch
        # batch[13] = seq_lens, batch[14] = train_sizes (per-dataset metadata).
        seq_lens, train_sizes = batch[13], batch[14]
        if seq_lens.unique().numel() > 1 or train_sizes.unique().numel() > 1:
            return batch
        perm = torch.randperm(batch[0].shape[0], device=batch[0].device)
        return [t[perm] for t in batch]

    def _post_step(self, results, prior_time, train_time):
        """Per-step bookkeeping after one ``run_batch``.

        Advances ``curr_step`` by exactly one and preserves the per-step cadence
        of checkpointing (``save_temp_every`` / ``save_perm_every``) and
        validation (``val_interval``). Returns the (possibly augmented) results.
        """
        self.curr_step += 1

        if self.master_process:
            # Add timing information to results
            results.update({"prior_time": prior_time, "train_time": train_time})
            if self.train_evaluator.ready(self.eval_buffer_episodes):
                buffered_train_metrics = self.train_evaluator.compute(
                    prefix="train_",
                    sync_ddp=False,
                    include_count=True,
                )
                results.update(buffered_train_metrics)
                self.train_evaluator.reset()

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
        if self.val_dataloader is not None and (
            self.curr_step % self.config.val_interval == 0 or self.curr_step == 1
        ):
            val_results = self.validate()
            results.update(val_results)

        # Route training logs through the new Logger class
        results["lr"] = self.scheduler.get_last_lr()[0]
        self.logger.log(results, step=self.curr_step)
        return results

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Amortizes prior generation by alternating between two phases:

        1. Generate a resident in-memory ``block`` of ``gen_block_size`` datasets
           (= ``ceil(gen_block_size / batch_size)`` generated batches) up front.
        2. Train ``gen_reuse_passes`` (K) passes over that block, reshuffling
           between every pass, before freeing it and regenerating a fresh one.

        The reuse factor is therefore exactly K. ``curr_step``, checkpointing and
        validation all still advance per training step (one ``run_batch`` call),
        not per block.
        """

        if self.master_process:
            step_progress = tqdm(
                initial=self.curr_step,
                total=self.config.max_steps,
                desc="Step",
                leave=True,
            )

        dataloader = iter(self.dataloader)
        # A block is gen_block_size datasets, drawn as whole generated batches.
        num_batches_per_block = max(
            1, math.ceil(self.config.gen_block_size / self.config.batch_size)
        )

        # Outer loop: generate a fresh block, then reuse it for K passes.
        while self.curr_step < self.config.max_steps:
            with Timer() as gen_timer:
                block = []
                for _ in range(num_batches_per_block):
                    try:
                        block.append(next(dataloader))
                    except RuntimeError as e:
                        if not is_oom_error(e):
                            raise
                        if self.master_process:
                            print(
                                "\n[Step {}] OutOfMemory caught during block "
                                "generation. Using a partial block.".format(
                                    self.curr_step
                                )
                            )
                        empty_accelerator_cache(self.config.device)
                        empty_accelerator_cache(self.config.prior_device)
                        break
            # Keep the block length identical across ranks so every rank issues the
            # same number of run_batch calls and reaches validation / internal
            # all-reduces in lockstep, even if generation OOM'd on only some ranks.
            if self.ddp:
                count = torch.tensor(
                    [len(block)], device=self.config.device, dtype=torch.int64
                )
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MIN)
                block = block[: int(count.item())]

            if not block:
                # No rank produced a usable batch; retry generation on all ranks.
                continue

            # Generation cost is amortized over the steps that actually reuse the
            # block. The final/truncated block runs fewer than K*len(block) steps,
            # so cap the denominator at the remaining steps (OOM-skipped batches
            # are not corrected for; prior_time is a diagnostic only).
            planned_steps = min(
                self.config.gen_reuse_passes * len(block),
                self.config.max_steps - self.curr_step,
            )
            amortized_prior_time = gen_timer.elapsed / max(1, planned_steps)

            reached_end = False
            for _pass in range(self.config.gen_reuse_passes):
                # Shuffle block order each pass so micro-batches are not drawn in
                # generation order, decorrelating gradients across the block.
                for bi in torch.randperm(len(block)).tolist():
                    if self.curr_step >= self.config.max_steps:
                        reached_end = True
                        break

                    batch = self._shuffle_batch_rows(block[bi])

                    try:
                        with Timer() as train_timer:
                            results = self.run_batch(batch)
                    except RuntimeError as e:
                        if not is_oom_error(e):
                            raise
                        if self.master_process:
                            print(
                                f"\n[Step {self.curr_step}] OutOfMemory caught during "
                                "training. Skipping batch."
                            )
                        empty_accelerator_cache(self.config.device)
                        continue

                    results = self._post_step(
                        results, amortized_prior_time, train_timer.elapsed
                    )

                    if self.master_process:
                        step_progress.update(1)
                        # Rounded values for cleaner display
                        step_progress.set_postfix(
                            **{
                                k: round(v, 3) if isinstance(v, float) else v
                                for k, v in results.items()
                            }
                        )

                if reached_end:
                    break

            # Free the block before generating the next one (bounded memory).
            del block

        if self.master_process:
            step_progress.close()

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
        micro_MU0,
        micro_MU1,
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
            micro_MU0 = micro_MU0[:, :seq_len]
            micro_MU1 = micro_MU1[:, :seq_len]

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

        return (
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
            micro_MU0,
            micro_MU1,
        )

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
        (
            micro_U,
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_A_probs,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
            micro_MU0,
            micro_MU1,
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
            micro_MU0,
            micro_MU1,
        ) = self.align_micro_batch(
            micro_X,
            micro_W,
            micro_Z,
            micro_A,
            micro_Y_obs,
            micro_Y0,
            micro_Y1,
            micro_MU0,
            micro_MU1,
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
        micro_MU0 = micro_MU0.to(self.config.device)
        micro_MU1 = micro_MU1.to(self.config.device)
        micro_d_X = micro_d_X.to(self.config.device)
        micro_d_W = micro_d_W.to(self.config.device)
        micro_d_Z = micro_d_Z.to(self.config.device)

        # 2. Standardize Targets based ONLY on the observed training context
        y_train_obs = micro_Y_obs[:, :train_size]
        y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
        y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(
            min=1e-5
        )

        # Test targets for loss.
        y_test_0_sampled = micro_Y0[:, train_size:]
        y_test_1_sampled = micro_Y1[:, train_size:]
        mu_test_0 = micro_MU0[:, train_size:]
        mu_test_1 = micro_MU1[:, train_size:]

        # Scale all targets using ONLY the observed training context.
        y_test_0_std = (mu_test_0 - y_context_mean) / y_context_std
        y_test_1_std = (mu_test_1 - y_context_mean) / y_context_std
        true_cate_std = y_test_1_std - y_test_0_std

        sampled_y_test_0_std = (y_test_0_sampled - y_context_mean) / y_context_std
        sampled_y_test_1_std = (y_test_1_sampled - y_context_mean) / y_context_std
        sampled_ite_std = sampled_y_test_1_std - sampled_y_test_0_std

        # Squeeze A down to (B, T) to match y_train_obs
        A_train = micro_A[:, :train_size].squeeze(-1)
        n0 = (A_train == 0).sum(dim=1)  # (B,)
        n1 = (A_train == 1).sum(dim=1)  # (B,)

        local_valid = not ((n0 == 0).any() or (n1 == 0).any())

        if self.ddp:
            valid_tensor = torch.tensor(
                [1 if local_valid else 0],
                device=self.config.device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(
                valid_tensor,
                op=torch.distributed.ReduceOp.MIN,
            )
            global_valid = bool(valid_tensor.item())
        else:
            global_valid = local_valid

        if not global_valid:
            return None

        with self.amp_ctx:
            # 3. Model predicts logits of shape (B, test_size, nbins * 2)
            pred_logits = self.model(
                micro_X,
                micro_Z,
                micro_W,
                micro_A,
                y_train_obs,
                micro_d_X,
                micro_d_Z,
                micro_d_W,
            )

            local_valid = not pred_logits.isnan().any()

            if self.ddp:
                valid_tensor = torch.tensor(
                    [1 if local_valid else 0],
                    device=self.config.device,
                    dtype=torch.int32,
                )
                torch.distributed.all_reduce(
                    valid_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                )
                global_valid = bool(valid_tensor.item())
            else:
                global_valid = local_valid

            if not global_valid:
                return None

            # Loss configuration routing
            loss_type = self.loss_type

            if loss_type == "bins_po":
                nbins = self.raw_model.nbins
                logits_y0 = pred_logits[..., :nbins]
                logits_y1 = pred_logits[..., nbins:]

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

                probs_y0 = F.softmax(logits_y0, dim=-1)
                probs_y1 = F.softmax(logits_y1, dim=-1)

                mean_pred_y0_std = torch.sum(
                    probs_y0 * self.raw_model.bin_centers, dim=-1
                )
                mean_pred_y1_std = torch.sum(
                    probs_y1 * self.raw_model.bin_centers, dim=-1
                )

                pred_cate_std = mean_pred_y1_std - mean_pred_y0_std

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.1)

                total_loss = loss_y0 + loss_y1 + (lambda_cate * loss_cate)

            elif loss_type == "cate_decomposition":
                # Network outputs: [ATE_mean, ATE_logvar, HE_bin_0, ..., HE_bin_N]
                ate_mu = pred_logits[..., 0]
                ate_logvar = pred_logits[..., 1]
                he_logits = pred_logits[..., 2:]

                # ATE is estimated via the average testing ITE per episode/batch element
                true_ate_std = true_cate_std.mean(dim=1, keepdim=True)

                # Heterogeneous effect is the residual structural shift isolated from ATE
                true_he_std = true_cate_std - true_ate_std

                loss_ate_nll = (
                    compute_gaussian_nll(ate_mu, ate_logvar, true_ate_std) / 10
                )
                loss_he_crps = compute_crps_loss(
                    he_logits,
                    true_he_std,
                    self.raw_model.bin_edges,
                )

                lambda_ate = getattr(self.config, "lambda_ate", 1.0)
                lambda_he = getattr(self.config, "lambda_he", 1.0)

                total_loss = (lambda_ate * loss_ate_nll) + (lambda_he * loss_he_crps)

                probs_he = F.softmax(he_logits, dim=-1)
                mean_he_pred_std = torch.sum(
                    probs_he * self.raw_model.bin_centers, dim=-1
                )

                # CATE prediction synthesizes the additive ATE and HE predictions
                pred_cate_std = ate_mu + mean_he_pred_std

                # Evaluator requirements zero-padding to avoid crash (MSE will be ignored)
                mean_pred_y0_std = torch.zeros_like(y_test_0_std)
                mean_pred_y1_std = torch.zeros_like(y_test_1_std)

            elif loss_type == "hl":
                nbins = self.raw_model.nbins
                logits_y0 = pred_logits[..., :nbins]
                logits_y1 = pred_logits[..., nbins:]

                hl_sigma = self.get_hl_sigma()

                loss_y0 = compute_hl_gaussian_cross_entropy_loss(
                    logits_y0,
                    y_test_0_std,
                    self.raw_model.bin_edges,
                    sigma=hl_sigma,
                )
                loss_y1 = compute_hl_gaussian_cross_entropy_loss(
                    logits_y1,
                    y_test_1_std,
                    self.raw_model.bin_edges,
                    sigma=hl_sigma,
                )

                probs_y0 = F.softmax(logits_y0, dim=-1)
                probs_y1 = F.softmax(logits_y1, dim=-1)

                mean_pred_y0_std = torch.sum(
                    probs_y0 * self.raw_model.bin_centers, dim=-1
                )
                mean_pred_y1_std = torch.sum(
                    probs_y1 * self.raw_model.bin_centers, dim=-1
                )

                pred_cate_std = mean_pred_y1_std - mean_pred_y0_std

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.1)

                total_loss = loss_y0 + loss_y1 + (lambda_cate * loss_cate)

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        # Optimization step
        if is_train:
            scaled_loss = total_loss / num_micro_batches
            self.scaler.scale(scaled_loss).backward()
            reported_loss_total = scaled_loss.item()
        else:
            reported_loss_total = total_loss.item()

        # Evaluation & Metric extraction block
        with torch.no_grad():
            if is_train:
                should_update_evaluator = self.master_process
                evaluator = self.train_evaluator
            else:
                should_update_evaluator = True
                evaluator = self.val_evaluator

            if should_update_evaluator:
                evaluator.update(
                    pred_y0=mean_pred_y0_std,
                    pred_y1=mean_pred_y1_std,
                    true_y0=y_test_0_std,
                    true_y1=y_test_1_std,
                    pred_cate=pred_cate_std,
                    true_cate=true_cate_std,
                )

            # Dictionary structure routes cleanly back to log aggregation
            if loss_type == "bins_po":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0": (loss_y0.item() / num_micro_batches)
                    if is_train
                    else loss_y0.item(),
                    "loss/y1": (loss_y1.item() / num_micro_batches)
                    if is_train
                    else loss_y1.item(),
                    "loss/cate": ((lambda_cate * loss_cate).item() / num_micro_batches)
                    if is_train
                    else (lambda_cate * loss_cate).item(),
                }
            elif loss_type == "cate_decomposition":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/ate_nll": (
                        (lambda_ate * loss_ate_nll).item() / num_micro_batches
                    )
                    if is_train
                    else (lambda_ate * loss_ate_nll).item(),
                    "loss/he_crps": (
                        (lambda_he * loss_he_crps).item() / num_micro_batches
                    )
                    if is_train
                    else (lambda_he * loss_he_crps).item(),
                }
            elif loss_type == "hl":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0_hl": (loss_y0.item() / num_micro_batches)
                    if is_train
                    else loss_y0.item(),
                    "loss/y1_hl": (loss_y1.item() / num_micro_batches)
                    if is_train
                    else loss_y1.item(),
                    "loss/cate": ((lambda_cate * loss_cate).item() / num_micro_batches)
                    if is_train
                    else (lambda_cate * loss_cate).item(),
                    "loss/hl_sigma": (hl_sigma / num_micro_batches)
                    if is_train
                    else hl_sigma,
                }
            scale = num_micro_batches if is_train else 1
            micro_results.update(
                {
                    "target/cate_mean": true_cate_std.mean().item() / scale,
                    "target/cate_std": true_cate_std.std(unbiased=False).item() / scale,
                    "target/sampled_ite_mean": sampled_ite_std.mean().item() / scale,
                    "target/ite_minus_cate_mse": torch.mean(
                        (sampled_ite_std - true_cate_std) ** 2
                    ).item()
                    / scale,
                }
            )

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

        # Dynamically build up results dictated by loss routing mode
        results = {}
        failed_batches = 0
        successful_micro_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, idx, num_micro_batches
                )

                if micro_results is None:
                    continue

                successful_micro_batches += 1

                for k, v in micro_results.items():
                    if k not in results:
                        results[k] = 0.0
                    results[k] += v

            except RuntimeError as e:
                if not is_oom_error(e):
                    raise
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                empty_accelerator_cache(self.config.device)
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        if successful_micro_batches == 0:
            return results

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

        for i in (4, 5, 6, 7, 8):
            if batch[i].dim() == 3 and batch[i].shape[-1] == 1:
                batch[i] = batch[i].squeeze(-1)

        seq_lens = batch[13]  # shape (B,)
        train_sizes = batch[14]  # shape (B,)
        sort_idx = torch.argsort(seq_lens * (train_sizes.max() + 1) + train_sizes)
        batch = [t[sort_idx] for t in batch]

        actual_batch_size = batch[0].shape[0]
        num_micro_batches = math.ceil(actual_batch_size / self.val_micro_batch_size)

        micro_batches = [
            torch.split(t, self.val_micro_batch_size, dim=0) for t in batch
        ]
        micro_batches = list(zip(*micro_batches))

        total_valid_samples = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                current_mb_size = micro_batch[0].shape[0]

                with self.amp_ctx:
                    micro_results = self.run_micro_batch(
                        micro_batch,
                        idx,
                        num_micro_batches,
                        is_train=False,
                    )

                if micro_results is None:
                    continue

                total_valid_samples += current_mb_size

            except RuntimeError as e:
                if not is_oom_error(e):
                    raise
                empty_accelerator_cache(self.config.device)
                continue

        return total_valid_samples

    @torch.inference_mode()
    def validate(self):
        """Run one full pass over the entire validation set.

        Metrics are accumulated across every validation episode and computed
        once, so each ``validate()`` call reports performance over the whole
        val_dir (i.e. logged exactly when all validation episodes have been
        visited once).
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        self.val_evaluator.reset()

        # One full epoch over the validation set. The dataloader yields batches
        # until all files are consumed, then stops.
        for batch in self.val_dataloader:
            self.run_val_batch(batch)

        final_metrics = self.val_evaluator.compute(
            prefix="val_",
            sync_ddp=self.ddp,
            device=self.config.device,
            include_count=True,
        )

        self.val_evaluator.reset()
        self.model.train()

        return final_metrics

    def get_hl_sigma(self) -> float:
        """
        Returns the current sigma for HL-Gaussian loss.

        Schedule:
        - use start_sigma for warmup_sigma_steps
        - linearly decay to final_sigma over linear_decay_steps
        - then stay at final_sigma

        If start_sigma == final_sigma, this returns a constant sigma.
        """
        start_sigma = float(self.hl_start_sigma)
        final_sigma = float(self.hl_final_sigma)
        warmup_steps = int(self.hl_warmup_sigma_steps)
        decay_steps = int(self.hl_linear_decay_steps)

        if start_sigma <= 0 or final_sigma <= 0:
            raise ValueError(
                f"HL sigmas must be positive, got start={start_sigma}, final={final_sigma}"
            )

        if start_sigma == final_sigma:
            return start_sigma

        if self.curr_step < warmup_steps:
            return start_sigma

        if decay_steps <= 0:
            return final_sigma

        decay_progress = (self.curr_step - warmup_steps) / decay_steps
        decay_progress = min(max(decay_progress, 0.0), 1.0)

        sigma = start_sigma + decay_progress * (final_sigma - start_sigma)

        return float(sigma)


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
