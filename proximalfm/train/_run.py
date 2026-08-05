from __future__ import annotations

import os
import warnings
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
from torch.distributed import init_process_group

from tqdm import tqdm
import yaml

from proximalfm._model.proximalfm import ProximalFM
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._genload import LoadPriorDataset
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP
from proximalfm.train._optim import get_scheduler
from proximalfm.train._train_config import build_parser
from proximalfm.train._eval import CausalEvaluator
from proximalfm.train._logger import ExperimentLogger
from proximalfm.train._losses import (
    compute_cate_distribution_loss,
    compute_cate_noise_aware_nll,
    compute_crps_loss,
    compute_gaussian_nll,
    compute_hl_gaussian_cross_entropy_loss,
)
from proximalfm.train._utils import (
    Timer,
    ddp_cleanup,
    empty_accelerator_cache,
    get_parameter_state,
    is_oom_error,
    load_transfer_rules,
    seed_dataloader_worker,
)


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


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

        self.val_evaluator = CausalEvaluator()

        self.val_micro_batch_size = (
            getattr(self.config, "val_micro_batch_size", None)
            or self.config.micro_batch_size
        )

        self.save_best = getattr(self.config, "save_best", True)
        self.best_metric = getattr(self.config, "best_metric", "val/point/cate/npehe")
        self.best_metric_mode = getattr(self.config, "best_metric_mode", "min")
        self.best_metric_value = None
        self._warned_missing_metric = False
        self._check_best_metric_config()
        self._load_best_metric_value()

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

        transfer_rules = load_transfer_rules(
            getattr(self.config, "transfer_rules", "transfer_rules.yaml")
        )

        self.model_config = {
            "loss_type": self.loss_type,
            "nbins": self.config.nbins,
            "vmin": self.config.vmin,
            "vmax": self.config.vmax,
            "cate_vmin": self.config.cate_vmin,
            "cate_vmax": self.config.cate_vmax,
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
                # Strip DDP's "module." prefix, then torch.compile's
                # "_orig_mod." prefix -- in that order, since a DDP-wrapped
                # compiled model nests them as "module._orig_mod.".
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key[10:]

                clean_old_state_dict[clean_key] = v

            # 1. Filter weights based on the transfer rules using the clean state dict.
            # Buffers (bin grids) are derived from the config, never learned: restoring
            # them from the checkpoint would let a stale vmin/vmax override the grid the
            # current run was configured with.
            buffer_names = set(dict(model.named_buffers()).keys())
            for name, weight in clean_old_state_dict.items():
                if name in buffer_names:
                    continue
                if (
                    name in model_state_dict
                    and weight.shape == model_state_dict[name].shape
                ):
                    state = get_parameter_state(name, transfer_rules)
                    if state in ["finetune", "freeze"]:
                        filtered_state_dict[name] = weight
                        # Track the count of weights successfully mapped over
                        loaded_params_count += weight.numel()

            # 2. Load the allowed weights (strict=False leaves "scratch" weights as random)
            model.load_state_dict(filtered_state_dict, strict=False)
        # 3. Apply freezing logic
        for name, param in model.named_parameters():
            state = get_parameter_state(name, transfer_rules)
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
                mc_cate=getattr(c, "mc_cate", 200),
                mc_targets_on_context=getattr(c, "mc_targets_on_context", False),
                scm_fixed_hp={
                    **DEFAULT_FIXED_HP,
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
            worker_init_fn=seed_dataloader_worker
            if c.train_dataloader_num_workers > 0
            else None,
        )

    def configure_validation_prior(self):
        """Set up the dataloader for the fixed validation set."""
        if self.config.val_dir is None:
            self.val_dataloader = None
            return

        all_val_files = [
            f
            for f in os.listdir(self.config.val_dir)
            if f.startswith("batch_") and f.endswith(".pt")
        ]
        total_val_files = len(all_val_files)

        if total_val_files == 0:
            raise ValueError(f"No validation files found in {self.config.val_dir}")

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

    @staticmethod
    def _list_step_checkpoints(ckpt_dir: str) -> list[str]:
        """Filenames in ``ckpt_dir`` matching the ``step-<N>.ckpt`` pattern."""
        return [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]

    @staticmethod
    def _checkpoint_step(filename: str) -> int:
        """Parse ``<N>`` out of a ``step-<N>.ckpt`` filename."""
        return int(filename.split("-")[1].split(".")[0])

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not ckpt_dir:
            return None

        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoints = self._list_step_checkpoints(ckpt_dir)

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=self._checkpoint_step)[-1]
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

    def save_checkpoint(self, name: str, extra: dict | None = None):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        extra : dict, optional
            Additional key/value pairs to store in the checkpoint (e.g. the
            monitored metric value for the best-model checkpoint). Values must be
            ``weights_only``-loadable primitives (numbers, strings, tensors).
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
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, checkpoint_path)

    def _check_best_metric_config(self):
        """Warn (once, on master) about best-model configs that silently no-op.

        Catches the two common footguns: (a) ``save_best`` requested but no
        validation set, so no best model can ever be saved; and (b) a
        ``best_metric`` whose name implies a direction opposite to the chosen
        ``best_metric_mode`` (e.g. an R2 tracked with ``min``).
        """
        if not self.save_best or not self.master_process:
            return

        if getattr(self.config, "val_dir", None) is None:
            print(
                "[save_best] Warning: save_best is enabled but val_dir is None; "
                "validation never runs, so no best_model.ckpt will be saved."
            )

        name = self.best_metric.lower()
        higher_is_better = any(k in name for k in ("r2", "sign_agreement"))
        lower_is_better = any(
            k in name for k in ("rmse", "npehe", "mse", "bias", "loss")
        )
        if higher_is_better and self.best_metric_mode == "min":
            print(
                f"[save_best] Warning: best_metric '{self.best_metric}' looks "
                "higher-is-better but best_metric_mode is 'min'. Did you mean 'max'?"
            )
        elif lower_is_better and self.best_metric_mode == "max":
            print(
                f"[save_best] Warning: best_metric '{self.best_metric}' looks "
                "lower-is-better but best_metric_mode is 'max'. Did you mean 'min'?"
            )

    def _load_best_metric_value(self):
        """Recover the best-so-far metric from an existing ``best_model.ckpt``.

        Lets best-model tracking survive a resume: if a best checkpoint already
        exists we only overwrite it when a genuinely better metric is seen.
        """
        if not self.save_best or not self.master_process:
            return
        if not getattr(self.config, "checkpoint_dir", None):
            return

        best_path = os.path.join(self.config.checkpoint_dir, "best_model.ckpt")
        if not os.path.exists(best_path):
            return

        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            value = ckpt.get("best_metric_value", None)
            if value is not None:
                self.best_metric_value = float(value)
                print(
                    f"Resuming best-model tracking: best {self.best_metric} so far "
                    f"= {self.best_metric_value:.6f}"
                )
        except Exception as e:
            print(f"Could not read best metric from {best_path}: {e}")

    def _maybe_save_best(self, val_results: dict):
        """Save ``best_model.ckpt`` if the monitored metric improved.

        Called on the master process only, right after a validation pass. The
        file is overwritten in place so only the single best checkpoint is kept.
        """
        metric = val_results.get(self.best_metric, None)
        if metric is None:
            # Metric absent this pass (e.g. all-homogeneous episodes for an R2
            # metric, or a mistyped best_metric). Nothing to compare against;
            # skip, but warn once so a permanent typo isn't silently ignored.
            if not self._warned_missing_metric:
                available = ", ".join(sorted(val_results)) or "(none)"
                print(
                    f"[save_best] Warning: monitored metric '{self.best_metric}' "
                    f"not found in validation results; not saving best model. "
                    f"Available keys: {available}"
                )
                self._warned_missing_metric = True
            return

        metric = float(metric)
        if not math.isfinite(metric):
            return

        if self.best_metric_value is None:
            improved = True
        elif self.best_metric_mode == "max":
            improved = metric > self.best_metric_value
        else:
            improved = metric < self.best_metric_value

        if not improved:
            return

        prev = self.best_metric_value
        self.best_metric_value = metric
        self.save_checkpoint(
            name="best_model.ckpt",
            extra={
                "best_metric_value": metric,
                "best_metric_name": self.best_metric,
            },
        )
        prev_str = "n/a" if prev is None else f"{prev:.6f}"
        print(
            f"[Step {self.curr_step}] New best {self.best_metric} = {metric:.6f} "
            f"(prev {prev_str}). Saved best_model.ckpt"
        )

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        limit = self.config.max_checkpoints

        checkpoints = self._list_step_checkpoints(ckpt_dir)
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = self._checkpoint_step(ckpt)
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
        # batch[15] = seq_lens, batch[16] = train_sizes (per-dataset metadata).
        seq_lens, train_sizes = batch[15], batch[16]
        if seq_lens.unique().numel() > 1 or train_sizes.unique().numel() > 1:
            return batch
        perm = torch.randperm(batch[0].shape[0], device=batch[0].device)
        return [t[perm] for t in batch]

    def _post_step(self, results, prior_time, train_time):
        """Per-step bookkeeping after one ``run_batch``.

        Advances ``curr_step`` by exactly one and preserves the per-step cadence
        of checkpointing (``save_temp_every`` / ``save_perm_every``) and
        validation (``point_val_interval``). Returns the (possibly augmented) results.
        """
        self.curr_step += 1

        if self.master_process:
            # Add timing information to results
            results.update({"prior_time": prior_time, "train_time": train_time})

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
            self.curr_step % self.config.point_val_interval == 0 or self.curr_step == 1
        ):
            val_results = self.validate()
            results.update(val_results)

            # Overwrite best_model.ckpt if the monitored metric improved. Saving
            # is master-only (like the step checkpoints); val_results is already
            # DDP-synced so every rank agrees on the value.
            if self.master_process and self.save_best:
                self._maybe_save_best(val_results)

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
        micro_S2,
        micro_d_X,
        micro_d_W,
        micro_d_Z,
        seq_len,
    ):
        """Truncate micro batch tensors to required dimensions.

        Truncates sequence length to the validated ``seq_len`` and each of
        X/W/Z's feature dimension to its own max active width
        (``micro_d_X.max()`` etc.), removing padding the micro batch doesn't
        need. ``micro_A`` and the Y-family tensors share the truncated
        sequence length but are not feature-truncated.

        Returns the same ten tensors, in the same order as the input args.
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
            micro_S2 = micro_S2[:, :seq_len]

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
            micro_S2,
        )

    def _ddp_all_valid(self, local_valid: bool) -> bool:
        """AND ``local_valid`` across all DDP ranks (no-op outside DDP).

        Every rank must skip the same micro batch together, or ranks that
        disagree on whether a step happened diverge and hang on the next
        collective (e.g. gradient all-reduce).
        """
        if not self.ddp:
            return local_valid
        valid_tensor = torch.tensor(
            [1 if local_valid else 0],
            device=self.config.device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(valid_tensor, op=torch.distributed.ReduceOp.MIN)
        return bool(valid_tensor.item())

    def run_micro_batch(
        self, micro_batch, micro_batch_idx, num_micro_batches, is_train=True
    ):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            The 19-tensor prior batch tuple, sliced along the dataset dimension.

        micro_batch_idx : int
            Index of the current micro batch (unused; kept for call-site clarity).

        num_micro_batches : int
            Total number of micro batches. Losses and reported metrics are divided
            by it during training so that the accumulated batch sums to one step.

        is_train : bool
            Whether to backpropagate and feed the training evaluator.

        Returns
        -------
        dict or None
            Per-micro-batch loss/target metrics, or ``None`` when the micro batch
            was skipped (a single-arm context, or NaN logits).
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
            micro_S2,
            micro_K_eff,
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
            micro_S2,
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
            micro_S2,
            micro_d_X,
            micro_d_W,
            micro_d_Z,
            seq_len,
        )

        # Move to device
        device = self.config.device
        micro_X = micro_X.to(device)
        micro_W = micro_W.to(device)
        micro_Z = micro_Z.to(device)
        micro_A = micro_A.to(device)
        micro_Y_obs = micro_Y_obs.to(device)
        micro_Y0 = micro_Y0.to(device)
        micro_Y1 = micro_Y1.to(device)
        micro_MU0 = micro_MU0.to(device)
        micro_MU1 = micro_MU1.to(device)
        micro_S2 = micro_S2.to(device)
        micro_d_X = micro_d_X.to(device)
        micro_d_W = micro_d_W.to(device)
        micro_d_Z = micro_d_Z.to(device)

        # 2. Standardize targets based ONLY on the observed training context
        y_train_obs = micro_Y_obs[:, :train_size]
        y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
        y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(
            min=1e-3
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

        # Monte Carlo noise on the CATE target, converted into the head's units.
        # S2 is a variance in raw y^2; the head lives on the cate_vmin/cate_vmax
        # grid in standardized-y space, so it scales by 1 / y_context_std^2 --
        # a *per-episode* factor. Skipping this division would leave the
        # convolution wrong by a random factor per episode, silently.
        # y_context_std is (B, 1) and broadcasts over the (B, n_q) query rows.
        target_var_std = micro_S2[:, train_size:] / (y_context_std**2)

        # Squeeze A down to (B, T) to match y_train_obs
        A_train = micro_A[:, :train_size].squeeze(-1)
        n0 = (A_train == 0).sum(dim=1)  # (B,)
        n1 = (A_train == 1).sum(dim=1)  # (B,)

        # A single-arm context (all-0 or all-1 treatment) makes the CATE
        # target undefined for this episode; skip it everywhere in the DDP
        # group rather than only where it happened to occur.
        if not self._ddp_all_valid(not ((n0 == 0).any() or (n1 == 0).any())):
            return None

        with self.amp_ctx:
            # Model predicts logits of shape (B, test_size, nbins * 2)
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

            if not self._ddp_all_valid(not pred_logits.isnan().any()):
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

            elif loss_type in ("cate_single", "cate_single_noise_aware"):
                # A single histogram over tau, so no slicing: out_dim == nbins.
                hl_sigma = self.get_hl_sigma()

                if loss_type == "cate_single_noise_aware":
                    loss_cate_dist = compute_cate_noise_aware_nll(
                        pred_logits,
                        true_cate_std,
                        self.raw_model.cate_bin_edges,
                        target_var=target_var_std,
                        sigma=hl_sigma,
                    )
                else:
                    loss_cate_dist = compute_cate_distribution_loss(
                        pred_logits,
                        true_cate_std,
                        self.raw_model.cate_bin_edges,
                        sigma=hl_sigma,
                    )

                probs_cate = F.softmax(pred_logits, dim=-1)
                pred_cate_std = torch.sum(
                    probs_cate * self.raw_model.cate_bin_centers, dim=-1
                )

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.0)

                # The MSE guard needs no adjustment for the convolution:
                # mean(P * N) == mean(P), so it constrains the same quantity
                # under either objective.
                total_loss = loss_cate_dist + (lambda_cate * loss_cate)

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
            # Point metrics are a validation-surface concern only: on the training
            # distribution they duplicate the loss at a batch size chosen for
            # throughput rather than estimator variance.
            if not is_train:
                # Only bins_po/hl have a real per-arm head; other loss types
                # predict the effect directly and have no honest po/* to report.
                if loss_type in ("bins_po", "hl"):
                    self.val_evaluator.update(
                        pred_y0=mean_pred_y0_std,
                        pred_y1=mean_pred_y1_std,
                        true_y0=y_test_0_std,
                        true_y1=y_test_1_std,
                        pred_cate=pred_cate_std,
                        true_cate=true_cate_std,
                    )
                else:
                    self.val_evaluator.update(
                        pred_cate=pred_cate_std,
                        true_cate=true_cate_std,
                    )

            # Training metrics are accumulated over num_micro_batches to sum to
            # one step's worth once every micro batch has been added in; eval
            # metrics are reported per micro batch as-is (scale=1 is a no-op).
            scale = num_micro_batches if is_train else 1

            # Dictionary structure routes cleanly back to log aggregation
            if loss_type == "bins_po":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0": loss_y0.item() / scale,
                    "loss/y1": loss_y1.item() / scale,
                    "loss/cate": (lambda_cate * loss_cate).item() / scale,
                }
            elif loss_type == "cate_decomposition":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/ate_nll": (lambda_ate * loss_ate_nll).item() / scale,
                    "loss/he_crps": (lambda_he * loss_he_crps).item() / scale,
                }
            elif loss_type == "hl":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0_hl": loss_y0.item() / scale,
                    "loss/y1_hl": loss_y1.item() / scale,
                    "loss/cate": (lambda_cate * loss_cate).item() / scale,
                    "loss/hl_sigma": hl_sigma / scale,
                }
            elif loss_type in ("cate_single", "cate_single_noise_aware"):
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/cate_dist": loss_cate_dist.item() / scale,
                    "loss/cate": (lambda_cate * loss_cate).item() / scale,
                    "loss/hl_sigma": hl_sigma / scale,
                }

                if loss_type == "cate_single_noise_aware":
                    micro_results.update(
                        self._target_noise_metrics(
                            target_var_std,
                            true_cate_std,
                            probs_cate,
                            pred_cate_std,
                            micro_K_eff,
                            scale=scale,
                        )
                    )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

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
        """Train the model on one batch of datasets (a single optimizer step).

        Splits the batch into micro-batches for gradient accumulation, padding
        nested tensors first. Micro-batches that OOM are skipped, and the logged
        metrics are rescaled so a partial batch still reports per-batch values.

        Parameters
        ----------
        batch : tuple
            The 19-tensor prior batch tuple. Entries may be nested tensors when
            sequence lengths vary across datasets.

        Returns
        -------
        dict
            Loss and target metrics for the step, summed over micro-batches.

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

        # Fix logged metrics for skipped micro-batches.
        if successful_micro_batches < num_micro_batches:
            correction = num_micro_batches / successful_micro_batches
            for k in results:
                results[k] *= correction

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

        seq_lens = batch[15]  # shape (B,)
        train_sizes = batch[16]  # shape (B,)
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
        with Timer() as val_timer:
            for batch in self.val_dataloader:
                self.run_val_batch(batch)

        final_metrics = self.val_evaluator.compute(
            prefix="val/point/",
            sync_ddp=self.ddp,
            device=self.config.device,
            include_count=True,
        )

        # Time (seconds) taken to pass over the whole validation set.
        final_metrics["val/time"] = val_timer.elapsed

        self.val_evaluator.reset()
        self.model.train()

        return final_metrics

    @torch.no_grad()
    def _target_noise_metrics(
        self,
        target_var_std,
        true_cate_std,
        probs_cate,
        pred_cate_std,
        micro_K_eff,
        scale: int,
    ) -> dict:
        """Diagnostics for the noise-aware CATE objective.

        ``s / sd_pred`` is the one to watch. It sets how much the noise
        correction is worth at all -- below ~0.2 the induced width inflation is
        under 2% and the correction is barely detectable -- but it is *not* a
        constant: ``sd_pred`` shrinks as the posterior sharpens, so a replicate
        count that was comfortably sufficient early can quietly become the
        binding constraint later in a run. Hence a logged series, not a one-off
        check.
        """
        centers = self.raw_model.cate_bin_centers.float()
        probs = probs_cate.float()
        mean = pred_cate_std.float().unsqueeze(-1)
        var_pred = torch.sum(probs * (centers.view(1, 1, -1) - mean) ** 2, dim=-1)

        s2 = torch.nan_to_num(target_var_std.float(), nan=0.0, posinf=0.0).clamp_min(
            0.0
        )
        mean_s = torch.sqrt(s2).mean()
        mean_sd_pred = torch.sqrt(var_pred.clamp_min(0.0)).mean()

        # Rows the convolution cannot act on: no replicate variance available
        # (K = 1, context rows, or data predating target-noise tracking). They
        # fall back to the plain sigma-smoothed NLL, which is correct but is not
        # the objective this loss_type exists for -- so surface the share.
        frac_fallback = (~(s2 > 0)).float().mean()

        # Targets past the grid ends get essentially no likelihood under any
        # prediction, so a non-trivial share here means cate_vmin/cate_vmax are
        # too tight and the loss is being driven by rows it cannot represent.
        edges = self.raw_model.cate_bin_edges
        out_of_grid = (
            ((true_cate_std < edges[0]) | (true_cate_std > edges[-1])).float().mean()
        )

        return {
            "noise/mean_s": mean_s.item() / scale,
            "noise/mean_sd_pred": mean_sd_pred.item() / scale,
            "noise/s_over_sd_pred": (mean_s / mean_sd_pred.clamp_min(1e-12)).item()
            / scale,
            "noise/frac_fallback": frac_fallback.item() / scale,
            "noise/mean_K": micro_K_eff.float().mean().item() / scale,
            "target/cate_out_of_grid_frac": out_of_grid.item() / scale,
        }

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
