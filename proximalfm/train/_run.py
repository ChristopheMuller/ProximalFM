from __future__ import annotations

import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method

from tqdm import tqdm
import yaml

from proximalfm._model.proximalfm import ProximalFM
from proximalfm.prior._dataset import PriorDataset
from proximalfm.prior._genload import LoadPriorDataset
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP
from proximalfm.train._optim import get_scheduler
from proximalfm.train._train_config import build_parser, resolve_hl_final_sigma
from proximalfm.train._dist_eval import require_single_cate_head, score_frozen_set
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


_UNSCALED_METRICS = frozenset({"loss/total"})


class Trainer:
    """Handles the complete training lifecycle for ProximalFM."""

    def __init__(self, config):
        self.config = config
        self.loss_type = getattr(self.config, "loss_type", "bins_po")

        self.hl_start_sigma = getattr(self.config, "hl_start_sigma", 0.3)
        self.hl_warmup_sigma_steps = getattr(self.config, "hl_warmup_sigma_steps", 1000)
        self.hl_linear_decay_steps = getattr(self.config, "hl_linear_decay_steps", 5000)
        # Derived from the grid (--hl_final_sigma_bins) unless --hl_final_sigma
        # was passed explicitly. Written back onto the config so the number the
        # loss actually uses is the number the run's wandb config shows.
        self.hl_final_sigma = resolve_hl_final_sigma(self.config)
        self.config.hl_final_sigma = self.hl_final_sigma

        self.configure_run()
        self.logger = ExperimentLogger(self.config)

        self.build_model()
        self.configure_prior()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

        self.save_best = getattr(self.config, "save_best", True)
        self.best_metric = getattr(self.config, "best_metric", "synt_point/cate/npehe")
        self.best_metric_mode = getattr(self.config, "best_metric_mode", "min")
        self.best_metric_value = None
        self._warned_missing_metric = False
        self._check_best_metric_config()
        self._load_best_metric_value()

        self.prior_eval_interval = getattr(self.config, "prior_eval_interval", 50)
        self.synt_eval_interval = getattr(self.config, "synt_eval_interval", 50)

        self.prior_eval_dir = getattr(self.config, "prior_eval_dir", None)
        self._prior_eval_meta = None
        if self.prior_eval_dir is not None:
            from prior_eval.build_prior_frozen_set import (
                frozen_content_digest,
                load_frozen_meta,
            )

            self._prior_eval_meta = load_frozen_meta(Path(self.prior_eval_dir))
            require_single_cate_head(self.model)
            self.logger.update_config(
                {
                    "prior_eval_content_digest": frozen_content_digest(
                        self._prior_eval_meta
                    ),
                    "prior_eval_n_datasets": self._prior_eval_meta["global"].get(
                        "n_datasets"
                    ),
                }
            )

        self.synt_eval_dir = getattr(self.config, "synt_eval_dir", None)
        self._synt_eval_meta = None
        if self.synt_eval_dir is not None:
            from prior_eval.build_prior_frozen_set import (
                frozen_content_digest,
                load_frozen_meta,
            )

            self._synt_eval_meta = load_frozen_meta(Path(self.synt_eval_dir))
            require_single_cate_head(self.model)
            self.logger.update_config(
                {
                    "synt_eval_content_digest": frozen_content_digest(
                        self._synt_eval_meta
                    ),
                    "synt_eval_n_datasets": self._synt_eval_meta["global"].get(
                        "n_datasets"
                    ),
                }
            )

    def configure_run(self):
        """Seed the run and set global backend flags.

        Training is single-process: one device, one optimizer step per batch.
        """
        self.curr_step = 0

        np.random.seed(self.config.np_seed)
        torch.manual_seed(self.config.torch_seed)
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
                if clean_key.startswith("module."):
                    clean_key = clean_key[7:]
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key[10:]

                clean_old_state_dict[clean_key] = v

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
                        loaded_params_count += weight.numel()

            model.load_state_dict(filtered_state_dict, strict=False)

        for name, param in model.named_parameters():
            state = get_parameter_state(name, transfer_rules)
            if state == "freeze":
                param.requires_grad = False
            else:
                param.requires_grad = True

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

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
            print("Model compiled successfully.")

        self.model = model

    def configure_prior(self):
        """Set up dataset generator for proximal synthetic data."""
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
            dataset = LoadPriorDataset(
                data_dir=c.prior_dir,
                batch_size=c.batch_size,
                start_from=c.load_prior_start,
                delete_after_load=c.delete_after_load,
                device=c.prior_device,
            )

        print(dataset)

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

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def configure_amp(self):
        """Configure automatic mixed precision (AMP)."""
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
            print(f"AMP enabled with dtype={dtype_str}.")
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            self.amp_ctx = nullcontext()

    @staticmethod
    def _list_step_checkpoints(ckpt_dir: str) -> list[str]:
        """Return list of step checkpoint filenames in directory."""
        return [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step-") and f.endswith(".ckpt")
        ]

    @staticmethod
    def _checkpoint_step(filename: str) -> int:
        """Parse step index from checkpoint filename."""
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
        """Save model and training state to checkpoint file."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        if extra:
            checkpoint.update(extra)
        tmp_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    def _monitored_surface_ran(
        self, *, ran_prior_eval: bool, ran_synt_eval: bool
    ) -> bool:
        """Check if the evaluation pass producing best_metric ran on this step."""
        by_prefix = (
            ("prior_dist/", ran_prior_eval),
            ("prior_point/", ran_prior_eval),
            ("synt_dist/", ran_synt_eval),
            ("synt_point/", ran_synt_eval),
        )
        for prefix, ran in by_prefix:
            if self.best_metric.startswith(prefix):
                return ran
        return False

    def _check_best_metric_config(self):
        """Warn about invalid best_metric configurations."""
        if not self.save_best:
            return

        if (
            getattr(self.config, "prior_eval_dir", None) is None
            and getattr(self.config, "synt_eval_dir", None) is None
        ):
            print(
                "[save_best] Warning: save_best is enabled but neither "
                "prior_eval_dir nor synt_eval_dir is set; no validation pass "
                "ever runs, so no best_model.ckpt will be saved."
            )

        for prefix, flag in (
            ("prior_dist/", "prior_eval_dir"),
            ("prior_point/", "prior_eval_dir"),
            ("synt_dist/", "synt_eval_dir"),
            ("synt_point/", "synt_eval_dir"),
        ):
            if self.best_metric.startswith(prefix) and (
                getattr(self.config, flag, None) is None
            ):
                print(
                    f"[save_best] Warning: best_metric '{self.best_metric}' is on "
                    f"the {prefix}* surface but {flag} is None; that pass never "
                    "runs, so this metric will never appear in validation results."
                )

        name = self.best_metric.lower()
        higher_is_better = any(k in name for k in ("r2", "sign_agreement"))
        lower_is_better = any(
            k in name for k in ("rmse", "npehe", "mse", "bias", "loss", "crps")
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
        """Recover best metric value from existing best_model.ckpt."""
        if not self.save_best:
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
        """Save best_model.ckpt if monitored metric improved."""
        metric = val_results.get(self.best_metric, None)
        if metric is None:
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
        """Manage temporary checkpoints by removing oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        limit = self.config.max_checkpoints

        checkpoints = self._list_step_checkpoints(ckpt_dir)
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = self._checkpoint_step(ckpt)
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except (ValueError, IndexError):
                continue

        temp_checkpoints.sort(key=lambda x: x[0])

        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    def _batch_row_permutation(self, batch):
        """Return dataset row permutation or None if batch has heterogeneous sequence lengths."""
        if any(t.is_nested for t in batch):
            return None
        seq_lens, train_sizes = batch[15], batch[16]
        if seq_lens.unique().numel() > 1 or train_sizes.unique().numel() > 1:
            return None
        return torch.randperm(batch[0].shape[0], device=batch[0].device)

    @staticmethod
    def _iter_micro_batches(batch, micro_batch_size, row_perm=None):
        """Yield micro batches from batch, optionally reordered by row_perm."""
        if row_perm is None:
            for micro_batch in zip(
                *[torch.split(t, micro_batch_size, dim=0) for t in batch]
            ):
                yield micro_batch
            return
        for idx in torch.split(row_perm, micro_batch_size):
            yield tuple(t[idx] for t in batch)

    def _post_step(self, results, prior_time, train_time):
        """Perform per-step bookkeeping and trigger validation passes."""
        self.curr_step += 1

        results.update({"prior_time": prior_time, "train_time": train_time})

        is_temp_save = self.curr_step % self.config.save_temp_every == 0
        is_perm_save = self.curr_step % self.config.save_perm_every == 0

        if is_temp_save or is_perm_save:
            ckpt_name = f"step-{self.curr_step}.ckpt"
            self.save_checkpoint(name=ckpt_name)

            if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                self.manage_checkpoint()

        surface_results = {}

        ran_prior_eval = (
            self.prior_eval_dir is not None
            and self.prior_eval_interval > 0
            and (self.curr_step % self.prior_eval_interval == 0 or self.curr_step == 1)
        )
        if ran_prior_eval:
            surface_results.update(self.prior_validate())

        ran_synt_eval = (
            self.synt_eval_dir is not None
            and self.synt_eval_interval > 0
            and (self.curr_step % self.synt_eval_interval == 0 or self.curr_step == 1)
        )
        if ran_synt_eval:
            surface_results.update(self.synt_validate())

        if surface_results:
            results.update(surface_results)

            surface_ran = self._monitored_surface_ran(
                ran_prior_eval=ran_prior_eval,
                ran_synt_eval=ran_synt_eval,
            )
            if self.save_best and surface_ran:
                self._maybe_save_best(surface_results)

        results["lr"] = self.scheduler.get_last_lr()[0]
        self.logger.log(results, step=self.curr_step)
        return results

    def train(self):
        """Execute the main training loop across configured steps."""

        step_progress = tqdm(
            initial=self.curr_step,
            total=self.config.max_steps,
            desc="Step",
            leave=True,
        )

        dataloader = iter(self.dataloader)
        num_batches_per_block = max(
            1, math.ceil(self.config.gen_block_size / self.config.batch_size)
        )

        while self.curr_step < self.config.max_steps:
            with Timer() as gen_timer:
                block = []
                for _ in range(num_batches_per_block):
                    try:
                        block.append(next(dataloader))
                    except RuntimeError as e:
                        if not is_oom_error(e):
                            raise
                        print(
                            "\n[Step {}] OutOfMemory caught during block "
                            "generation. Using a partial block.".format(self.curr_step)
                        )
                        empty_accelerator_cache(self.config.device)
                        empty_accelerator_cache(self.config.prior_device)
                        break
            if not block:
                # Generation OOM'd before the first batch; retry.
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

                    row_perm = self._batch_row_permutation(block[bi])

                    try:
                        with Timer() as train_timer:
                            results = self.run_batch(block[bi], row_perm=row_perm)
                    except RuntimeError as e:
                        if not is_oom_error(e):
                            raise
                        print(
                            f"\n[Step {self.curr_step}] OutOfMemory caught during "
                            "training. Skipping batch."
                        )
                        empty_accelerator_cache(self.config.device)
                        continue

                    results = self._post_step(
                        results, amortized_prior_time, train_timer.elapsed
                    )

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

        step_progress.close()

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """Validate consistent sequence length and train size within a micro batch."""
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
        """Truncate micro batch tensors to required sequence length and active feature dimensions."""
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

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Process a single micro batch for gradient accumulation."""
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

        device = self.config.device
        micro_X = micro_X.to(device, non_blocking=True)
        micro_W = micro_W.to(device, non_blocking=True)
        micro_Z = micro_Z.to(device, non_blocking=True)
        micro_A = micro_A.to(device, non_blocking=True)
        micro_Y_obs = micro_Y_obs.to(device, non_blocking=True)
        micro_Y0 = micro_Y0.to(device, non_blocking=True)
        micro_Y1 = micro_Y1.to(device, non_blocking=True)
        micro_MU0 = micro_MU0.to(device, non_blocking=True)
        micro_MU1 = micro_MU1.to(device, non_blocking=True)
        micro_S2 = micro_S2.to(device, non_blocking=True)
        micro_d_X = micro_d_X.to(device, non_blocking=True)
        micro_d_W = micro_d_W.to(device, non_blocking=True)
        micro_d_Z = micro_d_Z.to(device, non_blocking=True)

        y_train_obs = micro_Y_obs[:, :train_size]
        y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
        y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(
            min=1e-3
        )

        y_test_0_sampled = micro_Y0[:, train_size:]
        y_test_1_sampled = micro_Y1[:, train_size:]
        mu_test_0 = micro_MU0[:, train_size:]
        mu_test_1 = micro_MU1[:, train_size:]

        y_test_0_std = (mu_test_0 - y_context_mean) / y_context_std
        y_test_1_std = (mu_test_1 - y_context_mean) / y_context_std
        true_cate_std = y_test_1_std - y_test_0_std

        sampled_y_test_0_std = (y_test_0_sampled - y_context_mean) / y_context_std
        sampled_y_test_1_std = (y_test_1_sampled - y_context_mean) / y_context_std
        sampled_ite_std = sampled_y_test_1_std - sampled_y_test_0_std

        target_var_std = micro_S2[:, train_size:] / (y_context_std**2)

        A_train = micro_A[:, :train_size].squeeze(-1)
        n0 = (A_train == 0).sum(dim=1)
        n1 = (A_train == 1).sum(dim=1)

        if (n0 == 0).any() or (n1 == 0).any():
            return None

        with self.amp_ctx:
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

            if pred_logits.isnan().any():
                return None

            loss_type = self.loss_type

            if loss_type == "bins_po":
                nbins = self.model.nbins
                logits_y0 = pred_logits[..., :nbins]
                logits_y1 = pred_logits[..., nbins:]

                probs_y0 = F.softmax(logits_y0, dim=-1)
                probs_y1 = F.softmax(logits_y1, dim=-1)

                loss_y0 = compute_crps_loss(
                    logits_y0,
                    y_test_0_std,
                    self.model.bin_edges,
                    probs=probs_y0,
                )
                loss_y1 = compute_crps_loss(
                    logits_y1,
                    y_test_1_std,
                    self.model.bin_edges,
                    probs=probs_y1,
                )

                mean_pred_y0_std = torch.sum(probs_y0 * self.model.bin_centers, dim=-1)
                mean_pred_y1_std = torch.sum(probs_y1 * self.model.bin_centers, dim=-1)

                pred_cate_std = mean_pred_y1_std - mean_pred_y0_std

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.1)

                total_loss = loss_y0 + loss_y1 + (lambda_cate * loss_cate)

            elif loss_type == "cate_decomposition":
                ate_mu = pred_logits[..., 0]
                ate_logvar = pred_logits[..., 1]
                he_logits = pred_logits[..., 2:]

                probs_he = F.softmax(he_logits, dim=-1)

                true_ate_std = true_cate_std.mean(dim=1, keepdim=True)
                true_he_std = true_cate_std - true_ate_std

                loss_ate_nll = (
                    compute_gaussian_nll(ate_mu, ate_logvar, true_ate_std) / 10
                )
                loss_he_crps = compute_crps_loss(
                    he_logits,
                    true_he_std,
                    self.model.bin_edges,
                    probs=probs_he,
                )

                lambda_ate = getattr(self.config, "lambda_ate", 1.0)
                lambda_he = getattr(self.config, "lambda_he", 1.0)

                total_loss = (lambda_ate * loss_ate_nll) + (lambda_he * loss_he_crps)

                mean_he_pred_std = torch.sum(probs_he * self.model.bin_centers, dim=-1)

                pred_cate_std = ate_mu + mean_he_pred_std

            elif loss_type == "hl":
                nbins = self.model.nbins
                logits_y0 = pred_logits[..., :nbins]
                logits_y1 = pred_logits[..., nbins:]

                hl_sigma = self.get_hl_sigma()

                loss_y0 = compute_hl_gaussian_cross_entropy_loss(
                    logits_y0,
                    y_test_0_std,
                    self.model.bin_edges,
                    sigma=hl_sigma,
                )
                loss_y1 = compute_hl_gaussian_cross_entropy_loss(
                    logits_y1,
                    y_test_1_std,
                    self.model.bin_edges,
                    sigma=hl_sigma,
                )

                probs_y0 = F.softmax(logits_y0, dim=-1)
                probs_y1 = F.softmax(logits_y1, dim=-1)

                mean_pred_y0_std = torch.sum(probs_y0 * self.model.bin_centers, dim=-1)
                mean_pred_y1_std = torch.sum(probs_y1 * self.model.bin_centers, dim=-1)

                pred_cate_std = mean_pred_y1_std - mean_pred_y0_std

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.1)

                total_loss = loss_y0 + loss_y1 + (lambda_cate * loss_cate)

            elif loss_type in ("cate_single", "cate_single_noise_aware"):
                hl_sigma = self.get_hl_sigma()

                if loss_type == "cate_single_noise_aware":
                    loss_cate_dist = compute_cate_noise_aware_nll(
                        pred_logits,
                        true_cate_std,
                        self.model.cate_bin_edges,
                        target_var=target_var_std,
                        sigma=hl_sigma,
                    )
                else:
                    loss_cate_dist = compute_cate_distribution_loss(
                        pred_logits,
                        true_cate_std,
                        self.model.cate_bin_edges,
                        sigma=hl_sigma,
                    )

                probs_cate = F.softmax(pred_logits, dim=-1)
                pred_cate_std = torch.sum(
                    probs_cate * self.model.cate_bin_centers, dim=-1
                )

                cate_error_std = pred_cate_std - true_cate_std
                loss_cate = torch.mean(cate_error_std**2)

                lambda_cate = getattr(self.config, "lambda_cate", 0.0)

                total_loss = loss_cate_dist + (lambda_cate * loss_cate)

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        if not torch.isfinite(total_loss):
            warnings.warn(
                f"Non-finite {loss_type} loss at step {self.curr_step}; "
                "skipping this micro batch. If this repeats, check "
                "target/cate_out_of_grid_frac -- a target far outside "
                "[--cate_vmin, --cate_vmax] is the usual cause.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        scaled_loss = total_loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()
        reported_loss_total = scaled_loss

        with torch.no_grad():
            scale = num_micro_batches

            if loss_type == "bins_po":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0": loss_y0,
                    "loss/y1": loss_y1,
                    "loss/cate_mse_weighted": lambda_cate * loss_cate,
                    "loss/cate_mse": loss_cate,
                }
            elif loss_type == "cate_decomposition":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/ate_nll": lambda_ate * loss_ate_nll,
                    "loss/he_crps": lambda_he * loss_he_crps,
                }
            elif loss_type == "hl":
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/y0_hl": loss_y0,
                    "loss/y1_hl": loss_y1,
                    "loss/cate_mse_weighted": lambda_cate * loss_cate,
                    "loss/cate_mse": loss_cate,
                    "loss/hl_sigma": hl_sigma,
                }
            elif loss_type in ("cate_single", "cate_single_noise_aware"):
                micro_results = {
                    "loss/total": reported_loss_total,
                    "loss/cate_dist": loss_cate_dist,
                    "loss/cate_mse_weighted": lambda_cate * loss_cate,
                    "loss/cate_mse": loss_cate,
                    "loss/hl_sigma": hl_sigma,
                }

                if loss_type == "cate_single_noise_aware":
                    micro_results.update(
                        self._target_noise_metrics(
                            target_var_std,
                            true_cate_std,
                            probs_cate,
                            pred_cate_std,
                            micro_K_eff,
                        )
                    )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            micro_results.update(
                {
                    "target/cate_mean": true_cate_std.mean(),
                    "target/cate_std": true_cate_std.std(unbiased=False),
                    "target/sampled_ite_mean": sampled_ite_std.mean(),
                    "target/ite_minus_cate_mse": torch.mean(
                        (sampled_ite_std - true_cate_std) ** 2
                    ),
                }
            )

        return self._resolve_metrics(micro_results, scale)

    def run_batch(self, batch, row_perm=None):
        """Train the model on one batch of datasets for a single optimizer step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        num_micro_batches = math.ceil(
            self.config.batch_size / self.config.micro_batch_size
        )
        micro_batches = self._iter_micro_batches(
            batch, self.config.micro_batch_size, row_perm
        )

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

        # Clip the gradient. The norm is computed either way -- with
        # ``max_norm=inf`` the call measures and scales by 1 -- because it is
        # also the guard below, and one extra reduction per step is cheap next
        # to a silently dead model.
        max_norm = (
            self.config.gradient_clipping
            if self.config.gradient_clipping > 0
            else float("inf")
        )
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

        # A finite loss does not imply finite gradients, and clipping does not
        # rescue a NaN -- ``clip_grad_norm_`` multiplies by a NaN scale and
        # leaves every gradient NaN. On CUDA + AMP ``GradScaler`` would catch
        # this and skip the step; ``configure_amp`` disables the scaler off
        # CUDA, so on MPS/CPU nothing does. Skip the step explicitly instead.
        if not torch.isfinite(grad_norm):
            warnings.warn(
                f"Non-finite gradient norm at step {self.curr_step}; skipping "
                "the optimizer step. Repeated hits mean the run is diverging, "
                "not that one batch was unlucky.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            return results

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results

    @torch.inference_mode()
    def _validate_surface(self, surface: str):
        eval_dir = getattr(self, f"{surface}_eval_dir", None)
        if eval_dir is None:
            return {}

        eval_meta = getattr(self, f"_{surface}_eval_meta", None)
        max_episodes = getattr(self.config, f"{surface}_eval_max_episodes", None)

        self.model.eval()

        with Timer() as timer:
            metrics = score_frozen_set(
                self.model,
                eval_dir,
                device=self.config.device,
                max_episodes=max_episodes,
                meta=eval_meta,
                prefix_point=f"{surface}_point/",
                prefix_dist=f"{surface}_dist/",
            )

        metrics[f"Charts/{surface}_eval_time"] = timer.elapsed

        if surface == "prior":
            grid_scale = metrics.get("prior_dist/grid_scale")
            if grid_scale is not None:
                metrics["diag/sigma"] = self.get_hl_sigma() * grid_scale

        self.model.train()

        return metrics

    @torch.inference_mode()
    def prior_validate(self):
        return self._validate_surface("prior")

    @torch.inference_mode()
    def synt_validate(self):
        return self._validate_surface("synt")

    def _resolve_metrics(self, metrics: dict, scale: int) -> dict[str, float]:
        """Materialize metric dict of 0-d tensors into floats with a single device transfer."""
        tensor_keys = [k for k, v in metrics.items() if torch.is_tensor(v)]

        resolved: dict[str, float] = {}
        if tensor_keys:
            device = self.config.device
            stacked = torch.stack(
                [
                    metrics[key].detach().float().reshape(()).to(device)
                    for key in tensor_keys
                ]
            )
            resolved = dict(zip(tensor_keys, stacked.cpu().tolist()))

        out: dict[str, float] = {}
        for key, value in metrics.items():
            value = resolved[key] if key in resolved else float(value)
            out[key] = value if key in _UNSCALED_METRICS else value / scale
        return out

    @torch.no_grad()
    def _target_noise_metrics(
        self,
        target_var_std,
        true_cate_std,
        probs_cate,
        pred_cate_std,
        micro_K_eff,
    ) -> dict:
        """Diagnostics for the noise-aware CATE objective."""
        centers = self.model.cate_bin_centers.float()
        probs = probs_cate.float()
        mean = pred_cate_std.float().unsqueeze(-1)
        var_pred = torch.sum(probs * (centers.view(1, 1, -1) - mean) ** 2, dim=-1)

        s2 = torch.nan_to_num(target_var_std.float(), nan=0.0, posinf=0.0).clamp_min(
            0.0
        )
        mean_s = torch.sqrt(s2).mean()
        mean_sd_pred = torch.sqrt(var_pred.clamp_min(0.0)).mean()

        frac_fallback = (~(s2 > 0)).float().mean()

        edges = self.model.cate_bin_edges
        out_of_grid = (
            ((true_cate_std < edges[0]) | (true_cate_std > edges[-1])).float().mean()
        )

        return {
            "noise/mean_s": mean_s,
            "noise/mean_sd_pred": mean_sd_pred,
            "noise/s_over_sd_pred": mean_s / mean_sd_pred.clamp_min(1e-12),
            "noise/frac_fallback": frac_fallback,
            "noise/mean_K": micro_K_eff.float().mean(),
            "target/cate_out_of_grid_frac": out_of_grid,
        }

    def get_hl_sigma(self) -> float:
        """Return the current sigma for HL-Gaussian loss schedule."""
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
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = Trainer(config)
    trainer.train()
