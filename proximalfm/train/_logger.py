"""Wandb integration for ``proximalfm.train._run.Trainer``."""

from __future__ import annotations

import datetime
import os

import wandb


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

    def update_config(self, config_dict: dict):
        """Add keys to the run's config after ``wandb.init``.

        For provenance that is only known once something has been loaded -- the
        frozen eval set's ``content_digest``, say, which is read from a sidecar
        after the logger already exists. ``allow_val_change`` because a resumed
        run re-sets the same keys.
        """
        if self.use_wandb and self.run is not None:
            self.run.config.update(config_dict, allow_val_change=True)

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
