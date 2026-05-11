from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist


class CausalEvaluator:
    """
    Stateful streaming evaluator for causal metrics.

    It does not store predictions. Instead, update() computes per-episode metrics
    for each micro-batch and accumulates metric sums + episode counts.

    Workflow:
        evaluator.update(...)
        metrics = evaluator.compute(prefix="train_")
        evaluator.reset()
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.reset()

    def reset(self):
        self.metric_sums = defaultdict(float)
        self.num_episodes = 0
        self.num_updates = 0

    @property
    def empty(self) -> bool:
        return self.num_episodes == 0

    def ready(self, min_episodes: int) -> bool:
        return self.num_episodes >= min_episodes

    @torch.no_grad()
    def update(
        self,
        *,
        pred_y0: torch.Tensor,
        pred_y1: torch.Tensor,
        true_y0: torch.Tensor,
        true_y1: torch.Tensor,
        pred_cate: torch.Tensor,
        true_cate: torch.Tensor,
        y_obs_std: torch.Tensor,
        extra_metrics: dict[str, float | torch.Tensor] | None = None,
    ):
        """
        Update evaluator state with one micro-batch.

        All causal metrics are computed per episode, then summed. This makes the
        final compute() result an episode-weighted average across all updates.
        """

        pred_y0 = pred_y0.detach().float()
        pred_y1 = pred_y1.detach().float()
        true_y0 = true_y0.detach().float()
        true_y1 = true_y1.detach().float()
        pred_cate = pred_cate.detach().float()
        true_cate = true_cate.detach().float()

        y_obs_std = y_obs_std.detach().float().squeeze(-1)
        y_obs_std = y_obs_std.clamp_min(self.eps)

        batch_size = pred_y0.shape[0]

        # Potential outcome metrics
        mse_y0 = F.mse_loss(pred_y0, true_y0, reduction="none").mean(dim=1)
        mse_y1 = F.mse_loss(pred_y1, true_y1, reduction="none").mean(dim=1)

        nrmse_y0 = torch.sqrt(mse_y0) / y_obs_std
        nrmse_y1 = torch.sqrt(mse_y1) / y_obs_std

        # CATE metrics
        mse_cate = F.mse_loss(pred_cate, true_cate, reduction="none").mean(dim=1)
        nrmse_cate = torch.sqrt(mse_cate) / y_obs_std

        self.metric_sums["po/nrmse_y0"] += nrmse_y0.sum().item()
        self.metric_sums["po/nrmse_y1"] += nrmse_y1.sum().item()
        self.metric_sums["cate/nrmse"] += nrmse_cate.sum().item()

        # Optional scalar metrics, e.g. loss/total, loss/y0, loss/y1.
        # These are assumed to be micro-batch averages, so multiply by batch_size.
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().item()
                else:
                    value = float(value)

                self.metric_sums[key] += value * batch_size

        self.num_episodes += batch_size
        self.num_updates += 1

    def compute(
        self,
        *,
        prefix: str = "",
        sync_ddp: bool = False,
        device: str | torch.device | None = None,
        include_count: bool = True,
    ) -> dict[str, float]:
        """
        Compute averaged metrics.

        If sync_ddp=True, metric sums and counts are summed across DDP ranks.
        This must be called by all ranks when enabled.
        """

        if self.num_episodes == 0:
            return {}

        keys = sorted(self.metric_sums.keys())
        count = float(self.num_episodes)
        sums = [float(self.metric_sums[k]) for k in keys]

        if sync_ddp and dist.is_available() and dist.is_initialized():
            if device is None:
                raise ValueError("device must be provided when sync_ddp=True")

            payload = torch.tensor(
                [count, *sums],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)

            count = payload[0].item()
            sums = payload[1:].tolist()

        if count <= 0:
            return {}

        metrics = {f"{prefix}{key}": value / count for key, value in zip(keys, sums)}

        if include_count:
            metrics[f"{prefix}eval/episodes"] = count

        return metrics
