from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist


class CausalEvaluator:
    """
    Stateful streaming evaluator for causal metrics.

    It does not store predictions. Instead, update() computes per-episode metrics
    for each micro-batch and accumulates metric sums + episode counts.

    Metric philosophy:
        - CATE metrics: computed per episode, then averaged across episodes.
        - ATE metrics: computed over the full buffer of episodes (each episode
          contributes one ATE scalar; NRMSE and R² are evaluated across them).

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

        # 1. Potential outcome metrics (per-episode, then averaged)
        mse_y0 = F.mse_loss(pred_y0, true_y0, reduction="none").mean(dim=1)
        mse_y1 = F.mse_loss(pred_y1, true_y1, reduction="none").mean(dim=1)

        nrmse_y0 = torch.sqrt(mse_y0) / y_obs_std
        nrmse_y1 = torch.sqrt(mse_y1) / y_obs_std

        self.metric_sums["po/nrmse_y0"] += nrmse_y0.sum().item()
        self.metric_sums["po/nrmse_y1"] += nrmse_y1.sum().item()

        # 2. CATE metrics (per-episode, then averaged)
        mse_cate = F.mse_loss(pred_cate, true_cate, reduction="none").mean(dim=1)
        nrmse_cate = torch.sqrt(mse_cate) / y_obs_std

        var_cate = true_cate.var(dim=1, unbiased=False)
        r2_cate = torch.where(
            var_cate > self.eps,
            1.0 - (mse_cate / var_cate.clamp_min(self.eps)),
            torch.where(mse_cate < self.eps, 1.0, 0.0),
        )

        # R²-shape: removes the penalty from ATE mis-estimation so that only
        # the heterogeneity-capturing ability is measured.
        #   MSE = Bias² + Var(residuals)  →  R²_shape = 1 − Var(pred−true) / Var(true)
        # R²_shape ≥ R², with the gap equal to Bias²/Var(true).
        var_residuals = (pred_cate - true_cate).var(dim=1, unbiased=False)
        r2_shape_cate = torch.where(
            var_cate > self.eps,
            1.0 - (var_residuals / var_cate.clamp_min(self.eps)),
            torch.where(var_residuals < self.eps, 1.0, 0.0),
        )

        self.metric_sums["cate/nrmse"] += nrmse_cate.sum().item()
        self.metric_sums["cate/r2"] += r2_cate.sum().item()
        self.metric_sums["cate/r2_shape"] += r2_shape_cate.sum().item()

        # 3. ATE — accumulate buffer-tracking sums only.
        #    NRMSE and R² are both derived in compute() over the full episode buffer,
        #    so that each episode contributes one ATE scalar to a proper distribution.
        true_ate = true_cate.mean(dim=1)
        pred_ate = pred_cate.mean(dim=1)

        mse_ate = (true_ate - pred_ate) ** 2

        self.metric_sums["_ate/mse_sum"] += mse_ate.sum().item()
        self.metric_sums["_ate/true_sum"] += true_ate.sum().item()
        self.metric_sums["_ate/true_sq_sum"] += (true_ate**2).sum().item()

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

        CATE/PO metrics are per-episode averages.
        ATE metrics (nrmse, r2) are computed over the full episode buffer.
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

        raw_sums = {k: v for k, v in zip(keys, sums)}
        metrics = {}

        # 1. Per-episode averages (skips buffer-tracking keys starting with '_')
        for key in keys:
            if not key.startswith("_"):
                metrics[f"{prefix}{key}"] = raw_sums[key] / count

        # 2. ATE metrics — both derived from the same buffer-level statistics so
        #    that NRMSE and R² share a consistent normalizer (cross-episode std of
        #    true ATE) and both reflect performance across the episode distribution.
        if "_ate/mse_sum" in raw_sums:
            mse_ate = raw_sums["_ate/mse_sum"] / count
            mean_ate = raw_sums["_ate/true_sum"] / count
            sq_mean_ate = raw_sums["_ate/true_sq_sum"] / count

            # Clamp to zero before sqrt to guard against floating-point negatives.
            var_ate = max(sq_mean_ate - mean_ate**2, 0.0)

            if var_ate > self.eps:
                metrics[f"{prefix}ate/nrmse"] = math.sqrt(mse_ate) / math.sqrt(var_ate)
                metrics[f"{prefix}ate/r2"] = 1.0 - (mse_ate / var_ate)
            else:
                # Degenerate case: all true ATEs are identical across episodes.
                metrics[f"{prefix}ate/nrmse"] = (
                    0.0 if mse_ate < self.eps else float("inf")
                )
                metrics[f"{prefix}ate/r2"] = 1.0 if mse_ate < self.eps else 0.0

        if include_count:
            metrics[f"{prefix}eval/episodes"] = count

        return metrics
