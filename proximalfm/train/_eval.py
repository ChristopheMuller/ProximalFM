from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F


class CausalEvaluator:
    """
    Stateful streaming evaluator for causal metrics.

    It does not store predictions. Instead, update() computes per-episode metrics
    for each micro-batch and accumulates them.

    Metric philosophy:
        - CATE/PO metrics: computed per episode in standardized space. The
          per-episode values are retained so that compute() can report robust
          summaries (median, p25, p75) rather than a mean. R²-style metrics are
          unbounded below, so a handful of near-homogeneous episodes (tiny
          Var(τ)) can drive a *mean* arbitrarily negative even when the model is
          good; the median is unaffected. To match the benchmark
          (``benchmarks/bench_metrics.py``), episodes with
          ``Var(τ) < homogeneous_var_threshold`` are *excluded* from the R²
          metrics entirely.
        - ATE metrics: computed over the full buffer of episodes.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        homogeneous_var_threshold: float = 1e-6,
    ):
        self.eps = eps
        # Episodes whose true CATE variance is below this are "constant effect"
        # episodes; the scale-free R² metrics are undefined for them and they
        # are dropped (matches benchmarks.bench_metrics.ITE_VAR_THRESHOLD).
        self.homogeneous_var_threshold = homogeneous_var_threshold
        self.reset()

    def reset(self):
        # Running sums for buffer-level metrics (ATE) and extra_metrics.
        self.metric_sums = defaultdict(float)
        # Per-episode value tensors for metrics summarized via percentiles.
        self.metric_values = defaultdict(list)
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
        pred_cate: torch.Tensor,
        true_cate: torch.Tensor,
        pred_y0: torch.Tensor | None = None,
        pred_y1: torch.Tensor | None = None,
        true_y0: torch.Tensor | None = None,
        true_y1: torch.Tensor | None = None,
        extra_metrics: dict[str, float | torch.Tensor] | None = None,
    ):
        """
        Update evaluator state with one micro-batch. All inputs are assumed
        to be in standardized space.

        Per-arm predictions (``pred_y0``/``pred_y1``/``true_y0``/``true_y1``)
        are optional: loss types with no y0/y1 head (``cate_decomposition``,
        ``cate_single``, ``cate_single_noise_aware``) have nothing honest to
        pass here, so ``po/*`` metrics are only produced when all four are
        given.
        """

        pred_cate = pred_cate.detach().float()
        true_cate = true_cate.detach().float()

        batch_size = pred_cate.shape[0]

        # 1. Potential outcome metrics (per-episode; summarized via percentiles).
        # Opt-in: only loss types with a real per-arm head pass these.
        if pred_y0 is not None:
            pred_y0 = pred_y0.detach().float()
            pred_y1 = pred_y1.detach().float()
            true_y0 = true_y0.detach().float()
            true_y1 = true_y1.detach().float()

            mse_y0 = F.mse_loss(pred_y0, true_y0, reduction="none").mean(dim=1)
            mse_y1 = F.mse_loss(pred_y1, true_y1, reduction="none").mean(dim=1)

            rmse_y0 = torch.sqrt(mse_y0)
            rmse_y1 = torch.sqrt(mse_y1)

            self.metric_values["po/rmse_y0"].append(rmse_y0.cpu())
            self.metric_values["po/rmse_y1"].append(rmse_y1.cpu())

        # 2. CATE metrics (per-episode; summarized via percentiles)
        mse_cate = F.mse_loss(pred_cate, true_cate, reduction="none").mean(dim=1)
        rmse_cate = torch.sqrt(mse_cate)
        self.metric_values["cate/rmse"].append(rmse_cate.cpu())

        # Sign / decision agreement: fraction of test units whose predicted
        # effect direction matches the truth. Scale-free and defined for every
        # episode, so it is kept for all of them.
        sign_agreement = (
            (torch.sign(pred_cate) == torch.sign(true_cate)).float().mean(dim=1)
        )
        self.metric_values["cate/sign_agreement"].append(sign_agreement.cpu())

        # nPEHE / R² / ATE-bias are scale-free in Var(τ) and only defined for
        # heterogeneous episodes. Drop near-constant episodes (as the benchmark
        # does) rather than assigning them a fallback value that pollutes the
        # distribution.
        true_ate = true_cate.mean(dim=1)
        pred_ate = pred_cate.mean(dim=1)

        var_cate = true_cate.var(dim=1, unbiased=False)
        var_residuals = (pred_cate - true_cate).var(dim=1, unbiased=False)
        heterogeneous = var_cate >= self.homogeneous_var_threshold

        if heterogeneous.any():
            vc = var_cate[heterogeneous]
            sd_tau = vc.sqrt()
            npehe = (mse_cate[heterogeneous] / vc).sqrt()
            r2_cate = 1.0 - (mse_cate[heterogeneous] / vc)
            r2_shape_cate = 1.0 - (var_residuals[heterogeneous] / vc)
            # Signed, normalized ATE bias: (mean(τ̂) − mean(τ)) / SD(τ).
            ate_bias_norm = (pred_ate[heterogeneous] - true_ate[heterogeneous]) / sd_tau
            self.metric_values["cate/npehe"].append(npehe.cpu())
            self.metric_values["cate/r2"].append(r2_cate.cpu())
            self.metric_values["cate/r2_shape"].append(r2_shape_cate.cpu())
            self.metric_values["ate/bias_norm"].append(ate_bias_norm.cpu())

        # 3. ATE — accumulate buffer-tracking sums only.
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
        include_count: bool = True,
    ) -> dict[str, float]:
        """
        Compute summary metrics.

        Per-episode metrics (RMSE, R², R²-shape) are summarized by their median
        and inter-quartile bounds (p25/p75) instead of a mean, which is robust to
        the heavy left tail of the R² metrics. ATE metrics and ``extra_metrics``
        remain buffer-level means.
        """

        if self.num_episodes == 0:
            return {}

        count = float(self.num_episodes)
        raw_sums = {k: float(v) for k, v in self.metric_sums.items()}

        # Concatenate per-episode value tensors for the percentile metrics.
        value_tensors = {
            k: torch.cat(v).double() for k, v in self.metric_values.items() if v
        }

        if count <= 0:
            return {}

        keys = sorted(raw_sums.keys())
        metrics = {}

        # 1. Per-episode metrics: median + inter-quartile bounds.
        quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        for key, values in value_tensors.items():
            if values.numel() == 0:
                continue
            p25, median, p75 = torch.quantile(values, quantiles).tolist()
            metrics[f"{prefix}{key}"] = median
            metrics[f"{prefix}{key}/p25"] = p25
            metrics[f"{prefix}{key}/p75"] = p75

        # 2. extra_metrics: buffer-level means (non-underscore sum keys).
        for key in keys:
            if not key.startswith("_"):
                metrics[f"{prefix}{key}"] = raw_sums[key] / count

        # 3. ATE metrics
        if "_ate/mse_sum" in raw_sums:
            mse_ate = raw_sums["_ate/mse_sum"] / count
            mean_ate = raw_sums["_ate/true_sum"] / count
            sq_mean_ate = raw_sums["_ate/true_sq_sum"] / count

            var_ate = max(sq_mean_ate - mean_ate**2, 0.0)

            # Use raw RMSE for ATE since data is already standardized
            metrics[f"{prefix}ate/rmse"] = math.sqrt(mse_ate)

            if var_ate > self.eps:
                metrics[f"{prefix}ate/r2"] = 1.0 - (mse_ate / var_ate)
            else:
                metrics[f"{prefix}ate/r2"] = 1.0 if mse_ate < self.eps else 0.0

        if include_count:
            metrics[f"{prefix}eval_episodes"] = count

        return metrics
