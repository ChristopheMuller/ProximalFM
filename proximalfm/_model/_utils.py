from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor


def get_d_total(
    d_X: Optional[int], d_Z: Optional[int], d_W: Optional[int]
) -> Optional[int]:
    if d_X is not None and d_Z is not None and d_W is not None:
        return d_X + d_Z + d_W
    return None


def safe_standardize(
    x: Tensor,
    train_size: int,
    fit_on_context_only: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    fit_x = x[:, :train_size, :] if fit_on_context_only else x
    mu = fit_x.mean(dim=1, keepdim=True)
    std = fit_x.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std, mu, std


def hide_query_proxies(
    Z: Tensor,
    W: Tensor,
    train_size: int,
) -> Tuple[Tensor, Tensor]:
    if Z.shape[1] > train_size:
        Z = Z.clone()
        Z[:, train_size:, :] = 0.0
    if W.shape[1] > train_size:
        W = W.clone()
        W[:, train_size:, :] = 0.0
    return Z, W


def standardize_forward_inputs(
    X: Tensor,
    Z: Tensor,
    W: Tensor,
    y_train: Tensor,
    train_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    X_std, _, _ = safe_standardize(X, train_size=train_size, fit_on_context_only=True)
    Z_std, _, _ = safe_standardize(Z, train_size=train_size, fit_on_context_only=True)
    W_std, _, _ = safe_standardize(W, train_size=train_size, fit_on_context_only=True)
    Z_std, W_std = hide_query_proxies(Z_std, W_std, train_size)
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
    y_train_std = (y_train - y_mean) / y_std
    return X_std, Z_std, W_std, y_train_std, y_mean, y_std


def get_histogram_mean(probs: Tensor, vals: Tensor) -> Tensor:
    return torch.sum(probs * vals, dim=-1)


def get_histogram_quantiles(
    probs: Tensor, vals: Tensor, q_list: Tuple[Tensor, ...], nbins: int
) -> Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    results = []
    if vals.shape[1] == 1 and probs.shape[1] > 1:
        vals = vals.expand(-1, probs.shape[1], -1)
    for q in q_list:
        q_expanded = q.expand(*cdf.shape[:-1], 1).contiguous()
        idx = torch.searchsorted(cdf, q_expanded).clamp(max=nbins - 1)
        q_vals = torch.gather(vals, -1, idx).squeeze(-1)
        results.append(q_vals)
    return torch.stack(results, dim=-1)


def average_ensemble_outputs(outputs: list) -> Any:
    first = outputs[0]
    if isinstance(first, Tensor):
        return torch.stack(outputs, dim=0).mean(dim=0)
    if isinstance(first, dict):
        if "weights" in first:
            weights = torch.stack([o["weights"] for o in outputs], dim=0).mean(dim=0)
            logits = torch.stack([o["logits"] for o in outputs], dim=0).mean(dim=0)
            return {"weights": weights, "logits": logits}
        return {k: average_ensemble_outputs([o[k] for o in outputs]) for k in first}
    raise TypeError(f"Unsupported ensemble output type: {type(first)}")


class MemoryEstimator:
    coefficients: Dict[str, list] = {
        "tf_col": [1.456149314e-01, 1.94081457e-05, 4.88223400e-03],
        "tf_row": [-2.06831848e-05, 2.27205969e-04, 5.37117114e-03],
        "tf_icl": [-4.03932756e-02, 5.42811085e-07, 1.95312473e-02],
    }
    intercepts: Dict[str, float] = {
        "tf_col": 142.91294659096457,
        "tf_row": 138.53653545318957,
        "tf_icl": 142.84243874417552,
    }

    @staticmethod
    def estimate_peak_mem(
        batch_size: int,
        seq_len: int,
        enc_name: str,
        include_inputs: bool = True,
        in_dim: Optional[int] = None,
    ) -> float:
        coefs = MemoryEstimator.coefficients[enc_name]
        inter = MemoryEstimator.intercepts[enc_name]
        peak_activation_mem = (
            coefs[0] * batch_size
            + coefs[1] * seq_len
            + coefs[2] * batch_size * seq_len
            + inter
        )
        if include_inputs and in_dim is not None:
            input_mem = batch_size * seq_len * in_dim * 4 / (1024 * 1024)
            peak_activation_mem += input_mem
        return peak_activation_mem

    @staticmethod
    def estimate_batch_size(
        seq_len: int,
        target_memory: float,
        enc_name: str,
        include_inputs: bool = True,
        in_dim: Optional[int] = None,
    ) -> int:
        if target_memory <= 0:
            return 1
        coefs = MemoryEstimator.coefficients[enc_name]
        intercept = MemoryEstimator.intercepts[enc_name]
        numerator = target_memory - coefs[1] * seq_len - intercept
        denominator = coefs[0] + coefs[2] * seq_len
        if include_inputs and in_dim is not None:
            denominator += seq_len * in_dim * 4 / (1024**2)
        if denominator <= 0:
            return 1
        return max(1, int(numerator / denominator))
