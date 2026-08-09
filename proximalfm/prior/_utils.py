from __future__ import annotations

import random
from typing import Dict, Any, Optional, List
import numpy as np

import torch
from torch import nn, Tensor


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        if torch.is_tensor(std):
            self._is_zero = bool((std == 0).all())
        else:
            self._is_zero = std == 0

    def forward(self, X):
        if self._is_zero:
            return X
        return X + torch.randn_like(X).mul_(self.std)


class XSampler:
    def __init__(
        self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"
    ):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(
            means, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(
            0
        )

    def sample(self, return_numpy=False):
        samplers = {
            "normal": self.sample_normal_all,
            "mixed": self.sample_mixed,
            "uniform": self.sample_uniform,
        }
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()
        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            noise = torch.randn(self.seq_len, self.num_features, device=self.device)
            X = (self.means + self.stds * noise).float()
        else:
            X = torch.normal(
                0.0, 1.0, (self.seq_len, self.num_features), device=self.device
            ).float()
        return X

    def sample_uniform(self):
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        if self.pre_stats:
            noise = torch.randn(self.seq_len, device=self.device)
            return (self.means[0, n] + self.stds[0, n] * noise).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        std, mean = torch.std_mean(x)
        scale = torch.nan_to_num(std, nan=0.0).clamp(min=1e-8)
        return (x - mean) / scale

    def sample_zipf(self):
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        zipf_p, multi_p, normal_p = (
            random.random() * 0.66,
            random.random() * 0.66,
            random.random() * 0.66,
        )
        X = torch.empty(self.seq_len, self.num_features, device=self.device)
        normal_cols = []
        for n in range(self.num_features):
            if random.random() > normal_p:
                normal_cols.append(n)
            elif random.random() > multi_p:
                X[:, n] = self.sample_multinomial()
            elif random.random() > zipf_p:
                X[:, n] = self.sample_zipf()
            else:
                X[:, n] = torch.rand((self.seq_len,), device=self.device)
        if normal_cols:
            noise = torch.randn(self.seq_len, len(normal_cols), device=self.device)
            if self.pre_stats:
                cols = torch.tensor(normal_cols, device=self.device)
                noise = self.means[0, cols] + self.stds[0, cols] * noise
            X[:, normal_cols] = noise
        return X


def scalar_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (bool, str)):
            out[key] = value
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = float(value)
    return out


def mc_mean_var(replicates: Tensor, K: int) -> Tensor:
    if K < 2:
        return torch.zeros(
            replicates.shape[0], device=replicates.device, dtype=replicates.dtype
        )
    return replicates.var(dim=1, unbiased=True) / K


def dense2sparse(
    dense_tensor: Tensor,
    row_lengths: Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    assert dense_tensor.dim() == 2
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows
    assert (row_lengths <= num_cols).all()
    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    return dense_tensor[mask].to(dtype)


def sparse2dense(
    sparse_tensor: Tensor,
    row_lengths: Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    assert sparse_tensor.dim() == 1
    assert row_lengths.sum() == len(sparse_tensor)
    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)
    return dense


def sparse2dense_blocks(
    sparse_tensor: Tensor,
    dims: Tensor,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    assert sparse_tensor.dim() == 1
    dims = dims.to(torch.long)
    block_sizes = dims * seq_len
    assert block_sizes.sum() == len(sparse_tensor)
    num_datasets = len(dims)
    max_dim = int(dims.max()) if num_datasets else 0
    dense = torch.zeros(
        num_datasets,
        seq_len,
        max_dim,
        dtype=dtype,
        device=sparse_tensor.device,
    )
    offset = 0
    for i, (dim, block) in enumerate(zip(dims.tolist(), block_sizes.tolist())):
        dense[i, :, :dim] = sparse_tensor[offset : offset + block].view(seq_len, dim)
        offset += block
    return dense


def torch_nanstd(
    input: Tensor, dim=None, keepdim=False, ddof=0, *, dtype=None
) -> Tensor:
    if dtype is not None:
        input = input.to(dtype)
    reduce_dim = tuple(range(input.dim())) if dim is None else dim
    mask = ~torch.isnan(input)
    count = mask.sum(dim=reduce_dim, keepdim=True)
    total = torch.nansum(input, dim=reduce_dim, keepdim=True)
    mean = total / count.clamp(min=1)
    centered = torch.where(mask, input - mean, torch.zeros((), device=input.device))
    sq_sum = (centered * centered).sum(dim=reduce_dim, keepdim=True)
    var = sq_sum / (count - ddof).clamp(min=1)
    std = var.sqrt()
    if not keepdim:
        std = std.reshape(()) if dim is None else std.squeeze(dim)
    return std.to(torch.float)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std
    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clip(min=1e-6)
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)
    return input.clamp(min=lower, max=upper)
