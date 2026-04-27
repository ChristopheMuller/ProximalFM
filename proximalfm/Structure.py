from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

Activation = Callable[[Tensor], Tensor]


@dataclass
class NetworkSpec:
    num_layers: int = 3 # for prior with fixed depth
    min_layers: int = 2 # for prior with variable depth
    max_layers: int = 4 # for prior with variable depth
    hidden_size: int = 32
    activation: Activation = torch.tanh
    weight_scale: float = 0.75
    bias_scale: float = 0.1


@dataclass
class ProximalPriorConfig:
    type: str = "nonlinear"

    d_u: int = 3
    d_x: int = 5
    d_z: int = 4
    d_w: int = 4

    x_spec: NetworkSpec = field(default_factory=NetworkSpec)
    z_spec: NetworkSpec = field(default_factory=NetworkSpec)
    w_spec: NetworkSpec = field(default_factory=NetworkSpec)
    u_spec: NetworkSpec = field(default_factory=NetworkSpec)
    treatment_spec: NetworkSpec = field(
        default_factory=lambda: NetworkSpec(hidden_size=24)
    )
    outcome_spec: NetworkSpec = field(
        default_factory=lambda: NetworkSpec(hidden_size=24)
    )

    noise_scale_x: float = 0.05
    noise_scale_z: float = 0.05
    noise_scale_w: float = 0.05
    noise_scale_y: float = 0.05

    temperature: float = 0.7
    min_propensity: float = 0.05
    max_propensity: float = 0.95

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    confounding_strength: float = 1.0


@dataclass
class ProximalEpisode:
    X: Tensor
    Z: Tensor
    W: Tensor
    A: Tensor
    Y: Tensor
    Y0: Tensor
    Y1: Tensor
    U: Tensor
    propensity: Tensor


@dataclass
class ModelConfig:
    type: str = "gmm"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    embed_dim: int = 64
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    feature_group_size_x: int = 5
    feature_group_size_w: int = 4
    feature_group_size_z: int = 4
    n_cls_cols: int = 4
    n_cls_rows: int = 64
    n_gmm_components: int = 1
