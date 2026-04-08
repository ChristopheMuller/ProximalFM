from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

Activation = Callable[[Tensor], Tensor]


@dataclass
class NetworkSpec:
    num_layers: int = 3
    hidden_size: int = 32
    activation: Activation = torch.tanh
    weight_scale: float = 0.75
    bias_scale: float = 0.1


@dataclass
class ProximalPriorConfig:

    type: str = "nonlinear"

    d_u: int = 4
    d_x: int = 8
    d_z: int = 6
    d_w: int = 6

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
    normalize_episode: bool = (
        False  ## Important to be False: information leakage otherwise!
    )

    confounding_strength: float = 1.0 # for high confounding prior only


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
