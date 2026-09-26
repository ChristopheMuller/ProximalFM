"""Wu et al. (2024), Appendix H.2 high-dimensional synthetic design.

The source mechanism has continuous treatment

    A_cont = Lambda(3 X' beta_x + 3 Z' beta_z) + 0.25 U_w,

where ``Lambda(t) = 0.8 sigmoid(t) + 0.1``. ProximalFM requires a binary
treatment, so a literal transcription is impossible. This adapter targets the
paper's endpoint contrast ``a=1`` versus ``a=0`` and uses the following explicit
binary adaptation:

    P(A=1 | X,Z,U_w) = Lambda(3 X' beta_x + 3 Z' beta_z + 0.25 U_w).

This moves the additive confounding term inside the paper's own link, retaining
its sign and coefficient *before* the nonlinear link while guaranteeing overlap
in ``[0.1, 0.9]``. It is a new binary treatment law, not an algebraically
equivalent version of the paper's continuous law. All other equations are
transcribed directly. Results from this binary benchmark must not be presented
as the paper's continuous-dose cMSE experiment.

For the printed outcome equation,

    Y(a) = a^2 + 1.2a + 1.2(X' beta_x + W' beta_w) + a X_1 + 0.25 U_z,

the paired 1-vs-0 effect is exactly ``tau*(x) = 2.2 + x_1``. It contains no
latent variation, so paired Monte Carlo target variance ``S2`` is zero.
"""

from __future__ import annotations

import torch

from lit_eval._replicates import Replicates
from lit_eval.base import Dims, EpisodeSpec, ProximalDataset, RawEpisode
from lit_eval.registry import register_dataset

Tensor = torch.Tensor

DEFAULT_DIM_X = 100
DEFAULT_DIM_Z = 10
DEFAULT_DIM_W = 10


def quadratic_decay(dim: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """The paper's 1-based quadratically decaying coefficients, ``j^-2``."""
    j = torch.arange(1, dim + 1, dtype=dtype)
    return j.pow(-2)


def truncated_logistic(t: Tensor) -> Tensor:
    """``Lambda(t) = 0.8 sigmoid(t) + 0.1`` from Appendix H.2."""
    return 0.8 * torch.sigmoid(t) + 0.1


def cate(x: Tensor, a0: float = 0.0, a1: float = 1.0) -> Tensor:
    """Analytic conditional effect for an arbitrary pair of treatment arms."""
    delta = float(a1) - float(a0)
    theta_delta = float(a1) ** 2 - float(a0) ** 2 + 1.2 * delta
    return theta_delta + delta * x[..., 0]


def x_covariance(dim: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Appendix H.2 covariance at an arbitrary covariate dimension."""
    covariance = torch.eye(dim, dtype=dtype)
    idx = torch.arange(dim - 1)
    covariance[idx, idx + 1] = 0.5
    covariance[idx + 1, idx] = 0.5
    return covariance


@register_dataset
class WuAppH2(ProximalDataset):
    """Binary 0-vs-1 adaptation of Wu et al.'s high-dimensional DGP."""

    name = "wu_appH2"
    citation = (
        "Wu et al. (2024), Doubly Robust Proximal Causal Learning for "
        "Continuous Treatments, Appendix H.2"
    )
    overlap_eps = 0.0
    extra_observation_noise = 0.0

    @classmethod
    def option_defaults(cls):
        return {
            "dim_x": DEFAULT_DIM_X,
            "dim_z": DEFAULT_DIM_Z,
            "dim_w": DEFAULT_DIM_W,
        }

    def __init__(self, **options):
        super().__init__(**options)
        for name in ("dim_x", "dim_z", "dim_w"):
            value = self.options[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"wu_appH2: {name} must be an integer")
        self.dim_x = self.options["dim_x"]
        self.dim_z = self.options["dim_z"]
        self.dim_w = self.options["dim_w"]
        if self.dim_x < 1:
            raise ValueError("wu_appH2: dim_x must be >= 1")
        if self.dim_z < 2 or self.dim_w < 2:
            raise ValueError("wu_appH2: dim_z and dim_w must both be >= 2")
        self.options.update(
            {"dim_x": self.dim_x, "dim_z": self.dim_z, "dim_w": self.dim_w}
        )
        self._x_chol = torch.linalg.cholesky(x_covariance(self.dim_x))
        self._beta_x = quadratic_decay(self.dim_x)
        self._beta_z = quadratic_decay(self.dim_z)
        self._beta_w = quadratic_decay(self.dim_w)

    @property
    def max_dims(self) -> Dims:
        return Dims(d_U=2, d_X=self.dim_x, d_W=self.dim_w, d_Z=self.dim_z)

    def sample_episode(self, spec: EpisodeSpec, rep: Replicates) -> RawEpisode:
        # X is fixed for each logical query row across its Monte Carlo worlds.
        x_obs = self.randn(rep.T, self.dim_x) @ self._x_chol.T
        x = rep.expand(x_obs)

        eps = self.randn(rep.N, 3)
        uz = eps[:, 0:1] + eps[:, 2:3]
        uw = eps[:, 1:2] + eps[:, 2:3]

        nu_z = 2.0 * self.rand(rep.N, self.dim_z) - 1.0
        nu_w = 2.0 * self.rand(rep.N, self.dim_w) - 1.0
        z = nu_z + 0.25 * uz
        w = nu_w + 0.25 * uw

        assignment_index = (
            3.0 * (x @ self._beta_x) + 3.0 * (z @ self._beta_z) + 0.25 * uw[:, 0]
        )
        propensity = truncated_logistic(assignment_index).unsqueeze(1)

        baseline = 1.2 * (x @ self._beta_x + w @ self._beta_w) + 0.25 * uz[:, 0]
        y0 = baseline.unsqueeze(1)
        y1 = (2.2 + baseline + x[:, 0]).unsqueeze(1)

        tau_query = cate(x_obs[rep.n_ctx :]) if rep.replicated else None
        return RawEpisode(
            U=torch.cat([uz, uw], dim=1),
            X=x,
            W=w,
            Z=z,
            A_probs=propensity,
            Y0=y0,
            Y1=y1,
            d_U=2,
            d_X=self.dim_x,
            d_W=self.dim_w,
            d_Z=self.dim_z,
            tau_star_exact=tau_query,
            sampled_hp={
                "mechanism_version": 1.0,
                "binary_arm_low": 0.0,
                "binary_arm_high": 1.0,
                "propensity_min": float(propensity.min()),
                "propensity_max": float(propensity.max()),
            },
        )


__all__ = [
    "DEFAULT_DIM_W",
    "DEFAULT_DIM_X",
    "DEFAULT_DIM_Z",
    "WuAppH2",
    "cate",
    "quadratic_decay",
    "truncated_logistic",
    "x_covariance",
]
