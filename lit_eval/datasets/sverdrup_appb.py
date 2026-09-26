"""Appendix B of Sverdrup and Cui (2023).

The paper specifies the observational law in this order::

    X ~ N(0, 0.25 I_5)
    A | X ~ Bernoulli(1 / (1 + exp(0.125 (X_1 + X_2))))
    (Z, W, U) | A, X ~ N(mu(A, X), Sigma)
    Y | W, U, A, X ~ N(m(W, U, A, X), 0.25^2)

and declares ``tau*(X) = exp(X_1) - 3 X_2``. Sampling the proxies/latent again
under an intervention on A would instead create an extra arm contrast and
contradict that declared target. We therefore use the paper's factorization to
draw each observed/Monte-Carlo world, then construct paired potential outcomes
by holding its ``U``, ``W`` and outcome-noise draw fixed while toggling only the
direct treatment term. Consequently ``Y1 - Y0 == tau*(X)`` for every draw.

The three final X columns are deliberately irrelevant, exactly as in the
paper. ``Z`` appears in the conditioning set for Y but not in its mean.
"""

from __future__ import annotations

import torch

from lit_eval.base import Dims, EpisodeSpec, ProximalDataset, RawEpisode
from lit_eval._replicates import Replicates
from lit_eval.registry import register_dataset

Tensor = torch.Tensor

_X_COEF = torch.tensor([0.25, 0.25, 0.0, 0.0, 0.0])
_A_COEF = torch.tensor([0.125, 0.125, 0.0, 0.0, 0.0])
_ZWU_COV = torch.tensor([[1.0, 0.25, 0.5], [0.25, 1.0, 0.5], [0.5, 0.5, 1.0]])
_ZWU_CHOL = torch.linalg.cholesky(_ZWU_COV)


def cate(x: Tensor) -> Tensor:
    """The analytic CATE stated in Appendix B."""
    return torch.exp(x[..., 0]) - 3.0 * x[..., 1]


def treatment_probability(x: Tensor) -> Tensor:
    """Appendix-B treatment propensity ``P(A=1 | X=x)``."""
    return torch.sigmoid(-(x @ _A_COEF.to(dtype=x.dtype, device=x.device)))


def expected_w_given_u_x(u: Tensor, x: Tensor) -> Tensor:
    """The conditional expectation printed below Appendix B's outcome law."""
    x_term = x @ _X_COEF.to(dtype=x.dtype, device=x.device)
    return 0.25 + x_term + 0.5 * (u.reshape(-1) - 0.25 - x_term)


@register_dataset
class SverdrupAppB(ProximalDataset):
    """The five-covariate illustrative proximal CATE mechanism."""

    name = "sverdrup_appB"
    citation = (
        "Sverdrup and Cui (2023), Proximal Causal Learning of Conditional "
        "Average Treatment Effects, Appendix B"
    )
    max_dims = Dims(d_U=1, d_X=5, d_W=1, d_Z=1)
    overlap_eps = 0.0
    extra_observation_noise = 0.0

    def sample_episode(self, spec: EpisodeSpec, rep: Replicates) -> RawEpisode:
        # Logical covariate rows are fixed across the K query replicates.
        x_obs = 0.5 * self.randn(rep.T, 5)
        x = rep.expand(x_obs)

        propensity = treatment_probability(x).unsqueeze(1)
        a = (self.rand(rep.N, 1) < propensity).to(self.dtype)

        x_term = (x @ _X_COEF).unsqueeze(1)
        means = torch.cat(
            [
                0.25 + 0.25 * a + x_term,
                0.25 + 0.125 * a + x_term,
                0.25 + 0.25 * a + x_term,
            ],
            dim=1,
        )
        zwu = means + self.randn(rep.N, 3) @ _ZWU_CHOL.T
        z, w, u = (zwu[:, i : i + 1] for i in range(3))

        ew = expected_w_given_u_x(u, x).unsqueeze(1)
        eps_y = 0.25 * self.randn(rep.N, 1)
        y0 = 2.0 + x_term + 2.0 * ew + 2.0 * w + eps_y
        tau = cate(x).unsqueeze(1)
        y1 = y0 + tau

        tau_query = cate(x_obs[rep.n_ctx :]) if rep.replicated else None
        return RawEpisode(
            U=u,
            X=x,
            W=w,
            Z=z,
            A=a,
            A_probs=propensity,
            Y0=y0,
            Y1=y1,
            d_U=1,
            d_X=5,
            d_W=1,
            d_Z=1,
            tau_star_exact=tau_query,
            sampled_hp={
                "mechanism_version": 1.0,
                "x_variance": 0.25,
                "outcome_sd": 0.25,
            },
        )


__all__ = ["SverdrupAppB", "cate", "expected_w_given_u_x", "treatment_probability"]
