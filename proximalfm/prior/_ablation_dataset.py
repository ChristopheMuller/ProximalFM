"""A small proximal SCM for architecture and loss ablations."""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ._dataset import SCMPrior, _mc_reduce


class AblationSCMPrior(SCMPrior):
    """Standard-normal X and U, with linear proxy blocks."""

    def __init__(
        self,
        *,
        min_dim_U: int = 1,
        max_dim_U: int = 1,
        min_dim_X: int = 1,
        max_dim_X: int = 1,
        min_dim_W: int = 1,
        max_dim_W: int = 4,
        min_dim_Z: int = 1,
        max_dim_Z: int = 4,
        prior_type: str = "ablation_scm",
        **kwargs,
    ):
        if not 1 <= min_dim_X <= max_dim_X:
            raise ValueError("ablation_scm requires 1 <= min_dim_X <= max_dim_X")
        if not 1 <= min_dim_U <= max_dim_U:
            raise ValueError("ablation_scm requires 1 <= min_dim_U <= max_dim_U")
        if not (1 <= min_dim_W <= max_dim_W and 1 <= min_dim_Z <= max_dim_Z):
            raise ValueError("ablation_scm requires positive, ordered proxy bounds")
        if max_dim_U > min(max_dim_W, max_dim_Z):
            raise ValueError(
                "ablation_scm requires max_dim_W and max_dim_Z >= max_dim_U"
            )
        max_dims = (max_dim_U, max_dim_X, max_dim_W, max_dim_Z)
        if prior_type == "ablation_scm_slope" and max_dims != (1, 1, 1, 1):
            raise ValueError("ablation_scm_slope requires scalar U, X, W and Z")
        super().__init__(
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            prior_type=prior_type,
            **kwargs,
        )

    def hp_sampling(self) -> dict[str, Any]:
        # Coefficients are sampled once per episode, independently of MLP settings.
        return {}

    @staticmethod
    def _per_row_estimate(params: dict[str, Any]) -> int:
        # The replicate block carries only latent values, outcome noise and outcomes.
        return 12 + 2 * params["dim_U"]

    @torch.no_grad()
    def generate_dataset(self, params: dict[str, Any]) -> tuple[Tensor, ...]:
        """Generate one episode in the shared SCM format."""
        device = self.device
        rows = params["seq_len"]
        n_ctx = 0 if params["mc_targets_on_context"] else params["train_size"]
        n_targets = rows - n_ctx
        k_requested = int(params["mc_cate"])
        if k_requested < 1:
            raise ValueError("mc_cate must be >= 1")
        k = self._cap_mc_samples(k_requested, n_ctx, n_targets, params)
        d_u, d_w, d_z = params["dim_U"], params["dim_W"], params["dim_Z"]
        d_x = params["dim_X"]
        if min(d_w, d_z) < d_u:
            raise ValueError("ablation_scm requires dim_W and dim_Z >= dim_U")

        if self.prior_type == "ablation_scm_slope":
            # X is independent of treatment, so the arm slopes reveal beta.
            # U still confounds the outcome level and varies individual effects.
            c0 = cx = cxu = b0 = bx = tau0 = 0.0
            cu = bu = 0.5
            gamma = 1.0
            beta_sign = 2 * torch.randint(0, 2, (), device=device) - 1
            beta = beta_sign * (0.75 + 0.5 * torch.rand((), device=device))
            loading_w = loading_z = torch.ones((1, 1), device=device)
            confounding_direction = effect_direction = torch.ones(1, device=device)
            cz = torch.full((1,), 0.25, device=device)
        else:
            c0, cx, cu, cxu, b0, bx, bu, tau0, beta, gamma = torch.rand(
                10, device=device
            ).unbind()
            c0 = 2 * c0 - 1
            cx, b0, bx, tau0 = cx - 0.5, b0 - 0.5, bx - 0.5, tau0 - 0.5
            cu, cxu, bu = cu + 0.5, cxu + 0.75, bu + 0.75
            beta_sign, gamma_sign, bu_sign, cxu_sign = (
                2 * torch.randint(0, 2, (4,), device=device) - 1
            ).unbind()
            beta = beta_sign * (0.75 + 0.5 * beta)
            gamma = gamma_sign * (0.6 + 0.4 * gamma)
            # Vary the signs of level and shape confounding across episodes.
            bu, cxu = bu_sign * bu, cxu_sign * cxu
            # Unit loading strength in every latent direction at every proxy width.
            loading_w = torch.linalg.qr(
                torch.randn(d_w, d_u, device=device), mode="reduced"
            ).Q
            loading_z = torch.linalg.qr(
                torch.randn(d_z, d_u, device=device), mode="reduced"
            ).Q
            cz = 0.75 * F.normalize(torch.randn(d_z, device=device), dim=0)
            confounding_direction = F.normalize(torch.randn(d_u, device=device), dim=0)
            effect_direction = F.normalize(torch.randn(d_u, device=device), dim=0)

        x = torch.randn(rows, d_x, device=device)
        # Unit directions keep the covariate signal strength fixed across widths.
        if d_x == 1:
            x_directions = torch.ones(1, 4, device=device)
        else:
            x_directions = F.normalize(torch.randn(d_x, 4, device=device), dim=0)
        x_treatment, x_confounding, x_baseline, x_effect = (x @ x_directions).unbind(1)
        u = torch.randn(rows, d_u, device=device)
        u_confounding = u @ confounding_direction
        u_effect = u @ effect_direction
        w = u @ loading_w.T + torch.randn(rows, d_w, device=device)
        z = u @ loading_z.T + torch.randn(rows, d_z, device=device)
        logits = (
            c0
            + cx * x_treatment
            + cu * u_confounding
            + z @ cz
            + cxu * x_confounding * u_confounding
        )
        a_probs = 0.1 + 0.8 * logits.sigmoid()
        a = torch.bernoulli(a_probs).long()
        mu0 = b0 + bx * x_baseline
        cate = tau0 + beta * x_effect
        y0 = mu0 + bu * u_confounding + 0.35 * torch.randn_like(mu0)
        y1 = y0 + cate + gamma * u_effect
        y_obs = torch.where(a == 1, y1, y0)

        # Separate, draw-major streams keep every target prefix fixed as K changes.
        seed = int(torch.randint(0, 2**63 - 1, (), device="cpu").item())
        latent_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        shape = (k, n_targets)
        latent = np.random.default_rng(latent_seed).standard_normal(
            (*shape, d_u), dtype=np.float32
        )
        noise = np.random.default_rng(noise_seed).standard_normal(
            shape, dtype=np.float32
        )
        u_mc = torch.from_numpy(latent.transpose(1, 0, 2)).to(
            device=device, dtype=x.dtype
        )
        noise_mc = torch.from_numpy(noise.T).to(device=device, dtype=x.dtype)
        # A row-wise reduction avoids GEMM kernels changing rounding with K.
        confounding_mc = (u_mc * confounding_direction).sum(dim=-1)
        effect_mc = (u_mc * effect_direction).sum(dim=-1)
        y0_mc = mu0[n_ctx:, None] + bu * confounding_mc + 0.35 * noise_mc
        cate_mc = cate[n_ctx:, None] + gamma * effect_mc
        target0, target1, s2 = _mc_reduce(
            y0_mc.flatten(),
            (y0_mc + cate_mc).flatten(),
            cate_mc.flatten(),
            n_targets,
            k,
            pool_rows=False,
        )
        missing = x.new_full((n_ctx,), float("nan"))
        target0 = torch.cat((missing, target0))
        target1 = torch.cat((missing, target1))
        s2 = torch.cat((missing, s2))

        diagnostics = None
        if params.get("collect_mechanism_diagnostics", False):
            coefficients = dict(
                c0=c0,
                cx=cx,
                cu=cu,
                cxu=cxu,
                b0=b0,
                bx=bx,
                bu=bu,
                tau0=tau0,
                beta=beta,
                gamma=gamma,
            )
            diagnostics = {
                "params": {name: float(value) for name, value in coefficients.items()},
                "mechanisms": {
                    "loading_W": loading_w,
                    "loading_Z": loading_z,
                    "c_Z": cz,
                    "confounding_direction": confounding_direction,
                    "effect_direction": effect_direction,
                    "x_directions": x_directions,
                },
            }

        d_u_active, d_x_active, d_w_active, d_z_active, k_eff = torch.tensor(
            [d_u, d_x, d_w, d_z, k], dtype=torch.long, device=device
        ).unbind()
        return (
            F.pad(u, (0, params["max_dim_U"] - d_u)),
            F.pad(x, (0, params["max_dim_X"] - d_x)),
            F.pad(w, (0, params["max_dim_W"] - d_w)),
            F.pad(z, (0, params["max_dim_Z"] - d_z)),
            a,
            a_probs,
            y_obs,
            y0,
            y1,
            target0,
            target1,
            d_u_active,
            d_x_active,
            d_w_active,
            d_z_active,
            s2,
            k_eff,
            diagnostics,
        )
