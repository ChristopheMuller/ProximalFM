"""A fully linear-Gaussian proximal DGP -- the reference adapter.

This is **not** a literature dataset. It exists so the package's plumbing can be
built, tested and reasoned about before any external code is vendored, and it is
deliberately the simplest thing that still exercises every piece: a latent that
confounds both arms, two proxies that load on it, a treatment that reads one
proxy, and an effect that varies with both ``X`` and the latent.

The structure (``->`` reads "is an input of")::

    X   ~ N(0, I)
    U   = a_u * X B_xu + s_u * eps_U          (redrawn per replicate)
    W   = X B_wx + U B_wu + s_w * eps_W       X, U -> W
    Z   = X B_zx + U B_zu + s_z * eps_Z       X, U -> Z          (leaf)
    A   ~ Bern(sigmoid(std(X g_x + U g_u + Z g_z)))              (leaf)
    Y(a)= b0 + X b_x + U b_u + W b_w + a * tau + s_y * eps_Y
    tau = t0 + X t_x + U t_u

so ``Z`` never enters ``Y``, ``W`` never enters ``Z`` or ``A``, and ``A`` never
enters ``Y`` other than through ``tau`` -- the proximal exclusions, enforced
structurally rather than by rejection, exactly as in the SCM prior.

Why the closed form matters
---------------------------

Everything is linear and ``U`` is Gaussian given ``X``, so

    tau*(x) = E[Y1 - Y0 | X = x] = t0 + x t_x + a_u * (x B_xu) t_u

exactly. ``W`` and the shared ``eps_Y`` cancel in the difference; the latent does
not, so the per-replicate spread is real and ``S2`` is non-degenerate -- which is
what makes this a usable test of the replicate machinery rather than a
degenerate one. The exact value is emitted as ``tau_star_exact`` and stored in
the sidecar, so a test can assert that ``MU1 - MU0`` converges to it at the
Monte Carlo rate. That check is the only direct evidence that the replicate
layout estimates the estimand it claims to.

Nothing here is standardized on the fly for the same reason: a statistic taken
over the replicate block would be a random quantity, and the closed form would
stop being closed. (The propensity's logit *is* standardized, but ``A`` is a
leaf and cannot move ``tau*``.)
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from lit_eval._replicates import Replicates
from lit_eval.base import Dims, EpisodeSpec, ProximalDataset, RawEpisode
from lit_eval.registry import register_dataset

Tensor = torch.Tensor


@register_dataset
class DummyLinearProximal(ProximalDataset):
    """Linear-Gaussian proximal DGP with a closed-form ``tau*``."""

    name = "dummy_linear"
    citation = "placeholder DGP (not from the literature)"
    supports_replicates = True

    @classmethod
    def option_defaults(cls) -> Dict[str, Any]:
        return {
            # Per-episode dimension ranges (inclusive). d_W and d_Z are drawn
            # at or above d_U so categorical completeness can hold.
            "min_dim_U": 1,
            "max_dim_U": 3,
            "min_dim_X": 2,
            "max_dim_X": 8,
            "min_dim_W": 2,
            "max_dim_W": 6,
            "min_dim_Z": 2,
            "max_dim_Z": 6,
            # How much of the latent is explained by X. 0 = U fully orthogonal
            # to X (the hardest, and the case where backdoor on X is maximally
            # biased); 1 = U nearly a function of X (nearly no hidden
            # confounding left). Drawn per episode in [lo, hi].
            "u_signal_lo": 0.2,
            "u_signal_hi": 0.6,
            # Loading of U on the treatment and on the outcome -- the actual
            # confounding strength. Bounded away from zero on purpose.
            "confounding_lo": 0.6,
            "confounding_hi": 1.4,
            # Noise scales.
            "sigma_u": 1.0,
            "sigma_w": 0.5,
            "sigma_z": 0.5,
            "sigma_y_lo": 0.3,
            "sigma_y_hi": 0.8,
            # Effect size: |t0| and the heterogeneous part.
            "tau_base_lo": 0.3,
            "tau_base_hi": 1.0,
            "tau_het": 0.5,
        }

    @property
    def max_dims(self) -> Dims:
        o = self.options
        return Dims(
            d_U=int(o["max_dim_U"]),
            d_X=int(o["max_dim_X"]),
            d_W=int(o["max_dim_W"]),
            d_Z=int(o["max_dim_Z"]),
        )

    # ------------------------------------------------------------------

    def sample_episode(self, spec: EpisodeSpec, rep: Replicates) -> RawEpisode:
        o = self.options
        d_U = self.randint(int(o["min_dim_U"]), int(o["max_dim_U"]))
        d_X = self.randint(int(o["min_dim_X"]), int(o["max_dim_X"]))
        d_W = self.randint(max(int(o["min_dim_W"]), d_U), int(o["max_dim_W"]))
        d_Z = self.randint(max(int(o["min_dim_Z"]), d_U), int(o["max_dim_Z"]))

        a_u = self.uniform_scalar(float(o["u_signal_lo"]), float(o["u_signal_hi"]))
        cs = self.uniform_scalar(float(o["confounding_lo"]), float(o["confounding_hi"]))
        sigma_y = self.uniform_scalar(float(o["sigma_y_lo"]), float(o["sigma_y_hi"]))
        t0 = self.uniform_scalar(float(o["tau_base_lo"]), float(o["tau_base_hi"]))
        t0 = t0 if self.rand(1).item() < 0.5 else -t0

        # --- structural coefficients, all explicit so tau* stays closed-form ---
        B_xu = self._coefs(d_X, d_U)
        B_wx, B_wu = self._coefs(d_X, d_W), self._coefs(d_U, d_W)
        B_zx, B_zu = self._coefs(d_X, d_Z), self._coefs(d_U, d_Z)
        g_x, g_u, g_z = self._coefs(d_X, 1), self._coefs(d_U, 1), self._coefs(d_Z, 1)
        b_x, b_w = self._coefs(d_X, 1), self._coefs(d_W, 1)
        b_u = self._coefs(d_U, 1)
        t_x = float(o["tau_het"]) * self._coefs(d_X, 1)
        t_u = float(o["tau_het"]) * self._coefs(d_U, 1)

        # --- X on the observed rows, then held fixed across replicates -------
        X_obs = self.randn(rep.T, d_X)
        X = rep.expand(X_obs)

        # U is redrawn per replicate row: that is what marginalizes it out of
        # the target. a_u ties it to X; the rest is variance orthogonal to X,
        # which is what makes backdoor-on-X insufficient.
        U = a_u * (X @ B_xu) + float(o["sigma_u"]) * self.randn(rep.N, d_U)

        W = X @ B_wx + cs * (U @ B_wu) + float(o["sigma_w"]) * self.randn(rep.N, d_W)

        # Z and the propensity are leaves w.r.t. the target -> observed rows
        # only. Both need the *observed* realization of U.
        U_obs = rep.observed(U)
        Z = (
            X_obs @ B_zx
            + cs * (U_obs @ B_zu)
            + float(o["sigma_z"]) * self.randn(rep.T, d_Z)
        )
        A_probs = self.propensity(X_obs @ g_x + cs * (U_obs @ g_u) + 0.5 * (Z @ g_z))

        # --- outcomes --------------------------------------------------------
        tau = t0 + X @ t_x + cs * (U @ t_u)
        eps = sigma_y * self.randn(rep.N, 1)
        Y0 = X @ b_x + cs * (U @ b_u) + W @ b_w + eps
        Y1 = Y0 + tau

        # tau*(x) = t0 + x t_x + a_u * cs * (x B_xu) t_u  -- W and eps cancel in
        # the difference, and E[U | X = x] = a_u * x B_xu.
        X_query = X_obs[rep.n_ctx :]
        tau_star = t0 + X_query @ t_x + a_u * cs * ((X_query @ B_xu) @ t_u)

        return RawEpisode(
            X=X,
            W=W,
            Z=Z,
            A_probs=A_probs,
            Y0=Y0,
            Y1=Y1,
            U=U,
            d_U=d_U,
            d_X=d_X,
            d_W=d_W,
            d_Z=d_Z,
            tau_star_exact=tau_star.reshape(-1) if rep.replicated else None,
            sampled_hp={
                "u_signal": a_u,
                "confounding_strength": cs,
                "sigma_y": sigma_y,
                "tau_base": t0,
            },
        )

    # ------------------------------------------------------------------

    def _coefs(self, in_dim: int, out_dim: int) -> Tensor:
        """Signed coefficients scaled by ``1/sqrt(in_dim)``.

        The scaling keeps every linear block's output variance roughly constant
        as the dimensions vary, so ``d_X`` does not silently double as an
        effect-size knob across episodes of one build.
        """
        return self.signed_coefs(in_dim, out_dim) / math.sqrt(in_dim)
