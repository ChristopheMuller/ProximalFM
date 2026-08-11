"""The Demand benchmark -- the proximal literature's standard synthetic design.

Structural equations, verbatim from the benchmark (Xu et al., 2021; used as the
headline proximal benchmark by the NMMR paper), with ``eps_i ~ N(0, 1)``::

    U   ~ Uniform(0, 10)                                    demand
    Z   = [2 sin(2 pi U / 10) + eps_1, 2 cos(2 pi U / 10) + eps_2]   fuel cost
    W   = 7 g(U) + 45 + eps_3                               web page views
    A   = 35 + (Z_1 + 3) g(U) + Z_2 + eps_4                 price
    Y   = A * min(exp((W - A)/10), 5) - 5 g(U) + eps_5      sales

    g(u) = 2 ( (u-5)^4 / 600 + exp(-4 (u-5)^2) + u/10 - 2 )

The DAG is exactly this project's proximal skeleton, which is why the design
transfers with no structural surgery: ``U`` confounds both arms, ``Z`` reaches
``Y`` only through the price (treatment-inducing proxy), and ``W`` reaches the
price not at all (outcome-inducing proxy).

Two adaptations, both forced by the format and both consequential enough to
state up front.

**1. The price is continuous; this project's treatment is binary.** The
benchmark's estimand is the dose-response curve ``a -> E[Y^a]``. Here two price
levels are fixed (``price_low``/``price_high``, defaults 25 and 35 -- inside the
observed price support, roughly its 25th to 95th percentile) and the potential
outcomes are ``Y(price_low)`` and ``Y(price_high)`` computed from the *same*
latent, proxy and noise draw, so the target is one point of that curve:

    tau*(x) = E[Y^{price_high} - Y^{price_low}]  ~= -19.2 at the defaults.

The observed arm is drawn from the published price equation rather than
replacing it: ``A_cont`` is generated exactly as above, and

    P(A = 1 | .) = sigmoid( (A_cont - median(A_cont)) / (temp * sd(A_cont)) + shift )

so ``U`` and ``Z`` drive assignment with their published coefficients and the
confounding survives intact -- at the default ``temp = 1.0`` the naive
difference in means reads about -9.9 against a true effect of -19.2, i.e. the
benchmark stays a genuinely confounded problem, while propensities stay inside
roughly [0.02, 0.93] so positivity is not an issue. A hard threshold
(``temp -> 0``) would reproduce the deterministic binarization and destroy
overlap; that is why the temperature exists. Both the threshold and the scale
are computed on the ``T`` *observed* rows, so they are frozen statistics and
``mc_cate`` cannot move the estimand through them.

``assignment_shift`` (default ``0.0``) is a pure location move of that logit:
it sets how *imbalanced* the arms are without touching how strongly ``U`` and
``Z`` select into treatment. Measured treated fractions: 0.48 at 0, 0.66 at +1,
0.81 at +2, 0.29 at -1, 0.14 at -2. Past about ``|shift| = 2`` the far tail of
the propensity starts hitting the ``overlap_eps`` clamp (0.02), so positivity
there is imposed by the clamp rather than earned by the design -- read a heavily
shifted build with that in mind. It is the knob for asking whether a
method degrades when one arm is rare -- a regime the published design does not
cover, since it has no binary arm at all -- and it is deliberately orthogonal to
``temp``, which sets *selection strength* instead. Both slopes matter and they
fail differently: shift starves an arm of data, temp starves the overlap. The
shift is applied after every RNG draw in the episode, so ``Y0``/``Y1`` are
byte-identical across shifts and a sweep varies exactly one thing.

**2. The design has no observed covariates.** ``X`` is therefore emitted as
``n_covariates`` columns of independent noise that enter nothing -- the least
harmful encoding of "no covariates", since a constant column has zero variance
and breaks context standardisation downstream, and ``n_covariates = 0`` is
refused outright (the on-disk format has no zero-width block). At
``x_effect_scale = 0`` those columns are genuinely inert: the law of ``U``,
``W``, ``Z``, the price and both potential outcomes is identical at any
``n_covariates``, and only the RNG stream offset moves. The consequence is that
**tau*(x) is constant**: Demand is an ATE benchmark, not a CATE one. In
particular ``prior_eval.verify``'s group 4 will fail on a faithful build, and it
is right to: it reports ``s / sd(tau*)``, and with a constant target the
denominator is nothing but Monte Carlo noise. Read ``s`` against
``sd(y_context)`` on this set instead.

``x_effect_scale > 0`` (default ``0.0``, i.e. off) is the opt-in escape from
that: it adds ``x_effect_scale * (X @ c)`` to the high-price arm only, leaving
the published law of ``Y(price_low)``, of both proxies and of the price
untouched, and making ``tau*(x)`` genuinely vary with ``x``. **This is a
deviation from the published DGP** and must be reported as such wherever a
number built with it appears; it exists because a constant-CATE set exercises
nothing of a CATE model's conditional structure.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from lit_eval._replicates import Replicates
from lit_eval.base import Dims, EpisodeSpec, ProximalDataset, RawEpisode
from lit_eval.registry import register_dataset

Tensor = torch.Tensor


def g(u: Tensor) -> Tensor:
    """``g(u) = 2 ( (u-5)^4/600 + exp(-4 (u-5)^2) + u/10 - 2 )``.

    Ranges over about ``[-3.37, 0.08]`` on ``U ~ Uniform(0, 10)``, with mean
    ``-2.40`` -- the numbers the default price arms were chosen against.
    """
    c = u - 5.0
    return 2.0 * (c**4 / 600.0 + torch.exp(-4.0 * c**2) + u / 10.0 - 2.0)


def potential_outcome(price: float, W: Tensor, gU: Tensor) -> Tensor:
    """``Y^a`` without its noise term: ``a * min(exp((W-a)/10), 5) - 5 g(U)``.

    The noise ``eps_5`` is shared across the two arms (one draw per replicate
    row), so it cancels in ``Y1 - Y0`` -- but ``W`` and ``U`` do not, because of
    the ``exp``/``min`` link. The Monte Carlo average is doing real work here.
    """
    return price * torch.clamp(torch.exp((W - price) / 10.0), max=5.0) - 5.0 * gU


@register_dataset
class Demand(ProximalDataset):
    """The Demand benchmark, binarized at two price levels."""

    name = "demand"
    citation = (
        "Demand design (Xu et al., 2021); standard proximal benchmark, "
        "binarized at two price levels -- see module docstring"
    )
    supports_replicates = True

    # The design specifies its own eps_5; no extra measurement noise on top.
    extra_observation_noise = 0.0

    @classmethod
    def option_defaults(cls) -> Dict[str, Any]:
        return {
            "price_low": 25.0,
            "price_high": 35.0,
            "assignment_temp": 1.0,
            "assignment_shift": 0.0,
            "threshold_quantile": 0.5,
            "n_covariates": 1,
            "x_effect_scale": 0.0,
            "epsilon_sigma": 1.0,
        }

    @property
    def max_dims(self) -> Dims:
        return Dims(d_U=1, d_X=int(self.options["n_covariates"]), d_W=1, d_Z=2)

    # ------------------------------------------------------------------

    def sample_episode(self, spec: EpisodeSpec, rep: Replicates) -> RawEpisode:
        o = self.options
        sigma = float(o["epsilon_sigma"])
        p_lo, p_hi = float(o["price_low"]), float(o["price_high"])
        temp = float(o["assignment_temp"])
        if temp <= 0:
            raise ValueError(
                "assignment_temp must be > 0: a hard threshold makes assignment "
                "deterministic in (U, Z) and destroys positivity"
            )
        d_X = int(o["n_covariates"])

        # X plays no structural role in the published design; it is held fixed
        # across replicates so that, under x_effect_scale > 0, tau* is a
        # function of it in the usual way.
        X_obs = self.randn(rep.T, d_X)
        X = rep.expand(X_obs)

        # U is redrawn per replicate row -- that is what marginalizes it out of
        # the target, and it is the whole reason this set can be scored
        # distributionally.
        U = 10.0 * self.rand(rep.N, 1)
        gU = g(U)

        W = 7.0 * gU + 45.0 + sigma * self.randn(rep.N, 1)

        # Z and the price are leaves w.r.t. the target (neither enters Y other
        # than through the arm), so both are generated on the observed rows.
        U_obs = rep.observed(U)
        gU_obs = g(U_obs)
        ang = 2.0 * torch.pi * U_obs / 10.0
        Z = torch.cat(
            [
                2.0 * torch.sin(ang) + sigma * self.randn(rep.T, 1),
                2.0 * torch.cos(ang) + sigma * self.randn(rep.T, 1),
            ],
            dim=1,
        )

        A_cont = (
            35.0 + (Z[:, :1] + 3.0) * gU_obs + Z[:, 1:2] + sigma * self.randn(rep.T, 1)
        )
        # Frozen on the observed rows: A_cont exists nowhere else, so mc_cate
        # cannot move these two statistics and hence cannot move the estimand.
        threshold = torch.quantile(A_cont, float(o["threshold_quantile"]))
        scale = A_cont.std().clamp(min=1e-6)
        # The shift is a pure location move in logit space: it changes the
        # marginal treated fraction and nothing else. Applied last, after every
        # RNG draw of the episode, so the potential outcomes are byte-identical
        # across shifts and an imbalance sweep varies exactly one thing.
        shift = float(o["assignment_shift"])
        A_probs = torch.sigmoid((A_cont - threshold) / (temp * scale) + shift)

        eps_y = sigma * self.randn(rep.N, 1)
        Y0 = potential_outcome(p_lo, W, gU) + eps_y
        Y1 = potential_outcome(p_hi, W, gU) + eps_y

        # Opt-in heterogeneity: high-price arm only, so Y(price_low), both
        # proxies and the price keep their published law exactly.
        x_scale = float(o["x_effect_scale"])
        tau_x = None
        if x_scale:
            coefs = self.signed_coefs(d_X, 1)
            tau_x = x_scale * (X @ coefs)
            Y1 = Y1 + tau_x

        return RawEpisode(
            X=X,
            W=W,
            Z=Z,
            A_probs=A_probs,
            Y0=Y0,
            Y1=Y1,
            U=U,
            d_U=1,
            d_X=d_X,
            d_W=1,
            d_Z=2,
            # Not the target: only the X-dependent *part* of tau* is known in
            # closed form (the constant ATE is not analytic under the exp/min
            # link), so nothing here is exact enough to publish as tau*.
            tau_star_exact=None,
            sampled_hp={
                "price_low": p_lo,
                "price_high": p_hi,
                "assignment_temp": temp,
                "assignment_shift": shift,
                "a_threshold": float(threshold),
                "a_scale": float(scale),
                "treated_frac": float(A_probs.mean()),
                "x_effect_scale": x_scale,
            },
        )
