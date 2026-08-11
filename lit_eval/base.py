"""The adapter interface every ``lit_eval`` dataset implements.

A literature dataset only has to say how *one* episode is drawn:

    class MyDataset(ProximalDataset):
        name = "my_dataset"
        max_dims = Dims(d_U=2, d_X=5, d_W=3, d_Z=3)

        def sample_episode(self, spec, rep) -> RawEpisode:
            ...

Everything else -- the replicate layout, the Monte Carlo reduction, drawing the
observed treatment, forming ``Y_obs``, padding to a common feature width,
validation, and stacking episodes into the batch the writer wants -- is done
here, once, so no adapter can get the load-bearing parts wrong.

The two rules the base class enforces, restated because they are the ones that
silently produce plausible-looking wrong numbers:

* ``Y0``/``Y1`` are **structural**: observation noise reaches ``Y_obs`` only.
  Adding it to the potential outcomes would inflate the target and its Monte
  Carlo variance.
* ``MU0``/``MU1`` are the Monte Carlo mean over replicates of the *query* rows,
  NaN on the context prefix. They are estimates of ``E[Y_a | X = x]``, so the
  estimand is ``tau*(x) = E[Y1 - Y0 | X = x]`` -- marginalizing over the latent,
  over ``W``, and over the outcome noise.

Rows arriving from :meth:`sample_episode`
-----------------------------------------

Anything that feeds the target (``U``, ``X``, ``W``, ``Y0``, ``Y1``) must be on
the ``rep.N`` replicate rows; build it from ``rep.expand(x)`` so each query row's
covariates are held fixed while its latent and noise are redrawn. Anything that
is a *leaf* with respect to the target (``Z`` and the propensity, in every
mechanism in this project, because ``Z`` never enters ``f_Y`` and ``A`` never
enters either) may be generated on the ``T`` observed rows directly -- cheaper,
and :meth:`Replicates.observed` passes it through untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from lit_eval._replicates import Replicates

Tensor = torch.Tensor

# Field order of a generated episode, and of the stacked pool. Identical to
# ``RealisticProximalGenerator.get_batch`` (and to what ``build`` writes), so
# both builders feed ``write_batch_file`` the same names.
EPISODE_FIELDS = (
    "U X W Z A A_probs Y_obs Y0 Y1 d_U d_X d_W d_Z MU0 MU1 S2 K_eff"
).split()

POOL_FIELDS = (
    "U X W Z A A_probs Y_obs Y0 Y1 d_U d_X d_W d_Z seq_lens train_sizes "
    "MU0 MU1 S2 K_eff"
).split()


@dataclass(frozen=True)
class Dims:
    """Upper bound on each block's width, across every episode of a build.

    Episodes may use fewer columns (report the true width in
    :class:`RawEpisode`); the base pads to these so episodes stack. A dataset
    with fixed dimensions simply reports its own.
    """

    d_U: int
    d_X: int
    d_W: int
    d_Z: int


@dataclass(frozen=True)
class EpisodeSpec:
    """What the builder asks for; the dataset decides everything else."""

    index: int
    seed: int
    seq_len: int  # T -- observed rows
    train_size: int  # n_ctx -- context rows; the rest are query rows
    mc_cate: int


@dataclass
class RawEpisode:
    """One episode as the adapter produces it, before the shared plumbing.

    ``U`` is optional and purely diagnostic (nothing downstream may consume it);
    pass ``None`` for a dataset with no explicit latent, and a zero-width block
    is emitted.
    """

    X: Tensor  # (N, d_X)   replicate rows
    W: Tensor  # (N, d_W)   replicate rows
    Z: Tensor  # (T|N, d_Z) leaf -> observed rows are fine
    A_probs: Tensor  # (T|N, 1)   propensity, pre-clamp
    Y0: Tensor  # (N, 1)     structural, no observation noise
    Y1: Tensor  # (N, 1)     structural, no observation noise
    d_U: int
    d_X: int
    d_W: int
    d_Z: int
    U: Optional[Tensor] = None  # (N, d_U), diagnostic only

    # Closed-form ``tau*`` on the query rows, when the DGP has one. Never used
    # as the training/scoring target -- it is stored in the sidecar so a test
    # can check the Monte Carlo estimate against it, which is the only direct
    # check that the replicate machinery estimates what it claims to.
    tau_star_exact: Optional[Tensor] = None

    # Scalar per-episode knobs, recorded per episode in the sidecar. Anything
    # here becomes a stratification axis downstream (``prior_eval.strata``
    # cuts on ``sampled_hp``), so put the things you would want to slice by:
    # confounding strength, overlap, noise scale, sample size regime.
    sampled_hp: Dict[str, float] = field(default_factory=dict)


class ProximalDataset(ABC):
    """Base class for a literature / external proximal dataset adapter."""

    #: Registry key and the value recorded in every sidecar.
    name: str = "unnamed"
    #: Human-readable provenance, printed by the builder and stored in the sidecar.
    citation: str = ""
    #: ``False`` for a dataset with no forward model (real data): the builder
    #: then refuses ``--mc_cate > 1`` instead of writing a set whose targets are
    #: replicates of nothing. Such a set is point-only by construction.
    supports_replicates: bool = True

    #: Strict positivity floor: propensities are clamped to
    #: ``[overlap_eps, 1 - overlap_eps]`` before ``A`` is drawn.
    overlap_eps: float = 0.02
    #: Measurement noise added to ``Y_obs`` only.
    extra_observation_noise: float = 0.03

    def __init__(self, **options: Any) -> None:
        unknown = set(options) - set(self.option_defaults())
        if unknown:
            raise TypeError(
                f"{self.name}: unknown option(s) {sorted(unknown)}; "
                f"known: {sorted(self.option_defaults())}"
            )
        self.options: Dict[str, Any] = {**self.option_defaults(), **options}
        self.dtype: torch.dtype = torch.float32
        # Replaced per episode by :meth:`generate`.
        self.rng: torch.Generator = torch.Generator(device="cpu")

    # ------------------------------------------------------------------
    # What an adapter implements
    # ------------------------------------------------------------------

    @classmethod
    def option_defaults(cls) -> Dict[str, Any]:
        """Dataset-specific knobs and their defaults.

        The builder's ``--option k=v`` writes into this dict, and the resolved
        dict is part of the set's ``content_digest`` -- so a knob that changes
        the data must live here rather than being a module constant someone can
        edit in place without the set's identity noticing.
        """
        return {}

    @property
    @abstractmethod
    def max_dims(self) -> Dims:
        """Upper bound on ``(d_U, d_X, d_W, d_Z)`` across the whole build."""

    @abstractmethod
    def sample_episode(self, spec: EpisodeSpec, rep: Replicates) -> RawEpisode:
        """Draw one episode. See the module docstring for which rows are which."""

    def describe(self) -> str:
        """One line for the builder's console output."""
        return f"{self.name}: {self.citation or 'no citation recorded'}"

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def generate(
        self, spec: EpisodeSpec, *, mem_budget: int = 2_000_000_000
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Any]]:
        """Draw episode ``spec`` and return ``(tensors, info)``.

        ``tensors`` follows :data:`EPISODE_FIELDS`; ``info`` carries the
        out-of-band per-episode record (closed-form ``tau*``, ``sampled_hp``)
        that the sidecar keeps and the ``val/`` format cannot hold.
        """
        if spec.mc_cate > 1 and not self.supports_replicates:
            raise ValueError(
                f"{self.name} has no forward model, so it cannot be replicated: "
                "build it with --mc_cate 1 (point metrics only). A "
                "distributional score needs a dataset that can redraw the "
                "latent and the noise with the covariate row held fixed."
            )
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(spec.seed))

        rep = Replicates.make(
            spec.seq_len,
            spec.train_size,
            spec.mc_cate,
            rng=self.rng,
            mem_budget=mem_budget,
        )
        raw = self.sample_episode(spec, rep)
        return self._finalize(raw, rep)

    def _finalize(
        self, raw: RawEpisode, rep: Replicates
    ) -> Tuple[Tuple[Tensor, ...], Dict[str, Any]]:
        T = rep.T
        X = rep.observed(raw.X)
        W = rep.observed(raw.W)
        Z = rep.observed(raw.Z)
        Y0 = rep.observed(raw.Y0)
        Y1 = rep.observed(raw.Y1)
        U = (
            rep.observed(raw.U)
            if raw.U is not None
            else torch.zeros((T, 0), dtype=self.dtype)
        )

        A_probs = (
            rep.observed(raw.A_probs)
            .reshape(T, 1)
            .to(self.dtype)
            .clamp(self.overlap_eps, 1.0 - self.overlap_eps)
        )
        A = (self.rand(T, 1) < A_probs).to(self.dtype)

        # Observation noise reaches Y_obs only; Y0/Y1 stay structural, and the
        # Monte Carlo target is built from those.
        Y_obs = A * Y1 + (1.0 - A) * Y0
        if self.extra_observation_noise:
            Y_obs = Y_obs + float(self.extra_observation_noise) * self.randn(T, 1)

        MU0, MU1, S2, K_eff = rep.reduce(raw.Y0, raw.Y1, dtype=self.dtype)

        self._validate(
            rep=rep, U=U, X=X, W=W, Z=Z, A_probs=A_probs, MU0=MU0, MU1=MU1, raw=raw
        )

        m = self.max_dims
        tensors = (
            self._pad(U, m.d_U),
            self._pad(X, m.d_X),
            self._pad(W, m.d_W),
            self._pad(Z, m.d_Z),
            A.to(self.dtype),
            A_probs.to(self.dtype),
            Y_obs.to(self.dtype),
            Y0.to(self.dtype),
            Y1.to(self.dtype),
            torch.tensor(raw.d_U, dtype=torch.long),
            torch.tensor(raw.d_X, dtype=torch.long),
            torch.tensor(raw.d_W, dtype=torch.long),
            torch.tensor(raw.d_Z, dtype=torch.long),
            MU0.to(self.dtype),
            MU1.to(self.dtype),
            S2.to(self.dtype),
            torch.tensor(int(K_eff), dtype=torch.long),
        )
        info = {
            "sampled_hp": {k: float(v) for k, v in raw.sampled_hp.items()},
            "tau_star_exact": rep.query_pad(raw.tau_star_exact, self.dtype),
        }
        return tensors, info

    def _validate(
        self,
        *,
        rep: Replicates,
        U: Tensor,
        X: Tensor,
        W: Tensor,
        Z: Tensor,
        A_probs: Tensor,
        MU0: Tensor,
        MU1: Tensor,
        raw: RawEpisode,
    ) -> None:
        T, m = rep.T, self.max_dims
        for nm, t, d, cap in (
            ("U", U, raw.d_U if raw.U is not None else 0, m.d_U),
            ("X", X, raw.d_X, m.d_X),
            ("W", W, raw.d_W, m.d_W),
            ("Z", Z, raw.d_Z, m.d_Z),
        ):
            if t.shape[0] != T:
                raise ValueError(
                    f"{self.name}: {nm} has {t.shape[0]} rows, expected {T}"
                )
            if t.shape[1] != d:
                raise ValueError(
                    f"{self.name}: {nm} has {t.shape[1]} columns but d_{nm}={d}"
                )
            if d > cap:
                raise ValueError(
                    f"{self.name}: d_{nm}={d} exceeds max_dims.d_{nm}={cap}; "
                    "widen max_dims or the episodes will not stack"
                )
            if nm != "U" and not torch.isfinite(t).all():
                raise ValueError(f"{self.name}: {nm} contains non-finite values")

        # X, W and Z must each be at least one column wide. A zero-width block
        # is not a smaller version of the format, it is outside it: the sparse
        # writer's `X.view(-1, 0)` raises an unreadable reshape error, and a
        # design with no covariates has to say so with an inert column rather
        # than an absent one. Caught here, before generation time is spent.
        for nm, d in (("X", raw.d_X), ("W", raw.d_W), ("Z", raw.d_Z)):
            if d < 1:
                raise ValueError(
                    f"{self.name}: d_{nm}={d}; the on-disk format has no "
                    f"zero-width block. A design with no {nm} needs one inert "
                    "column (independent noise), not zero columns."
                )

        # Categorical completeness, the structural precondition for proximal
        # identification. Heuristic, exactly as in the prior: necessary, not
        # sufficient -- but a violation here is unambiguous.
        if raw.d_U and min(raw.d_W, raw.d_Z) < raw.d_U:
            raise ValueError(
                f"{self.name}: min(d_W={raw.d_W}, d_Z={raw.d_Z}) < d_U={raw.d_U}; "
                "the proxies cannot be complete for the latent"
            )

        if not torch.isfinite(A_probs).all():
            raise ValueError(f"{self.name}: A_probs contains non-finite values")

        if rep.replicated:
            # NaN on the context prefix and finite on the query rows is the
            # contract every downstream consumer reads MU under.
            for nm, mu in (("MU0", MU0), ("MU1", MU1)):
                head, tail = mu[: rep.n_ctx], mu[rep.n_ctx :]
                if not torch.isnan(head).all():
                    raise ValueError(f"{self.name}: {nm} is not NaN on context rows")
                if not torch.isfinite(tail).all():
                    raise ValueError(f"{self.name}: {nm} is not finite on query rows")

    def _pad(self, x: Tensor, width: int) -> Tensor:
        if x.shape[1] == width:
            return x.to(self.dtype)
        pad = torch.zeros((x.shape[0], width - x.shape[1]), dtype=self.dtype)
        return torch.cat([x.to(self.dtype), pad], dim=1)

    # ------------------------------------------------------------------
    # RNG helpers -- every draw goes through this generator, so an episode is
    # reproducible from its seed alone.
    # ------------------------------------------------------------------

    def rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, generator=self.rng, dtype=self.dtype)

    def randn(self, *shape: int) -> Tensor:
        return torch.randn(*shape, generator=self.rng, dtype=self.dtype)

    def randint(self, low: int, high: int) -> int:
        """Inclusive on both ends, matching the other generators in this repo."""
        return int(torch.randint(low, high + 1, (1,), generator=self.rng).item())

    def uniform(self, low: float, high: float, *shape: int) -> Tensor:
        return low + (high - low) * self.rand(*shape)

    def uniform_scalar(self, low: float, high: float) -> float:
        return float(self.uniform(low, high, 1).item())

    def signed_coefs(self, *shape: int, low: float = 0.3, high: float = 1.5) -> Tensor:
        """Random coefficients with magnitude in ``[low, high]`` and random sign.

        Bounding the magnitude away from zero is what keeps a "confounded" DGP
        actually confounded: unconstrained Gaussian coefficients occasionally
        draw a near-zero loading on the latent and produce an episode that
        backdoor adjustment solves.
        """
        mag = self.uniform(low, high, *shape)
        sign = torch.where(self.rand(*shape) < 0.5, -1.0, 1.0)
        return mag * sign

    def linear(self, x: Tensor, out_dim: int, low: float = 0.3, high: float = 1.5):
        """A random linear map ``x @ B``, coefficients from :meth:`signed_coefs`."""
        B = self.signed_coefs(x.shape[1], out_dim, low=low, high=high)
        return x @ B

    @staticmethod
    def standardize(x: Tensor, eps: float = 1e-6) -> Tensor:
        """Column-standardize over the rows given.

        Call it on the **replicate** block, never separately per split: a
        statistic computed over a different row set per call would make ``K``
        move the estimand instead of only reducing its variance.
        """
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)

    def propensity(self, logit: Tensor, clip: float = 6.0) -> Tensor:
        """Standardized, clipped logit -> probability in ``(0, 1)``."""
        z = self.standardize(logit).clamp(-clip, clip)
        return torch.sigmoid(z)


def stack_episodes(
    episodes: Sequence[Tuple[Tensor, ...]], seq_len: int, train_size: int
) -> Tuple[Tensor, ...]:
    """Stack per-episode tuples into the 19-field pool the writer consumes."""
    if not episodes:
        raise ValueError("no episodes to stack")
    n_fields = len(episodes[0])
    stacked: List[Tensor] = [
        torch.stack([e[f] for e in episodes]) for f in range(n_fields)
    ]
    B = stacked[0].shape[0]
    seq_lens = torch.full((B,), int(seq_len), dtype=torch.long)
    train_sizes = torch.full((B,), int(train_size), dtype=torch.long)
    # seq_lens/train_sizes sit at indices 13/14, exactly where the realistic
    # generator puts them, so POOL_FIELDS reads both builders' pools.
    return tuple(stacked[:13] + [seq_lens, train_sizes] + stacked[13:])
