"""Replicate layout and Monte Carlo reduction for one episode.

This is the same layout ``SCMPrior.generate_dataset`` and
``synt_eval.realistic_proximal_mechanisms`` use::

    [ ctx 0 .. n_ctx-1 ][ q0 x K ][ q1 x K ] ... [ q_{n_q-1} x K ]

The replicate block is a contiguous suffix, so the reduction is a single
``view(n_q, K).mean(1)`` with no gather -- and ``view``, which raises on a
non-contiguous input, doubles as the layout assertion.

It is re-implemented here rather than imported from ``synt_eval`` on purpose:
that copy is welded to the byte-identity constraint of the pre-replicate
realistic generator (``mc_cate=1`` must consume exactly zero RNG in the same
order it used to), which is a constraint this package does not have and should
not inherit. The *semantics* are identical and the tests pin that they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from proximalfm.prior._dataset import _mc_mean_var

Tensor = torch.Tensor


class MonteCarloBudgetError(RuntimeError):
    """Raised when the requested ``mc_cate`` exceeds the memory budget."""


# Live float32 columns per replicate row, summed over a wide episode: X, U, W
# (raw + standardized), a hidden activation, and a dozen single-column
# intermediates. A calibration dial for the budget check, not a hard ceiling;
# kept equal to ``synt_eval``'s so the two builders refuse the same K.
COLS_PER_ROW = 96


@dataclass(frozen=True)
class Replicates:
    """Replicate layout for one episode.

    ``obs_idx`` maps each logical row to the replicate row realized as the
    *observed* world: identity on the context prefix, one uniformly chosen
    replicate per query row. Everything the model ever sees is indexed through
    it; the full ``N``-row block exists only to build the target.
    """

    T: int
    n_ctx: int
    n_q: int
    K: int
    N: int
    obs_idx: Tensor

    @property
    def replicated(self) -> bool:
        return self.K > 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        T: int,
        train_size: int,
        K: int,
        *,
        rng: torch.Generator,
        mem_budget: int = 2_000_000_000,
    ) -> "Replicates":
        K = int(K)
        if K < 1:
            raise ValueError("mc_cate must be >= 1")
        if K == 1:
            return cls(T=T, n_ctx=T, n_q=0, K=1, N=T, obs_idx=torch.arange(T))

        # train_size is metadata elsewhere and is not guaranteed to be < T;
        # clamp so at least one query row survives (else view(0, K)).
        n_ctx = max(0, min(int(train_size), T - 1))
        n_q = T - n_ctx
        _check_budget(n_ctx, n_q, K, mem_budget)

        obs_k = torch.randint(K, size=(n_q,), generator=rng)
        obs_idx = torch.cat(
            [torch.arange(n_ctx), n_ctx + torch.arange(n_q) * K + obs_k]
        )
        return cls(T=T, n_ctx=n_ctx, n_q=n_q, K=K, N=n_ctx + n_q * K, obs_idx=obs_idx)

    # ------------------------------------------------------------------
    # Row plumbing
    # ------------------------------------------------------------------

    def expand(self, x: Tensor) -> Tensor:
        """``(T, d) -> (N, d)``: hold each query row fixed across ``K`` draws.

        Broadcast into one preallocated buffer; ``expand().reshape()`` + ``cat``
        would copy the query block twice.
        """
        if not self.replicated:
            return x
        if x.shape[0] == self.N:
            return x
        d = x.shape[1]
        out = x.new_empty((self.N, d))
        out[: self.n_ctx] = x[: self.n_ctx]
        out[self.n_ctx :].view(self.n_q, self.K, d)[:] = x[self.n_ctx :, None, :]
        return out

    def observed(self, x: Tensor) -> Tensor:
        """Index an ``N``-row tensor down to the ``T`` observed rows.

        Identity at ``K = 1``, and identity on a tensor that is already ``T``
        rows -- so a leaf quantity (one that never feeds the target, e.g. ``Z``
        in most mechanisms) may be generated on the observed rows only.
        """
        if not self.replicated or x.shape[0] == self.T:
            return x
        return x[self.obs_idx]

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def reduce(
        self, Y0_mc: Tensor, Y1_mc: Tensor, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Monte Carlo CEPO targets ``(MU0, MU1, S2, K_eff)``. Consumes no RNG.

        ``S2`` is the *paired* sampling variance of ``Y1 - Y0``: both arms come
        from the same replicate rows and share their noise draw, so differencing
        first cancels it. ``[Var(Y1) + Var(Y0)]/K`` would materially over-state
        the target noise.

        Context rows get NaN rather than the realized draw: a consumer reading
        ``MU`` on the context prefix is reading something that was never
        computed and should fail loudly, not silently receive ITE-quality
        values. ``S2 = 0`` (not NaN) is the project-wide "no replicates" marker.
        """
        if not self.replicated:
            return Y0_mc, Y1_mc, torch.zeros_like(Y0_mc), 1

        q0 = Y0_mc[self.n_ctx :, 0].view(self.n_q, self.K)
        q1 = Y1_mc[self.n_ctx :, 0].view(self.n_q, self.K)

        nan_pad = torch.full((self.n_ctx,), float("nan"), dtype=dtype)
        MU0 = torch.cat([nan_pad, q0.mean(dim=1)]).unsqueeze(1)
        MU1 = torch.cat([nan_pad, q1.mean(dim=1)]).unsqueeze(1)
        S2 = torch.cat([nan_pad, _mc_mean_var(q1 - q0, self.K)]).unsqueeze(1)
        return MU0, MU1, S2, self.K

    def query_pad(
        self, values: Optional[Tensor], dtype: torch.dtype
    ) -> Optional[Tensor]:
        """NaN-pad a per-query-row vector up to ``T`` rows, matching ``MU``.

        Used for quantities an adapter knows in closed form on the query rows
        (a closed-form ``tau*``, say), so they align with ``MU1 - MU0``.
        """
        if values is None:
            return None
        v = values.reshape(-1).to(dtype)
        if v.numel() == self.T:
            return v
        if v.numel() != self.n_q:
            raise ValueError(
                f"expected {self.n_q} query values or {self.T} row values, "
                f"got {v.numel()}"
            )
        return torch.cat([torch.full((self.n_ctx,), float("nan"), dtype=dtype), v])


def _check_budget(n_ctx: int, n_q: int, K: int, mem_budget: int) -> None:
    """Refuse a build that cannot afford the requested ``K``.

    The prior caps-and-warns because it generates online, where a noisier target
    beats a crashed run. A frozen evaluation set is the opposite object: an
    offline artifact whose whole value is a documented, *uniform* ``K``. A
    silent per-episode cap would make ``K_eff`` vary across episodes and across
    machines, invisibly, and every calibration number downstream would be scored
    against a target noisier than the sidecar claims.
    """
    N = n_ctx + n_q * K
    elements = N * COLS_PER_ROW
    if elements <= int(mem_budget):
        return
    raise MonteCarloBudgetError(
        f"mc_cate={K} needs N = n_ctx + n_q*K = {n_ctx:,} + {n_q:,}*{K} = {N:,} "
        f"replicate rows (~{elements * 4 / 1e9:.1f} GB at {COLS_PER_ROW} float32 "
        f"cols/row), over mc_mem_budget={int(mem_budget):,} elements. Lower "
        "--mc_cate, or lower n_q, or raise the budget."
    )
