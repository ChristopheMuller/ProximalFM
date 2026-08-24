"""P-learner (Sverdrup & Cui, 2023) — proximal CATE by pseudo-outcome regression.

The method is two independent halves, and this module is built so they stay that
way:

1. **The bridges.** Estimate the confounding bridge functions ``h(w, a, x)`` and
   ``q(z, a, x)`` from the proximal moment conditions. This is the hard,
   method-defining half, and it is the only part that changes between the
   literature's proposals (kernel minimax, 2SLS, deep features, …).
2. **The driver.** Cross-fit those bridges, form the doubly robust proximal
   score, and regress it on ``X``. This is Algorithm 1 of the paper and it is
   fixed: it does not care how the bridges were obtained.

Everything below the ``Bridge`` interface is half 1; everything above it is
half 2. A new bridge is one class and one registry entry.

The score (paper eq. 10), for cross-fit nuisances::

    Gamma_i = (-1)^(1 - A_i) q(Z_i, A_i, X_i) (Y_i - h(W_i, A_i, X_i))
              + h(W_i, 1, X_i) - h(W_i, 0, X_i)

and the estimator is ``tau(x) = E[Gamma | X = x]``, fit by ordinary regression.
The sample mean of ``Gamma`` is the semiparametric-efficient proximal ATE, so it
comes out of the same fit for free and is exposed as ``ate_``.

Arrays convention, matching the rest of ``benchmarks.methods``::

    X: (n, d_x) covariates       A: (n,) binary treatment in {0, 1}
    W: (n, d_w) outcome proxy    Y: (n,) observed outcome
    Z: (n, d_z) treatment proxy

The production bridge is ``bridge="kernel"``: the minimax RKHS bridge of the
proximal literature, solved in an explicit Nyström feature space so no ``n x n``
matrix is ever formed. Passing ``lam_grid`` selects its two regularizers per
episode on a held-out *moment* loss rather than hand setting them. See
:class:`KernelBridge` and :class:`_MomentSystem`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from benchmarks.methods.base import CATEEstimator

__all__ = [
    "Bridge",
    "KernelBridge",
    "BRIDGES",
    "build_bridge",
    "proximal_scores",
    "PLearner",
]


# ─────────────────────────────────────────────────────────────────────────────
# Input handling
# ─────────────────────────────────────────────────────────────────────────────


def _as_2d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {arr.shape}")
    return arr


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_binary(a: ArrayLike, name: str = "A") -> np.ndarray:
    """Return A as a 0/1 float vector, refusing anything else.

    Loud rather than lenient: every formula below indexes an arm with it, and a
    silently-coerced {1, 2} or {-1, 1} encoding would flip the sign of the score
    instead of raising.
    """
    arr = _as_1d(a, name)
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(f"{name} must be binary and encoded as 0/1")
    return arr


def _check_lengths(n: int, **arrays: Optional[np.ndarray]) -> None:
    for name, arr in arrays.items():
        if arr is not None and arr.shape[0] != n:
            raise ValueError(
                f"{name} has {arr.shape[0]} rows, expected {n} — X, W, Z, A and Y "
                "must all describe the same observations"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Half 1 — the bridges
# ─────────────────────────────────────────────────────────────────────────────


class Bridge(ABC):
    """Estimates of the two proximal confounding bridge functions.

    A bridge is fit on one set of rows and evaluated on another (the driver
    cross-fits it), so it must carry whatever it needs — scalers, kernel centres,
    coefficients — from ``fit`` to ``predict_*``.

    Both predict methods return **both arms at once**, shape ``(n, 2)`` with
    column ``a`` holding the function evaluated at ``A = a``. That is not a
    convenience: the score needs ``h`` at the observed arm *and* at both
    counterfactual arms, so a one-arm-per-call interface makes callers either
    recompute shared work or cache it themselves. Returning the pair is the
    cheapest correct shape, and it is what lets a kernel bridge build its
    cross-kernel once.
    """

    #: Registry key; set by @register_bridge.
    name: str = ""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "Bridge":
        """Estimate h and q from these rows."""

    @abstractmethod
    def predict_h(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Outcome bridge, ``(n, 2)``: column a is ``h(W_i, a, X_i)``."""

    @abstractmethod
    def predict_q(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Treatment bridge, ``(n, 2)``: column a is ``q(Z_i, a, X_i)``."""


BRIDGES: dict[str, type[Bridge]] = {}


def register_bridge(name: str):
    """Class decorator: make a Bridge reachable as ``bridge: <name>`` in YAML."""

    def wrap(cls: type[Bridge]) -> type[Bridge]:
        if name in BRIDGES:
            raise ValueError(f"bridge {name!r} is already registered")
        cls.name = name
        BRIDGES[name] = cls
        return cls

    return wrap


def build_bridge(name: str, params: Optional[dict] = None) -> Bridge:
    """Instantiate a registered bridge by name.

    Bridge hyperparameters travel as a plain dict from the config's
    ``bridge_params:`` block, so every knob a bridge has is reachable from YAML
    without this function knowing what any of them mean.
    """
    if name not in BRIDGES:
        raise ValueError(f"unknown bridge {name!r}; registered: {sorted(BRIDGES)}")
    return BRIDGES[name](**(params or {}))


# ── the kernel bridge, and the feature map that keeps it small ───────────────


def _median_bandwidth(
    P: np.ndarray, rng: np.random.Generator, max_pairs: int = 2000
) -> float:
    """RBF bandwidth by the median heuristic, from a *subsample* of pairs.

    The heuristic wants the median pairwise distance, but computing it exactly
    means an n x n distance matrix — the very thing this bridge is built to
    avoid, and for a median of O(n^2) numbers the sampling error is negligible
    anyway. So it draws ``max_pairs`` random pairs instead: O(max_pairs) memory
    and an estimate whose relative error is a fraction of a percent.

    Returns a strictly positive width; degenerate inputs (all rows identical)
    fall back to 1.0 rather than producing a zero divisor.
    """
    n = P.shape[0]
    if n < 2:
        return 1.0
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    keep = i != j
    if not np.any(keep):
        return 1.0
    d = np.linalg.norm(P[i[keep]] - P[j[keep]], axis=1)
    med = float(np.median(d))
    return med if med > 1e-12 else 1.0


class _NystromFeatures:
    """An explicit feature map ``Phi`` with ``Phi Phi' ~= K``, built in chunks.

    This is the whole memory story. The textbook kernel bridge forms ``K`` at
    ``n x n`` and solves against it; here we pick ``r`` landmark rows, form the
    thin ``K_nr`` and the small ``K_rr``, and hand back

        Phi = K_nr @ K_rr^(-1/2)          (n, r)

    whose Gram is the Nystrom approximation of ``K``. Every downstream solve is
    then ``r x r``. Two consequences worth stating plainly:

    - **``rank=None`` is not a different algorithm.** It takes every row as a
      landmark, so ``Phi Phi' = K`` exactly and the bridge becomes the textbook
      one. That is the reference path — correct, and quadratic in memory. The
      approximation is a dial, not a fork in the code.
    - ``K_rr^(-1/2)`` is formed by eigendecomposition with a relative cutoff,
      so a rank-deficient landmark set (duplicated rows, a tiny bandwidth)
      truncates rather than blowing up. ``Phi`` may come back narrower than
      ``r``; that is reported in ``rank_``.
    """

    def __init__(self, sigma: float, rcond: float = 1e-10, chunk: int = 1024):
        self.sigma = float(sigma)
        self.rcond = float(rcond)
        self.chunk = int(chunk)

    def _kernel_block(self, P: np.ndarray) -> np.ndarray:
        """RBF kernel of ``P`` against the landmarks, ``(len(P), r)``."""
        d2 = (
            np.sum(P**2, axis=1)[:, None]
            + self._land_sq[None, :]
            - 2.0 * (P @ self.landmarks_.T)
        )
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-d2 / (2.0 * self.sigma**2))

    def fit(self, P: np.ndarray, idx: np.ndarray) -> "_NystromFeatures":
        self.landmarks_ = P[idx]
        self._land_sq = np.sum(self.landmarks_**2, axis=1)

        K_rr = self._kernel_block(self.landmarks_)
        vals, vecs = np.linalg.eigh(K_rr)  # K_rr is PSD by construction
        keep = vals > self.rcond * max(vals[-1], 1e-300)
        self.whiten_ = vecs[:, keep] / np.sqrt(vals[keep])
        self.rank_ = int(keep.sum())
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """``Phi`` for new rows. Chunked, so the temporary is (chunk, r)."""
        out = np.empty((P.shape[0], self.rank_), dtype=float)
        for s in range(0, P.shape[0], self.chunk):
            e = min(s + self.chunk, P.shape[0])
            out[s:e] = self._kernel_block(P[s:e]) @ self.whiten_
        return out


def _solve_psd(M: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``(M + ridge I) v = b`` for symmetric PSD ``M``.

    Cholesky, because every system this file hands it is a Gram matrix plus a
    strictly positive multiple of the identity and is therefore positive
    definite — no pseudo-inverse or SVD is warranted. Falls back to a symmetric
    least-squares solve if the factorisation fails, which it can only do if the
    caller passed ``ridge=0`` on a singular matrix.
    """
    A = M + ridge * np.eye(M.shape[0])
    try:
        return cho_solve(
            cho_factor(A, lower=True, check_finite=False), b, check_finite=False
        )
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


class _MomentSystem:
    """Everything the two saddle points need from a block of rows, with the
    regularizers **not yet chosen** — and the held-out criterion that chooses them.

    Fix the feature maps and this object is all that a candidate ``lam_h`` or
    ``lam_q`` ever touches::

        S_Q = (Phi_Q'Phi_Q/n + lam_g I)^-1     S_H = (Phi_H'Phi_H/n + lam_g I)^-1
        B_a = Phi_Q' D_a Phi_H / n             c_a = Phi_Q' D_a Y / n
        f   = Phi_H' 1 / n

    all of them free of ``lam_h`` and ``lam_q``. A candidate therefore costs one
    ``r x r`` Cholesky and two mat-vecs, against the ~0.4 s the feature build
    costs — which is what makes per-episode tuning affordable at all, and is a
    property of the step-3 rearrangement rather than of the tuner.

    **The criterion.** The h-bridge is defined by a *conditional moment*,
    ``E[Y - h(W, A, X) | Z, A, X] = 0``, not by a prediction target. Scoring a
    candidate by held-out ``mean (Y - h)^2`` therefore selects the wrong function
    outright: that loss is minimised by ``E[Y | W, A, X]``, the regression of Y on
    the outcome proxy, which under confounding differs from the bridge by exactly
    the quantity the method exists to remove. It would drive ``lam_h`` to zero and
    hand back the naive answer.

    What is well posed is the *violation* of the moment, measured against the same
    test class the fit maximises over. Writing ``g = Phi_Q omega``, the empirical
    moment is the vector ``v(theta) = c_a - B_a theta``, and

        max_omega  2 v'omega - omega'(Phi_Q'Phi_Q/n + lam_g I) omega  =  v' S_Q v

    in closed form. That is :meth:`loss_h`: the RKHS-smoothed
    ``||E[Y - h | Z, A, X]||^2``. It is zero at the true bridge, non-negative
    everywhere, and indifferent to how well ``h`` predicts Y. The q-side mirrors
    it with ``v(gamma) = B_a' gamma - f`` and ``S_H``.

    The reason to trust it is that it is not a new metric. Differentiating the
    *penalised* criterion,

        d/dtheta [ (c - B theta)' S_Q (c - B theta) + lam_h theta'theta ] = 0
          <=>  (B' S_Q B + lam_h I) theta = B' S_Q c

    reproduces :meth:`solve_h` exactly. So the selection score **is** the training
    objective with the penalty term dropped, evaluated on rows the bridge never
    saw — the penalty being precisely the term one must not grade a fit on.

    Two consequences worth internalising before changing anything here:

    - **The searches decouple.** ``theta`` depends only on ``lam_h`` and
      ``loss_h`` only on ``theta``; likewise ``gamma``/``lam_q``/``loss_q``. So
      tuning both is two 1-D searches costing ``|grid| + |grid|``, not a grid
      costing ``|grid|^2``.
    - **``lam_g`` must not be tuned.** ``S_Q`` *is* the norm the violation is
      measured in, so a candidate carrying its own ``lam_g`` would be scored in
      its own units, and the search would simply pick whichever value shrinks the
      yardstick. It stays a fixed constant — which costs nothing, since it was
      measured flat over two decades.
    """

    def __init__(self, Phi_H, Phi_Q, A, Y, lam_g: float):
        n = len(A)
        self.S_Q = np.linalg.inv(Phi_Q.T @ Phi_Q / n + lam_g * np.eye(Phi_Q.shape[1]))
        self.S_H = np.linalg.inv(Phi_H.T @ Phi_H / n + lam_g * np.eye(Phi_H.shape[1]))
        self.f = Phi_H.sum(axis=0) / n  # E[g(W, X)] — the q-system's RHS

        self.B, self.c = [], []
        for a in (0, 1):
            in_arm = A == a
            if not np.any(in_arm):
                raise ValueError(
                    f"no rows with A={a} in this block; the kernel bridge has "
                    "nothing to estimate. Reduce n_splits (or tune_val_frac) or "
                    "check the episode."
                )
            Ph_a, Pq_a = Phi_H[in_arm], Phi_Q[in_arm]
            self.B.append(Pq_a.T @ Ph_a / n)  # (r_Q, r_H) — serves both systems
            self.c.append(Pq_a.T @ Y[in_arm] / n)

    # ── the two solves ───────────────────────────────────────────────────────

    def solve_h(self, a: int, lam_h: float) -> np.ndarray:
        BS = self.B[a].T @ self.S_Q  # (r_H, r_Q)
        return _solve_psd(BS @ self.B[a], BS @ self.c[a], lam_h)

    def solve_q(self, a: int, lam_q: float) -> np.ndarray:
        SB = self.B[a] @ self.S_H  # (r_Q, r_H)
        return _solve_psd(SB @ self.B[a].T, SB @ self.f, lam_q)

    # ── the two held-out losses ──────────────────────────────────────────────

    def loss_h(self, a: int, theta: np.ndarray) -> float:
        v = self.c[a] - self.B[a] @ theta
        return float(v @ self.S_Q @ v)

    def loss_q(self, a: int, gamma: np.ndarray) -> float:
        v = self.B[a].T @ gamma - self.f
        return float(v @ self.S_H @ v)


def _stratified_split(A: np.ndarray, val_frac: float, rng) -> tuple:
    """(train, val) row indices, splitting **within each arm**.

    Stratified for the same reason :meth:`PLearner._folds` is: every block handed
    to :class:`_MomentSystem` needs both arms, and a plain permutation loses one
    on an imbalanced episode — the failure mode being an exception mid-search
    rather than a bad score.
    """
    train, val = [], []
    for a in (0, 1):
        idx = np.flatnonzero(A == a)
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(len(idx) * val_frac))
        n_val = min(max(n_val, 1), len(idx) - 1)  # both halves keep this arm
        val.append(idx[:n_val])
        train.append(idx[n_val:])
    return np.concatenate(train), np.concatenate(val)


@register_bridge("kernel")
class KernelBridge(Bridge):
    """The minimax kernel bridge (Ghassami et al. 2022; Dikkala et al. 2020),
    solved in an explicit feature space so nothing is ever ``n x n``.

    The same three-step reasoning applies: bridge equation, then a
    conditional moment restriction, then testing against a function class — with
    the linear span replaced by an RKHS. Because the test class is now infinite
    dimensional, "test against every ``g``" becomes a saddle-point problem,

        min_h  max_g  (2/n) sum_i d_i (Y_i - h(F_i)) g(G_i)
                      - (1/n) sum_i g(G_i)^2 - lam_g ||g||^2  + lam_h ||h||^2

    whose inner maximisation is a quadratic with a closed form. Writing
    ``h = Phi_H theta`` and ``g = Phi_Q omega`` and eliminating ``omega``:

        theta_a = (B' S_Q B + lam_h I)^-1 B' S_Q c
        gamma_a = (B  S_H B' + lam_q I)^-1 B  S_H f

        B   = Phi_Q' D_a Phi_H / n        S_Q = (Phi_Q'Phi_Q/n + lam_g I)^-1
        c   = Phi_Q' D_a Y     / n        S_H = (Phi_H'Phi_H/n + lam_g I)^-1
        f   = Phi_H' 1         / n

    Three structural facts this buys, all of which the archive's kernel-matrix
    formulation gives up:

    - **``B`` is shared.** The h-system uses ``B'`` and the q-system uses ``B``.
      One cross-product per arm, not four.
    - **``S_Q`` and ``S_H`` do not depend on the arm**, so they are factored once
      per fold rather than once per (arm, bridge). The archive recomputes the
      equivalent object four times, and at the default lambdas two of the four
      are bit-identical.
    - **Every solve is ``r x r``** and every matrix is at most ``n x r``. At
      n = 2048, r = 256 that is 4 MB of features and a 256-cubed factorisation,
      against 565 MB and 1024-cubed for the archive.

    The bridge regularizers play the usual Tikhonov role: the bridge equation is
    an ill-posed inverse problem, and
    undamped it inverts sampling noise. ``lam_g`` regularises the *test* class,
    which is what makes the inner maximisation well posed at all.

    **On the lambdas: prefer ``lam_grid`` to the defaults.** ``lam_h = lam_q =
    1e-4`` was chosen by sweeping on 25 episodes of the realistic validation set,
    where it is the optimum (median nPEHE 0.88, against 1.01 at ``1e-5`` and 1.11
    at ``1e-3``). That is tuning on an evaluation set — the thing one is not
    supposed to do — and worse, **the optimum moves with n**, so no constant can
    be right everywhere.

    Passing ``lam_grid`` replaces both defects with a per-fit choice on held-out
    moment loss (:class:`_MomentSystem`). Measured on those same episodes with
    ``lam_grid = [1e-7 ... 1e-1]``, ``rank = 256``, median over 25 episodes:

        n_ctx   fixed 1e-4      tuned          tuned picks
        ─────   ──────────      ─────          ───────────
          256   nPEHE 2.087     1.892          lam_h 1e-3
          512         1.425     1.339          lam_h 1e-3
         2048         0.884     0.866          lam_h 1e-4

    Read that as the transport claim, not as a headline gain: at 2048 the tuner
    *reproduces* the hand-picked value and so can only match it, which is the
    correct outcome and the reason the last row moves so little. The gain appears
    where the fixed value was never fitted, and it appears in the right direction —
    less data, more damping. Cost is +25 to +50% wall clock (0.55 -> 0.70 s per
    episode at n = 2048), which buys a value that no longer depends on having
    seen the evaluation set. One number ran the other way and is worth watching:
    at n = 512 the ATE error rose (0.094 against 0.063) while nPEHE fell, i.e.
    the extra damping helped the shape and hurt the level.

    ``lam_g``, by contrast, is genuinely flat — ``1e-4`` through ``1e-2`` move
    nothing measurable — and is not tunable in any case; see
    :class:`_MomentSystem` for why.

    ``rank`` is close to free accuracy: on those episodes 256 landmarks reproduce
    the exact bridge to within 2% of nPEHE (0.879 against 0.865) at a sixteenth
    of the time (0.46 s against 7.45 s per episode) and ~600 MB less peak memory.
    Even 64 landmarks give 0.948.

    Args:
        rank: Nystrom landmark count. ``None`` uses every row, giving the exact
            kernel bridge at quadratic memory — the reference path, appropriate
            for tests and small n, not for a 2048-row episode.
        bandwidth_scale: multiplies the median-heuristic width. ``1.0`` is the
            heuristic itself. This is the knob the archive got wrong by pinning
            an absolute ``kernel_sigma = 1.0``, which is not a bandwidth at all
            — it ignores the data scale and the proxy dimension.
        lam_h, lam_q: Tikhonov damping on the two bridges. Used as given when
            ``lam_grid`` is None, and as the fallback if every candidate scores
            non-finite.
        lam_g: damping on the test class, shared by both saddle points. Not
            tunable by construction — it defines the units the selection
            criterion is measured in (:class:`_MomentSystem`).
        lam_grid: candidate values for ``lam_h`` and ``lam_q``, selected per fit
            on held-out moment loss. ``None`` (the default) keeps the fixed
            lambdas. The chosen values land in ``lam_h_``/``lam_q_`` and the full
            curves in ``lam_scores_``.
        tune_val_frac: fraction of the fold held out for that selection,
            stratified on A. Only the *choice* uses the split; the returned
            bridge is refit on every row.
        random_state: seeds the landmark draw, the bandwidth subsample and the
            tuning split.
        chunk: row block size for the feature transform.
    """

    def __init__(
        self,
        rank: Optional[int] = 256,
        bandwidth_scale: float = 1.0,
        lam_h: float = 1e-4,
        lam_q: float = 1e-4,
        lam_g: float = 1e-3,
        lam_grid: Optional[Sequence[float]] = None,
        tune_val_frac: float = 0.25,
        random_state: Optional[int] = 0,
        chunk: int = 1024,
    ):
        self.rank = rank
        self.bandwidth_scale = float(bandwidth_scale)
        self.lam_h = float(lam_h)
        self.lam_q = float(lam_q)
        self.lam_g = float(lam_g)
        self.lam_grid = lam_grid
        self.tune_val_frac = float(tune_val_frac)
        self.random_state = random_state
        self.chunk = int(chunk)

    # ── feature construction ─────────────────────────────────────────────────

    def _build(self, P: np.ndarray, rng) -> tuple:
        """Scaler + feature map for one block, and the block's own bandwidth.

        Standardization already removes the raw scale, so what a per-block
        bandwidth buys is *dimension*: ``[W, X]`` and ``[Z, X]`` have widths
        ``d_W + d_X`` and ``d_Z + d_X``, and the median pairwise distance grows
        like the square root of that. With ``d_W`` and ``d_Z`` drawn
        independently in 1..10 these routinely differ by a factor of two, and one
        shared sigma then over-smooths the wider block and under-smooths the
        narrower.
        """
        scaler = StandardScaler().fit(P)
        Ps = scaler.transform(P)

        sigma = _median_bandwidth(Ps, rng) * self.bandwidth_scale
        n = Ps.shape[0]
        r = n if self.rank is None else min(int(self.rank), n)
        idx = np.arange(n) if r >= n else rng.choice(n, size=r, replace=False)

        feats = _NystromFeatures(sigma, chunk=self.chunk).fit(Ps, idx)
        return scaler, feats

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, A, Y, *, W, Z) -> "KernelBridge":
        rng = np.random.default_rng(self.random_state)

        self.h_scaler_, self.h_feats_ = self._build(np.column_stack([W, X]), rng)
        self.q_scaler_, self.q_feats_ = self._build(np.column_stack([Z, X]), rng)

        Phi_H = self.h_feats_.transform(
            self.h_scaler_.transform(np.column_stack([W, X]))
        )
        Phi_Q = self.q_feats_.transform(
            self.q_scaler_.transform(np.column_stack([Z, X]))
        )

        self.lam_h_, self.lam_q_ = self.lam_h, self.lam_q
        if self.lam_grid is not None:
            self.lam_h_, self.lam_q_ = self._select_lambdas(Phi_H, Phi_Q, A, Y, rng)

        # One system over all the rows, whether or not a split was used to pick
        # the lambdas: selection is the only thing the held-out half is for.
        full = _MomentSystem(Phi_H, Phi_Q, A, Y, self.lam_g)

        self.theta_ = np.stack([full.solve_h(a, self.lam_h_) for a in (0, 1)])
        self.gamma_ = np.stack([full.solve_q(a, self.lam_q_) for a in (0, 1)])
        return self

    # ── tuning ───────────────────────────────────────────────────────────────

    def _select_lambdas(self, Phi_H, Phi_Q, A, Y, rng) -> tuple[float, float]:
        """Pick ``(lam_h, lam_q)`` by held-out moment loss — see :class:`_MomentSystem`.

        Two 1-D searches rather than one 2-D grid, because the objectives
        decouple. Both are summed over arms: the two arms are separate systems
        but share a regularizer, and the alternative — a per-arm lambda — would
        halve the rows each choice is made on for no reason anyone has argued.

        ``lam_scores_`` keeps the full curve. Reach for it before trusting a
        selected value: a monotone curve means the grid does not bracket the
        optimum, and that is invisible from the chosen lambda alone.
        """
        grid = sorted(float(v) for v in self.lam_grid)
        if not grid:
            raise ValueError("lam_grid is empty; pass None to disable tuning")

        tr, va = _stratified_split(A, self.tune_val_frac, rng)
        train = _MomentSystem(Phi_H[tr], Phi_Q[tr], A[tr], Y[tr], self.lam_g)
        val = _MomentSystem(Phi_H[va], Phi_Q[va], A[va], Y[va], self.lam_g)

        self.lam_scores_ = {
            "lam_h": [
                (lam, sum(val.loss_h(a, train.solve_h(a, lam)) for a in (0, 1)))
                for lam in grid
            ],
            "lam_q": [
                (lam, sum(val.loss_q(a, train.solve_q(a, lam)) for a in (0, 1)))
                for lam in grid
            ],
        }
        return tuple(
            min(
                (p for p in self.lam_scores_[k] if np.isfinite(p[1])),
                key=lambda p: p[1],
                default=(getattr(self, k), np.inf),
            )[0]
            for k in ("lam_h", "lam_q")
        )

    # ── predict ──────────────────────────────────────────────────────────────

    def predict_h(self, X, W) -> np.ndarray:
        Phi = self.h_feats_.transform(self.h_scaler_.transform(np.column_stack([W, X])))
        return Phi @ self.theta_.T

    def predict_q(self, X, Z) -> np.ndarray:
        Phi = self.q_feats_.transform(self.q_scaler_.transform(np.column_stack([Z, X])))
        return Phi @ self.gamma_.T


# ─────────────────────────────────────────────────────────────────────────────
# Half 2 — the score and the driver
# ─────────────────────────────────────────────────────────────────────────────


def proximal_scores(
    A: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    q: np.ndarray,
    *,
    q_clip: Optional[float] = None,
) -> np.ndarray:
    """The doubly robust proximal score, paper eq. (10). One value per row.

    Pure and array-only, so it can be tested against a hand-computed example
    without constructing a bridge.

    Args:
        A: (n,) observed arm, 0/1.
        Y: (n,) observed outcome.
        h: (n, 2) outcome bridge at both arms.
        q: (n, 2) treatment bridge at both arms.
        q_clip: if given, clip ``q`` to ``[-q_clip, q_clip]``. ``q`` is an
            inverse propensity in disguise, so a badly estimated value can have
            excessive magnitude and dominate the score. This is a signed
            magnitude cap: it preserves the sign of a negative estimate and
            does not enforce the positivity of the population bridge.

    Returns:
        (n,) array of scores. Its mean is the proximal ATE estimate; regressing
        it on X is the P-learner.
    """
    arm = A.astype(np.intp)
    rows = np.arange(len(A))

    h_obs = h[rows, arm]  # h(W_i, A_i, X_i)
    q_obs = q[rows, arm]  # q(Z_i, A_i, X_i)
    if q_clip is not None:
        q_obs = np.clip(q_obs, -abs(q_clip), abs(q_clip))

    sign = np.where(A == 1.0, 1.0, -1.0)  # (-1)^(1 - A)
    return sign * q_obs * (Y - h_obs) + h[:, 1] - h[:, 0]


class PLearner(CATEEstimator):
    """Algorithm 1: cross-fit the bridges, form the scores, regress them on X.

    One ``fit`` produces one score vector, and any number of final-stage models
    can be trained on it — the expensive half is the bridge, and it is shared.
    That is what ``final_models`` is for: pass ``{"P-RF": RandomForestRegressor(),
    "P-Linear": LinearRegression()}`` and both come back from a single
    ``predict_variants`` call, having paid for the bridges once.

    Args:
        bridge: registered bridge name (currently ``"kernel"``; see ``BRIDGES``).
        bridge_params: keyword arguments for that bridge.
        final_models: ``{output_name: unfitted sklearn regressor}``. Each is
            cloned before fitting, so the caller's instances are untouched.
        n_splits: cross-fitting folds. Stratified on A, so no fold can lose an
            arm — an unstratified split does exactly that on imbalanced episodes
            and takes the bridge down with it.
        random_state: seeds the fold shuffle.
        q_clip: magnitude cap on the observed-arm q; see ``proximal_scores``.

    Attributes set by ``fit``:
        scores_: (n,) the cross-fit Gamma.
        ate_: float, ``mean(scores_)`` — the proximal ATE.
        bridges_: the per-fold fitted bridges, kept for inspection.
    """

    def __init__(
        self,
        *,
        bridge: str = "kernel",
        bridge_params: Optional[dict] = None,
        final_models: Optional[dict[str, object]] = None,
        n_splits: int = 3,
        random_state: Optional[int] = 42,
        q_clip: Optional[float] = None,
    ):
        self.bridge = bridge
        self.bridge_params = bridge_params
        self.final_models = final_models
        self.n_splits = n_splits
        self.random_state = random_state
        self.q_clip = q_clip

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, A, Y, *, W=None, Z=None) -> "PLearner":
        if W is None or Z is None:
            raise ValueError(
                "PLearner requires both proxies: W (outcome) and Z (treatment)."
            )

        X = _as_2d(X, "X")
        W = _as_2d(W, "W")
        Z = _as_2d(Z, "Z")
        A = _as_binary(A)
        Y = _as_1d(Y, "Y")
        n = X.shape[0]
        _check_lengths(n, W=W, Z=Z, A=A, Y=Y)

        self._clip_hits_ = 0
        self._clip_total_ = 0
        self._lam_h_choices_: list[float] = []
        self._lam_q_choices_: list[float] = []

        self.scores_ = self._cross_fit_scores(X, A, Y, W, Z)
        self.ate_ = float(np.mean(self.scores_))

        models = self.final_models or {}
        self.final_models_ = {
            name: clone(m).fit(X, self.scores_) for name, m in models.items()
        }
        return self

    def _effective_bridge_params(self) -> dict:
        """``bridge_params`` with ``self.random_state`` filled in when the bridge
        takes one and the caller didn't already set it.

        Without this, a bridge's own ``random_state`` default (e.g.
        :class:`KernelBridge`'s ``0``) silently wins over ``PLearner``'s
        ``random_state`` (default ``42``) whenever ``bridge_params`` doesn't
        mention it — the same "wrapper default doesn't reach the nested
        component's own default" trap as ``KPV_Wrapper``'s seed. That decouples
        the landmark draw, bandwidth subsample and tuning split from the seed
        the caller thinks controls reproducibility.
        """
        params = dict(self.bridge_params or {})
        cls = BRIDGES.get(self.bridge)
        if cls is not None and "random_state" not in params:
            if "random_state" in signature(cls.__init__).parameters:
                params["random_state"] = self.random_state
        return params

    def _cross_fit_scores(self, X, A, Y, W, Z) -> np.ndarray:
        """Fit the bridge on each training fold, score its held-out fold.

        The held-out half is what makes the score orthogonal: a bridge evaluated
        on rows it was fit on carries its own overfitting into Gamma, and the
        oracle bound in the paper (Thm 1) assumes it does not.
        """
        n = X.shape[0]
        scores = np.empty(n, dtype=float)
        self.bridges_ = []
        bridge_params = self._effective_bridge_params()

        for train_idx, test_idx in self._folds(A):
            bridge = build_bridge(self.bridge, bridge_params).fit(
                X[train_idx],
                A[train_idx],
                Y[train_idx],
                W=W[train_idx],
                Z=Z[train_idx],
            )
            self.bridges_.append(bridge)

            h = bridge.predict_h(X[test_idx], W[test_idx])
            q = bridge.predict_q(X[test_idx], Z[test_idx])
            scores[test_idx] = proximal_scores(
                A[test_idx], Y[test_idx], h, q, q_clip=self.q_clip
            )
            self._record_fold_diagnostics(bridge, A[test_idx], q)

        return scores

    def _record_fold_diagnostics(self, bridge: Bridge, A_test, q) -> None:
        """Accumulate this fold's contribution to :meth:`diagnostics`.

        Mirrors, rather than reuses, the observed-arm selection inside
        ``proximal_scores`` — that function stays pure and score-only; this is
        purely a side-observation of the same ``q`` for reporting.
        """
        if self.q_clip is not None:
            arm = A_test.astype(np.intp)
            q_obs = q[np.arange(len(A_test)), arm]
            self._clip_hits_ += int(np.sum(np.abs(q_obs) > abs(self.q_clip)))
            self._clip_total_ += len(q_obs)

        tuned = bool((self.bridge_params or {}).get("lam_grid"))
        if tuned and hasattr(bridge, "lam_h_") and hasattr(bridge, "lam_q_"):
            self._lam_h_choices_.append(float(bridge.lam_h_))
            self._lam_q_choices_.append(float(bridge.lam_q_))

    def _folds(self, A: np.ndarray):
        """(train, test) index pairs. ``n_splits=1`` means no cross-fitting.

        The degenerate split is supported for debugging and small experiments,
        although it is not a sensible way to run the estimator because the
        scores are then evaluated in-sample.
        """
        n = len(A)
        if self.n_splits == 1:
            everything = np.arange(n)
            return [(everything, everything)]

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        return list(skf.split(np.zeros(n), A))

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X, *, W=None, Z=None) -> np.ndarray:
        """CATE from the first configured final model.

        W and Z are accepted and ignored: the P-learner's prediction is a
        function of x alone, which is why its estimand tag is ``cate(x)``.
        """
        if not self.final_models_:
            raise ValueError(
                "PLearner was fit with no final models; nothing to predict."
            )
        return next(iter(self.predict_variants(X).values()))

    def predict_variants(self, X, *, W=None, Z=None) -> dict[str, np.ndarray]:
        X = _as_2d(X, "X")
        return {
            name: np.asarray(model.predict(X), dtype=float)
            for name, model in self.final_models_.items()
        }

    def predict_ate(self, X) -> np.ndarray:
        """The single proximal ATE, broadcast to one value per row."""
        return np.full(_as_2d(X, "X").shape[0], self.ate_, dtype=float)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, float]:
        """How often ``q_clip`` bound, and where a tuned kernel bridge landed.

        - ``q_clip_frac``: fraction of held-out, observed-arm ``q`` evaluations
          across all folds whose magnitude exceeded ``q_clip`` before clipping.
          Close to 0 means the cap is a safety net; close to 1 means it is
          doing the estimating instead of the bridge. Only emitted when
          ``q_clip`` is set.
        - ``lam_h_frac@<v>`` / ``lam_q_frac@<v>``: for :class:`KernelBridge`
          tuned with ``lam_grid``, the fraction of cross-fitting folds
          (``n_splits``) whose held-out moment-loss search landed on grid
          value ``v`` — one pair of keys per **configured grid value**
          (explicit ``0.0`` for a value no fold picked), so a distribution
          piled on the smallest or largest value flags a grid that does not
          bracket the optimum (see :class:`KernelBridge`, "prefer lam_grid to
          the defaults"). Only emitted when the bridge was actually tuned.
          Emitting the full grid rather than only the values folds happened
          to pick is load-bearing, not cosmetic: `run_method` accumulates
          diagnostics across episodes by key, padding an episode that never
          emitted a key with NaN rather than 0. A grid value silently
          restricted to "values chosen this episode" would make each key's
          across-episode NaN-mean run over a different subset of episodes,
          so the printed fractions would no longer sum to 1 — the fractions
          are exact per episode, but averaging fractions computed over
          different populations is not.
        """
        diag: dict[str, float] = {}

        if self.q_clip is not None and getattr(self, "_clip_total_", 0):
            diag["q_clip_frac"] = self._clip_hits_ / self._clip_total_

        lam_h = getattr(self, "_lam_h_choices_", [])
        lam_q = getattr(self, "_lam_q_choices_", [])
        grid = sorted(float(v) for v in (self.bridge_params or {}).get("lam_grid", []))
        if lam_h and grid:
            n_folds = len(lam_h)
            for lam in grid:
                diag[f"lam_h_frac@{lam:g}"] = lam_h.count(lam) / n_folds
                diag[f"lam_q_frac@{lam:g}"] = lam_q.count(lam) / n_folds

        # GridSearchCV is intentionally detected by capability rather than an
        # import: PLearner accepts any sklearn-compatible final regressor, and
        # the driver should not depend on the registry's concrete CV wrapper.
        # Candidate index remains numeric even when a grid includes values such
        # as max_depth=None; the (negative) validation MSE makes it possible to
        # inspect whether the selected final model was genuinely predictive.
        for name, model in getattr(self, "final_models_", {}).items():
            if hasattr(model, "best_index_") and hasattr(model, "best_score_"):
                diag[f"base_cv/{name}/best_index"] = float(model.best_index_)
                diag[f"base_cv/{name}/mse"] = -float(model.best_score_)
                for param, value in model.best_params_.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        param = param.removeprefix("model__")
                        diag[f"base_cv/{name}/{param}"] = float(value)

        return diag
