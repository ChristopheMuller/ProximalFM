"""
PMMR — Proximal Maximum Moment Restriction
==========================================
Single-stage kernel ridge estimator for proximal causal inference. Learns an
outcome bridge function h(A, W) by minimising a kernelised maximum-moment
restriction with an RKHS-norm penalty.

Reference: Mastouri et al. "Proximal Causal Learning with Kernels" (ICML 2021)

Let, over the n training points, with product kernels over (A, Z, X) and (A, W, X):
    Φ = K_A ⊙ K_Z ⊙ K_X      (instrument-side feature gram)
    L = K_A ⊙ K_W ⊙ K_X      (bridge-side feature gram)

K_X is shared by both grams; K_Z alone carries an extra widening factor
(kernel_scale_z). See PMMRConfig.

The bridge coefficients solve the regularised normal equations:
    α = (L Φ L + λ n² L + ε n² I)⁻¹ L Φ Y

where λ is the paper's objective-level regularisation parameter and ε is a
fixed numerical jitter. Parameterising the objective by λ keeps its meaning
unchanged when tuning on a split and refitting on the full context.

For binary A this system is block diagonal in the arms — k_A is an indicator and
multiplies both grams — so `fit` assembles and solves it one arm at a time. That
is an exact decomposition of the equation above, not a second estimator; see
`PMMREstimator.fit`.

Prediction of E[Y | do(a)] marginalises over the *joint* empirical (W, X):
    ŷ(a) = αᵀ [ k_A(A_train, a) ⊙ wx̄ ]
    where wx̄ = mean_row(K_W ⊙ K_X)

The mean is taken over the elementwise product, not of each gram separately —
E_{W,X}[·] under the joint empirical measure is (1/n) Σ_j k_W(w_i,w_j) k_X(x_i,x_j),
which is not mean_row(K_W) ⊙ mean_row(K_X) unless W ⊥ X in sample.

CATE = ŷ(1) - ŷ(0)  (for binary treatment).

All computation is pure numpy/scipy — no JAX dependency.

Tuning entry point: PMMRConfig, whose every field except `jitter` may appear in
the wrapper's `tune.grid` (benchmarks/methods/__init__.py). PMMR is *not* a
fixed-hyperparameter method — the reference grid-searches λ and selects on a
held-out V-statistic, which is what `PMMREstimator.score` reproduces — so the
defaults here are starting points rather than tuned values. `lam1` is the
paper's λ directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.linalg import solve, LinAlgError

from benchmarks.methods.kernels import ColumnWiseGaussianKernel, as_2d, BinaryKernel


#: PMMRConfig fields that enter *only* the assembled left-hand side, never a
#: kernel gram — so a fitted system can be re-solved at new values of these
#: without recomputing anything. `_PreparedSystem.solve` accepts exactly these,
#: and the benchmark wrapper validates `tune.sweep` against the same set, so the
#: two cannot drift apart.
SOLVE_ONLY_PARAMS = frozenset({"lam1"})


# The reference implementation sets the instrument-side bandwidth to
# `median / 0.6` while leaving the bridge side at the plain median heuristic.
# Their kernel is exp(-d²/(2σ²)) and ours is exp(-d²/(2·median²·scale)), so
# their divisor of 0.6 on σ is a scale of 1/0.6² here.
_REFERENCE_Z_WIDENING = 1.0 / 0.6**2  # ≈ 2.778


def _solve_pd(
    lhs: np.ndarray, rhs: np.ndarray, jitter: float, context: str = ""
) -> np.ndarray:
    """
    Solve the symmetric positive-definite system ``lhs @ x = rhs`` by Cholesky,
    with a fixed diagonal jitter.

    The released binary-treatment implementation uses ``JITTER_LW = 1e-8`` in
    its normalized system. The caller scales that value by n² when assembling
    our equivalent unnormalized system. We apply it on every solve so the full
    matrix and its exact per-arm decomposition remain identical: conditionally
    jittering only the blocks whose Cholesky fails would define a different
    matrix from conditionally jittering the assembled system. The jitter is
    numerical insurance, not a statistical parameter, and is never tuned.

    ``lhs`` is freshly allocated by ``_PreparedSystem.solve_into`` for this
    one solve, so adding the diagonal in place is safe.  In particular, it
    cannot mutate the prepared ``LPhiL`` matrix shared by a lambda sweep.
    This avoids allocating a dense identity and its dense sum at every arm.
    """
    where = f" {context}" if context else ""
    try:
        lhs[np.diag_indices_from(lhs)] += jitter
        return solve(lhs, rhs, assume_a="pos")
    except LinAlgError as exc:
        raise LinAlgError(
            f"PMMR normal equations{where} remained non-positive-definite "
            f"after adding the fixed {jitter:g} diagonal jitter. Widen "
            "kernel_scale or check for duplicate/constant proxy columns."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PMMRConfig:
    # ── Kernel ────────────────────────────────────────────────────────────────
    # Global multiplicative rescaling of the median-heuristic bandwidth σ.
    # Values < 1 → narrower kernel (more local); > 1 → wider (smoother).
    kernel_scale: float = 1.0

    # Extra widening applied to the treatment-inducing proxy kernel K_Z alone —
    # the factor that distinguishes Φ from L. Composes with kernel_scale, so
    # Z's bandwidth is kernel_scale * kernel_scale_z and everything else's is
    # kernel_scale. This has to be its own knob rather than a different default
    # on a shared one: the two grams want different bandwidths, and no single
    # scale can express that.
    #
    # Scope note: X enters *both* Φ and L, and we deliberately give it one
    # shared bandwidth rather than widening its Φ-side copy. The reference
    # cannot settle this — its experiments set X = ∅ throughout ("we assume
    # that X is a null set"), so the 0.6 divisor was only ever applied to
    # (A, Z). Widening X on the Φ side would make the objective less sensitive
    # to moment violations that vary with X, which is the wrong trade when X is
    # exactly what CATE conditions on. The competing consideration — a product
    # kernel decays faster the more factors it has, so Φ drifts toward I as d_X
    # grows — is better handled by tuning kernel_scale globally.
    #
    # Why it is not a nuisance parameter. k decides which moment violations the
    # MMR objective can see, and both limits are degenerate:
    #   narrow → Φ → I    → the objective collapses to Σᵢ dᵢ², i.e. plain kernel
    #                       ridge of Y on (A, W, X) — the paper's KernelRidge-W
    #                       baseline. The proximal correction is gone; how much
    #                       bias that costs depends on how good a proxy W is.
    #   wide   → Φ → 11ᵀ  → the objective collapses to (mean d)², a single
    #                       moment, and h is badly under-identified.
    # Measured, n=1500, true ATE 2.0, naive contrast 4.27, lam1=1e-3, over
    # kernel_scale_z = 1e-4 … 1e6:
    #   informative W (proxy noise 0.5; KernelRidge-W = 2.74)
    #       2.45  2.26  2.20  2.17  2.17  2.14  2.30  2.13  3.82
    #   noisy W       (proxy noise 1.5; KernelRidge-W = 3.74)
    #       3.58  3.35  2.89  2.44  2.27  2.17  2.15  2.55  4.09
    # The wide end is the sharper failure — it walks back to the naive contrast.
    # The narrow end degrades gracefully with an informative W and badly with a
    # noisy one. The default sits in a broad flat basin in both cases.
    #
    # The default reproduces the reference implementation, which hardcodes
    # `ak = median(AZ) / 0.6` (`W_kern_divisor`) against a plain median
    # heuristic on the bridge side — a deliberate step away from the Φ ≈ I
    # cliff. Note their Appendix D describes a plain median heuristic on *both*
    # sides; only the released code widens Φ. We follow the code.
    kernel_scale_z: float = _REFERENCE_Z_WIDENING

    # ── Regularisation ────────────────────────────────────────────────────────
    # lam1 is the paper's objective-level RKHS penalty λ, not the coefficient
    # multiplying L in the unnormalised normal equations. For a fit on s rows,
    # the latter is λ s². Keeping lam1 unchanged between a tuning split and a
    # full-data refit therefore keeps the statistical objective unchanged.
    # The paper searches λ over [4.9e-6, 0.25] using a held-out V-statistic.
    lam1: float = 1e-3

    # Fixed jitter matching the released binary-treatment implementation's
    # JITTER_LW value. It is deliberately not a tuning parameter.
    jitter: float = 1e-8

    # No seed: fit() is fully deterministic. Do not add a global
    # np.random.seed() call here — estimators share a process in the episode
    # pool, so it would reseed whatever runs next.


# ─────────────────────────────────────────────────────────────────────────────
# The lam-free half of a fit
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ArmBlock:
    """One arm's share of the normal equations, with lam1 not yet applied.

    ``LPhiL`` and ``LPhiY`` are the two cubic products; ``L`` is kept because it
    multiplies ``lam1``. Together they are everything the solve consumes.
    """

    arm: float
    idx: np.ndarray  # (n_a,) rows of the full sample in this arm
    L: np.ndarray  # (n_a, n_a)
    LPhiL: np.ndarray  # (n_a, n_a)
    LPhiY: np.ndarray  # (n_a,)


class _PreparedSystem:
    """Everything a PMMR fit needs *before* lam1 is chosen.

    The split is not cosmetic. Writing the per-arm system out,

        lhs = L Φ L + λ n² L + ε n² I        rhs = L Φ Y

    the regulariser and fixed jitter enter only in the assembly — ``L Φ L``
    and ``L Φ Y`` are free of both, and they are the whole cost of a fit (the
    grams plus two cubic matmuls, ~1.5 s of ~1.7 s at n = 2048, against ~0.2 s
    for the Cholesky). So a search over λ costs one prepare and one *solve* per
    candidate rather than one fit per candidate, which is what makes an
    exhaustive λ sweep affordable inside a per-episode tuner. Same reasoning,
    and the same shape, as ``plearner._MomentSystem``.

    What is *not* free is anything touching a bandwidth: ``kernel_scale`` and
    ``kernel_scale_z`` change the grams and therefore need a new prepare. That
    is why :meth:`solve` accepts only :data:`SOLVE_ONLY_PARAMS` and raises on
    anything else — silently honouring a ``kernel_scale`` override would return
    a model whose ``cfg`` disagrees with the grams it was actually built from,
    which is a wrong answer rather than an error.

    Built by :meth:`PMMREstimator.prepare`; not constructed directly.
    """

    def __init__(
        self,
        cfg: PMMRConfig,
        n: int,
        blocks: list[_ArmBlock],
        *,
        wx_mean: np.ndarray,
        K_W_train: np.ndarray,
        A_train: np.ndarray,
        W_train: np.ndarray,
        X_train: np.ndarray,
        k_A,
        k_W,
        k_X,
    ) -> None:
        self.cfg = cfg
        self.n = n
        self.blocks = blocks
        self._wx_mean = wx_mean
        self._K_W_train = K_W_train
        self._A_train = A_train
        self._W_train = W_train
        self._X_train = X_train
        self._k_A = k_A
        self._k_W = k_W
        self._k_X = k_X

    def solve_into(self, est: "PMMREstimator") -> "PMMREstimator":
        """Solve at ``est.cfg``'s lam1 and write the fit into ``est``.

        The predict-side state is *shared*, not copied: several estimators
        produced from one prepared system point at the same grams. They are
        read-only after this call, so that is safe and is what keeps a sweep's
        memory flat in the number of candidates.
        """
        cfg = est.cfg
        alpha = np.zeros(self.n, dtype=float)
        for b in self.blocks:
            lhs = b.LPhiL + cfg.lam1 * self.n**2 * b.L
            alpha[b.idx] = _solve_pd(
                lhs,
                b.LPhiY,
                cfg.jitter * self.n**2,
                context=f"(arm A={b.arm:.0f})",
            )

        est._alpha = alpha
        est._wx_mean = self._wx_mean
        est._K_W_train = self._K_W_train
        est._A_train = self._A_train
        est._W_train = self._W_train
        est._X_train = self._X_train
        est._k_A = self._k_A
        est._k_W = self._k_W
        est._k_X = self._k_X
        return est

    def solve(self, **lams: float) -> "PMMREstimator":
        """A fitted estimator at this lam1, sharing this system's state.

        Keyword arguments must be drawn from :data:`SOLVE_ONLY_PARAMS`; omitted
        ones keep the value the system was prepared with.
        """
        unknown = set(lams) - SOLVE_ONLY_PARAMS
        if unknown:
            raise ValueError(
                f"_PreparedSystem.solve got {sorted(unknown)}, which "
                f"change the kernel grams and so need a fresh prepare(). "
                f"Only {sorted(SOLVE_ONLY_PARAMS)} can be re-solved."
            )
        return self.solve_into(PMMREstimator(replace(self.cfg, **lams)))

    def bridge_gram(self, A: np.ndarray, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Bridge-kernel Gram from this fit sample to another row set.

        All estimators obtained by :meth:`solve` share the same fitted kernels
        and training rows.  A lambda sweep can therefore build this matrix once
        for its validation rows and reuse it for every candidate alpha.
        """
        A = as_2d(A)
        W = as_2d(W)
        X = as_2d(X)

        K = self._k_A(self._A_train, A) * self._k_W(self._W_train, W)
        if self._k_X is not None:
            K *= self._k_X(self._X_train, X)
        return K


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class PMMREstimator:
    """
    Sklearn-style CATE estimator implementing PMMR.

    Usage::

        cfg = PMMRConfig(lam1=1e-3)
        est = PMMREstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: PMMRConfig | None = None) -> None:
        self.cfg = cfg or PMMRConfig()
        # Learnt quantities stored after fit()
        self._alpha: np.ndarray | None = None  # (n,)
        self._wx_mean: np.ndarray | None = (
            None  # (n,) — joint empirical mean embedding of (W, X)
        )
        self._A_train: np.ndarray | None = None  # (n, d_A)
        self._W_train: np.ndarray | None = None  # (n, d_W)
        self._X_train: np.ndarray | None = None  # (n, d_X)
        self._k_A: BinaryKernel | ColumnWiseGaussianKernel | None = None
        self._k_W: ColumnWiseGaussianKernel | None = None
        self._k_X: ColumnWiseGaussianKernel | None = None
        self._K_W_train: np.ndarray | None = None  # (n, n) train-train W gram

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "PMMREstimator":
        """
        Fit the PMMR bridge function.

        Args:
            X: Covariates,                shape (n, d_X).
            A: Binary treatment encoded as 0/1, shape (n,) or (n, 1).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).

        This is exactly ``prepare(...).solve_into(self)`` — one code path, so a
        single fit and a swept one cannot diverge. See :meth:`prepare` for the
        split and :class:`_PreparedSystem` for why it is worth making.
        """
        return self.prepare(X, A, Y, W=W, Z=Z).solve_into(self)

    def prepare(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> _PreparedSystem:
        """
        Build the kernel grams and the two cubic products, stopping short of the
        solve — everything that does not depend on ``lam1``.

        Call this directly when several λ values are to be tried on one sample;
        :meth:`_PreparedSystem.solve` then costs one Cholesky each. For a single
        fit, use :meth:`fit`, which wraps it.

        Args are as :meth:`fit`.

        The normal equations are assembled and solved **once per arm**, not once
        over all n rows. That is exact, not an approximation: ``k_A`` is an
        indicator, so it is identically zero across arms, and it multiplies
        *both* grams — hence ``Φ``, ``L`` and ``L Φ L`` are all block diagonal
        in the arms and the system decouples into two independent per-arm MMR
        regressions. Pinned against the assembled n×n solve by
        ``tests/test_bench_pmmr_blocks.py``.

        Why bother: the solve and the two matmuls are the whole cost of a fit
        (~1.7 s of ~2.0 s at n = 2048), they are cubic, and two halves cost a
        quarter of one whole. Measured 1.47x end to end, with peak memory down
        by more than that — ``Φ`` is now built per block and ``K_A`` never
        materialises at all.

        Note that ``n`` in the two penalty terms stays the **full** row count in
        every block (applied in :meth:`_PreparedSystem.solve_into`). The blocks
        are a decomposition of one system, not two smaller problems, so
        ``lam1`` keeps meaning exactly what it meant before; using a per-arm
        count here would silently change the regularisation.
        """
        cfg = self.cfg

        A = as_2d(A)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        W = as_2d(W)
        Z = as_2d(Z)
        X = as_2d(X)
        n = len(Y)

        # ── Fit kernel bandwidths (median heuristic) ───────────────────────
        # Everything uses kernel_scale; the proxy Z alone is widened further by
        # kernel_scale_z. K_A has no bandwidth (indicator) and K_X is shared by
        # both grams, so Z is the only factor that differs between Φ and L.
        # k_A also validates that A is scalar and encoded in {0, 1}, which the
        # per-arm decomposition below relies on.
        k_A = BinaryKernel().fit(A)
        k_Z = ColumnWiseGaussianKernel().fit(
            Z, scale=cfg.kernel_scale * cfg.kernel_scale_z
        )
        k_W = ColumnWiseGaussianKernel().fit(W, scale=cfg.kernel_scale)

        # Bandwidths are fitted on the *full* sample, so slicing a gram to an arm
        # below carries the same numbers the assembled matrix would have.
        K_W = k_W(W, W)

        if X.shape[1] > 0:
            k_X = ColumnWiseGaussianKernel().fit(X, scale=cfg.kernel_scale)
            K_X = k_X(X, X)
        else:
            k_X = None
            K_X = None

        # Empirical joint mean embedding of (W, X), used by the marginal ATE
        # path. Taken here so the n x n product stays transient.
        wx_mean = (K_W if K_X is None else K_W * K_X).mean(axis=1)

        # Assemble the normal equations arm by arm — see the docstring. Within
        # an arm k_A is identically 1, so it drops out of both grams; X is
        # shared, so only K_Z carries the extra widening.
        blocks: list[_ArmBlock] = []
        arm_of = A.ravel()
        for a in (0.0, 1.0):
            idx = np.flatnonzero(arm_of == a)
            if idx.size == 0:
                continue  # an arm with no rows contributes no coefficients
            sub = np.ix_(idx, idx)

            # Both are fresh arrays — fancy indexing copies, and k_Z returns a
            # new gram — so the in-place products cannot reach back into K_W,
            # which `predict` still needs whole.
            L_a = K_W[sub]
            Phi_a = k_Z(Z[idx], Z[idx])
            if K_X is not None:
                K_X_a = K_X[sub]  # shared by both grams; slice it once
                L_a *= K_X_a
                Phi_a *= K_X_a

            LPhi = L_a @ Phi_a
            blocks.append(
                _ArmBlock(arm=a, idx=idx, L=L_a, LPhiL=LPhi @ L_a, LPhiY=LPhi @ Y[idx])
            )

        return _PreparedSystem(
            cfg,
            n,
            blocks,
            wx_mean=wx_mean,
            K_W_train=K_W,
            A_train=A,
            W_train=W,
            X_train=X,
            k_A=k_A,
            k_W=k_W,
            k_X=k_X,
        )

    def score(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
        kernel_scale: float = 1.0,
        loss: Literal["v_stat", "u_stat"] = "v_stat",
        moment_kernel: np.ndarray | None = None,
        bridge_kernel: np.ndarray | None = None,
    ) -> float:
        """Held-out MMR statistic (lower is better) for model selection.

        This is the quantity the paper selects on: "the metric we use for
        hyperparameter selection is the empirical estimate of the V-statistic,
        R̂_V ... over a held-out validation set" (Appendix D). With residuals
        d_i = h(a_i,w_i,x_i) − y_i and K the instrument-side gram over the
        held-out (A, Z, X),

            R̂_V = dᵀ K d / n²            (keeps the diagonal; what PMMR trains on)
            R̂_U = dᵀ (K − diagK) d / n(n−1)   (drops it; unbiased but noisier)

        V is the default because it is the objective PMMR is fitted to — Φ in
        the normal equations includes its own diagonal — so the ranking is on
        the criterion the candidates were trained for.

        ``kernel_scale`` is deliberately **fixed and independent of the fitted
        model's own bandwidths**, exactly as in
        :meth:`NMMREstimator.score`. R̂ is a squared norm in H_k, so its scale
        is set by k; letting each candidate score itself with its own k would
        rank candidates by how wide their kernel is rather than by how well
        their bridge fits, and a search over ``kernel_scale``/
        ``kernel_scale_z`` would be won by whichever value shrinks the norm.
        Because the split is fixed within an episode, every candidate is then
        scored by the identical functional.

        Returns NaN for fewer than two held-out rows (and for ``u_stat``, which
        needs an off-diagonal pair to be defined at all).
        """
        self._check_fitted()

        X = as_2d(X)
        A = as_2d(A)
        W = as_2d(W)
        Z = as_2d(Z)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        n = len(Y)
        if n < 2:
            return float("nan")

        if bridge_kernel is None:
            d = self.predict_bridge(A, W=W, X=X) - Y
        else:
            bridge_kernel = np.asarray(bridge_kernel, dtype=float)
            expected_shape = (self._alpha.shape[0], n)
            if bridge_kernel.shape != expected_shape:
                raise ValueError(
                    "bridge_kernel must have shape (n_train, n_validation); "
                    f"got {bridge_kernel.shape}, expected {expected_shape}."
                )
            d = (self._alpha @ bridge_kernel).ravel() - Y

        # Fixed reference kernel over (A, Z, X) — see the docstring.  The
        # wrapper builds it once per validation split and shares it across all
        # candidates; accepting it here keeps the public score calculation and
        # the cached tuning calculation bit-for-bit the same.
        if moment_kernel is None:
            K = self.validation_moment_kernel(X, A, Z, kernel_scale=kernel_scale)
        else:
            K = np.asarray(moment_kernel, dtype=float)
            if K.shape != (n, n):
                raise ValueError(
                    "moment_kernel must have shape (n_validation, n_validation); "
                    f"got {K.shape} for n_validation={n}."
                )

        if loss == "u_stat":
            K = K - np.diag(np.diag(K))
            return float(d @ K @ d / (n * (n - 1)))
        if loss != "v_stat":
            raise ValueError(f"Unknown loss {loss!r}; choose 'v_stat' or 'u_stat'.")
        return float(d @ K @ d / (n * n))

    @staticmethod
    def validation_moment_kernel(
        X: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
        *,
        kernel_scale: float = 1.0,
    ) -> np.ndarray:
        """Reference V-statistic kernel over a fixed validation split.

        This intentionally has no dependence on a fitted PMMR model.  Tuning
        builds it once and shares it across candidates so bandwidth candidates
        are judged by one common moment norm, rather than paying to rebuild an
        identical matrix for every value of ``lam1``.
        """
        X = as_2d(X)
        A = as_2d(A)
        Z = as_2d(Z)

        K = BinaryKernel().fit(A)(A, A)
        K *= ColumnWiseGaussianKernel().fit(Z, scale=kernel_scale)(Z, Z)
        if X.shape[1] > 0:
            K *= ColumnWiseGaussianKernel().fit(X, scale=kernel_scale)(X, X)
        return K

    def predict(
        self,
        X: np.ndarray,
        *,
        W: np.ndarray | None = None,
        strategy: Literal[
            "marginal", "paired", "independent", "conditional"
        ] = "marginal",
    ) -> np.ndarray:
        """
        Predict a binary treatment contrast.

        strategy="marginal":
            Return the global ATE repeated for each test row.

        strategy="paired":
            Requires W. Return h(1,W_i,X_i) - h(0,W_i,X_i).

        strategy="independent":
            Return mean_j h(1,W_j,X_i) - h(0,W_j,X_i),
            averaging over marginal training W.

        strategy="conditional":
            Return kernel-smoothed approximation to
            E[h(1,W,X_i)-h(0,W,X_i) | X_i].
        """
        self._check_fitted()

        X = as_2d(X)

        if strategy == "marginal":
            return self.predict_mean_outcome(
                1.0, n=X.shape[0]
            ) - self.predict_mean_outcome(0.0, n=X.shape[0])

        if strategy == "paired":
            if W is None:
                raise ValueError("strategy='paired' requires W.")
            return self._predict_effect_paired_w(X, as_2d(W))

        if strategy == "independent":
            return self._predict_effect_independent_w(X)

        if strategy == "conditional":
            return self._predict_effect_conditional_w(X)

        raise ValueError(f"Unknown prediction strategy: {strategy!r}")

    def _predict_effect_paired_w(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        return self.predict_bridge(1.0, W=W, X=X) - self.predict_bridge(0.0, W=W, X=X)

    def predict_bridge(
        self,
        a: float | np.ndarray,
        W: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Evaluate h(a, W, X) at paired test points.
        """
        self._check_fitted()

        W = as_2d(W)
        n_test = W.shape[0]

        if np.isscalar(a):
            A_test = np.full((n_test, self._A_train.shape[1]), float(a))
        else:
            A_test = as_2d(a)

        K_A = self._k_A(self._A_train, A_test)
        K_W = self._k_W(self._W_train, W)

        if self._k_X is not None:
            if X is None:
                raise ValueError("X must be provided because the model was fit with X.")
            X = as_2d(X)
            K_X = self._k_X(self._X_train, X)
        else:
            K_X = 1.0

        K_test = K_A * K_W * K_X
        return (self._alpha @ K_test).ravel()

    def predict_mean_outcome(self, a: float | np.ndarray, n: int = 1) -> np.ndarray:
        self._check_fitted()

        if np.isscalar(a):
            a_arr = np.full((n, self._A_train.shape[1]), float(a))
        else:
            a_arr = as_2d(a)

        K_A = self._k_A(self._A_train, a_arr)
        K_test = K_A * self._wx_mean[:, None]

        return (self._alpha @ K_test).ravel()

    def _predict_mean_outcome_independent_w(
        self, a: float, X: np.ndarray
    ) -> np.ndarray:
        self._check_fitted()
        X = as_2d(X)
        n_test = X.shape[0]

        A_test = np.full((n_test, self._A_train.shape[1]), float(a))

        K_A = self._k_A(self._A_train, A_test)

        if self._k_X is not None:
            K_X = self._k_X(self._X_train, X)
        else:
            K_X = np.ones_like(K_A)

        # mean over marginal W_train
        w_mean = self._K_W_train.mean(axis=1)

        K_test = K_A * K_X * w_mean[:, None]
        return (self._alpha @ K_test).ravel()

    def _predict_effect_independent_w(self, X: np.ndarray) -> np.ndarray:
        return self._predict_mean_outcome_independent_w(
            1.0, X
        ) - self._predict_mean_outcome_independent_w(0.0, X)

    def _predict_mean_outcome_conditional_w(
        self,
        a: float,
        X: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        self._check_fitted()

        if self._k_X is None:
            # No X was used in training, so conditional and marginal collapse.
            return self.predict_mean_outcome(a, n=X.shape[0])

        X = as_2d(X)
        n_test = X.shape[0]

        A_test = np.full((n_test, self._A_train.shape[1]), float(a))

        # K_X_train_test[j, t] = k_X(X_j, x_t)
        K_X_train_test = self._k_X(self._X_train, X)

        # Kernel weights q_j(x_t) approx P(W_j | X=x_t)
        weights = K_X_train_test / (K_X_train_test.sum(axis=0, keepdims=True) + eps)

        # cond_w_mean[i, t] = sum_j k_W(W_i, W_j) q_j(x_t)
        cond_w_mean = self._K_W_train @ weights

        K_A = self._k_A(self._A_train, A_test)

        # h(a, W, x) also contains k_X(X_i, x)
        K_test = K_A * K_X_train_test * cond_w_mean

        return (self._alpha @ K_test).ravel()

    def _predict_effect_conditional_w(self, X: np.ndarray) -> np.ndarray:
        return self._predict_mean_outcome_conditional_w(
            1.0, X
        ) - self._predict_mean_outcome_conditional_w(0.0, X)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._alpha is None:
            raise RuntimeError("Call fit() before predict().")
