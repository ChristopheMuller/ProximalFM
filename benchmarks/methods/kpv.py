"""
KPV — Kernel Proxy Variable (two-stage kernel regression)
==========================================================
Non-parametric proximal causal estimator that learns a bridge function in two
regularised kernel regression stages.

Reference: Mastouri et al. "Proximal Causal Learning with Kernels" (ICML 2021)

Pipeline:
  Stage 1 (data split D1):
      Γ = (K_ZAX + λ₁ n₁ I)⁻¹ K_ZAX(D1, D2)
      Bridges outcome proxy W to the (A, Z, X) space.

  Stage 2 (data split D2):
      Σ = (Γᵀ K_W Γ) ⊙ K_A(D2, D2) ⊙ K_X(D2, D2)
      α = Γ ⊙ [(Σ + λ₂ n₂ I)⁻¹ Y₂]     (n₁ × n₂)

  Prediction of E[Y | do(a)] marginalises over the *joint* empirical (X, W):
      β̂(a) = (1/n) Σ_ijk α_ij k_A(ã_j, a) k_X(x̃_j, x_k) k_W(w_i, w_k)
  which is the paper's estimator of the causal effect. The mean is taken over
  the elementwise product, not of each gram separately — mean_k[k_X · k_W] is
  not mean_k[k_X] · mean_k[k_W] unless X ⊥ W in sample. (The released reference
  code factorises the two, i.e. silently assumes independence.)

  CATE = β̂(1) − β̂(0)  (for binary treatment)

For binary A the whole pipeline is block diagonal in the arms — k_A is an
indicator and multiplies *every* gram, K_ZAX on both splits and Σ — so both
stages, both LOOCV criteria, and α itself decouple by arm, and `_fit_stages`
assembles them one arm at a time. That is an exact decomposition of the
equations above, not a second estimator; see `KPVEstimator._fit_stages` and
`tests/test_bench_kpv_blocks.py`.

All computation is pure numpy/scipy — no JAX dependency.

Hyperparameters
───────────────
λ₁ and λ₂ are selected *inside* a fit by the paper's own closed-form LOOCV
(eqs. stage1_reg / stage2_reg), so they are not search axes by default. What the
paper leaves undetermined is the kernel length-scale: it prescribes the median
heuristic as an *initialisation* and then says to pick "the optimal length-scale
from a narrow range around this level" without naming a criterion, and the
released code does not implement even that. `kernel_scale` is therefore an
ordinary hyperparameter searched from outside, through the wrapper's `tune:`
block, ranked on `KPVEstimator.loocv_score_` — the Stage-2 LOOCV loss, whose
target Y₂ is fixed across bandwidths and so gives comparable scores. TUNABLE_PARAMS
is the whitelist and carries the reasoning for every field left out of it.

Read that score for what it is: Stage 2 fits G_h(a,x,z) = E[Y | a,x,z], the
*reduced-form* regression. A bandwidth that predicts Y well need not identify
the bridge h well, and nothing here scores CATE. This is the open problem the
paper's own conclusion names, so expect the criterion to separate candidates
weakly — check `tune_score_range` in the wrapper's diagnostics before reading a
selected value as an optimum.

Tuning entry point: KPVConfig — every knob lives there with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve

from benchmarks.methods.kernels import ColumnWiseGaussianKernel, as_2d, BinaryKernel


#: The only :class:`KPVConfig` fields a `tune:` block may search. The benchmark
#: wrapper validates `tune.grid` against this set; everything else in the
#: dataclass is excluded on purpose, and for three different reasons:
#:
#: - ``split_ratio`` and ``seed`` are **unscoreable**. `_loocv_stage2_block`
#:   drops the ``1/m₂`` prefactor, so its scores are not on a common scale once
#:   ``m₂`` changes — a search over `split_ratio` would rank candidates by how
#:   many terms their sum has. `seed` is scoreable but selecting *which rows
#:   land in D1* on the fit criterion is fitting the split, not the model.
#: - ``lam1_min``/``lam1_max``/``n_lam1_search`` (and λ₂'s) are the *grid* the
#:   paper's LOOCV searches over, not statistics. Widening or refining a grid
#:   can only lower the LOOCV minimum, so a search over them is monotone and
#:   would always return the widest, finest option offered.
#: - ``lam1``/``lam2`` are in the set because setting either *fixes* it and
#:   bypasses the LOOCV. That is a legitimate thing to compare against, and the
#:   score stays comparable because Y₂ is unchanged.
TUNABLE_PARAMS = frozenset(
    {
        "kernel_scale",
        "kernel_scale_z",
        "kernel_scale_w",
        "kernel_scale_x",
        "lam1",
        "lam2",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class KPVConfig:
    # ── Data split ────────────────────────────────────────────────────────────
    # Fraction of training data used in Stage 1 (remainder goes to Stage 2).
    split_ratio: float = 0.5

    # ── Kernel length-scales ──────────────────────────────────────────────────
    # Multiplicative rescalings of the median-heuristic bandwidth σ², in the
    # same global × per-variable spelling PMMR uses (see PMMRConfig): the
    # effective scale of Z's kernel is `kernel_scale * kernel_scale_z`, and
    # likewise for W and X. Values < 1 → narrower kernel (more local); > 1 →
    # wider (smoother). All-1.0 is the plain per-dimension median heuristic.
    #
    # Two levels rather than one because they answer different questions.
    # `kernel_scale` moves every bandwidth together — "is the median heuristic
    # too local for this sample size?" — while the per-variable factors change
    # the *shape* of the joint kernel, which is what actually trades Stage 1's
    # fit of W against Stage 2's fit of Y. A search over the three relatives at
    # a fixed global covers the space without redundancy; searching all four
    # over-parameterises it, since only the products enter.
    #
    # A has no entry: its kernel is the Kronecker delta (BinaryKernel), which
    # has no bandwidth to scale.
    #
    # These are genuine hyperparameters with no procedure attached: the paper
    # says to pick "the optimal length-scale from a narrow range around this
    # level" (Experiments_append.tex §D.2.1) without naming a criterion, and the
    # released reference never searches it at all — its scale loop is
    # `np.logspace(mn, mx, 1)`, a one-point grid that always returns the lower
    # bound. Searching them is therefore a `tune:` block on the benchmark
    # wrapper, not something this estimator does to itself; see TUNABLE_PARAMS.
    kernel_scale: float = 1.0
    kernel_scale_z: float = 1.0
    kernel_scale_w: float = 1.0
    # Inert when the episode has no covariates (d_X = 0): k_X is never fitted,
    # so every value of this gives the identical fit. Searching it on a
    # covariate-free set is not wrong, just empty — `tune_score_range` collapses
    # to 0 and says so.
    kernel_scale_x: float = 1.0

    # ── Stage 1 regularisation ────────────────────────────────────────────────
    # Set lam1 to a fixed value to skip cross-validation.
    # Leave as None to tune automatically over the log-grid [lam1_min, lam1_max].
    lam1: Optional[float] = None
    lam1_min: float = 1e-5
    lam1_max: float = 1e1
    n_lam1_search: int = 10

    # ── Stage 2 regularisation ────────────────────────────────────────────────
    # Same convention as Stage 1.
    lam2: Optional[float] = None
    lam2_min: float = 1e-5
    lam2_max: float = 1e1
    n_lam2_search: int = 10

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 1234


# ─────────────────────────────────────────────────────────────────────────────
# LOOCV helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ridge_influence(K: np.ndarray, lam: float, n_full: int) -> tuple:
    """The LOO influence matrix H and the reciprocal of its diagonal.

    H = I − K (K + cI)⁻¹ = c (K + cI)⁻¹ with c = λ·n_full, by the standard ridge
    identity — one solve, no extra matrix multiply.

    ``n_full`` is the row count the penalty is scaled by, which is *not*
    ``K.shape[0]`` when K is one arm's block of a larger system: λ has to keep
    meaning what it meant over the whole sample. See `_fit_stages`.
    """
    m = K.shape[0]
    c = lam * n_full
    H = c * solve(K + c * np.eye(m), np.eye(m), assume_a="pos")
    diag_H = np.diag(H)
    # Avoid division by zero in degenerate cases.
    diag_H = np.where(np.abs(diag_H) < 1e-12, 1e-12, diag_H)
    return H, 1.0 / diag_H


def _loocv_stage1_block(
    K_ZAX: np.ndarray, K_W: np.ndarray, lam1: float, n_full: int
) -> float:
    """One arm's contribution to the Stage-1 LOOCV score. Lower is better.

    Measures how well the kernel embedding predicts the outcome proxy W, as
    tr(D H K_W H D) with D = diag(1/H_ii). Evaluated as tr(G K_W Gᵀ) with
    G = D H, i.e. as one matmul and an elementwise sum: D is diagonal, so the
    literal form's ``np.diag(...) @ ...`` would spend two full n³ matmuls
    scaling rows and columns. Same number to machine precision, ~3x less work.

    The scores of the arms *add*: H is block diagonal when k_A is an indicator
    (see `_fit_stages`), so row i of H is supported on i's own arm and every
    term of the trace stays inside one block.
    """
    H, inv_diag = _ridge_influence(K_ZAX, lam1, n_full)
    G = inv_diag[:, np.newaxis] * H
    return float(np.einsum("ij,ij->", G @ K_W, G))


def _loocv_stage2_block(
    Sigma: np.ndarray, Y: np.ndarray, lam2: float, n_full: int
) -> float:
    """One arm's **squared** contribution to the Stage-2 LOOCV score.

    The score is ‖D H Y‖ with D = diag(1/H_ii); this returns ‖·‖² per arm so
    the caller can add the blocks and take one square root at the end. As
    above, D is applied by broadcasting on a matrix-*vector* product rather
    than by two dense matmuls.

    Uses the identity D_t M_Γ = Σ from the paper, so the (m₂ × m₁m₂)
    ``modif_kron`` matrix of the reference implementation is never formed:
        H = I − Σ (Σ + λ₂ n I)⁻¹ = λ₂ n (Σ + λ₂ n I)⁻¹

    **What the resulting score may be compared across.** Dropping ``1/m₂`` is
    free over a λ grid and over kernel bandwidths — the target Y₂ is the same
    vector and the sum has the same number of terms — which is what makes this
    a valid selection criterion for `kernel_scale`. It is *not* free across
    anything that changes ``m₂``: a different `split_ratio` changes the term
    count, so the raw scores are not on a common scale. See TUNABLE_PARAMS.
    """
    H, inv_diag = _ridge_influence(Sigma, lam2, n_full)
    r = inv_diag * (H @ Y)
    return float(r @ r)


def _loocv_stage1(K_ZAX: np.ndarray, K_W: np.ndarray, lam1: float) -> float:
    """Stage-1 LOOCV over a single undecomposed system (the dense reference)."""
    return _loocv_stage1_block(K_ZAX, K_W, lam1, K_ZAX.shape[0])


def _loocv_stage2(Sigma: np.ndarray, Y: np.ndarray, lam2: float) -> float:
    """Stage-2 LOOCV over a single undecomposed system (the dense reference)."""
    return float(np.sqrt(_loocv_stage2_block(Sigma, Y, lam2, Sigma.shape[0])))


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class KPVEstimator:
    """
    Sklearn-style CATE estimator implementing the Kernel Proxy Variable method.

    Usage::

        cfg = KPVConfig(split_ratio=0.5, lam1=1e-3, lam2=1e-3)
        est = KPVEstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: KPVConfig | None = None) -> None:
        self.cfg = KPVConfig() if cfg is None else cfg

        self._alpha: np.ndarray | None = None  # (n1, n2)

        self._A2: np.ndarray | None = None  # (n2, d_A)
        self._W1: np.ndarray | None = None  # (n1, d_W)
        self._X2: np.ndarray | None = None  # (n2, d_X), if X exists

        self._W_ref: np.ndarray | None = None  # usually full training W
        self._X_ref: np.ndarray | None = None  # usually full training X
        self._K_W1_ref: np.ndarray | None = None  # K_W(W1, W_ref)

        self._marginal_xw_mean: np.ndarray | None = None  # (n1, n2), or (n1, 1) if no X
        self._w_marginal_mean: np.ndarray | None = None  # (n1,)

        self._k_A: ColumnWiseGaussianKernel | None = None
        self._k_W: ColumnWiseGaussianKernel | None = None
        self._k_X: ColumnWiseGaussianKernel | None = None

        self._has_x: bool = False

        self.lam1_: float | None = None
        self.lam2_: float | None = None
        self.kernel_scales_: dict[str, float] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "KPVEstimator":
        """
        Fit the two-stage KPV model.

        Args:
            X: Covariates,                shape (n, d_X).
            A: Treatment (binary/cont.),  shape (n,) or (n, d_A).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).
        """
        cfg = self.cfg
        # A local Generator, not np.random.seed: the benchmark runs many
        # estimators inside one worker process, and reseeding the global RNG
        # here would silently reset every other method's stream too.
        rng = np.random.default_rng(cfg.seed)

        A = as_2d(A)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        W = as_2d(W)
        Z = as_2d(Z)
        X = as_2d(X)

        # ── Split ──────────────────────────────────────────────────────────
        n = len(Y)
        n1 = int(n * cfg.split_ratio)
        idx = rng.permutation(n)
        i1, i2 = idx[:n1], idx[n1:]

        A1, A2 = A[i1], A[i2]
        W1, W2 = W[i1], W[i2]
        Z1, Z2 = Z[i1], Z[i2]
        X1, X2 = X[i1], X[i2]
        Y2 = Y[i2]
        n1, n2 = len(i1), len(i2)
        has_x = X1.shape[1] > 0

        # ── Fit both stages at the configured length-scales ────────────────
        # One fit, one set of bandwidths. An outer search over them is the
        # wrapper's `tune:` block (benchmarks/methods/__init__.py), ranked on
        # `loocv_score_` below.
        best = self._fit_stages(A1, A2, W1, Z1, Z2, X1, X2, Y2, n1, n2, has_x)

        # The Stage-2 LOOCV loss at the selected λ's — the selection score for
        # any outer search, and the only in-sample signal of fit quality KPV
        # produces. Comparable across kernel scales because the target Y₂ is
        # fixed; see `_loocv_stage2_block` for what it is *not* comparable
        # across.
        self.loocv_score_ = best["score"]
        # The *effective* per-variable scales (global x relative), which is what
        # you want when reading a tuned fit back: `cfg` holds the two factors,
        # this holds the product actually handed to each kernel. Absent when a
        # kernel was never fitted (k_X on a covariate-free episode).
        self.kernel_scales_ = best["scales"]
        self.lam1_ = best["lam1"]
        self.lam2_ = best["lam2"]
        self._alpha = best["alpha"]  # (n1, n2)

        k_A, k_W, k_X = best["k_A"], best["k_W"], best["k_X"]

        # Store bridge basis points
        self._A2 = A2
        self._W1 = W1
        self._X2 = X2

        # Store full empirical reference distribution for prediction heads.
        # This is what lets the prediction head choose joint, marginal,
        # independent, or conditional averaging over (X, W).
        self._W_ref = W
        self._X_ref = X
        self._has_x = has_x

        # Store kernels
        self._k_A = k_A
        self._k_W = k_W
        self._k_X = k_X

        # Cache K_W(W1, W_ref): shape (n1, n)
        K_W1_ref = k_W(W1, W)
        self._K_W1_ref = K_W1_ref

        # Marginal W embedding over the full training distribution:
        # w_mean[i] = mean_l k_W(W1_i, W_l)
        self._w_marginal_mean = K_W1_ref.mean(axis=1)  # (n1,)

        if self._has_x:
            K_X2_ref = k_X(X2, X)  # (n2, n)
            self._marginal_xw_mean = K_W1_ref @ K_X2_ref.T / len(Y)  # (n1, n2)
        else:
            self._marginal_xw_mean = self._w_marginal_mean[:, np.newaxis]  # (n1, 1)

        return self

    # ── Two-stage fit for a single kernel scale ─────────────────────────────────

    def _fit_stages(
        self,
        A1: np.ndarray,
        A2: np.ndarray,
        W1: np.ndarray,
        Z1: np.ndarray,
        Z2: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        Y2: np.ndarray,
        n1: int,
        n2: int,
        has_x: bool,
    ) -> dict:
        """
        Run both KPV stages at the configured kernel length-scales.

        Fits the kernel bandwidths on D1 (each rescaled by its own
        ``kernel_scale * kernel_scale_<v>``), tunes λ₁/λ₂ by closed-form LOOCV,
        and returns the resulting bridge weights together with the Stage-2
        LOOCV score an outer search ranks on. Called once per ``fit``.
        """
        cfg = self.cfg

        # ── Fit kernel bandwidths on D1 ────────────────────────────────────
        # Appendix §D.2.1, "Kernel's length-scale": an exponentiated-quadratic
        # kernel per *dimension*, bandwidth from the median inter-point distance
        # of that dimension, the joint kernel being the product over dimensions
        # (that product is why `kx`/`k_Z` multiply below rather than concatenate).
        # The scales are the "narrow range around this level" the appendix
        # searches; each multiplies its variable's σ², so scale > 1 = wider,
        # exactly the reference's `rbf_ker(x, y, scale)` with
        # coef = 1/(2·scale·median²). The appendix ties them together; we allow
        # one per variable, because the shape of the joint kernel is what trades
        # Stage 1's fit of W against Stage 2's fit of Y (see KPVConfig).
        #
        # One deliberate departure: we fit each bandwidth **once, on D1**, and
        # reuse it for every gram. The reference recomputes the median inside
        # every `rbf_ker` call, so K_11, K_12 and K_22 there are built with three
        # *different* bandwidths and are not blocks of one kernel matrix. Ours is
        # the consistent reading of the equations; the numbers differ slightly.
        #
        # A is integer-valued, so it takes the Kronecker-delta kernel — the
        # `identifier_k` branch the reference selects for integer dtypes — and
        # therefore takes no scale at all.
        k_A = BinaryKernel()
        k_Z = ColumnWiseGaussianKernel()
        k_W = ColumnWiseGaussianKernel()
        k_X = ColumnWiseGaussianKernel()

        # Global x per-variable, so a config that sets only `kernel_scale` still
        # moves every bandwidth together (see KPVConfig).
        scales = {
            "Z": float(cfg.kernel_scale * cfg.kernel_scale_z),
            "W": float(cfg.kernel_scale * cfg.kernel_scale_w),
        }
        k_A.fit(A1)
        k_Z.fit(Z1, scale=scales["Z"])
        k_W.fit(W1, scale=scales["W"])
        if has_x:
            scales["X"] = float(cfg.kernel_scale * cfg.kernel_scale_x)
            k_X.fit(X1, scale=scales["X"])

        # X-kernel factor: an empty (d_X = 0) covariate set contributes a
        # neutral factor of 1 (empty product), so the bridge collapses to the
        # (A, Z, W)-only model without ever touching the unfitted k_X.
        def kx(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
            return k_X(a, b) if has_x else 1.0

        # ── Arm decomposition ──────────────────────────────────────────────
        # k_A is an indicator, and it multiplies *every* gram in the pipeline:
        # K_ZAX = K_A ⊙ K_Z ⊙ K_X on both splits, and Σ carries K_A(D2, D2).
        # So K_ZAX(D1,D1) and Σ are block diagonal in the arms, K_ZAX(D1,D2) is
        # zero off the arm blocks, and hence so are Γ and α. Both stages
        # therefore decouple into two independent per-arm problems whose
        # results are *assembled*, not approximated — pinned against the
        # undecomposed solve by tests/test_bench_kpv_blocks.py.
        #
        # Both LOOCV criteria decompose with them: H inherits the block
        # structure, so the Stage-1 trace is a sum over arms and the Stage-2
        # squared norm is too (one square root at the end, not one per arm).
        #
        # Why bother: the two LOOCV scans are ~86% of a fit at n = 2048 (0.68 s
        # + 0.20 s of 1.62 s) and every term in them is cubic, so two half-size
        # arms cost a quarter. K_ZAX(D1,D1), K_A and the full K_W(D1,D1) are
        # never materialised at all.
        #
        # Note λ₁ and λ₂ keep multiplying the **full** split sizes n1 and n2
        # inside every block. The blocks are a decomposition of one system, not
        # two smaller problems; a per-arm count would silently double the
        # regularisation. Bandwidths are likewise fitted on the whole of D1
        # above, so a block carries the same numbers the dense gram would.
        arm1, arm2 = A1.ravel(), A2.ravel()
        arms = (0.0, 1.0)

        # ── Stage 1 ────────────────────────────────────────────────────────
        # Within an arm k_A ≡ 1, so it drops out of the product entirely.
        stage1: dict[float, tuple] = {}
        for a in arms:
            i1 = np.flatnonzero(arm1 == a)
            if i1.size == 0:
                continue
            K11_a = k_Z(Z1[i1], Z1[i1]) * kx(X1[i1], X1[i1])
            stage1[a] = (i1, K11_a, k_W(W1[i1], W1[i1]))

        lam1 = cfg.lam1
        if lam1 is None:
            lam1 = self._tune(
                lambda l: sum(
                    _loocv_stage1_block(K11_a, K_W_a, l, n1)
                    for _, K11_a, K_W_a in stage1.values()
                ),
                cfg.lam1_min,
                cfg.lam1_max,
                cfg.n_lam1_search,
            )

        # Γ = (K_ZAX_11 + λ₁ n₁ I)⁻¹ K_ZAX_12  → (n1, n2), zero off the blocks.
        # Kept dense: it is only n1×n2 floats, and every prediction head then
        # reads it exactly as before.
        Gamma = np.zeros((n1, n2), dtype=float)
        for a, (i1, K11_a, _) in stage1.items():
            i2 = np.flatnonzero(arm2 == a)
            if i2.size == 0:
                continue
            K12_a = k_Z(Z1[i1], Z2[i2]) * kx(X1[i1], X2[i2])
            Gamma[np.ix_(i1, i2)] = solve(
                K11_a + lam1 * n1 * np.eye(i1.size), K12_a, assume_a="pos"
            )

        # ── Stage 2 ────────────────────────────────────────────────────────
        # Σ_a = (Γ_aᵀ K_W^{aa} Γ_a) ⊙ K_X(D2_a, D2_a). Only the *within-arm*
        # block of K_W is reachable: Γ is zero off the blocks, so the cross-arm
        # entries of K_W are annihilated on both sides.
        sigma_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for a in arms:
            i2 = np.flatnonzero(arm2 == a)
            if i2.size == 0:
                continue
            if a in stage1:
                i1, _, K_W_a = stage1[a]
                G_a = Gamma[np.ix_(i1, i2)]
                Sigma_a = (G_a.T @ K_W_a @ G_a) * kx(X2[i2], X2[i2])
            else:
                # No D1 rows in this arm, so Γ has no column here and Σ_a = 0
                # — matching what the undecomposed product would give.
                Sigma_a = np.zeros((i2.size, i2.size), dtype=float)
            sigma_blocks.append((i2, Sigma_a))

        def stage2_score(lam: float) -> float:
            return float(
                np.sqrt(
                    sum(
                        _loocv_stage2_block(Sigma_a, Y2[i2], lam, n2)
                        for i2, Sigma_a in sigma_blocks
                    )
                )
            )

        lam2 = cfg.lam2
        if lam2 is None:
            lam2 = self._tune(
                stage2_score, cfg.lam2_min, cfg.lam2_max, cfg.n_lam2_search
            )

        # Stage-2 LOOCV loss at the chosen λ₂ — the criterion for ranking
        # scales. Comparable across scales because the target Y₂ is fixed.
        score = stage2_score(lam2)

        # v = (Σ + λ₂ n₂ I)⁻¹ Y₂  → (n2,)
        v = np.zeros(n2, dtype=float)
        for i2, Sigma_a in sigma_blocks:
            v[i2] = solve(Sigma_a + lam2 * n2 * np.eye(i2.size), Y2[i2], assume_a="pos")

        # α[i, j] = Γ[i, j] * v[j]  → (n1, n2)
        alpha = Gamma * v[np.newaxis, :]

        return {
            "scales": scales,
            "lam1": lam1,
            "lam2": lam2,
            "alpha": alpha,
            "k_A": k_A,
            "k_W": k_W,
            "k_X": k_X,
            "score": score,
        }

    # ── Prediction heads ──────────────────────────────────────────────────────

    def predict_mean_outcome(
        self,
        a: float | np.ndarray,
        *,
        mode: str = "marginal",
        X: np.ndarray | None = None,
        W: np.ndarray | None = None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Predict E[Y | do(A=a)] or bridge-based conditional variants.

        Modes:
            "marginal":
                ATE / causal dose-response head.
                Averages h(a, X, W) over the empirical joint training
                distribution of (X, W).

            "paired":
                Bridge contrast/value head at supplied paired observations.
                Requires W, and requires X if the model was fit with X.
                Returns h(a_i, x_i, w_i).

            "independent":
                X-specific head assuming W ⟂ X.
                For each supplied x_i, averages h(a, x_i, W) over the
                marginal training distribution of W.

            "conditional":
                X-specific head using kernel smoothing to approximate
                E_W[h(a, x_i, W) | X=x_i].

        Args:
            a:
                Scalar treatment, treatment grid of shape (n,), or treatment
                matrix of shape (n, d_A).
            mode:
                One of {"marginal", "paired", "independent", "conditional"}.
            X:
                Test covariates, required for paired/independent/conditional
                when the model was fit with non-empty X.
            W:
                Test outcome proxies, required for paired mode.
            n:
                Optional number of outputs when `a` is scalar and the chosen
                mode is constant.

        Returns:
            Predictions, shape depends on mode:
                - marginal with scalar a: shape (1,) unless n or X/W is given
                - marginal with treatment grid: shape (n_grid,)
                - paired/independent/conditional: shape (n_test,)
        """
        self._check_fitted()

        mode = mode.lower()
        if mode in {"ate", "marginal"}:
            pred = self._predict_marginal_mean(a)
            n_out = self._infer_n(X=X, W=W, n=n)
            return self._repeat_if_scalar(pred, n_out)

        if mode == "paired":
            return self._predict_paired_mean(a, X=X, W=W)

        if mode == "independent":
            return self._predict_independent_mean(a, X=X, n=n)

        if mode == "conditional":
            return self._predict_conditional_mean(a, X=X, n=n)

        raise ValueError(
            "Unknown prediction mode. Expected one of "
            "{'marginal', 'paired', 'independent', 'conditional'}."
        )

    def predict(
        self,
        X: np.ndarray | None = None,
        *,
        W: np.ndarray | None = None,
        Z=None,
        mode: str = "marginal",
        a1: float | np.ndarray = 1.0,
        a0: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """
        Predict treatment contrast under the selected prediction head.

        For binary treatment this returns:

            head(a1) - head(a0)

        Modes:
            "marginal":
                ATE contrast. Returned as a constant vector if X/W is supplied.

            "paired":
                h(a1, x_i, w_i) - h(a0, x_i, w_i).
                Requires W, and requires X if the model has X.

            "independent":
                E_{W marginal}[h(a1, x_i, W)] -
                E_{W marginal}[h(a0, x_i, W)].

            "conditional":
                Kernel-smoothed approximation of
                E[h(a1, x_i, W) | X=x_i] -
                E[h(a0, x_i, W) | X=x_i].

        Z is accepted for benchmark/API compatibility but is not used at
        prediction time.
        """
        self._check_fitted()

        n = self._infer_n(X=X, W=W, n=None)
        if n is None:
            n = 1

        y1 = self.predict_mean_outcome(a1, mode=mode, X=X, W=W, n=n)
        y0 = self.predict_mean_outcome(a0, mode=mode, X=X, W=W, n=n)

        return np.asarray(y1 - y0, dtype=float).reshape(-1)

    def predict_ate(
        self,
        a1: float | np.ndarray = 1.0,
        a0: float | np.ndarray = 0.0,
    ) -> float | np.ndarray:
        """
        Convenience wrapper for the marginal ATE / dose-response contrast.
        """
        y1 = self.predict_mean_outcome(a1, mode="marginal")
        y0 = self.predict_mean_outcome(a0, mode="marginal")
        out = y1 - y0
        return float(out.item()) if out.size == 1 else out

    def predict_bridge(
        self,
        treatment: float | np.ndarray,
        *,
        X: np.ndarray | None = None,
        W: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the learned bridge h(a, x, w) at supplied points.

        This is equivalent to predict_mean_outcome(..., mode="paired"), but
        without taking a treatment contrast.
        """
        return self.predict_mean_outcome(treatment, mode="paired", X=X, W=W)

    # ── Individual prediction heads ───────────────────────────────────────────

    def _predict_marginal_mean(self, a: float | np.ndarray) -> np.ndarray:
        """
        Marginal ATE / dose-response head:

            mean_l h(a, X_l, W_l)

        using the empirical joint training distribution of (X, W).
        """
        A_test = self._as_treatment_matrix(a)

        # K_A(A2, a_test): (n2, n_test)
        K_A_test = self._k_A(self._A2, A_test)

        # M[i, j] = E_hat[k_W(W1_i, W) k_X(X2_j, X)]
        # If no X, M has shape (n1, 1) and broadcasts over alpha.
        M = self._marginal_xw_mean  # (n1, n2) or (n1, 1)

        # coef[j] = sum_i alpha[i, j] M[i, j]
        coef_over_A2 = (self._alpha * M).sum(axis=0)  # (n2,)

        return (coef_over_A2 @ K_A_test).ravel()

    def _predict_paired_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        W: np.ndarray | None,
    ) -> np.ndarray:
        """
        Paired bridge head:

            h(a_i, x_i, w_i)

        If a is scalar, it is repeated for all supplied pairs.
        """
        if W is None:
            raise ValueError("mode='paired' requires W.")

        W = as_2d(W)
        n = W.shape[0]

        X_checked = self._validate_X_for_prediction(X, n=n, required=self._has_x)
        A_test = self._as_treatment_matrix(a, n=n)

        # K_W(W1, W_test): (n1, n)
        K_W_test = self._k_W(self._W1, W)

        # left[i, j] = sum_r k_W(W1_r, W_i) alpha[r, j]
        # shape: (n, n2)
        left = K_W_test.T @ self._alpha

        # K_A(A2, A_test): (n2, n)
        right = self._k_A(self._A2, A_test)

        if self._has_x:
            # K_X(X2, X_test): (n2, n)
            right = right * self._k_X(self._X2, X_checked)

        # pred_i = left_i dot right_i
        return np.einsum("ij,ji->i", left, right).ravel()

    def _predict_independent_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Independent head:

            E_{W marginal}[h(a, x_i, W)]

        This assumes W is independent of X for the prediction target.
        """
        if self._has_x:
            X_checked = self._validate_X_for_prediction(X, n=None, required=True)
            n_test = X_checked.shape[0]
        else:
            X_checked = None
            n_test = n

        A_test = self._as_treatment_matrix(a, n=n_test)
        n_test = A_test.shape[0]

        # E_W[k_W(W1_i, W)]
        w_mean = self._w_marginal_mean  # (n1,)

        # left[j] = sum_i E_W[k_W(W1_i, W)] alpha[i, j]
        left = w_mean @ self._alpha  # (n2,)

        right = self._k_A(self._A2, A_test)  # (n2, n_test)

        if self._has_x:
            right = right * self._k_X(self._X2, X_checked)  # (n2, n_test)

        return np.sum(left[:, np.newaxis] * right, axis=0).ravel()

    def _predict_conditional_mean(
        self,
        a: float | np.ndarray,
        *,
        X: np.ndarray | None,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Conditional head:

            E_hat[h(a, x_i, W) | X=x_i]

        where the conditional expectation over W is approximated by
        Nadaraya-Watson kernel smoothing over the training pairs (X_ref, W_ref):

            weights_l(x_i) ∝ k_X(X_ref_l, x_i).
        """
        if not self._has_x:
            # With no X, P(W | X) = P(W) by construction of the model input.
            return self._predict_independent_mean(a, X=None, n=n)

        X_checked = self._validate_X_for_prediction(X, n=None, required=True)
        n_test = X_checked.shape[0]
        A_test = self._as_treatment_matrix(a, n=n_test)

        # K_X(X_ref, X_test): (n_ref, n_test)
        K_X_ref_test = self._k_X(self._X_ref, X_checked)

        denom = K_X_ref_test.sum(axis=0, keepdims=True)
        weights = K_X_ref_test / np.maximum(denom, 1e-12)

        # If all weights underflow for a test point, fall back to uniform.
        bad = denom.ravel() <= 1e-12
        if np.any(bad):
            weights[:, bad] = 1.0 / K_X_ref_test.shape[0]

        # Conditional W mean features:
        # W_cond_mean[:, i] = sum_l weight_l(x_i) k_W(W1, W_ref_l)
        # shape: (n1, n_test)
        W_cond_mean = self._K_W1_ref @ weights

        # left[i, j] = sum_r W_cond_mean[r, i] alpha[r, j]
        # shape: (n_test, n2)
        left = W_cond_mean.T @ self._alpha

        right = self._k_A(self._A2, A_test) * self._k_X(self._X2, X_checked)

        return np.einsum("ij,ji->i", left, right).ravel()

    # ── Prediction utilities ──────────────────────────────────────────────────

    def _as_treatment_matrix(
        self,
        a: float | np.ndarray,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Convert treatment input to shape (n_test, d_A).

        Rules:
            scalar a:
                repeated n times if n is given, else shape (1, d_A)

            1D a with scalar treatment:
                interpreted as treatment grid, shape (len(a), 1)

            1D a with vector treatment:
                interpreted as one treatment vector, shape (1, d_A)

            2D a:
                must already have shape (n_test, d_A)
        """
        d_A = self._A2.shape[1]
        arr = np.asarray(a, dtype=float)

        if arr.ndim == 0:
            n_out = 1 if n is None else n
            return np.full((n_out, d_A), float(arr))

        if arr.ndim == 1:
            if d_A == 1:
                out = arr.reshape(-1, 1)
            elif arr.size == d_A:
                out = arr.reshape(1, d_A)
            else:
                raise ValueError(
                    f"Treatment has shape {arr.shape}; expected scalar, "
                    f"(*, 1), or (*, {d_A})."
                )

        elif arr.ndim == 2:
            if arr.shape[1] != d_A:
                raise ValueError(
                    f"Treatment has shape {arr.shape}; expected second "
                    f"dimension {d_A}."
                )
            out = arr

        else:
            raise ValueError(
                f"Treatment has shape {arr.shape}; expected scalar, 1D, or 2D."
            )

        if n is not None:
            if out.shape[0] == 1 and n > 1:
                out = np.repeat(out, n, axis=0)
            elif out.shape[0] != n:
                raise ValueError(
                    f"Treatment has {out.shape[0]} rows, but expected {n}."
                )

        return out

    def _validate_X_for_prediction(
        self,
        X: np.ndarray | None,
        *,
        n: int | None,
        required: bool,
    ) -> np.ndarray | None:
        if not self._has_x:
            return None

        if X is None:
            if required:
                raise ValueError("This prediction mode requires X.")
            return None

        X = as_2d(X)

        if X.shape[1] != self._X2.shape[1]:
            raise ValueError(
                f"X has shape {X.shape}; expected second dimension "
                f"{self._X2.shape[1]}."
            )

        if n is not None and X.shape[0] != n:
            raise ValueError(f"X has {X.shape[0]} rows, but expected {n}.")

        return X

    @staticmethod
    def _infer_n(
        *,
        X: np.ndarray | None,
        W: np.ndarray | None,
        n: int | None,
    ) -> int | None:
        if n is not None:
            return n
        if X is not None:
            return as_2d(X).shape[0]
        if W is not None:
            return as_2d(W).shape[0]
        return None

    @staticmethod
    def _repeat_if_scalar(pred: np.ndarray, n: int | None) -> np.ndarray:
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if n is not None and pred.size == 1:
            return np.full(n, pred.item(), dtype=float)
        return pred

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._alpha is None:
            raise RuntimeError("Call fit() before predict().")

    @staticmethod
    def _tune(
        criterion,
        low: float,
        high: float,
        n_points: int,
    ) -> float:
        """Grid search over a log-spaced λ grid; returns the best value."""
        candidates = np.logspace(np.log10(low), np.log10(high), n_points)
        scores = {lam: criterion(lam) for lam in candidates}
        return min(scores, key=scores.get)
