"""
DFPV — Deep Feature Proxy Variable (deep two-stage least squares)
================================================================
Proximal causal estimator that learns *adaptive* feature maps for the
treatment, treatment proxy (Z), outcome proxy (W), and covariates (X) with
neural networks, then closes the bridge equation in closed form via a two-stage
ridge regression (2SLS) on those features.

Reference: Xu et al. "Deep Proxy Causal Learning and its Application to
Confounded Bandit Policy Evaluation" (NeurIPS 2021).

Pipeline (per epoch, alternating gradient updates on the feature nets):
  Stage 1 (split D1):  regress  φ_W(W)  on  ψ₁ = f₁(A) ⊗ g(Z) [⊗ b₁(X)]
                       → learns the conditional mean embedding of W given (A, Z, X).
  Stage 2 (split D2):  regress  Y  on  ψ₂ = f₂(A) ⊗ φ̂_W [⊗ b₂(X)]
                       where φ̂_W is the Stage-1 prediction of φ_W.

After the nets are trained, a final closed-form `_fit_2sls` produces the ridge
weights and the reference feature means used at prediction time.

Bridge function:
    h(a, w, x) = ψ₂(f₂(a), φ_W(w), b₂(x)) · β₂

Prediction of E[Y | do(a)] marginalises the bridge over the empirical (W, X):
    ŷ(a) ≈ ψ₂(f₂(a), E[φ_W], E[b₂(X)]) · β₂
(the mean-embedding plug-in; exact under W ⟂ X for the cross terms, matching the
"independent" framing shared by the other proximal baselines).

CATE = ŷ(1) − ŷ(0)  (for binary treatment).

Pure numpy/torch — no JAX and no dependency on the reference `src` package.

Hyperparameters are searched from outside, through the benchmark wrapper's
`tune:` block, ranked on `DFPVEstimator.oos_score_` — the paper's own
out-of-sample Stage-2 loss (Appendix "Hyper-parameters"). TUNABLE_PARAMS is the
whitelist and carries the reasoning for every field left out of it.

Read that score for what it is: Stage 2 fits E[Y | a, z], the *reduced-form*
regression. A candidate that predicts Y well need not identify the bridge h
well, and nothing here scores CATE — the same limitation KPV's LOOCV criterion
has. Check `tune_score_range` in the wrapper's diagnostics before reading a
selected value as an optimum.

Tuning entry point: DFPVConfig — every knob lives there with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn


#: The only :class:`DFPVConfig` fields a `tune:` block may search. The benchmark
#: wrapper validates `tune.grid` against this set; everything else in the
#: dataclass is configurable but not searchable, because the selection score
#: cannot rank it:
#:
#: - ``seed`` and ``device`` are not model choices. Searching the seed would
#:   select the luckiest initialisation on the criterion and report it as a
#:   tuned optimum.
#: - ``x_cond_length_scale`` / ``x_cond_top_k`` enter *only*
#:   ``_mean_outcome_conditional``, i.e. only the `conditional` read-out. They
#:   are invisible to `oos_score_`, which is computed from the bridge and the
#:   two stage fits, so every candidate differing only in them scores
#:   identically. They are honest ablation axes, not tunable ones — ranking
#:   them would need the true CATE.
#:
#: Everything else is in, including the structural flags: refusing a field needs
#: a reason the *criterion* cannot rank it, and `use_backdoor`,
#: `feature_relu_output` and the two intercept flags all change the fit that
#: `oos_score_` scores. Two carry caveats worth knowing before reading a winner:
#:
#: - ``split_ratio`` changes which rows are scored (the criterion evaluates on
#:   D1, whose size it moves). The score is a *mean*, so this is unbiased and
#:   the comparison is legitimate — unlike KPV, whose LOOCV score drops its
#:   ``1/m₂`` prefactor and is therefore refused this field — but candidates are
#:   scored on different row sets, so expect more noise across this axis than
#:   across the others.
#: - ``feature_relu_output`` is a known-bad option, not a neutral one: the
#:   paper's own footnote reports that a ReLU on the final feature layer "causes
#:   numerical instability" and disowns the results produced with it. Searchable,
#:   but `False` is the corrected setting and a grid over it is mostly a way to
#:   rediscover that.
TUNABLE_PARAMS = frozenset(
    {
        "feature_dim",
        "hidden_dim",
        "n_hidden_layers",
        "feature_relu_output",
        "use_backdoor",
        "add_stage1_intercept",
        "add_stage2_intercept",
        "lam1",
        "lam2",
        "n_epochs",
        "stage1_iter",
        "stage2_iter",
        "lr",
        "treatment_weight_decay",
        "treatment_proxy_weight_decay",
        "outcome_proxy_weight_decay",
        "backdoor_weight_decay",
        "split_ratio",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DFPVConfig:
    # ── Feature networks ──────────────────────────────────────────────────────
    # Output dimensionality of every feature extractor (treatment, proxies, X).
    # NOTE: cost scales steeply with this. The stage features are Kronecker
    # products, so the ridge system has size (feature_dim+1)^2 without X and
    # (feature_dim+1)^3 with X — e.g. 8 → 729, 16 → 4913. Keep it small.
    feature_dim: int = 8
    # Width and depth of the hidden trunk shared by the extractor architecture.
    hidden_dim: int = 32
    n_hidden_layers: int = 2
    # ReLU on the final feature layer (faithful to the reference demand nets).
    feature_relu_output: bool = False

    # ── 2SLS structure ────────────────────────────────────────────────────────
    # Whether covariates X enter as a "backdoor" feature. Auto-disabled when X
    # has zero columns.
    use_backdoor: bool = True
    add_stage1_intercept: bool = True
    add_stage2_intercept: bool = True

    # ── Ridge regularisation ──────────────────────────────────────────────────
    lam1: float = 0.1  # Stage-1 ridge penalty
    lam2: float = 0.1  # Stage-2 ridge penalty

    # ── Optimisation ──────────────────────────────────────────────────────────
    # Full-batch alternating updates: each epoch runs `stage1_iter` Stage-1 steps
    # then `stage2_iter` Stage-2 steps.
    n_epochs: int = 50
    stage1_iter: int = 20
    stage2_iter: int = 1
    lr: float = 1e-3

    # Per-network L2 weight decay (Adam). Mirrors the reference trainer, which
    # tunes these independently.
    treatment_weight_decay: float = 0.0
    treatment_proxy_weight_decay: float = 0.0
    outcome_proxy_weight_decay: float = 0.0
    backdoor_weight_decay: float = 0.0

    # ── Data split ────────────────────────────────────────────────────────────
    # Fraction of training data used in Stage 1 (remainder goes to Stage 2).
    split_ratio: float = 0.5

    # ── Conditional W | X smoothing (strategy="conditional") ──────────────────
    x_cond_length_scale: float = 1.0
    x_cond_top_k: Optional[int] = 512

    # ── Reproducibility / device ──────────────────────────────────────────────
    seed: int = 42
    # "cpu", "cuda", or None to auto-select (cuda if available, else cpu).
    device: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Linear-algebra helpers (self-contained ports of the reference 2SLS utilities)
# ─────────────────────────────────────────────────────────────────────────────


def _add_const_col(mat: torch.Tensor) -> torch.Tensor:
    """Append a column of ones — the intercept term for the ridge feature map."""
    ones = torch.ones((mat.shape[0], 1), device=mat.device, dtype=mat.dtype)
    return torch.cat([mat, ones], dim=1)


def _outer_prod(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Row-wise outer product.

    For mat1 (n, d1) and mat2 (n, d2) returns (n, d1, d2); generalises to higher
    feature ranks the same way as the reference implementation.
    """
    s1 = tuple(mat1.shape)
    s2 = tuple(mat2.shape)
    n = s1[0]
    a1 = mat1.reshape(s1 + (1,) * (len(s2) - 1))
    a2 = mat2.reshape((n,) + (1,) * (len(s1) - 1) + s2[1:])
    return a1 * a2


def _fit_ridge(target: torch.Tensor, feature: torch.Tensor, reg: float) -> torch.Tensor:
    """
    Closed-form ridge weight  W = (FᵀF + reg·n·I)⁻¹ Fᵀ target.

    The normal equations are symmetric positive-definite, so we factorise with
    Cholesky (≈2× faster than a generic LU solve and more stable). A small
    deterministic jitter keeps them invertible even when `reg` is tiny.

    The factorisation runs on the CPU on purpose: Apple-MPS LU/Cholesky kernels
    are unreliable for the large (feature_dim+1)^k systems DFPV produces and have
    been observed to return garbage pivots — which surface downstream as
    nonsensical out-of-bounds indices. The (cheap) feature products stay on the
    network's device; only the (d×d) solve is moved to the CPU.
    """
    n, d = feature.shape
    A = (feature.t() @ feature).cpu()
    b = (feature.t() @ target).cpu()
    A = A + (reg * n + 1e-6) * torch.eye(d, dtype=A.dtype)
    try:
        L = torch.linalg.cholesky(A)
        weight = torch.cholesky_solve(b, L)
    except torch.linalg.LinAlgError:
        weight = torch.linalg.solve(A, b)
    return weight.to(feature.device)


def _ridge_pred(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return feature @ weight


def _ridge_loss(
    target: torch.Tensor, feature: torch.Tensor, reg: float
) -> torch.Tensor:
    """Stage-1 objective: refit the ridge weight, return penalised residual norm."""
    weight = _fit_ridge(target, feature, reg)
    pred = feature @ weight
    n = feature.shape[0]
    return torch.norm(target - pred) ** 2 + reg * torch.norm(weight) ** 2 * n


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor
# ─────────────────────────────────────────────────────────────────────────────


def _build_extractor(
    input_dim: int,
    cfg: DFPVConfig,
) -> nn.Module:
    """MLP feature map: input_dim → hidden trunk → feature_dim."""
    dims = [input_dim] + [cfg.hidden_dim] * cfg.n_hidden_layers + [cfg.feature_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last or cfg.feature_relu_output:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class DFPVEstimator:
    """
    Sklearn-style CATE estimator implementing Deep Feature Proxy Variable.

    Usage::

        cfg = DFPVConfig(feature_dim=16, n_epochs=50, lam1=0.1, lam2=0.1)
        est = DFPVEstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)              # marginal ATE, shape (n_test,)
        cate = est.predict(X_test, W=W_test,    # paired bridge contrast
                           strategy="paired")

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: DFPVConfig | None = None) -> None:
        self.cfg = cfg or DFPVConfig()
        self._device = self._select_device(self.cfg.device)

        # Feature networks
        self._f1_A: nn.Module | None = None  # treatment, stage 1
        self._f2_A: nn.Module | None = None  # treatment, stage 2
        self._g_Z: nn.Module | None = None  # treatment proxy
        self._g_W: nn.Module | None = None  # outcome proxy
        self._b1_X: nn.Module | None = None  # backdoor (X), stage 1
        self._b2_X: nn.Module | None = None  # backdoor (X), stage 2

        # Learnt closed-form quantities
        self._stage1_weight: torch.Tensor | None = None
        self._stage2_weight: torch.Tensor | None = None
        self._mean_W_feat: torch.Tensor | None = None  # (1, feat)
        self._mean_X_feat: torch.Tensor | None = None  # (1, feat) or None

        # Reference state for the per-x prediction strategies
        self._W_feat_ref: torch.Tensor | None = None  # (m, feat) g_W(W2)
        self._X_ref: torch.Tensor | None = None  # (m, d_X) raw X2
        self._X_scale: torch.Tensor | None = None  # (1, d_X)

        # Held-out selection scores, set by `_finalise`. See `_oos_scores`.
        self.oos_score_: float = float("nan")
        self.oos_stage1_score_: float = float("nan")

        self._dim_A: int = 1
        self._has_x: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "DFPVEstimator":
        """
        Train the DFPV feature maps and close the two-stage ridge.

        Args:
            X: Covariates,                shape (n, d_X)  (d_X may be 0).
            A: Treatment (binary),        shape (n,) or (n, d_A).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        A_t = self._to_tensor(A)
        Y_t = self._to_tensor(Y).view(-1, 1)
        W_t = self._to_tensor(W)
        Z_t = self._to_tensor(Z)
        X_t = self._to_tensor(X)

        self._dim_A = A_t.shape[1]
        self._has_x = cfg.use_backdoor and X_t.shape[1] > 0

        # ── Split D1 / D2 ──────────────────────────────────────────────────
        n = Y_t.shape[0]
        perm = torch.randperm(n, device=self._device)
        n1 = int(n * cfg.split_ratio)
        i1, i2 = perm[:n1], perm[n1:]

        d1 = self._Split(A_t[i1], Z_t[i1], W_t[i1], X_t[i1], Y_t[i1])
        d2 = self._Split(A_t[i2], Z_t[i2], W_t[i2], X_t[i2], Y_t[i2])

        # ── Build networks ─────────────────────────────────────────────────
        self._f1_A = _build_extractor(self._dim_A, cfg).to(self._device)
        self._f2_A = _build_extractor(self._dim_A, cfg).to(self._device)
        self._g_Z = _build_extractor(Z_t.shape[1], cfg).to(self._device)
        self._g_W = _build_extractor(W_t.shape[1], cfg).to(self._device)
        if self._has_x:
            self._b1_X = _build_extractor(X_t.shape[1], cfg).to(self._device)
            self._b2_X = _build_extractor(X_t.shape[1], cfg).to(self._device)

        opt_s1 = torch.optim.Adam(self._stage1_params(), lr=cfg.lr)
        opt_s2 = torch.optim.Adam(self._stage2_params(), lr=cfg.lr)

        # ── Alternating feature training ───────────────────────────────────
        for _ in range(cfg.n_epochs):
            self._stage1_update(d1, opt_s1)
            self._stage2_update(d1, d2, opt_s2)

        # ── Final closed-form 2SLS on the trained features ─────────────────
        self._finalise(d1, d2)
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        W: np.ndarray | None = None,
        Z: np.ndarray | None = None,  # accepted for API parity, unused
        strategy: Literal[
            "marginal", "paired", "independent", "conditional"
        ] = "marginal",
    ) -> np.ndarray:
        """
        Predict the binary treatment contrast h(1, ·) − h(0, ·).

        strategy="marginal":
            Global ATE, repeated for each test row.
        strategy="paired":
            Requires W. h(1, W_i, X_i) − h(0, W_i, X_i).
        strategy="independent":
            Per-x contrast, marginalising W over its empirical marginal
            (assumes W ⟂ X).
        strategy="conditional":
            Per-x contrast, marginalising W over W | X via kernel smoothing.
        """
        self._check_fitted()
        X_t = self._to_tensor(X)
        n = X_t.shape[0]

        if strategy == "marginal":
            return self.predict_mean_outcome(1.0, n=n) - self.predict_mean_outcome(
                0.0, n=n
            )

        if strategy == "paired":
            if W is None:
                raise ValueError("strategy='paired' requires W.")
            W_t = self._to_tensor(W)
            return self._bridge(1.0, W_t, X_t) - self._bridge(0.0, W_t, X_t)

        if strategy == "independent":
            return self._mean_outcome_independent(
                1.0, X_t
            ) - self._mean_outcome_independent(0.0, X_t)

        if strategy == "conditional":
            return self._mean_outcome_conditional(
                1.0, X_t
            ) - self._mean_outcome_conditional(0.0, X_t)

        raise ValueError(f"Unknown prediction strategy: {strategy!r}")

    def predict_mean_outcome(self, a: float | np.ndarray, n: int = 1) -> np.ndarray:
        """E[Y | do(A=a)] via the mean-embedding plug-in over the empirical (W, X)."""
        self._check_fitted()
        A_t = self._treatment_matrix(a, n)
        m = A_t.shape[0]

        gW = self._mean_W_feat.expand(m, -1)
        gX = self._mean_X_feat.expand(m, -1) if self._has_x else None

        with torch.no_grad():
            feat = self._augment_stage2(gW, self._f2_A(A_t), gX)
            pred = _ridge_pred(feat, self._stage2_weight)
        return pred.squeeze(1).cpu().numpy()

    def predict_bridge(
        self,
        a: float | np.ndarray,
        W: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the bridge h(a, W, X) at paired test points."""
        self._check_fitted()
        W_t = self._to_tensor(W)
        X_t = self._to_tensor(X) if X is not None else None
        return self._bridge(a, W_t, X_t)

    # ── Bridge evaluation ──────────────────────────────────────────────────────

    def _bridge(
        self,
        a: float | np.ndarray,
        W_t: torch.Tensor,
        X_t: torch.Tensor | None,
    ) -> np.ndarray:
        n = W_t.shape[0]
        A_t = self._treatment_matrix(a, n)

        if self._has_x:
            if X_t is None:
                raise ValueError("X must be provided because the model was fit with X.")
            gX = self._b2_X(X_t)
        else:
            gX = None

        with torch.no_grad():
            feat = self._augment_stage2(self._g_W(W_t), self._f2_A(A_t), gX)
            pred = _ridge_pred(feat, self._stage2_weight)
        return pred.squeeze(1).cpu().numpy()

    def _mean_outcome_independent(self, a: float, X_t: torch.Tensor) -> np.ndarray:
        n = X_t.shape[0]
        A_t = self._treatment_matrix(a, n)
        gW = self._mean_W_feat.expand(n, -1)
        gX = self._b2_X(X_t) if self._has_x else None

        with torch.no_grad():
            feat = self._augment_stage2(gW, self._f2_A(A_t), gX)
            pred = _ridge_pred(feat, self._stage2_weight)
        return pred.squeeze(1).cpu().numpy()

    def _mean_outcome_conditional(self, a: float, X_t: torch.Tensor) -> np.ndarray:
        # Without X, P(W | X) = P(W): collapse to the marginal/independent head.
        if not self._has_x:
            return self.predict_mean_outcome(a, n=X_t.shape[0])

        cfg = self.cfg
        n = X_t.shape[0]
        A_t = self._treatment_matrix(a, n)

        with torch.no_grad():
            # Nadaraya–Watson weights over the reference X (softmax RBF on scaled
            # X). Computed on the CPU: topk's index tensor is unreliable on MPS.
            Xs = (X_t / self._X_scale).cpu()
            Xref_s = (self._X_ref / self._X_scale).cpu()
            W_feat_ref = self._W_feat_ref.cpu()
            log_w = -torch.cdist(Xs, Xref_s).pow(2) / (
                2.0 * (cfg.x_cond_length_scale + 1e-8) ** 2
            )

            if cfg.x_cond_top_k is not None:
                k = min(cfg.x_cond_top_k, W_feat_ref.shape[0])
                top_log_w, top_idx = torch.topk(log_w, k=k, dim=1)
                weights = torch.softmax(top_log_w, dim=1)  # (n, k)
                W_feat_local = W_feat_ref[top_idx]  # (n, k, feat)
                gW = torch.einsum("nk,nkf->nf", weights, W_feat_local)
            else:
                weights = torch.softmax(log_w, dim=1)  # (n, m)
                gW = weights @ W_feat_ref  # (n, feat)

            gW = gW.to(self._device)
            feat = self._augment_stage2(gW, self._f2_A(A_t), self._b2_X(X_t))
            pred = _ridge_pred(feat, self._stage2_weight)
        return pred.squeeze(1).cpu().numpy()

    # ── Feature augmentation (2SLS feature maps) ───────────────────────────────

    def _augment_stage1(
        self,
        t_feat: torch.Tensor,
        tp_feat: torch.Tensor,
        bd_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cfg.add_stage1_intercept:
            t_feat = _add_const_col(t_feat)
            tp_feat = _add_const_col(tp_feat)
            if bd_feat is not None:
                bd_feat = _add_const_col(bd_feat)

        feat = torch.flatten(_outer_prod(t_feat, tp_feat), start_dim=1)
        if bd_feat is not None:
            feat = torch.flatten(_outer_prod(feat, bd_feat), start_dim=1)
        return feat

    def _augment_stage2(
        self,
        w_feat: torch.Tensor,
        t_feat: torch.Tensor,
        bd_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.cfg.add_stage2_intercept:
            w_feat = _add_const_col(w_feat)
            t_feat = _add_const_col(t_feat)
            if bd_feat is not None:
                bd_feat = _add_const_col(bd_feat)

        feat = torch.flatten(_outer_prod(t_feat, w_feat), start_dim=1)
        if bd_feat is not None:
            feat = torch.flatten(_outer_prod(feat, bd_feat), start_dim=1)
        return feat

    # ── Training steps ─────────────────────────────────────────────────────────

    def _stage1_update(
        self, d1: "DFPVEstimator._Split", opt: torch.optim.Optimizer
    ) -> None:
        self._set_train(stage1=True)
        with torch.no_grad():
            target = self._g_W(d1.W)  # outcome-proxy feature is the regression target

        for _ in range(self.cfg.stage1_iter):
            opt.zero_grad()
            feat = self._augment_stage1(
                self._f1_A(d1.A),
                self._g_Z(d1.Z),
                self._b1_X(d1.X) if self._has_x else None,
            )
            loss = _ridge_loss(target, feat, self.cfg.lam1)
            loss.backward()
            opt.step()

    def _stage2_update(
        self,
        d1: "DFPVEstimator._Split",
        d2: "DFPVEstimator._Split",
        opt: torch.optim.Optimizer,
    ) -> None:
        self._set_train(stage1=False)
        with torch.no_grad():
            # Stage-1 features are frozen during the Stage-2 update.
            f1_A_1 = self._f1_A(d1.A)
            f1_A_2 = self._f1_A(d2.A)
            gZ_1 = self._g_Z(d1.Z)
            gZ_2 = self._g_Z(d2.Z)
            b1_1 = self._b1_X(d1.X) if self._has_x else None
            b1_2 = self._b1_X(d2.X) if self._has_x else None

        for _ in range(self.cfg.stage2_iter):
            opt.zero_grad()
            loss = self._stage2_loss(d1, d2, f1_A_1, f1_A_2, gZ_1, gZ_2, b1_1, b1_2)
            loss.backward()
            opt.step()

    def _stage2_loss(
        self,
        d1: "DFPVEstimator._Split",
        d2: "DFPVEstimator._Split",
        f1_A_1: torch.Tensor,
        f1_A_2: torch.Tensor,
        gZ_1: torch.Tensor,
        gZ_2: torch.Tensor,
        b1_1: torch.Tensor | None,
        b1_2: torch.Tensor | None,
    ) -> torch.Tensor:
        _, _, stage2_loss = self._two_stage(
            d1_W_feat=self._g_W(d1.W),
            f1_A_1=f1_A_1,
            f1_A_2=f1_A_2,
            gZ_1=gZ_1,
            gZ_2=gZ_2,
            b1_1=b1_1,
            b1_2=b1_2,
            f2_A_2=self._f2_A(d2.A),
            b2_2=self._b2_X(d2.X) if self._has_x else None,
            Y2=d2.Y,
        )
        return stage2_loss

    def _two_stage(
        self,
        *,
        d1_W_feat: torch.Tensor,
        f1_A_1: torch.Tensor,
        f1_A_2: torch.Tensor,
        gZ_1: torch.Tensor,
        gZ_2: torch.Tensor,
        b1_1: torch.Tensor | None,
        b1_2: torch.Tensor | None,
        f2_A_2: torch.Tensor,
        b2_2: torch.Tensor | None,
        Y2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closed-form 2SLS, returning (stage1_weight, stage2_weight, stage-2 loss)."""
        # Stage 1: predict the outcome-proxy feature from the instrument side.
        feat1 = self._augment_stage1(f1_A_1, gZ_1, b1_1)
        stage1_weight = _fit_ridge(d1_W_feat, feat1, self.cfg.lam1)

        feat1_pred = self._augment_stage1(f1_A_2, gZ_2, b1_2)
        W_feat_hat = _ridge_pred(feat1_pred, stage1_weight)

        # Stage 2: regress Y on the predicted outcome-proxy feature.
        feat2 = self._augment_stage2(W_feat_hat, f2_A_2, b2_2)
        stage2_weight = _fit_ridge(Y2, feat2, self.cfg.lam2)

        n = feat2.shape[0]
        pred = _ridge_pred(feat2, stage2_weight)
        loss = (
            torch.norm(Y2 - pred) ** 2
            + self.cfg.lam2 * n * torch.norm(stage2_weight) ** 2
        )
        return stage1_weight, stage2_weight, loss

    def _finalise(self, d1: "DFPVEstimator._Split", d2: "DFPVEstimator._Split") -> None:
        """Recompute the closed-form weights and reference means with frozen nets."""
        self._set_eval()
        with torch.no_grad():
            stage1_weight, stage2_weight, _ = self._two_stage(
                d1_W_feat=self._g_W(d1.W),
                f1_A_1=self._f1_A(d1.A),
                f1_A_2=self._f1_A(d2.A),
                gZ_1=self._g_Z(d1.Z),
                gZ_2=self._g_Z(d2.Z),
                b1_1=self._b1_X(d1.X) if self._has_x else None,
                b1_2=self._b1_X(d2.X) if self._has_x else None,
                f2_A_2=self._f2_A(d2.A),
                b2_2=self._b2_X(d2.X) if self._has_x else None,
                Y2=d2.Y,
            )
            self._stage1_weight = stage1_weight
            self._stage2_weight = stage2_weight

            # Reference features (Stage-2 split) for marginal / per-x heads.
            W_feat_ref = self._g_W(d2.W)
            self._W_feat_ref = W_feat_ref
            self._mean_W_feat = W_feat_ref.mean(dim=0, keepdim=True)

            if self._has_x:
                X_feat_ref = self._b2_X(d2.X)
                self._mean_X_feat = X_feat_ref.mean(dim=0, keepdim=True)
                self._X_ref = d2.X
                self._X_scale = d2.X.std(dim=0, unbiased=False, keepdim=True).clamp_min(
                    1e-6
                )

            self.oos_score_, self.oos_stage1_score_ = self._oos_scores(d1, d2)

    def _oos_scores(
        self, d1: "DFPVEstimator._Split", d2: "DFPVEstimator._Split"
    ) -> tuple[float, float]:
        """The paper's two out-of-sample stage losses, as means. Lower is better.

        From the DFPV appendix ("Hyper-parameters"): with the full joint
        (A, Y, Z, W) available on both splits — which every episode here has —
        each stage can be scored on the data the *other* stage was fitted on.

            L₂-oos = (1/m) Σ_{i ∈ D1} ( y_i − û ᵀ(ψ_A(a_i) ⊗ V̂(φ_A(a_i) ⊗ φ_Z(z_i))) )²
            L₁-oos = (1/n) Σ_{j ∈ D2} ‖ ψ_W(w̃_j) − V̂(φ_A(ã_j) ⊗ φ_Z(z̃_j)) ‖²

        Two things to know before ranking anything on these:

        - **Only `û` is held out in L₂-oos.** `V̂` was fitted on D1, which is
          where L₂-oos evaluates, so the Stage-1 solve is in-sample there. That
          asymmetry is the paper's own construction, and it is why L₂-oos is a
          clean criterion for Stage-2 quantities and only an indirect one for
          `lam1`.
        - **L₁-oos is a distance in ψ_W's own learned feature space**, and ψ_W
          differs between candidates. It is comparable only with the Stage-2
          nets held fixed, so it is returned as a *diagnostic* and never as the
          rank key — anything touching `feature_dim`, `hidden_dim`,
          `outcome_proxy_weight_decay` or the Stage-2 schedule would otherwise
          be compared across different spaces, which means nothing. It is
          reported because a large value flags a Stage-1 fit that failed, which
          L₂-oos alone can hide.

        Both are means, so they stay comparable when a candidate changes the
        number of rows in either split (`split_ratio`).
        """
        # L₂-oos: the full bridge pipeline, run forward on the Stage-1 rows.
        feat1_d1 = self._augment_stage1(
            self._f1_A(d1.A),
            self._g_Z(d1.Z),
            self._b1_X(d1.X) if self._has_x else None,
        )
        W_hat_d1 = _ridge_pred(feat1_d1, self._stage1_weight)
        feat2_d1 = self._augment_stage2(
            W_hat_d1,
            self._f2_A(d1.A),
            self._b2_X(d1.X) if self._has_x else None,
        )
        resid = d1.Y - _ridge_pred(feat2_d1, self._stage2_weight)
        stage2_oos = float(torch.mean(resid**2))

        # L₁-oos: the Stage-1 conditional mean embedding, on the Stage-2 rows.
        feat1_d2 = self._augment_stage1(
            self._f1_A(d2.A),
            self._g_Z(d2.Z),
            self._b1_X(d2.X) if self._has_x else None,
        )
        W_err = self._g_W(d2.W) - _ridge_pred(feat1_d2, self._stage1_weight)
        stage1_oos = float(torch.mean(torch.sum(W_err**2, dim=1)))

        return stage2_oos, stage1_oos

    # ── Internals ─────────────────────────────────────────────────────────────

    class _Split:
        """Lightweight container for one data split, all tensors on-device."""

        __slots__ = ("A", "Z", "W", "X", "Y")

        def __init__(self, A, Z, W, X, Y):
            self.A, self.Z, self.W, self.X, self.Y = A, Z, W, X, Y

    def _stage1_params(self):
        groups = [
            {
                "params": self._f1_A.parameters(),
                "weight_decay": self.cfg.treatment_weight_decay,
            },
            {
                "params": self._g_Z.parameters(),
                "weight_decay": self.cfg.treatment_proxy_weight_decay,
            },
        ]
        if self._has_x:
            groups.append(
                {
                    "params": self._b1_X.parameters(),
                    "weight_decay": self.cfg.backdoor_weight_decay,
                }
            )
        return groups

    def _stage2_params(self):
        groups = [
            {
                "params": self._f2_A.parameters(),
                "weight_decay": self.cfg.treatment_weight_decay,
            },
            {
                "params": self._g_W.parameters(),
                "weight_decay": self.cfg.outcome_proxy_weight_decay,
            },
        ]
        if self._has_x:
            groups.append(
                {
                    "params": self._b2_X.parameters(),
                    "weight_decay": self.cfg.backdoor_weight_decay,
                }
            )
        return groups

    def _set_train(self, *, stage1: bool) -> None:
        # Stage-1 nets train during the Stage-1 update; Stage-2 nets during Stage-2.
        self._f1_A.train(stage1)
        self._g_Z.train(stage1)
        self._f2_A.train(not stage1)
        self._g_W.train(not stage1)
        if self._has_x:
            self._b1_X.train(stage1)
            self._b2_X.train(not stage1)

    def _set_eval(self) -> None:
        for net in (
            self._f1_A,
            self._f2_A,
            self._g_Z,
            self._g_W,
            self._b1_X,
            self._b2_X,
        ):
            if net is not None:
                net.eval()

    def _treatment_matrix(self, a: float | np.ndarray, n: int) -> torch.Tensor:
        if np.isscalar(a):
            arr = np.full((n, self._dim_A), float(a))
        else:
            arr = np.asarray(a, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, self._dim_A)
        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    @staticmethod
    def _select_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _check_fitted(self) -> None:
        if self._stage2_weight is None:
            raise RuntimeError("Call fit() before predict().")
