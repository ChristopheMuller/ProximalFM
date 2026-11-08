"""Reproducible causal episodes for the r3mo sensitivity experiments.

Only ``binary_completeness`` certifies the stated finite-state completeness
conditions.  Gaussian families have exact CATEs and explicit bridges; their
linear measurement rank is NOT presented as a general completeness theorem.
Separate random streams preserve pairing when proxy reliability or irrelevant
feature counts change.  No query outcomes/proxies are needed by the estimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    X: np.ndarray
    Z: np.ndarray
    W: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    tau: np.ndarray
    n_context: int
    metadata: dict[str, Any] = field(default_factory=dict)
    U: np.ndarray | None = None
    tau_all: np.ndarray | None = None
    A_all: np.ndarray | None = None
    Y_all: np.ndarray | None = None
    bridge_all: np.ndarray | None = None
    mc_target: np.ndarray | None = None
    mc_variance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.n_context = int(self.n_context)
        for name in ("X", "Z", "W", "A", "Y", "tau", "U", "tau_all",
                     "A_all", "Y_all", "bridge_all", "mc_target", "mc_variance"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} contains nonfinite values")
                setattr(self, name, value)
        n = self.X.shape[0]
        if not 0 < self.n_context < n:
            raise ValueError("An episode needs positive context and query sizes")
        for name in ("X", "Z", "W"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1:
                raise ValueError(f"{name} must have shape (context+query, d>=1)")
        for name in ("A", "Y"):
            if getattr(self, name).shape != (self.n_context,):
                raise ValueError(f"{name} must contain context rows only")
        if self.tau.shape != (n - self.n_context,):
            raise ValueError("tau must contain exact query CATEs")
        if not np.isin(self.A, (0, 1)).all():
            raise ValueError("Treatment must be binary")
        for name in ("mc_target", "mc_variance"):
            value = getattr(self, name)
            if value is not None and value.shape != self.tau.shape:
                raise ValueError(f"{name} must have one value per query")
        if self.mc_variance is not None and np.any(self.mc_variance < 0):
            raise ValueError("MC variances cannot be negative")

    @property
    def n_query(self) -> int:
        return self.X.shape[0] - self.n_context

    @property
    def X_context(self) -> np.ndarray:
        return self.X[:self.n_context]

    @property
    def Z_context(self) -> np.ndarray:
        return self.Z[:self.n_context]

    @property
    def W_context(self) -> np.ndarray:
        return self.W[:self.n_context]

    @property
    def X_query(self) -> np.ndarray:
        return self.X[self.n_context:]

    @property
    def tau_query(self) -> np.ndarray:
        return self.tau


def _rng(seed: int, stream: int, *parts: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream, *parts]))


def _columns(seed: int, stream: int, nc: int, nq: int, d: int,
             uniform: bool = False) -> np.ndarray:
    """Column-wise and split-wise streams preserve dimension/context prefixes."""
    arrays = []
    for split, size in enumerate((nc, nq)):
        cols = []
        for j in range(d):
            r = _rng(seed, stream, split, j)
            cols.append(r.uniform(size=size) if uniform else r.normal(size=size))
        arrays.append(np.column_stack(cols))
    return np.concatenate(arrays, axis=0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _validate_sizes(n_context: int, n_query: int, mc_replicates: int) -> None:
    if n_context < 1 or n_query < 1:
        raise ValueError("n_context and n_query must be positive")
    if mc_replicates < 0:
        raise ValueError("mc_replicates must be nonnegative")


def binary_channel_certificate(rho: float, rho_w: float, p: float = 0.5) -> dict:
    """Exact matrices within a fixed (A=a,X=x) stratum with Pr(U=+1)=p.

    Both stratumwise ranks drop at rho=0.  Even if treatment alone preserves
    injectivity of the first condition interpreted globally, the second global
    condition fails: E[Z|W,A,X]=0 for the nondegenerate random variable Z.
    """
    if not 0 <= rho <= 1 or not 0 < rho_w < 1 or not 0 < p < 1:
        raise ValueError("Need rho in [0,1], rho_w and p in (0,1)")
    cz = 0.5 * np.array([[1 + rho, 1 - rho], [1 - rho, 1 + rho]])
    cw = 0.5 * np.array([[1 + rho_w, 1 - rho_w], [1 - rho_w, 1 + rho_w]])
    pu = np.array([1 - p, p])
    zu = cz @ np.diag(pu)
    z_given_u = cz
    u_given_z = zu / zu.sum(axis=1, keepdims=True)
    zw = cz @ np.diag(pu) @ cw.T
    z_given_w = zw.T / zw.sum(axis=0)[:, None]
    return {
        "channel_matrix": z_given_u,
        "u_given_z_a_x": u_given_z,
        "z_given_w_a_x": z_given_w,
        "joint_z_w_a_x": zw,
        "channel_singular_values": np.linalg.svd(cz, compute_uv=False),
        "stratum_rank_u_from_z": int(np.linalg.matrix_rank(u_given_z)),
        "stratum_rank_z_from_w": int(np.linalg.matrix_rank(z_given_w)),
        "det_joint": float(np.linalg.det(zw)),
    }


def binary_completeness(seed: int, n_context: int, n_query: int,
                        rho: float = 0.5, rho_w: float = 0.8, d_x: int = 5,
                        confounding: float = 1.5, gamma: float = 1.0,
                        eta: float = 0.6, noise_sd: float = 0.5,
                        overlap: float = 0.05, mc_replicates: int = 0,
                        **kwargs: Any) -> Episode:
    """Finite binary channel sweep with exact CATE and bridge at rho=0.

    Y(a)=b(X)+a*tau(X)+(gamma+a*eta)*U+epsilon,
    h(W,a,X)=b(X)+a*tau(X)+(gamma+a*eta)*W/rho_w.
    U is independent of X, so CATE=tau(X), including at complete rank failure.
    ``d_u``, ``d_z``, ``d_w`` may be supplied only as 1 for generic callers.
    """
    _validate_sizes(n_context, n_query, mc_replicates)
    for name, value in kwargs.items():
        if name not in ("d_u", "d_z", "d_w") or value != 1:
            raise TypeError(f"Unsupported binary argument {name}={value!r}")
    if d_x < 1 or not 0 < overlap < 0.5 or noise_sd < 0:
        raise ValueError("Invalid dimensions, overlap, or noise scale")
    certificate = binary_channel_certificate(rho, rho_w)
    X = _columns(seed, 10, n_context, n_query, d_x)
    U = np.where(_columns(seed, 11, n_context, n_query, 1, True)[:, 0] < .5, -1., 1.)
    ez = np.where(_columns(seed, 12, n_context, n_query, 1, True)[:, 0] < (1 + rho) / 2, 1., -1.)
    ew = np.where(_columns(seed, 13, n_context, n_query, 1, True)[:, 0] < (1 + rho_w) / 2, 1., -1.)
    Z, W = (U * ez)[:, None], (U * ew)[:, None]
    coeff_rng = _rng(seed, 14)
    tau0 = float(coeff_rng.uniform(.5, 1.5))
    tau_coef = np.zeros(d_x)
    tau_coef[0] = coeff_rng.choice([-1., 1.]) * .8
    b = .3 * X[:, 0]
    if d_x > 1:
        b += .2 * X[:, 1]
    tau = tau0 + X @ tau_coef
    propensity = overlap + (1 - 2 * overlap) * _sigmoid(.3 * X[:, 0] + confounding * U)
    A = (_columns(seed, 15, n_context, n_query, 1, True)[:, 0] < propensity).astype(float)
    epsilon = noise_sd * _columns(seed, 16, n_context, n_query, 1)[:, 0]
    Y = b + A * tau + (gamma + A * eta) * U + epsilon
    bridge = b + A * tau + (gamma + A * eta) * W[:, 0] / rho_w
    target = variance = None
    if mc_replicates:
        # Draws in replicate-major order are nested when K changes.
        draws = _rng(seed, 17).uniform(size=(mc_replicates, n_query))
        effects = tau[n_context:][None, :] + eta * np.where(draws < .5, -1., 1.)
        target = effects.mean(axis=0)
        variance = (effects.var(axis=0, ddof=1) / mc_replicates
                    if mc_replicates > 1 else np.full(n_query, eta ** 2))
    metadata = {
        "kind": "binary", "family": "binary_channel", "seed": int(seed),
        "n_ctx": n_context, "n_query": n_query, "d_x": d_x, "d_u": 1,
        "d_z": 1, "d_w": 1, "rho": rho, "rho_w": rho_w,
        "confounding": confounding, "gamma": gamma, "eta": eta,
        "overlap": overlap, "noise_sd": noise_sd, "mc_replicates": mc_replicates,
        "mc_effect_variance": eta ** 2,
        "channel_singular_values": certificate["channel_singular_values"].tolist(),
        "stratum_operator_rank": 2 if rho > 0 else 1,
        "completeness_certified": bool(rho > 0),
        "completeness_failure_certified": bool(rho == 0),
        "bridge_exists": True,
        "certification": "finite_state_stratum_rank; second global condition fails at rho=0",
        "tau_intercept": tau0, "tau_coefficients": tau_coef.tolist(),
        "graph_edges": ["X->A", "U->A", "X->Y", "U->Y", "A->Y", "U->Z", "U->W"],
        "independent_noise_streams": True,
    }
    return Episode(X, Z, W, A[:n_context], Y[:n_context], tau[n_context:],
                   n_context, metadata, U[:, None], tau, A, Y, bridge, target, variance)


def _dag_noise(d: int, family: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Acyclic linear SEM and population-standardized noise loading."""
    G = np.zeros((d, d))
    if family == "chain":
        for j in range(1, d):
            G[j, j - 1] = rng.choice([-1., 1.]) * .55
    elif family == "sparse":
        for j in range(1, d):
            mask = rng.uniform(size=j) < min(.35, 2 / j)
            G[j, :j] = mask * rng.uniform(-.5, .5, size=j)
    L = np.linalg.solve(np.eye(d) - G, np.eye(d))
    scale = np.sqrt(np.sum(L * L, axis=1))
    # Row scaling leaves an acyclic SEM; recover its standardized coefficients.
    L = L / scale[:, None]
    G = G * scale[None, :] / scale[:, None]
    return G, L


def _unit_variance(vector: np.ndarray, covariance: np.ndarray,
                   target_sd: float) -> np.ndarray:
    sd = np.sqrt(max(float(vector @ covariance @ vector), 1e-12))
    return vector * (target_sd / sd)


def _proxy_loading(seed: int, stream: int, d: int, du: int,
                   mode: str) -> np.ndarray:
    Q, _ = np.linalg.qr(_rng(seed, stream).normal(size=(du, du)))
    C = np.zeros((d, du))
    C[:du] = Q
    if d > du:
        if mode == "relevant":
            more = _rng(seed, stream + 1).normal(size=(d - du, du))
            more /= np.linalg.norm(more, axis=1, keepdims=True)
            C[du:] = more
        elif mode == "redundant":
            C[du:] = C[np.arange(d - du) % du]
    return C


def gaussian_episode(seed: int, n_context: int, n_query: int,
                     d_x: int = 5, d_u: int = 5, d_z: int | None = None,
                     d_w: int | None = None, family: str = "dense",
                     width_mode: str = "relevant", base_dim: int = 5,
                     confounding: float = 1.5, proxy_noise: float = .6,
                     effect_modifier_scale: float = .6,
                     heterogeneity_sd: float = .8, noise_sd: float = .5,
                     overlap: float = .05, mc_replicates: int = 0,
                     direct_z_strength: float = 0.,
                     direct_w_strength: float = 0.) -> Episode:
    """Linear SCMs with analytic heterogeneous CATE and an explicit bridge.

    Families ``dense``, ``chain``, ``sparse``, ``role_partition`` differ in
    coordinate DAGs and latent direct-parent roles.  The latter three can be
    held out from training, but are not automatically held out from a released
    model's original prior.

    Optional direct Z->A and W->Y terms distinguish the proxy roles while
    preserving exclusions. Strength is the population SD of each direct term.
    The W term affects both potential outcomes equally, so CATE/MC targets are
    unchanged. Zero defaults preserve previously generated episodes exactly.

    In ``irrelevant``/``redundant`` mode at most base_dim X/U coordinates are
    causal. Extra irrelevant proxy columns are independent noise, which can
    invalidate completeness for the entire enlarged proxy block. This is
    disclosed in metadata; do not call this an identification-preserving test.
    Redundant columns are exact copies and add no proxy information.
    """
    _validate_sizes(n_context, n_query, mc_replicates)
    if family not in {"dense", "chain", "sparse", "role_partition"}:
        raise ValueError(f"Unknown family {family!r}")
    if width_mode not in {"relevant", "irrelevant", "redundant"}:
        raise ValueError(f"Unknown width_mode {width_mode!r}")
    if min(d_x, d_u, base_dim) < 1 or min(proxy_noise, noise_sd) < 0:
        raise ValueError("Invalid dimensions or noise scale")
    if not 0 < overlap < .5 or heterogeneity_sd <= 0 or min(confounding, direct_z_strength, direct_w_strength) < 0:
        raise ValueError("Invalid overlap or signal scale")
    du = d_u if width_mode == "relevant" else min(d_u, base_dim)
    dx = d_x if width_mode == "relevant" else min(d_x, base_dim)
    dz, dw = int(d_z if d_z is not None else d_u), int(d_w if d_w is not None else d_u)
    if min(dz, dw) < du:
        raise ValueError("Explicit full-column-rank bridge requires d_z,d_w >= effective d_u")
    rg = _rng(seed, 100)
    Gx, Lx = _dag_noise(dx, family, _rng(seed, 101))
    Gu, Lu = _dag_noise(du, family, _rng(seed, 102))
    covariance_x = Lx @ Lx.T
    X = _columns(seed, 103, n_context, n_query, d_x)
    X[:, :dx] = X[:, :dx] @ Lx.T
    if width_mode == "redundant" and d_x > dx:
        X[:, dx:] = X[:, np.arange(d_x - dx) % dx]
    B = rg.normal(size=(du, dx)) * (.35 / np.sqrt(dx))
    if family == "sparse":
        B *= rg.uniform(size=B.shape) < .35
    # Standardize U in population while preserving its conditional mean and DAG.
    scales = np.sqrt(np.diag(B @ covariance_x @ B.T + Lu @ Lu.T))
    B, Lu = B / scales[:, None], Lu / scales[:, None]
    Gu = Gu * scales[None, :] / scales[:, None]
    covariance_u = B @ covariance_x @ B.T + Lu @ Lu.T
    Umean = X[:, :dx] @ B.T
    U = Umean + _columns(seed, 104, n_context, n_query, du) @ Lu.T
    beta_a, beta_y, beta_m = (rg.normal(size=du) for _ in range(3))
    roles = {"shared": list(range(du)), "treatment_only": [],
             "outcome_only": [], "effect_modifier": list(range(du))}
    if family == "role_partition":
        groups = np.array_split(np.arange(du), 4)
        # For very small du retain at least one actual shared confounder/modifier.
        shared = groups[0] if len(groups[0]) else np.array([0])
        treatment_only, outcome_only, modifiers = groups[1:]
        if not len(modifiers):
            modifiers = shared
        keep_a = np.concatenate([shared, treatment_only])
        keep_y = np.concatenate([shared, outcome_only])
        beta_a[~np.isin(np.arange(du), keep_a)] = 0
        beta_y[~np.isin(np.arange(du), keep_y)] = 0
        beta_m[~np.isin(np.arange(du), modifiers)] = 0
        roles = {"shared": shared.tolist(), "treatment_only": treatment_only.tolist(),
                 "outcome_only": outcome_only.tolist(), "effect_modifier": modifiers.tolist()}
    beta_a = _unit_variance(beta_a, covariance_u, confounding)
    beta_y = _unit_variance(beta_y, covariance_u, confounding)
    beta_m = _unit_variance(beta_m, Lu @ Lu.T, effect_modifier_scale)
    bcoef = rg.normal(size=dx) * (.35 / np.sqrt(dx))
    acoef = rg.normal(size=dx) * (.25 / np.sqrt(dx))
    tcoef = rg.normal(size=dx) * (.8 / np.sqrt(dx))
    tau0 = float(rg.uniform(.5, 1.5))
    total_tcoef = tcoef + B.T @ beta_m
    factor = heterogeneity_sd / np.sqrt(max(float(total_tcoef @ covariance_x @ total_tcoef), 1e-12))
    tcoef, beta_m = tcoef * factor, beta_m * factor
    total_tcoef = tcoef + B.T @ beta_m
    tau_base = tau0 + X[:, :dx] @ tcoef
    tau = tau0 + X[:, :dx] @ total_tcoef
    baseline = X[:, :dx] @ bcoef
    Cz = _proxy_loading(seed, 110, dz, du, width_mode)
    Cw = _proxy_loading(seed, 112, dw, du, width_mode)
    Znoise = _columns(seed, 114, n_context, n_query, dz)
    Wnoise = _columns(seed, 115, n_context, n_query, dw)
    Z, W = U @ Cz.T + proxy_noise * Znoise, U @ Cw.T + proxy_noise * Wnoise
    if width_mode == "redundant":
        if dz > du:
            Z[:, du:] = Z[:, np.arange(dz - du) % du]
        if dw > du:
            W[:, du:] = W[:, np.arange(dw - du) % du]
    direct_z_coef, direct_w_coef = np.zeros(dz), np.zeros(dw)
    def proxy_covariance(C, count):
        noise_covariance = np.eye(count)
        if width_mode == "redundant":
            noise_index = np.arange(count) % du
            noise_covariance = (noise_index[:, None] == noise_index[None, :]).astype(float)
        return C @ covariance_u @ C.T + proxy_noise ** 2 * noise_covariance
    treatment_score = X[:, :dx] @ acoef + U @ beta_a
    if direct_z_strength:
        direct_z_coef = _unit_variance(_rng(seed, 119).normal(size=dz), proxy_covariance(Cz, dz), direct_z_strength)
        treatment_score = treatment_score + Z @ direct_z_coef
    direct_w_term = np.zeros(len(X))
    if direct_w_strength:
        direct_w_coef = _unit_variance(_rng(seed, 120).normal(size=dw), proxy_covariance(Cw, dw), direct_w_strength)
        direct_w_term = W @ direct_w_coef
    propensity = overlap + (1 - 2 * overlap) * _sigmoid(treatment_score)
    A = (_columns(seed, 116, n_context, n_query, 1, True)[:, 0] < propensity).astype(float)
    epsilon = noise_sd * _columns(seed, 117, n_context, n_query, 1)[:, 0]
    Y = baseline + U @ beta_y + A * (tau_base + U @ beta_m) + epsilon
    if direct_w_strength:
        Y = Y + direct_w_term
    # A known left inverse recovers E[U|U] from W; extra independent columns
    # receive zero coefficients automatically, and exact copies remain valid.
    Cw_left_inverse = np.linalg.pinv(Cw)
    U_from_W = W @ Cw_left_inverse.T
    bridge = baseline + U_from_W @ beta_y + A * (tau_base + U_from_W @ beta_m)
    if direct_w_strength:
        bridge = bridge + direct_w_term
    effect_variance = float(beta_m @ Lu @ Lu.T @ beta_m)
    target = variance = None
    if mc_replicates:
        # Since beta_m^T(U-E[U|X]) is Gaussian, direct scalar simulation is exact.
        effects = tau[n_context:][None, :] + np.sqrt(effect_variance) * _rng(seed, 118).normal(size=(mc_replicates, n_query))
        target = effects.mean(axis=0)
        variance = (effects.var(axis=0, ddof=1) / mc_replicates
                    if mc_replicates > 1 else np.full(n_query, effect_variance))
    all_tcoef = np.zeros(d_x)
    all_tcoef[:dx] = total_tcoef
    metadata = {
        "kind": "gaussian", "family": family, "seed": int(seed),
        "n_ctx": n_context, "n_query": n_query, "d_x": d_x, "d_u": du,
        "requested_d_u": d_u, "d_z": dz, "d_w": dw,
        "effective_d_x": dx, "width_mode": width_mode, "base_dim": base_dim,
        "confounding": confounding, "proxy_noise": proxy_noise,
        "direct_z_strength": direct_z_strength, "direct_w_strength": direct_w_strength,
        "direct_z_coefficients": direct_z_coef.tolist(), "direct_w_coefficients": direct_w_coef.tolist(),
        "noise_sd": noise_sd, "overlap": overlap, "mc_replicates": mc_replicates,
        "mc_effect_variance": effect_variance, "population_cate_sd": heterogeneity_sd,
        "bridge_exists": True, "completeness_certified": False,
        "certification": "structural exclusions, overlap, exact CATE, explicit linear bridge; general completeness not asserted",
        "irrelevant_proxy_completeness_warning": bool(width_mode == "irrelevant" and (dz > du or dw > du)),
        "tau_intercept": tau0, "tau_coefficients": all_tcoef.tolist(),
        "latent_direct_parent_roles": roles,
        "x_dag_edges": [[int(j), int(i), float(Gx[i, j])] for i, j in zip(*np.nonzero(Gx))],
        "u_dag_edges": [[int(j), int(i), float(Gu[i, j])] for i, j in zip(*np.nonzero(Gu))],
        "graph_edges": (["X->U", "X->A", "U->A", "X->Y", "U->Y", "A->Y", "U->Z", "U->W"]
                        + (["Z->A"] if direct_z_strength else [])
                        + (["W->Y"] if direct_w_strength else [])),
        "independent_noise_streams": True,
    }
    return Episode(X, Z, W, A[:n_context], Y[:n_context], tau[n_context:],
                   n_context, metadata, U, tau, A, Y, bridge, target, variance)


def generate_episode(kind: str = "gaussian", **kwargs: Any) -> Episode:
    if kind in {"binary", "binary_completeness"}:
        return binary_completeness(**kwargs)
    if kind in {"gaussian", "linear"}:
        return gaussian_episode(**kwargs)
    raise ValueError(f"Unknown episode kind {kind!r}")
