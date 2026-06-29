"""
realistic_proximal_mechanisms.py

Realistic data-generating families for proximal causal learning.

Every mechanism is a plausible real-world data-generating process 
(clinical, economic, marketing, education, epidemiological). 

All mechanisms obey the proximal causal skeleton implied by Tchetgen Tchetgen
et al., "An Introduction to Proximal Causal Learning":

    X ............. measured Type-(a) common causes  (mixed tabular types)
    U ............. UNMEASURED confounder (severity / type / frailty / ability)
    Z = f_Z(X, U) + noise ........ treatment-inducing proxy  (Type b)
    W = f_W(X, U) + noise ........ outcome-inducing  proxy   (Type c, pre-treatment)
    A ~ Bernoulli(sigmoid(g_A(X, U, Z))) ......... NO direct W input
    Y(a) = g_Y(a, X, U, W) + shared_noise ........ NO direct Z input
    Y_obs = A * Y1 + (1 - A) * Y0

The structural exclusions enforce the negative-control conditions:

    NC.1   Y  _||_ Z | A, U, X
    NC.2   W  _||_ (A, Z) | U, X

and every mechanism guarantees:

    * U enters BOTH g_A and g_Y with non-trivial weight  -> real confounding,
    * U has variance orthogonal to X (independent noise)  -> X-adjustment is
      insufficient, so the bias survives backdoor on the measured covariates,
    * Z and W load on U strongly enough to be U-relevant   -> proximal
      identification is possible,
    * dim(Z), dim(W) >= dim(U)                              -> categorical
      completeness `min(d_Z, d_W) >= d_U`.

Output contract (unchanged) -- get_batch returns the 15 objects expected by
save_batch_sparse:

    U, X, W, Z, A, A_probs, Y_obs, Y0, Y1,
    d_U, d_X, d_W, d_Z, seq_lens, train_sizes
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math

import torch

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Sparse helper (kept for compatibility with the original save layout)
# -----------------------------------------------------------------------------


def dense2sparse(x: Tensor, dims: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Convert padded dense rows to a compact 1D format (drop padding columns)."""
    if x.ndim != 2:
        raise ValueError(f"dense2sparse expects [N, D], got shape {tuple(x.shape)}")
    device = x.device
    dims = dims.to(device=device, dtype=torch.long).view(-1)
    n, d = x.shape
    if dims.numel() != n:
        raise ValueError(f"dims has length {dims.numel()}, expected {n}")
    if torch.any(dims < 0):
        raise ValueError("dims contains negative entries")
    cols = torch.arange(d, device=device).view(1, d)
    mask = cols < dims.clamp(min=0, max=d).view(n, 1)
    return x[mask].to(dtype=dtype).contiguous()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class RealisticGeneratorConfig:
    save_dir: str | Path

    # Independently parameterized datasets per mechanism batch.
    K: int = 10

    # Rows per synthetic dataset.
    seq_len: int = 2048

    # Integer number of training rows, or float fraction of seq_len.
    train_size: int | float = 0.5

    min_dim_U: int = 1
    max_dim_U: int = 3
    min_dim_X: int = 1
    max_dim_X: int = 8
    min_dim_W: int = 1
    max_dim_W: int = 8
    min_dim_Z: int = 1
    max_dim_Z: int = 8

    # Strict positivity floor: propensities mapped to [overlap_eps, 1 - overlap_eps].
    overlap_eps: float = 0.02

    # Clamp standardized logits before sigmoid for numerical stability.
    logit_clip: float = 6.0

    # Measurement noise added to OBSERVED labels only (Y0/Y1 stay structural).
    extra_observation_noise: float = 0.03
    missing_prob: float = 0.0

    # Lower bound on how strongly U must drive treatment and outcome.
    # Mechanisms scale their U-coefficients to respect this floor.
    min_confounding_strength: float = 0.8

    seed: int = 123
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    seq_len_per_gp: bool = False


N_MECHANISMS = 12


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------


class RealisticProximalGenerator:
    """Generate and save realistic proximal data-generating families."""

    def __init__(self, config: RealisticGeneratorConfig):
        self.cfg = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(config.seed))

        self.device = torch.device(config.device)
        self.dtype = config.dtype

        if config.seq_len_per_gp:
            raise NotImplementedError(
                "Set seq_len_per_gp=False for fixed-length batches."
            )

        self._check_dims()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(
        self,
        mechanism_id: int,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        train_size: Optional[int | float] = None,
    ) -> Tuple[Tensor, ...]:
        if not 1 <= mechanism_id <= N_MECHANISMS:
            raise ValueError(f"mechanism_id must be in 1..{N_MECHANISMS}")

        B = int(batch_size or self.cfg.K)
        T = int(seq_len or self.cfg.seq_len)
        train_n = self._resolve_train_size(
            train_size if train_size is not None else self.cfg.train_size, T
        )

        results = []
        for ds_idx in range(B):
            seed = int(self.cfg.seed + 100_000 * mechanism_id + ds_idx)
            params = {
                "mechanism_id": mechanism_id,
                "seq_len": T,
                "train_size": train_n,
                "dataset_seed_index": ds_idx,
                "seed": seed,
            }
            with self._with_seed(seed):
                results.append(self.generate_dataset(params))

        (
            U_list,
            X_list,
            W_list,
            Z_list,
            A_list,
            A_probs_list,
            Y_obs_list,
            Y0_list,
            Y1_list,
            d_U_list,
            d_X_list,
            d_W_list,
            d_Z_list,
        ) = zip(*results)

        U = torch.stack(U_list).to(self.device)
        X = torch.stack(X_list).to(self.device)
        W = torch.stack(W_list).to(self.device)
        Z = torch.stack(Z_list).to(self.device)
        A = torch.stack(A_list).to(self.device)
        A_probs = torch.stack(A_probs_list).to(self.device)
        Y_obs = torch.stack(Y_obs_list).to(self.device)
        Y0 = torch.stack(Y0_list).to(self.device)
        Y1 = torch.stack(Y1_list).to(self.device)

        d_U = torch.stack(d_U_list).to(self.device)
        d_X = torch.stack(d_X_list).to(self.device)
        d_W = torch.stack(d_W_list).to(self.device)
        d_Z = torch.stack(d_Z_list).to(self.device)

        seq_lens = torch.full((B,), T, device=self.device, dtype=torch.long)
        train_sizes = torch.full((B,), train_n, device=self.device, dtype=torch.long)

        self._validate_batch_shapes(
            U=U,
            X=X,
            W=W,
            Z=Z,
            A=A,
            A_probs=A_probs,
            Y_obs=Y_obs,
            Y0=Y0,
            Y1=Y1,
            d_U=d_U,
            d_X=d_X,
            d_W=d_W,
            d_Z=d_Z,
            seq_lens=seq_lens,
            train_sizes=train_sizes,
        )

        return (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            d_U,
            d_X,
            d_W,
            d_Z,
            seq_lens,
            train_sizes,
        )

    def save_batch_sparse(
        self,
        U: Tensor,
        X: Tensor,
        W: Tensor,
        Z: Tensor,
        A: Tensor,
        A_probs: Tensor,
        Y_obs: Tensor,
        Y0: Tensor,
        Y1: Tensor,
        d_U: Tensor,
        d_X: Tensor,
        d_W: Tensor,
        d_Z: Tensor,
        seq_lens: Tensor,
        train_sizes: Tensor,
    ) -> None:
        """Standardize + merge to the benchmark `data.pt` structure."""
        B, T, _ = X.shape
        train_size = int(train_sizes[0].item())
        n_test = T - train_size

        def safe_standardize(tensor: Tensor) -> Tensor:
            if tensor.shape[-1] == 0:
                return tensor
            mu = tensor.mean(dim=1, keepdim=True)
            std = tensor.std(dim=1, keepdim=True)
            std = torch.where(std < 1e-5, torch.ones_like(std), std)
            return (tensor - mu) / std

        X_std = safe_standardize(X)
        W_std = safe_standardize(W)
        Z_std = safe_standardize(Z)
        U_std = safe_standardize(U)

        y_train_obs = Y_obs[:, :train_size]
        y_context_mean = y_train_obs.mean(dim=1, keepdim=True)
        y_context_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(
            min=1e-5
        )

        Y_obs_std = (Y_obs - y_context_mean) / y_context_std

        y_test_0 = Y0[:, train_size:]
        y_test_1 = Y1[:, train_size:]
        true_ITEs_std = ((y_test_1 - y_test_0) / y_context_std).cpu().numpy()

        final_data = {
            "U": U_std.cpu(),
            "X": X_std.cpu(),
            "Z": Z_std.cpu(),
            "W": W_std.cpu(),
            "A": A.cpu(),
            "Y": Y_obs_std.cpu(),
            "d_X": d_X.cpu(),
            "d_Z": d_Z.cpu(),
            "d_W": d_W.cpu(),
            "true_ITEs": true_ITEs_std,
            "N_TRAIN": train_size,
            "N_TEST": n_test,
            "N_BATCH": B,
        }

        batch_file = self.save_dir / "data.pt"
        temp_file = self.save_dir / "data.pt.tmp"
        torch.save(final_data, temp_file)
        temp_file.replace(batch_file)

    # ------------------------------------------------------------------
    # Dataset dispatch
    # ------------------------------------------------------------------

    def generate_dataset(self, params: Dict) -> Tuple[Tensor, ...]:
        mechanism_id = int(params["mechanism_id"])
        T = int(params["seq_len"])
        train_n = int(params.get("train_size", max(1, T // 2)))

        dispatch = {
            1: lambda: self._m01_confounding_by_indication(T),
            2: lambda: self._m02_discrete_patient_subtypes(T),
            3: lambda: self._m03_threshold_treatment_policy(T),
            4: lambda: self._m04_count_utilization(T),
            5: lambda: self._m05_censored_survival(T),
            6: lambda: self._m06_customer_uplift(T),
            7: lambda: self._m07_self_selection_education(T),
            8: lambda: self._m08_returns_to_treatment(T),
            9: lambda: self._m09_multisite_random_effects(T),
            10: lambda: self._m10_nonlinear_measurement_factor(T),
            11: lambda: self._m11_interaction_confounding(T),
            12: lambda: self._m12_near_null_strong_confounding(T),
        }
        if mechanism_id not in dispatch:
            raise ValueError(f"Unknown mechanism_id={mechanism_id}")
        return dispatch[mechanism_id]()

    @staticmethod
    def mechanism_name(mechanism_id: int) -> str:
        names = {
            1: "confounding_by_indication",
            2: "discrete_patient_subtypes",
            3: "threshold_treatment_policy",
            4: "count_utilization_frailty",
            5: "censored_survival_frailty",
            6: "customer_uplift_selection",
            7: "self_selection_education",
            8: "returns_to_treatment_ability",
            9: "multisite_random_effects",
            10: "nonlinear_measurement_factor",
            11: "interaction_confounding",
            12: "near_null_strong_confounding",
        }
        return names.get(int(mechanism_id), f"unknown_{mechanism_id}")

    @staticmethod
    def mechanism_story(mechanism_id: int) -> str:
        stories = {
            1: "Sicker patients (latent severity U) are preferentially treated and "
            "have worse outcomes; Z=biomarker proxy of severity drives the "
            "physician, W=baseline functional score is a pre-treatment severity proxy.",
            2: "Latent patient subtype U (responder phenotypes) governs propensity, "
            "baseline risk and effect; proxies are noisy readouts of subtype.",
            3: "Guideline threshold policy on severity + proxy creates near-deterministic "
            "treatment in regions (limited overlap), strong confounding by indication.",
            4: "Frailty U (gamma) inflates both treatment uptake and a positive, "
            "over-dispersed utilization-rate outcome; proxies are prior-use signals.",
            5: "Latent frailty confounds a log-normal time-to-event outcome with "
            "administrative right-censoring (heavy right tail).",
            6: "Promotions targeted at high latent-value customers; spend outcome is "
            "skewed/zero-inflated; engagement proxies measure latent value.",
            7: "Motivated/able students (latent U) self-select into a program using a "
            "test-score proxy; ability also drives later achievement.",
            8: "Returns-to-treatment: latent ability confounds a skewed log-wage "
            "outcome; proxies are imperfect ability measurements.",
            9: "Hierarchical sites/hospitals: latent site effects + within-site "
            "severity drive both assignment and outcome.",
            10: "Low-dimensional latent factor measured through nonlinear, saturating, "
            "ordinal proxies; treatment and outcome nonlinear in the factor.",
            11: "Non-additive confounding driven by X*U interactions with realistic "
            "effect modification.",
            12: "True effect near zero but strong U-confounding makes the naive "
            "contrast look large (specificity stress test).",
        }
        return stories.get(int(mechanism_id), "")

    # ==================================================================
    # Mechanism 1: Confounding by indication (clinical severity)
    # ==================================================================
    def _m01_confounding_by_indication(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()

        # X: age (continuous), sex (binary), comorbidity index (ordinal 0..4).
        X = self._patient_covariates(T, d_X)

        # U: latent disease severity, right-skewed, partly explained by X.
        severity_lin = self._mlp_feature(X, d_U)
        U = self._skew_positive(severity_lin, base_sigma=0.8)  # >0, right-skewed
        U = self._standardize(U)

        cs = self._confound_scale()  # ensures U-coefficients clear the floor

        # Z: biomarker the physician reads (noisy, skewed proxy of severity).
        Z = self._skew_positive(
            0.4 * self._linear(X, d_Z) + 1.0 * self._linear(U, d_Z), base_sigma=0.5
        )
        # W: baseline functional/questionnaire score (bounded, ordinal-ish proxy).
        W_latent = 0.4 * self._linear(X, d_W) + 1.0 * self._linear(U, d_W)
        W = self._bounded_ordinal(W_latent, n_levels=self._randint(4, 7))

        # Treatment: physician treats sicker patients (confounding by indication).
        raw_logit = (
            0.6 * self._mlp_feature(X, 1)
            + cs * self._linear(U, 1)
            + 0.7 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # Outcome: disease progression score (higher = worse). Severity worsens it.
        f0 = (
            0.6 * self._mlp_feature(X, 1)
            + cs * torch.tanh(self._linear(U, 1))
            + 0.5 * cs * self._linear(U, 1)
            + 0.5 * self._linear(W, 1)
        )
        # Treatment helps, but less in the most severe (saturating benefit).
        tau = -(0.8 + 0.6 * torch.sigmoid(-self._linear(U, 1))) * (
            0.7 + 0.3 * torch.tanh(self._linear(X, 1))
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.3, 0.8))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 2: Discrete patient subtypes (latent classes)
    # ==================================================================
    def _m02_discrete_patient_subtypes(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._patient_covariates(T, d_X)

        # --- latent discrete subtype structure (used ONLY to shape U) ---
        n_types = self._randint(3, 6)
        type_logits = self._linear(X, n_types)
        type_probs = torch.softmax(type_logits, dim=1)
        type_id = torch.multinomial(
            type_probs, 1, replacement=True, generator=self.rng
        ).squeeze(1)
        ctr_U = self._rsu(n_types, d_U, low=0.8, high=2.0)

        # U centred on its subtype centroid -> well-separated, multi-modal latent.
        U = self._standardize(ctr_U[type_id] + self._noise(T, d_U, sigma=0.35))
        cs = self._confound_scale()

        # --- proxies: noisy continuous read-out of U (+ a little X) ---
        # one ordinal channel (>=3 levels) instead of a single 0/1 bit, and only
        # when d_Z/d_W exceed d_U so the discretised column never eats the rank
        # required for completeness (min(d_Z, d_W) >= d_U).
        Z = self._linear(U, d_Z) + 0.3 * self._linear(X, d_Z) + self._noise(T, d_Z, 0.4)
        W = self._linear(U, d_W) + 0.3 * self._linear(X, d_W) + self._noise(T, d_W, 0.4)
        if d_Z > d_U:
            Z[:, 0:1] = self._bounded_ordinal(Z[:, 0:1], n_levels=4)
        if d_W > d_U:
            W[:, 0:1] = self._bounded_ordinal(W[:, 0:1], n_levels=4)

        # --- confounding: subtype baselines for A and Y are BOTH functions of U ---
        a_base = self._linear(U, 1)  # subtype propensity baseline
        y_base = torch.tanh(self._linear(U, 1))  # subtype outcome baseline

        raw_logit = (
            cs * 0.9 * a_base
            + 0.4 * self._linear(X, 1)
            + 0.6 * torch.tanh(self._linear(Z, 1))
        )  # Z -> A only (NC.1)
        p = self._propensity(raw_logit)

        f0 = (
            cs * 0.9 * y_base
            + 0.5 * self._linear(X, 1)
            + 0.5 * torch.tanh(self._linear(W, 1))
        )  # W -> Y only (NC.2)

        # heterogeneous effect varies with subtype (via U) and X -> real CATE
        # structure, but does NOT enter assignment, so ignorability still holds.
        tau = (
            0.8
            + 0.5 * torch.tanh(self._linear(U, 1))
            + 0.25 * torch.tanh(self._linear(X, 1))
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.25, 0.8))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 3: Threshold treatment policy (limited overlap)
    # ==================================================================
    def _m03_threshold_treatment_policy(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._patient_covariates(T, d_X)
        U = self._standardize(self._mlp_feature(X, d_U) + self._noise(T, d_U, 0.7))
        cs = self._confound_scale()

        Z = self._mlp_feature(self._cat(X, U), d_Z) + self._noise(T, d_Z, 0.35)
        W = self._mlp_feature(self._cat(X, U), d_W) + self._noise(T, d_W, 0.35)

        # Latent clinical severity (confounder + covariates) -- NO Z.
        # Drives both the policy baseline and the effect heterogeneity.
        latent_risk = cs * self._linear(U, 1) + 0.6 * self._linear(X, 1)
        latent_cut = torch.quantile(latent_risk, self._uniform_scalar(0.45, 0.65))

        # Guideline score adds the treatment-inducing proxy Z ON TOP of latent
        # severity: Z perturbs WHO crosses the assignment threshold, but only A.
        risk_score = latent_risk + 0.7 * torch.tanh(Z[:, :1])
        cut = torch.quantile(risk_score, self._uniform_scalar(0.45, 0.65))
        sharp = self._uniform_scalar(3.0, 6.0)  # steep policy -> limited overlap
        raw_logit = sharp * (risk_score - cut)
        p = self._propensity(raw_logit)

        # Outcome depends on X, U, W only (Z-free).
        f0 = (
            self._mlp_feature(X, 1)
            + cs * torch.tanh(self._linear(U, 1))
            + 0.5 * cs * self._linear(U, 1)
            + 0.6 * torch.tanh(self._linear(W, 1))
        )
        # Effect modified across the LATENT severity threshold (U, X), not Z.
        tau = (
            0.4
            + 0.8 * (X[:, :1] > 0.0).to(self.dtype)
            - 0.4 * (latent_risk > latent_cut).to(self.dtype)
            + 0.2 * torch.sin(self._linear(X, 1))
        )
        hsig = 0.15 + 0.5 * torch.sigmoid(torch.abs(self._linear(U, 1)))
        eps = hsig * self._randn(T, 1)
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 4: Count / utilization outcome with frailty
    # ==================================================================
    def _m04_count_utilization(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._patient_covariates(T, d_X)

        # U: frailty (gamma-like, positive, heavy right tail), partly from X.
        frailty = self._gamma_like(
            self._linear(X, d_U), shape=self._uniform_scalar(1.2, 3.0)
        )
        U = self._standardize(frailty)
        cs = self._confound_scale()

        # Proxies: prior-utilization-like signals (over-dispersed) + a frailty index.
        Z = self._skew_positive(0.4 * self._linear(X, d_Z) + self._linear(U, d_Z), 0.4)
        W = self._skew_positive(0.4 * self._linear(X, d_W) + self._linear(U, d_W), 0.4)

        raw_logit = (
            0.5 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.6 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # Expected event rate (positive). Treatment lowers the rate multiplicatively.
        log_rate0 = (
            0.3 * self._mlp_feature(X, 1)
            + cs * 0.7 * self._linear(U, 1)
            + 0.4 * torch.tanh(self._linear(W, 1))
        )
        log_irr = -(
            0.3 + 0.4 * torch.sigmoid(self._linear(X, 1))
        )  # incidence-rate ratio<1
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.5))
        Y0 = torch.exp((log_rate0 + eps).clamp(-6, 6))  # positive rate
        Y1 = torch.exp((log_rate0 + log_irr + eps).clamp(-6, 6))
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 5: Censored survival outcome with frailty
    # ==================================================================
    def _m05_censored_survival(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._patient_covariates(T, d_X)
        U = self._standardize(self._gamma_like(self._linear(X, d_U), shape=2.0))
        cs = self._confound_scale()

        Z = self._mlp_feature(self._cat(X, U), d_Z) + self._noise(T, d_Z, 0.35)
        W = self._mlp_feature(self._cat(X, U), d_W) + self._noise(T, d_W, 0.35)

        raw_logit = (
            0.6 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.7 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # Log survival time: frailty shortens survival; treatment prolongs it.
        log_t0 = (
            1.0
            + 0.4 * self._mlp_feature(X, 1)
            - cs * 0.6 * self._linear(U, 1)
            + 0.3 * torch.tanh(self._linear(W, 1))
        )
        benefit = 0.3 + 0.5 * torch.sigmoid(self._linear(X, 1))
        eps = self._noise(
            T, 1, sigma=self._log_unif_scalar(0.3, 0.7)
        )  # Weibull-ish scatter
        t0 = torch.exp((log_t0 + eps).clamp(-6, 6))
        t1 = torch.exp((log_t0 + benefit + eps).clamp(-6, 6))
        horizon = torch.quantile(t1, self._uniform_scalar(0.7, 0.9))  # admin censoring
        Y0 = torch.clamp(t0, max=horizon)
        Y1 = torch.clamp(t1, max=horizon)
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 6: Customer uplift / promotion targeting
    # ==================================================================
    def _m06_customer_uplift(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)
        # U: latent customer value/intent, partly from observed features.
        U = self._standardize(self._mlp_feature(X, d_U) + self._noise(T, d_U, 0.7))
        cs = self._confound_scale()

        # Engagement proxies (clicks/opens): skewed, some binary.
        Z = self._skew_positive(0.4 * self._linear(X, d_Z) + self._linear(U, d_Z), 0.4)
        W = self._skew_positive(0.4 * self._linear(X, d_W) + self._linear(U, d_W), 0.4)
        if d_W >= 2:
            W[:, 0:1] = self._binary_from_score(W[:, 0:1])

        # Promotions targeted at predicted-high-value customers.
        raw_logit = (
            0.5 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.8 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # Spend outcome: zero-inflated + skewed. Promotion lifts spend, more for
        # high-intent customers (positive effect modification).
        log_spend0 = (
            0.3 * self._mlp_feature(X, 1)
            + cs * 0.7 * self._linear(U, 1)
            + 0.4 * torch.tanh(self._linear(W, 1))
        )
        uplift = 0.3 + 0.6 * torch.sigmoid(self._linear(U, 1))
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.25, 0.6))
        spend0 = torch.exp((log_spend0 + eps).clamp(-6, 6))
        spend1 = torch.exp((log_spend0 + uplift + eps).clamp(-6, 6))
        # Zero-inflation: some customers never spend regardless of treatment.
        active = (
            self._rand(T, 1) < (0.55 + 0.4 * torch.sigmoid(self._linear(X, 1)))
        ).to(self.dtype)
        Y0 = active * spend0
        Y1 = active * spend1
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 7: Self-selection into an educational program
    # ==================================================================
    def _m07_self_selection_education(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)  # demographics, prior schooling
        # U: unobserved ability/motivation.
        U = self._standardize(0.5 * self._linear(X, d_U) + self._noise(T, d_U, 0.9))
        cs = self._confound_scale()

        # Z: standardized test score (treatment-inducing proxy of ability).
        Z = 0.4 * self._linear(X, d_Z) + self._linear(U, d_Z) + self._noise(T, d_Z, 0.5)
        # W: motivation/engagement survey (outcome-inducing proxy), ordinal.
        W = self._bounded_ordinal(
            0.3 * self._linear(X, d_W) + self._linear(U, d_W),
            n_levels=self._randint(4, 6),
        )

        # Able/motivated students self-select into the program.
        raw_logit = (
            0.4 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.8 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # Later achievement: ability raises it; program helps, more for low-baseline.
        f0 = (
            0.5 * self._mlp_feature(X, 1)
            + cs * 0.8 * self._linear(U, 1)
            + 0.4 * self._linear(W, 1)
        )
        tau = (
            0.5
            + 0.6 * torch.sigmoid(-self._linear(U, 1))
            + 0.2 * torch.tanh(self._linear(X, 1))
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.3, 0.8))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 8: Returns to treatment with ability confounding
    # ==================================================================
    def _m08_returns_to_treatment(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)
        U = self._standardize(0.4 * self._linear(X, d_U) + self._noise(T, d_U, 1.0))
        cs = self._confound_scale()

        # Two distinct imperfect ability proxies.
        Z = 0.3 * self._linear(X, d_Z) + self._linear(U, d_Z) + self._noise(T, d_Z, 0.6)
        W = 0.3 * self._linear(X, d_W) + self._linear(U, d_W) + self._noise(T, d_W, 0.6)

        raw_logit = (
            0.4 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.6 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        # log-wage outcome: ability raises wage; heterogeneous positive returns.
        log_w0 = (
            2.0
            + 0.4 * self._mlp_feature(X, 1)
            + cs * 0.7 * self._linear(U, 1)
            + 0.3 * torch.tanh(self._linear(W, 1))
        )
        ret = (
            0.15
            + 0.25 * torch.sigmoid(self._linear(X, 1))
            + 0.1 * torch.tanh(self._linear(U, 1))
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.5))
        Y0 = torch.exp((log_w0 + eps).clamp(-3, 8))
        Y1 = torch.exp((log_w0 + ret + eps).clamp(-3, 8))
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 9: Multisite / hierarchical random effects
    # ==================================================================
    def _m09_multisite_random_effects(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        n_sites = self._randint(4, 10)
        site = torch.randint(0, n_sites, (T,), generator=self.rng)
        X = self._correlated_X(T, d_X)

        site_lat = self._rsu(n_sites, d_U, low=0.5, high=1.4)
        U = self._standardize(
            self._linear(X, d_U)
            + site_lat[site]
            + self._noise(T, d_U, self._log_unif_scalar(0.3, 0.8))
        )
        cs = self._confound_scale()

        # Proxies are clean read-outs of U (no separate site injection).
        Z = self._mlp_feature(self._cat(X, U), d_Z) + self._noise(T, d_Z, 0.4)
        W = self._mlp_feature(self._cat(X, U), d_W) + self._noise(T, d_W, 0.4)

        raw_logit = (
            0.5 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.7 * torch.tanh(Z[:, :1])
        )  # Z -> A only (NC.1)
        p = self._propensity(raw_logit)

        f0 = (
            self._mlp_feature(X, 1)
            + cs * torch.tanh(self._linear(U, 1))
            + 0.5 * cs * self._linear(U, 1)
            + 0.6 * torch.tanh(self._linear(W, 1))
        )  # W -> Y only
        subgroup = torch.where(X[:, :1] > 0.0, 1.0, -1.0)
        tau = 0.6 + 0.4 * subgroup + 0.3 * torch.tanh(self._linear(X, 1))
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.7))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 10: Nonlinear measurement model of a latent factor
    # ==================================================================
    def _m10_nonlinear_measurement_factor(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)
        U = self._standardize(0.5 * self._linear(X, d_U) + self._noise(T, d_U, 0.9))
        cs = self._confound_scale()

        # Nonlinear, saturating, ordinal proxy measurements of the latent factor.
        lam_Z = self._rsu(d_U, d_Z, low=0.7, high=1.6)
        lam_W = self._rsu(d_U, d_W, low=0.7, high=1.6)
        Z = torch.tanh(U @ lam_Z + 0.3 * self._linear(X, d_Z)) + self._noise(
            T, d_Z, 0.3
        )
        W = torch.tanh(U @ lam_W + 0.3 * self._linear(X, d_W)) + self._noise(
            T, d_W, 0.3
        )
        if d_Z >= 2:
            Z[:, 0:1] = self._bounded_ordinal(Z[:, 0:1], n_levels=5)
        if d_W >= 2:
            W[:, 0:1] = self._bounded_ordinal(W[:, 0:1], n_levels=5)

        raw_logit = (
            0.5 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.7 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        w_proj = self._linear(W, 1)
        f0 = (
            0.5 * self._linear(X, 1)
            + cs * torch.tanh(self._linear(U, 1))
            + 0.5 * cs * self._linear(U, 1)
            + 0.6 * torch.sign(w_proj) * torch.sqrt(w_proj.abs() + 1e-3)
        )
        tau = (
            0.3
            + 0.8 * (X[:, :1] > 0.0).to(self.dtype) * torch.sigmoid(self._linear(X, 1))
            + 0.2 * torch.tanh(self._linear(U, 1))
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.7))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 11: Non-additive (interaction) confounding
    # ==================================================================
    def _m11_interaction_confounding(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)
        U = self._standardize(0.5 * self._linear(X, d_U) + self._noise(T, d_U, 0.6))
        cs = self._confound_scale()

        UX = U[:, :1] * X[:, :1]
        cross = self._linear(self._linear(X, d_U) * U, 1)

        Z = (
            self._linear(U, d_Z)
            + self._rsu_scalar(0.2, 0.8) * X[:, :1]
            + self._noise(T, d_Z, 0.4)
        )
        W = (
            self._linear(U, d_W)
            + self._rsu_scalar(0.2, 0.8) * X[:, :1]
            + self._noise(T, d_W, 0.4)
        )

        raw_logit = (
            0.4 * self._linear(X, 1)
            + cs * 0.7 * self._linear(U, 1)
            + cs * self._rsu_scalar(0.5, 1.2) * UX
            + 0.4 * cross
            + 0.6 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        f0 = (
            0.4 * self._linear(X, 1)
            + cs
            * self._rsu_scalar(0.6, 1.0)
            * (U[:, :1] * torch.tanh(self._linear(W, 1)))
            + cs
            * self._rsu_scalar(0.4, 0.8)
            * torch.sigmoid(self._linear(U, 1) + self._linear(X, 1))
        )
        tau = 0.4 + 0.7 * torch.tanh(self._linear(X, 1) + UX)
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.2, 0.7))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ==================================================================
    # Mechanism 12: Near-null effect under strong confounding
    # ==================================================================
    def _m12_near_null_strong_confounding(self, T: int) -> Tuple[Tensor, ...]:
        d_U, d_X, d_W, d_Z = self._sample_dims()
        X = self._correlated_X(T, d_X)
        U = self._standardize(0.6 * self._linear(X, d_U) + self._noise(T, d_U, 0.8))
        cs = max(self._confound_scale(), 1.2)  # force strong confounding here

        Z = self._linear(self._cat(X, U), d_Z) + self._noise(T, d_Z, 0.5)
        W = self._linear(self._cat(X, U), d_W) + self._noise(T, d_W, 0.5)

        raw_logit = (
            0.7 * self._linear(X, 1)
            + cs * self._linear(U, 1)
            + 0.7 * torch.tanh(self._linear(Z, 1))
        )
        p = self._propensity(raw_logit)

        w_proj = self._linear(W, 1)
        f0 = (
            0.6 * self._linear(X, 1)
            + cs * 0.9 * self._linear(U, 1)
            + 0.7 * torch.sign(w_proj) * torch.sqrt(w_proj.abs() + 1e-3)
        )
        # Sparse, tiny true effect: mostly exactly zero.
        null_frac = self._uniform_scalar(0.7, 0.95)
        is_null = self._rand(T, 1) < null_frac
        tau = torch.where(
            is_null,
            torch.zeros(T, 1, dtype=self.dtype),
            self._rsu_scalar(0.05, 0.3) * torch.tanh(self._linear(X, 1)),
        )
        eps = self._noise(T, 1, sigma=self._log_unif_scalar(0.3, 0.9))
        Y0 = f0 + eps
        Y1 = f0 + tau + eps
        return self._finalize(U, X, W, Z, p, Y0, Y1, d_U, d_X, d_W, d_Z)

    # ------------------------------------------------------------------
    # Realistic structural building blocks
    # ------------------------------------------------------------------

    def _confound_scale(self) -> float:
        """U-coefficient magnitude, floored by min_confounding_strength."""
        lo = float(self.cfg.min_confounding_strength)
        return self._uniform_scalar(lo, lo + 0.9)

    def _patient_covariates(self, T: int, d_X: int) -> Tensor:
        """Mixed clinical-style covariates: continuous + binary + ordinal, correlated."""
        X = self._correlated_X(T, d_X)
        if d_X >= 1:  # a binary indicator (e.g. sex / a flag)
            X[:, 0:1] = self._binary_from_score(X[:, 0:1] + 0.5 * self._randn(T, 1))
        if d_X >= 3:  # an ordinal index (e.g. comorbidity 0..4)
            X[:, 2:3] = self._bounded_ordinal(X[:, 2:3], n_levels=5)
        return self._standardize(X)

    def _skew_positive(self, lin: Tensor, base_sigma: float = 0.6) -> Tensor:
        """Right-skewed positive variable via exponentiation of a noisy linear term."""
        T, d = lin.shape
        raw = 0.7 * self._standardize(lin) + self._noise(T, d, base_sigma)
        return torch.exp(raw.clamp(-3.0, 3.0))

    def _gamma_like(self, lin: Tensor, shape: float = 2.0) -> Tensor:
        """Gamma-ish positive frailty whose mean shifts with a linear term."""
        T, d = lin.shape
        # Sum of `k` squared normals ~ chi2 (gamma), then modulate the mean by lin.
        k = max(1, int(round(shape)))
        g = sum(self._randn(T, d) ** 2 for _ in range(k)) / k
        return g * torch.exp(0.4 * self._standardize(lin).clamp(-3, 3))

    def _bounded_ordinal(self, score: Tensor, n_levels: int = 5) -> Tensor:
        """Map a latent score to ordinal levels 0..n_levels-1 (e.g. a questionnaire)."""
        s = self._standardize(score)
        qs = torch.linspace(0.0, 1.0, n_levels + 1, dtype=self.dtype)[1:-1]
        thr = torch.quantile(s, qs.clamp(0, 1))
        return torch.bucketize(s.contiguous(), thr).to(self.dtype)

    def _cat(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.cat([a, b], dim=1)

    # ------------------------------------------------------------------
    # Finalize / generic helpers (kept from the original contract)
    # ------------------------------------------------------------------

    def _finalize(
        self, U, X, W, Z, A_probs, Y0, Y1, d_U, d_X, d_W, d_Z
    ) -> Tuple[Tensor, ...]:
        T = X.shape[0]
        U = self._inject_missing(U)
        X = self._inject_missing(X)
        W = self._inject_missing(W)
        Z = self._inject_missing(Z)

        A_probs = (
            A_probs.view(T, 1)
            .to(self.dtype)
            .clamp(self.cfg.overlap_eps, 1.0 - self.cfg.overlap_eps)
        )
        A = (self._rand(T, 1) < A_probs).to(self.dtype)

        obs_noise = float(self.cfg.extra_observation_noise) * self._randn(T, 1)
        Y_obs = A * Y1 + (1.0 - A) * Y0 + obs_noise

        self._validate_dataset(
            U=U,
            X=X,
            W=W,
            Z=Z,
            A=A,
            A_probs=A_probs,
            Y_obs=Y_obs,
            Y0=Y0,
            Y1=Y1,
            d_U=d_U,
            d_X=d_X,
            d_W=d_W,
            d_Z=d_Z,
        )
        return (
            self._pad(U, self.cfg.max_dim_U),
            self._pad(X, self.cfg.max_dim_X),
            self._pad(W, self.cfg.max_dim_W),
            self._pad(Z, self.cfg.max_dim_Z),
            A.to(self.dtype),
            A_probs.to(self.dtype),
            Y_obs.to(self.dtype),
            Y0.to(self.dtype),
            Y1.to(self.dtype),
            torch.tensor(d_U, dtype=torch.long),
            torch.tensor(d_X, dtype=torch.long),
            torch.tensor(d_W, dtype=torch.long),
            torch.tensor(d_Z, dtype=torch.long),
        )

    def _check_dims(self) -> None:
        c = self.cfg
        for name in ["U", "X", "W", "Z"]:
            lo = getattr(c, f"min_dim_{name}")
            hi = getattr(c, f"max_dim_{name}")
            if lo < 1 or hi < lo:
                raise ValueError(f"Invalid dimensions for {name}: min={lo}, max={hi}")
        if c.max_dim_W < max(c.min_dim_W, c.min_dim_U):
            raise ValueError("max_dim_W must be >= max(min_dim_W, min_dim_U)")
        if c.max_dim_Z < max(c.min_dim_Z, c.min_dim_U):
            raise ValueError("max_dim_Z must be >= max(min_dim_Z, min_dim_U)")
        if not (0.0 <= c.missing_prob < 1.0):
            raise ValueError("missing_prob must be in [0, 1)")
        if not (0.0 < c.overlap_eps < 0.5):
            raise ValueError("overlap_eps must be in (0, 0.5)")

    def _sample_dims(
        self, force_d_U: Optional[int] = None
    ) -> Tuple[int, int, int, int]:
        c = self.cfg
        d_U = (
            int(force_d_U)
            if force_d_U is not None
            else self._randint(c.min_dim_U, c.max_dim_U)
        )
        d_X = self._randint(c.min_dim_X, c.max_dim_X)
        min_W = max(c.min_dim_W, d_U)
        min_Z = max(c.min_dim_Z, d_U)
        if min_W > c.max_dim_W or min_Z > c.max_dim_Z:
            raise ValueError("Cannot enforce dim(W), dim(Z) >= dim(U).")
        d_W = self._randint(min_W, c.max_dim_W)
        d_Z = self._randint(min_Z, c.max_dim_Z)
        return d_U, d_X, d_W, d_Z

    def _resolve_train_size(self, train_size: int | float, T: int) -> int:
        if isinstance(train_size, float):
            if train_size == 1.0:
                return T
            if not 0.0 < train_size < 1.0:
                raise ValueError("float train_size must be in (0, 1]")
            return max(1, min(T - 1, int(round(train_size * T))))
        return max(1, min(T, int(train_size)))

    def _with_seed(self, seed: int):
        outer = self

        class SeedContext:
            def __enter__(self):
                self.old = outer.rng
                outer.rng = torch.Generator(device="cpu")
                outer.rng.manual_seed(int(seed))

            def __exit__(self, *a):
                outer.rng = self.old

        return SeedContext()

    def _rand(self, *s):
        return torch.rand(*s, generator=self.rng, dtype=self.dtype)

    def _randn(self, *s):
        return torch.randn(*s, generator=self.rng, dtype=self.dtype)

    def _randint(self, low, high):
        if high < low:
            high = low
        return int(torch.randint(low, high + 1, (1,), generator=self.rng).item())

    def _uniform_scalar(self, low, high):
        return float(low + (high - low) * self._rand(1).item())

    def _log_unif_scalar(self, low, high):
        if low <= 0 or high <= 0:
            raise ValueError("log-uniform bounds must be positive")
        return float(
            math.exp(
                math.log(low) + (math.log(high) - math.log(low)) * self._rand(1).item()
            )
        )

    def _rsu(self, *s, low=0.3, high=1.5):
        mag = low + (high - low) * self._rand(*s)
        signs = torch.where(self._rand(*s) < 0.5, -1.0, 1.0).to(self.dtype)
        return signs * mag

    def _rsu_scalar(self, low=0.3, high=1.5):
        if low < 0:
            return self._uniform_scalar(low, high)
        mag = self._uniform_scalar(low, high)
        return float((-1.0 if self._rand(1).item() < 0.5 else 1.0) * mag)

    def _linear(self, x, out_dim, low=0.3, high=1.5):
        if out_dim == 0:
            return self._empty(x.shape[0])
        if x.ndim != 2:
            raise ValueError(f"_linear expects [N, D], got {tuple(x.shape)}")
        in_dim = x.shape[1]
        if in_dim == 0:
            return torch.zeros(x.shape[0], out_dim, dtype=self.dtype)
        w = self._rsu(in_dim, out_dim, low=low, high=high) / math.sqrt(float(in_dim))
        return x @ w

    def _mlp_feature(self, x, out_dim, hidden=None):
        if out_dim == 0:
            return self._empty(x.shape[0])
        if hidden is None:
            hidden = max(4, min(16, 2 * x.shape[1]))
        bias = self._rsu(1, hidden, low=0.0, high=0.5)
        h = torch.tanh(self._linear(x, hidden, low=0.2, high=1.2) + bias)
        return self._linear(h, out_dim, low=0.2, high=1.2)

    def _correlated_X(self, T, d_X):
        r = min(max(2, d_X // 2), 4)
        factors = self._randn(T, r)
        loadings = self._rsu(r, d_X, low=0.2, high=1.2)
        eps = 0.4 * self._randn(T, d_X)
        return self._standardize(factors @ loadings / math.sqrt(float(r)) + eps)

    def _noise(self, n, d, sigma=1.0):
        return self._empty(n) if d == 0 else float(sigma) * self._randn(n, d)

    def _standardize(self, x, eps=1e-6):
        if x.numel() == 0:
            return x
        return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(
            eps
        )

    def _propensity(self, raw_logit):
        z = self._standardize(raw_logit).clamp(
            -self.cfg.logit_clip, self.cfg.logit_clip
        )
        eps = self.cfg.overlap_eps
        return eps + (1.0 - 2.0 * eps) * torch.sigmoid(z)

    def _binary_from_score(self, score):
        p = torch.sigmoid(self._standardize(score))
        return (self._rand(*p.shape) < p).to(self.dtype)

    def _inject_missing(self, x):
        p = float(self.cfg.missing_prob)
        if p <= 0 or x.numel() == 0:
            return x
        mask = self._rand(*x.shape) < p
        out = x.clone()
        out[mask] = 0.0
        return out

    def _pad(self, x, max_dim):
        if x.ndim != 2:
            raise ValueError(f"_pad expects [T, d], got {tuple(x.shape)}")
        T, d = x.shape
        if d > max_dim:
            raise ValueError(f"Active dim {d} exceeds max_dim={max_dim}")
        out = torch.zeros(T, max_dim, dtype=self.dtype)
        out[:, :d] = x.to(self.dtype)
        return out

    def _empty(self, n):
        return torch.empty(n, 0, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataset(
        self, *, U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z
    ):
        T = X.shape[0]
        expected = {
            "U": (T, d_U),
            "X": (T, d_X),
            "W": (T, d_W),
            "Z": (T, d_Z),
            "A": (T, 1),
            "A_probs": (T, 1),
            "Y_obs": (T, 1),
            "Y0": (T, 1),
            "Y1": (T, 1),
        }
        tensors = {
            "U": U,
            "X": X,
            "W": W,
            "Z": Z,
            "A": A,
            "A_probs": A_probs,
            "Y_obs": Y_obs,
            "Y0": Y0,
            "Y1": Y1,
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensors[name].shape)}, expected {shape}"
                )
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if not ((A_probs >= 0).all() and (A_probs <= 1).all()):
            raise ValueError("A_probs outside [0, 1]")
        if not ((A == 0) | (A == 1)).all():
            raise ValueError("A is not binary")

    def _validate_batch_shapes(self, **t):
        B, T, _ = t["X"].shape
        expected = {
            "U": (B, T, self.cfg.max_dim_U),
            "X": (B, T, self.cfg.max_dim_X),
            "W": (B, T, self.cfg.max_dim_W),
            "Z": (B, T, self.cfg.max_dim_Z),
            "A": (B, T, 1),
            "A_probs": (B, T, 1),
            "Y_obs": (B, T, 1),
            "Y0": (B, T, 1),
            "Y1": (B, T, 1),
        }
        for name, exp in expected.items():
            if tuple(t[name].shape) != exp:
                raise ValueError(f"{name} shape {tuple(t[name].shape)}, expected {exp}")
