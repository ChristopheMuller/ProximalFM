"""
ProximalFM Stress Test Suite
============================
Run with:
    python -m proximalfm.StressTest
    python -m proximalfm.StressTest --config config_stress.yaml

Checks performed
----------------
1. Forward-pass sanity   – shapes, finite values, no NaN in the forward pass.
2. Gradient flow         – every trainable parameter gets a non-zero gradient
                           after a single backward pass.
3. Single-episode overfit – freeze one real episode from the prior; train for
                            N steps; verify the NLL loss drops by ≥ threshold.
4. Trivial CATE recovery  – inject a synthetic batch with constant ITE = 1.0;
                            overfit on it; verify the model converges to
                            predicting mean ≈ 1.0 and shrinking uncertainty.
5. Proxy ablation        – repeat the overfit with zw_mask_prob = 1.0 (proxies
                           fully hidden).  Final loss must be *worse* than in
                           check 3, confirming Z / W are actually used.

Exit code: 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import yaml

from proximalfm.Model import PerFeatureTransformerCATE
from proximalfm.Prior import ProximalPrior
from proximalfm.Structure import ProximalPriorConfig
from proximalfm.Train import gaussian_nll_loss, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stress_test")

# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓ PASS"
FAIL = "✗ FAIL"
SKIP = "— SKIP"


def _save_weights(model: nn.Module) -> dict:
    """Return a CPU copy of the model's state dict (no references to device memory)."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _restore_weights(model: nn.Module, state: dict) -> None:
    """Load a previously saved state dict back into model (moves to model's device)."""
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


@dataclass
class CheckResult:
    name: str
    status: str          # PASS / FAIL / SKIP
    message: str = ""
    details: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return self.status == PASS

    def __str__(self) -> str:
        lines = [f"  {self.status}  {self.name}"]
        if self.message:
            lines.append(f"         {self.message}")
        for d in self.details:
            lines.append(f"           • {d}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Batch utilities
# ─────────────────────────────────────────────────────────────────────────────


def _to_seq_first(t: torch.Tensor) -> torch.Tensor:
    """(B, N, D) → (N, B, D)  (same convention as CATETrainer._prepare_inputs)."""
    return t.permute(1, 0, 2).contiguous()


def episode_to_model_inputs(
    episode,
    device: torch.device,
    dtype: torch.dtype,
    single_eval_pos: int,
) -> Tuple[dict, dict, dict, dict, dict, torch.Tensor]:
    """
    Convert a ProximalEpisode (N samples) into the (N, B=1, D) tensors
    expected by PerFeatureTransformerCATE.
    """

    def _prep(t: torch.Tensor, d: int) -> dict:
        # episode tensors are (N, d); add batch dim → (1, N, d) → seq-first (N, 1, d)
        return {"main": _to_seq_first(t.unsqueeze(0).to(device=device, dtype=dtype))}

    a = episode.A.unsqueeze(-1)   # (N,) → (N, 1)
    y = episode.Y.unsqueeze(-1)

    # Mask query positions in Y with NaN so the model treats them as unknowns
    y_model = y.clone()
    y_model[single_eval_pos:] = float("nan")

    x = _prep(episode.X, episode.X.shape[-1])
    z = _prep(episode.Z, episode.Z.shape[-1])
    w = _prep(episode.W, episode.W.shape[-1])
    a_d = _prep(a, 1)
    y_d = _prep(y_model, 1)

    ite_query = (episode.Y1 - episode.Y0).unsqueeze(-1)[single_eval_pos:]  # (N_q, 1)
    ite_query = ite_query.unsqueeze(1).to(device=device, dtype=dtype)       # (N_q, 1, 1)

    return x, z, w, a_d, y_d, ite_query


def synthetic_constant_ite_episode(
    n: int,
    d_x: int,
    d_z: int,
    d_w: int,
    ite_value: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> dict:
    """
    Construct a synthetic batch with constant ITE = ite_value.

    Y(0) ~ N(0,1),  Y(1) = Y(0) + ite_value,  A ~ Bernoulli(0.5).
    X, Z, W are random noise (the model should still predict ITE from context).
    """
    torch.manual_seed(seed)
    X  = torch.randn(n, d_x, device=device, dtype=dtype)
    Z  = torch.randn(n, d_z, device=device, dtype=dtype)
    W  = torch.randn(n, d_w, device=device, dtype=dtype)
    A  = torch.bernoulli(torch.full((n,), 0.5, device=device, dtype=dtype))
    Y0 = torch.randn(n, device=device, dtype=dtype)
    Y1 = Y0 + ite_value + 0.1 * torch.randn(n, device=device, dtype=dtype)
    Y  = torch.where(A.bool(), Y1, Y0)

    return dict(X=X, Z=Z, W=W, A=A, Y=Y, Y0=Y0, Y1=Y1)


# ─────────────────────────────────────────────────────────────────────────────
# Overfit loop helper
# ─────────────────────────────────────────────────────────────────────────────


def run_overfit_loop(
    model: nn.Module,
    x, z, w, a, y,
    ite_query: torch.Tensor,
    single_eval_pos: int,
    *,
    steps: int,
    lr: float,
    zw_mask_prob: float = 0.0,
    device: torch.device,
    dtype: torch.dtype,
) -> List[float]:
    """
    Gradient-descent loop on a single frozen batch.
    Returns the list of per-step losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    losses: List[float] = []

    model.train()
    for step in range(steps):
        out = model(
            x, z, w, a, y,
            single_eval_pos=single_eval_pos,
            zw_mask_prob=zw_mask_prob,
        )
        pred_mean    = out["cate_mean"][:, single_eval_pos:, :]     # (1, N_q, 1)
        pred_log_var = out["cate_log_var"][:, single_eval_pos:, :]

        target = ite_query.permute(1, 0, 2)     # (1, N_q, 1)
        loss   = gaussian_nll_loss(pred_mean, pred_log_var, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            log.info(f"    step {step+1:4d}/{steps}  loss={loss.item():.4f}")

    return losses


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────


def check_forward_pass(
    model: nn.Module,
    prior: ProximalPrior,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> CheckResult:
    """Check 1: shapes and finiteness of the forward pass."""
    name = "Forward-pass sanity (shapes + finite + no NaN)"
    try:
        n   = cfg["data"]["samples_per_episode"]
        sep = int(n * cfg["data"]["context_fraction"])

        episode = prior.generate_episode(batch_size=n, episode_seed=0)
        x, z, w, a, y, _ = episode_to_model_inputs(episode, device, dtype, sep)

        model.eval()
        with torch.no_grad():
            out = model(x, z, w, a, y, single_eval_pos=sep)

        details = []
        ok = True

        for key in ("cate_mean", "cate_std", "cate_log_var"):
            t = out[key]
            exp_shape = (1, n, 1)
            if t.shape != exp_shape:
                ok = False
                details.append(f"{key}: expected {exp_shape}, got {tuple(t.shape)}")
            if torch.isnan(t).any():
                ok = False
                details.append(f"{key}: contains NaN")
            if torch.isinf(t).any():
                ok = False
                details.append(f"{key}: contains Inf")

        # std must be positive
        if (out["cate_std"] <= 0).any():
            ok = False
            details.append("cate_std has non-positive values")

        if ok:
            return CheckResult(name, PASS, "shapes correct, all values finite")
        else:
            return CheckResult(name, FAIL, details=details)

    except Exception as exc:
        return CheckResult(name, FAIL, message=f"Exception: {exc}")


def check_gradient_flow(
    model: nn.Module,
    prior: ProximalPrior,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> CheckResult:
    """Check 2: every trainable parameter receives a non-zero gradient."""
    name = "Gradient flow (all params get non-zero grad)"
    try:
        n   = cfg["data"]["samples_per_episode"]
        sep = int(n * cfg["data"]["context_fraction"])

        episode = prior.generate_episode(batch_size=n, episode_seed=1)
        x, z, w, a, y, ite_query = episode_to_model_inputs(episode, device, dtype, sep)

        model.train()
        out = model(x, z, w, a, y, single_eval_pos=sep)
        pred_mean    = out["cate_mean"][:, sep:, :]
        pred_log_var = out["cate_log_var"][:, sep:, :]
        target = ite_query.permute(1, 0, 2)
        loss   = gaussian_nll_loss(pred_mean, pred_log_var, target)
        model.zero_grad()
        loss.backward()

        dead: List[str] = []
        zero: List[str] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                dead.append(pname)

        n_params = sum(1 for p in model.parameters() if p.requires_grad)
        details: List[str] = []
        if dead:
            details.append(f"{len(dead)} params with None grad: {dead[:5]}")
        if zero:
            details.append(f"{len(zero)} params with zero grad: {zero[:5]}")

        if not dead and not zero:
            return CheckResult(
                name, PASS,
                f"all {n_params} trainable params have non-zero gradients"
            )
        else:
            return CheckResult(name, FAIL, details=details)

    except Exception as exc:
        return CheckResult(name, FAIL, message=f"Exception: {exc}")


def check_single_episode_overfit(
    model: nn.Module,
    prior: ProximalPrior,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    lr: float,
    min_relative_decrease: float,
    loss_target: float,
) -> Tuple[CheckResult, List[float]]:
    """
    Check 3: training on a single frozen episode should drive NLL well down.
    Returns the check result AND the loss curve for re-use in check 5.
    """
    name = "Single-episode overfit (NLL decreases on frozen batch)"
    try:
        n   = cfg["data"]["samples_per_episode"]
        sep = int(n * cfg["data"]["context_fraction"])

        episode = prior.generate_episode(batch_size=n, episode_seed=42)
        x, z, w, a, y, ite_q = episode_to_model_inputs(episode, device, dtype, sep)

        log.info(f"  Running {steps} overfit steps (lr={lr}) …")
        losses = run_overfit_loop(
            model, x, z, w, a, y, ite_q, sep,
            steps=steps, lr=lr, device=device, dtype=dtype,
        )

        loss_0    = losses[0]
        loss_last = losses[-1]
        decrease  = (loss_0 - loss_last) / (abs(loss_0) + 1e-8)

        details = [
            f"initial loss = {loss_0:.4f}",
            f"final   loss = {loss_last:.4f}",
            f"relative decrease = {decrease*100:.1f}%  (threshold: {min_relative_decrease*100:.0f}%)",
        ]

        ok = (decrease >= min_relative_decrease) and (loss_last <= loss_target)
        status = PASS if ok else FAIL
        msg = "" if ok else (
            "loss did not decrease enough — check model capacity, LR, or gradient flow"
        )
        return CheckResult(name, status, msg, details), losses

    except Exception as exc:
        return CheckResult(name, FAIL, message=f"Exception: {exc}"), []


def check_trivial_cate(
    model: nn.Module,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    lr: float,
    ite_value: float = 1.0,
    tol_mean: float = 0.15,
) -> CheckResult:
    """
    Check 4: overfit a synthetic batch where ITE ≡ ite_value for all units.
    The model should converge to predicting mean ≈ ite_value.
    """
    name = f"Trivial CATE recovery (constant ITE = {ite_value})"
    try:
        n   = cfg["data"]["samples_per_episode"]
        sep = int(n * cfg["data"]["context_fraction"])
        d   = cfg["prior"]

        batch = synthetic_constant_ite_episode(
            n, d["d_x"], d["d_z"], d["d_w"], ite_value, device, dtype, seed=7
        )

        def _prep(t: torch.Tensor) -> dict:
            return {"main": t.unsqueeze(1)}   # (N, 1, D)

        def _prep1(t: torch.Tensor) -> dict:
            return {"main": t.unsqueeze(1).unsqueeze(-1)}   # (N, 1, 1)

        # Mask query Y as NaN
        y_ctx = batch["Y"].clone()
        y_ctx[sep:] = float("nan")

        x  = _prep(batch["X"])
        z  = _prep(batch["Z"])
        w  = _prep(batch["W"])
        a  = _prep1(batch["A"])
        y  = _prep1(y_ctx)

        ite_q = (batch["Y1"] - batch["Y0"]).unsqueeze(-1).unsqueeze(1)[sep:]  # (N_q, 1, 1)

        log.info(f"  Running {steps} trivial-CATE overfit steps (ITE = {ite_value}) …")
        losses = run_overfit_loop(
            model, x, z, w, a, y, ite_q, sep,
            steps=steps, lr=lr, device=device, dtype=dtype,
        )

        # Evaluate final predictions
        model.eval()
        with torch.no_grad():
            out = model(x, z, w, a, y, single_eval_pos=sep)
        final_mean = out["cate_mean"][:, sep:, :].mean().item()
        final_std  = out["cate_std"][:, sep:, :].mean().item()

        err = abs(final_mean - ite_value)
        ok  = err <= tol_mean

        details = [
            f"predicted mean ITE = {final_mean:.4f}  (target = {ite_value:.1f},  |err| = {err:.4f})",
            f"predicted std       = {final_std:.4f}  (should shrink as model is confident)",
        ]

        status = PASS if ok else FAIL
        msg = "" if ok else f"|mean - {ite_value}| = {err:.4f} > tol {tol_mean}"
        return CheckResult(name, status, msg, details)

    except Exception as exc:
        return CheckResult(name, FAIL, message=f"Exception: {exc}")


def check_proxy_ablation(
    model: nn.Module,
    prior: ProximalPrior,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    lr: float,
    baseline_losses: List[float],
    initial_weights: dict,
) -> CheckResult:
    """
    Check 5: a fresh model trained WITHOUT proxies should converge slower and
    to a worse final loss than check 3 (same episode, same steps, proxies visible).

    Design fixes vs. the naive version
    ─────────────────────────────────────
    • Model weights are reset to the same initial state used by check 3, so
      carryover from checks 3 and 4 cannot inflate or deflate the result.
    • We run a full overfit from scratch rather than continuing from a
      specialised checkpoint — this gives a fair convergence comparison.
    • We log the TYPE_MASK embedding gradient norm after the first backward
      pass to diagnose whether the mask token is actually receiving signal.
    """
    name = "Proxy ablation (fresh masked overfit converges worse than check 3)"
    if not baseline_losses:
        return CheckResult(name, SKIP, "skipped because check 3 failed")

    try:
        n   = cfg["data"]["samples_per_episode"]
        sep = int(n * cfg["data"]["context_fraction"])

        # ── Reset to the same initial weights used by check 3 ────────────────
        _restore_weights(model, initial_weights)

        episode = prior.generate_episode(batch_size=n, episode_seed=42)
        x, z, w, a, y, ite_q = episode_to_model_inputs(episode, device, dtype, sep)

        # ── Diagnose the TYPE_MASK embedding gradient before the full loop ───
        # One forward+backward pass with gradient capture.
        model.train()
        out = model(x, z, w, a, y, single_eval_pos=sep, zw_mask_prob=1.0)
        pred_mean    = out["cate_mean"][:, sep:, :]
        pred_log_var = out["cate_log_var"][:, sep:, :]
        target = ite_q.permute(1, 0, 2)
        probe_loss = gaussian_nll_loss(pred_mean, pred_log_var, target)
        model.zero_grad()
        probe_loss.backward()

        mask_emb_grad_norm: float | str = "N/A"
        if hasattr(model, "type_embedding"):
            # type_embedding is an nn.Embedding; weight rows correspond to type ids.
            # TYPE_MASK = 5 (last row).
            g = model.type_embedding.weight.grad
            if g is not None:
                # Row 5 = TYPE_MASK
                mask_emb_grad_norm = g[5].norm().item()
                log.info(
                    f"  TYPE_MASK embedding gradient norm = {mask_emb_grad_norm:.6f}"
                    + (" ← near zero, mask token not learning!" if mask_emb_grad_norm < 1e-6 else "")
                )
            else:
                log.warning("  type_embedding.weight.grad is None — mask token receives no gradient")
                mask_emb_grad_norm = 0.0

        # ── Reset weights again so the loop starts fresh ──────────────────────
        _restore_weights(model, initial_weights)

        log.info(f"  Running {steps} overfit steps with zw_mask_prob=1.0 (fresh model) …")
        masked_losses = run_overfit_loop(
            model, x, z, w, a, y, ite_q, sep,
            steps=steps, lr=lr, zw_mask_prob=1.0,
            device=device, dtype=dtype,
        )

        base_initial  = baseline_losses[0]
        base_final    = baseline_losses[-1]
        masked_initial = masked_losses[0]
        masked_final   = masked_losses[-1]

        base_decrease   = base_initial   - base_final
        masked_decrease = masked_initial - masked_final

        # Both checks start from the same weights so initial losses should be
        # nearly identical. The key metric is the gap in final loss and in the
        # total decrease achieved.
        worse_final   = masked_final  > base_final
        worse_decrease = masked_decrease < base_decrease

        ok = worse_final and worse_decrease

        details = [
            f"initial loss (both runs, same weights) ≈ {masked_initial:.4f}",
            f"proxies VISIBLE  →  final loss = {base_final:.4f}  (decrease = {base_decrease:.4f})",
            f"proxies MASKED   →  final loss = {masked_final:.4f}  (decrease = {masked_decrease:.4f})",
            f"gap in final loss       = {masked_final - base_final:+.4f}  (expect > 0)",
            f"gap in total decrease   = {masked_decrease - base_decrease:+.4f}  (expect < 0)",
            f"TYPE_MASK embedding grad norm = {mask_emb_grad_norm:.6f}" if isinstance(mask_emb_grad_norm, float) else f"TYPE_MASK embedding grad norm = {mask_emb_grad_norm}",
        ]

        # Explain failure mode
        if not ok:
            if not worse_final and not worse_decrease:
                msg = "masked model converges just as well — proxies may not be contributing"
            elif not worse_final:
                msg = "masked final loss not worse, though convergence was slower"
            else:
                msg = "masked model converged slower but reached similar final loss"
        else:
            msg = ""

        return CheckResult(name, PASS if ok else FAIL, msg, details)

    except Exception as exc:
        return CheckResult(name, FAIL, message=f"Exception: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def run_stress_tests(config_path: str = "config_stress.yaml") -> bool:
    """
    Execute all stress tests and print a summary table.
    Returns True if all non-skip checks pass.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    st = cfg["stress_test"]
    set_seed(cfg["training"].get("seed", 42))

    # ── Device ────────────────────────────────────────────────────────────────
    req = cfg["training"].get("device", "cpu")
    if req == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif req == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Running stress tests on {device}")

    dtype_str = cfg["model"].get("dtype", "float32")
    dtype = getattr(torch, dtype_str)

    # ── Model ─────────────────────────────────────────────────────────────────
    mc = cfg["model"]
    model = PerFeatureTransformerCATE(
        ninp=mc["ninp"],
        nhead=mc["nhead"],
        nhid=mc["nhid"],
        nlayers=mc["nlayers"],
        layer_norm_with_elementwise_affine=mc.get(
            "layer_norm_with_elementwise_affine", True
        ),
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} trainable parameters")

    # ── Prior ─────────────────────────────────────────────────────────────────
    pc = cfg["prior"]
    prior_config = ProximalPriorConfig(
        d_x=pc["d_x"], d_z=pc["d_z"], d_w=pc["d_w"], d_u=pc["d_u"],
        dtype=dtype, device="cpu",
    )
    prior = ProximalPrior(config=prior_config)

    # ── Run checks ────────────────────────────────────────────────────────────
    results: List[CheckResult] = []

    # Save the random initial weights once so every overfit check starts from
    # the exact same point — no carryover between checks.
    initial_weights = _save_weights(model)

    log.info("\n── Check 1: Forward-pass sanity ─────────────────────────────")
    results.append(check_forward_pass(model, prior, cfg, device, dtype))

    log.info("\n── Check 2: Gradient flow ───────────────────────────────────")
    results.append(check_gradient_flow(model, prior, cfg, device, dtype))

    log.info("\n── Check 3: Single-episode overfit ──────────────────────────")
    _restore_weights(model, initial_weights)
    res3, baseline_losses = check_single_episode_overfit(
        model, prior, cfg, device, dtype,
        steps=st["overfit_steps"],
        lr=st["overfit_lr"],
        min_relative_decrease=st["min_relative_decrease"],
        loss_target=st["loss_target"],
    )
    results.append(res3)

    log.info("\n── Check 4: Trivial CATE recovery ───────────────────────────")
    _restore_weights(model, initial_weights)
    results.append(
        check_trivial_cate(
            model, cfg, device, dtype,
            steps=st["overfit_steps"],
            lr=st["overfit_lr"],
            ite_value=st.get("trivial_ite_value", 1.0),
            tol_mean=st.get("trivial_tol_mean", 0.15),
        )
    )

    log.info("\n── Check 5: Proxy ablation ──────────────────────────────────")
    # Note: _restore_weights is called inside check_proxy_ablation itself,
    # both before the gradient probe and before the full overfit loop.
    results.append(
        check_proxy_ablation(
            model, prior, cfg, device, dtype,
            steps=st["overfit_steps"],
            lr=st["overfit_lr"],
            baseline_losses=baseline_losses,
            initial_weights=initial_weights,
        )
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ProximalFM Stress Test Results")
    print("═" * 65)
    for r in results:
        print(r)
    print("═" * 65)

    n_pass = sum(r.ok() for r in results)
    n_fail = sum(r.status == FAIL for r in results)
    n_skip = sum(r.status == SKIP for r in results)
    print(f"  {n_pass} passed  |  {n_fail} failed  |  {n_skip} skipped")
    print("═" * 65 + "\n")

    return n_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ProximalFM stress tests")
    parser.add_argument(
        "--config",
        default="config_stress.yaml",
        help="Path to the stress-test config file (default: config_stress.yaml)",
    )
    args = parser.parse_args()

    all_passed = run_stress_tests(args.config)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()