# %%
"""Interactive visualization of the ``cate_single_noise_aware`` loss.

Drives the *actual* training-code functions (``gaussian_bin_mass``,
``compute_cate_noise_aware_nll``, ``compute_cate_distribution_loss`` from
:mod:`proximalfm.train._losses`, and ``Trainer.get_hl_sigma`` from
:mod:`proximalfm.train._run`) with sliders, so nothing here can silently
drift from what training does. See the docstrings of those functions (and
``cate_single_noise_aware`` in ``proximalfm/train/_run.py``) for the math.

Run standalone:

    uv run python visu/visu_noise_aware_loss.py

Panels:
  A (top-left)     Predicted histogram over tau, with the noisy observed
                    target ``t`` and the per-bin likelihood weight used to
                    score it at the current ``(s2, sigma)``.
  B (top-right)     Why ``gaussian_bin_mass`` *integrates* over each bin
                    instead of sampling the density at the bin centre: the
                    two coincide for a wide kernel but diverge once the
                    kernel narrows below a bin width.
  C (bottom-left)   Loss vs. the target's sampling std ``s`` (with a twin
                    axis in ``K``, since ``s = mc_std / sqrt(K)``): the
                    noise-aware NLL vs. the plain sigma-only HL loss, which
                    is blind to ``s`` entirely.
  D (bottom-right)  The ``hl_sigma`` warmup/decay schedule from
                    ``Trainer.get_hl_sigma``, with the current sigma slider
                    value marked, plus the ``s / sd_pred`` diagnostic from
                    ``Trainer._target_noise_metrics`` (below ~0.2 the noise
                    correction is nearly invisible).

Sliders: predicted-posterior mean/sd, the observed noisy target ``t``,
per-replicate CATE std ``mc_std``, replicate count ``K`` (together giving
``s2 = mc_std^2 / K``), and the smoothing ``sigma``.
"""

import types

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from proximalfm.train._run import Trainer
from proximalfm.train._losses import (
    gaussian_bin_mass,
    compute_cate_noise_aware_nll,
    compute_cate_distribution_loss,
)
from proximalfm.train._train_config import build_parser, resolve_hl_final_sigma

# %%
# 1. Configuration
# Bin grid mirrors the model default (cate_vmin/cate_vmax) but with far fewer
# bins so individual bars stay legible; the loss math is scale-free in nbins.
NBINS = 121
VMIN, VMAX = -3.0, 3.0
EPS = 1e-12

DEFAULTS = build_parser().parse_args([])
# --hl_final_sigma defaults to None (derived from the grid); resolve it here so
# the schedule panel plots the number a real run would use.
DEFAULTS.hl_final_sigma = resolve_hl_final_sigma(DEFAULTS)

bin_edges = torch.linspace(VMIN, VMAX, NBINS + 1)
bin_width = float(bin_edges[1] - bin_edges[0])
bin_centers = (bin_edges[:-1] + 0.5 * bin_width).numpy()


def predicted_logits(mean: float, sd: float) -> torch.Tensor:
    """Logits for a Gaussian(mean, sd^2) predictive, built via the *integrated*
    bin mass (the same primitive the loss itself uses), not a sampled density.
    """
    probs = gaussian_bin_mass(bin_edges, torch.tensor(float(mean)), float(sd))
    probs = probs / probs.sum().clamp_min(EPS)
    return torch.log(probs.clamp_min(EPS)).view(1, 1, -1)


def hl_sigma_schedule(step: np.ndarray, start, final, warmup, decay) -> np.ndarray:
    """Vectorized wrapper around the exact ``Trainer.get_hl_sigma`` schedule."""
    out = np.empty_like(step, dtype=float)
    mock = types.SimpleNamespace(
        hl_start_sigma=start,
        hl_final_sigma=final,
        hl_warmup_sigma_steps=warmup,
        hl_linear_decay_steps=decay,
    )
    for i, s in enumerate(step):
        mock.curr_step = int(s)
        out[i] = Trainer.get_hl_sigma(mock)
    return out


# %%
# 2. Figure & static (non-slider) content
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.93, bottom=0.34, hspace=0.8)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
fig.suptitle("cate_single_noise_aware: bins, sigma-smoothing and target noise (s2)")

axA_twin = axA.twinx()

# Panel D's schedule curve is fixed (uses the CLI-configurable defaults from
# _train_config.py); only the "current step / sigma" markers move.
sched_steps = np.linspace(
    0, DEFAULTS.hl_warmup_sigma_steps + DEFAULTS.hl_linear_decay_steps * 1.3, 400
)
sched_sigma = hl_sigma_schedule(
    sched_steps,
    DEFAULTS.hl_start_sigma,
    DEFAULTS.hl_final_sigma,
    DEFAULTS.hl_warmup_sigma_steps,
    DEFAULTS.hl_linear_decay_steps,
)
axD.plot(sched_steps, sched_sigma, color="0.4", lw=1.5, label="get_hl_sigma(step)")
axD.set_yscale("log")
axD.set_xlabel("training step")
axD.set_ylabel("hl_sigma")
axD.set_title("sigma schedule (defaults from _train_config.py)")
sigma_hline = axD.axhline(
    DEFAULTS.hl_start_sigma, color="C1", lw=1.5, label="current sigma"
)
ratio_text = axD.text(0.02, 0.05, "", transform=axD.transAxes, fontsize=9, va="bottom")
axD.legend(loc="upper right", fontsize=8)

# %%
# 3. Sliders
slider_specs = [
    ("pred_mean", -1.5, 1.5, 0.30),
    ("pred_sd", 0.02, 1.2, 0.20),
    ("t (observed target)", -2.0, 2.0, 0.55),
    ("mc_std (per-replicate)", 0.05, 3.0, 1.0),
    ("K (replicates)", 1, 200, 8),
    ("log10(sigma)", -3.0, 0.30, np.log10(0.05)),
]
sliders = {}
slider_axes_bottom = 0.04
for i, (label, lo, hi, init) in enumerate(slider_specs):
    ax = fig.add_axes([0.12, slider_axes_bottom + i * 0.035, 0.76, 0.02])
    step = 1 if label.startswith("K") else None
    sliders[label] = Slider(ax, label, lo, hi, valinit=init, valstep=step)


def get_params():
    mean = sliders["pred_mean"].val
    sd = sliders["pred_sd"].val
    t = sliders["t (observed target)"].val
    mc_std = sliders["mc_std (per-replicate)"].val
    K = sliders["K (replicates)"].val
    sigma = 10 ** sliders["log10(sigma)"].val
    s2 = (mc_std**2) / K
    return mean, sd, t, mc_std, K, sigma, s2


# %%
# 4. Redraw
# axC's secondary (K) axis is a child Axes that ax.cla() does not remove, so
# the stale one from the previous redraw must be torn down explicitly -
# otherwise they stack up call after call and their tick labels overlap.
_state = {"axC_top": None}


def redraw(_event=None):
    mean, sd, t, mc_std, K, sigma, s2 = get_params()
    s = np.sqrt(s2)

    logits = predicted_logits(mean, sd)
    t_tensor = torch.tensor([[t]])
    s2_tensor = torch.tensor([[s2]])

    nll_noise_aware = compute_cate_noise_aware_nll(
        logits, t_tensor, bin_edges, target_var=s2_tensor, sigma=sigma
    ).item()
    nll_plain = compute_cate_distribution_loss(
        logits, t_tensor, bin_edges, sigma=sigma
    ).item()

    probs = torch.softmax(logits, dim=-1).view(-1).numpy()
    kernel_sd = float(np.sqrt(s2 + sigma**2))
    # Per-bin likelihood weight at t: same construction as
    # compute_cate_noise_aware_nll, exposed here for display.
    mass = gaussian_bin_mass(bin_edges, t_tensor, kernel_sd).view(-1).numpy()
    kernel_weight = mass / bin_width
    kernel_weight = kernel_weight / max(kernel_weight.max(), EPS)

    pred_sd_hat = float(np.sqrt(np.sum(probs * (bin_centers - mean) ** 2)))

    # --- Panel A: predicted histogram + likelihood kernel around t ---
    axA.cla()
    axA_twin.cla()
    window = max(4 * sd, 4 * kernel_sd, 6 * bin_width)
    lo, hi = min(t, mean) - window, max(t, mean) + window
    mask = (bin_centers >= lo) & (bin_centers <= hi)
    axA.bar(
        bin_centers[mask],
        probs[mask],
        width=bin_width * 0.9,
        color="C0",
        alpha=0.7,
        label="predicted P(tau) [softmax(logits)]",
    )
    axA.axvline(mean, color="C0", ls="--", lw=1.5, label="predicted mean")
    axA.axvline(t, color="C3", lw=2, label="observed target t")
    axA_twin.plot(
        bin_centers[mask],
        kernel_weight[mask],
        color="C1",
        lw=1.5,
        label="likelihood weight per bin (normalized)",
    )
    axA.set_xlim(lo - window * 0.2, hi + window * 0.2)
    axA.set_xlabel("tau (standardized)")
    axA.set_ylabel("bin probability", color="C0")
    axA_twin.set_ylabel("relative likelihood weight", color="C1")
    axA.set_title(
        f"noise-aware NLL={nll_noise_aware:.3f}  |  plain HL NLL={nll_plain:.3f}"
        f"  |  kernel_sd=sqrt(s2+sigma^2)={kernel_sd:.3f}"
    )
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA_twin.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)

    # --- Panel B: integrated bin mass vs. naive point-sampled density ---
    axB.cla()
    demo_center = 0.0
    window_b = max(4 * kernel_sd, 6 * bin_width)
    maskB = np.abs(bin_centers - demo_center) <= window_b
    integrated = gaussian_bin_mass(
        bin_edges, torch.tensor(demo_center), kernel_sd
    ).numpy()
    naive_density = (
        np.exp(-0.5 * ((bin_centers - demo_center) / kernel_sd) ** 2)
        / (np.sqrt(2 * np.pi) * kernel_sd)
    ) * bin_width
    axB.bar(
        bin_centers[maskB],
        integrated[maskB],
        width=bin_width,
        color="C2",
        alpha=0.6,
        label="gaussian_bin_mass (integrated over bin)",
    )
    axB.plot(
        bin_centers[maskB],
        naive_density[maskB],
        "k.-",
        lw=1,
        ms=3,
        label="density(bin center) x width (naive)",
    )
    ratio = kernel_sd / bin_width
    axB.set_title(
        f"bin integration vs point-sampling  (kernel_sd / bin_width = {ratio:.2f})"
    )
    axB.set_xlabel("tau (standardized), Gaussian centered at 0")
    axB.set_ylabel("mass in bin")
    note = (
        "kernel narrower than a bin: naive sampling collapses toward a spike\n"
        "and misses mass -- integration is what keeps the loss well-behaved."
        if ratio < 1.0
        else "kernel wide relative to a bin: the two agree closely here."
    )
    axB.text(0.02, 0.95, note, transform=axB.transAxes, fontsize=8, va="top")
    axB.legend(loc="upper right", fontsize=7)

    # --- Panel C: loss vs s (target sampling std), twin axis in K ---
    axC.cla()
    s_max = max(mc_std, s, 1e-3) * 1.5
    s_grid = np.linspace(0, s_max, 200)
    # compute_cate_noise_aware_nll reduces over its batch with .mean(), so a
    # per-s curve needs one call per grid point rather than a single batched
    # call across s values.
    nll_per_s = np.array(
        [
            compute_cate_noise_aware_nll(
                logits,
                torch.tensor([[t]]),
                bin_edges,
                target_var=torch.tensor([[s_val**2]]),
                sigma=sigma,
            ).item()
            for s_val in s_grid
        ]
    )
    axC.plot(s_grid, nll_per_s, color="C0", label="noise-aware NLL(s)")
    axC.axhline(nll_plain, color="C4", ls="--", label="plain HL NLL (blind to s)")
    axC.axvline(s, color="C3", lw=1.5, label=f"current s={s:.3f}")
    axC.set_xlabel("s = sqrt(s2)  (target sampling std)")
    axC.set_ylabel("NLL")
    axC.set_title("loss vs. target-noise magnitude", pad=6)
    axC.legend(loc="upper left", fontsize=7)

    # Floor tied to s_max (not a fixed epsilon): s=0 would otherwise imply
    # K -> inf, which blows up the secondary axis's tick range and forces a
    # distracting 1e12-style offset label near the panel title.
    k_floor = max(s_max / 100.0, 1e-6)

    def s_to_K(s_vals):
        return (mc_std / np.maximum(s_vals, k_floor)) ** 2

    def K_to_s(k_vals):
        return mc_std / np.sqrt(np.maximum(k_vals, 1.0))

    if _state["axC_top"] is not None:
        _state["axC_top"].remove()
    _state["axC_top"] = axC.secondary_xaxis("top", functions=(s_to_K, K_to_s))
    axC_top = _state["axC_top"]
    axC_top.set_xlabel(f"implied K (mc_std={mc_std:.2f})", fontsize=8, labelpad=2)
    axC_top.tick_params(labelsize=7)
    # K = (mc_std/s)^2 is steep near s=0, so linear ticks on the primary axis
    # map to K-ticks that crowd and overlap; cap the count instead.
    axC_top.xaxis.set_major_locator(plt.MaxNLocator(4))

    # --- Panel D: schedule marker + s/sd_pred diagnostic ---
    sigma_hline.set_ydata([sigma, sigma])
    s_over_sd = s / max(pred_sd_hat, EPS)
    flag = "  (< 0.2: correction ~invisible)" if s_over_sd < 0.2 else ""
    ratio_text.set_text(
        f"s={s:.3f}   sd_pred={pred_sd_hat:.3f}   s/sd_pred={s_over_sd:.3f}{flag}\n"
        f"K={int(K)}   s2={s2:.4f}"
    )

    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(redraw)

redraw()

# %%
if __name__ == "__main__":
    plt.show()
