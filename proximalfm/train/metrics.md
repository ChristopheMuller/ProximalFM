# Metrics catalogue

Every key ProximalFM logs, what it means, what value to want, and the caveat that makes it misleading.

---

## 1. Three surfaces

Each answers a different question, on its own data, in its own namespace. Nothing is shared between
them except the metric *functions*.

| Surface | Prefix | Data | Question | Kinds |
|---|---|---|---|---|
| **Train** | `loss/`, `target/`, `noise/` | on-the-fly prior | is the optimization healthy? | loss & batch diagnostics **only** |
| **Val** | `val/` | `synthetic/` 480 realistic mechanisms (out-of-prior) | does it generalize off the prior? | point only |
| **Prior** | `prior/` | `posterior_eval/` frozen high-`K` set (in-prior) | is the posterior any good? | point **and** distributional |

Plus two non-surfaces: `diag/*` (schedules and derived quantities put on another surface's axis) and
a handful of flat keys (`lr`, `prior_time`, `train_time`).

The three boundaries are load-bearing, and each is a thing someone will try to "fix":

- **Train carries no point metrics.** They would be computed on the same distribution as the loss, at
  a batch size chosen for throughput rather than estimator variance. They tell you nothing the loss
  does not. Removed in step 1.
- **Val carries no distributional metrics.** See [§7](#7-why-valdist-does-not-exist). This is the
  single most likely wrong contribution to this codebase.
- **Prior carries both.** It is the only surface with a low-noise `tau_star`, so it is the only place
  a distribution can be scored honestly — and its point metrics are worth having *separately* from
  Val's precisely because they are in-prior, which makes them the sensitive guard against a
  distributional loss quietly trading away point accuracy.

### Cadence

| Surface | When | Gate |
|---|---|---|
| `loss/*`, `target/*`, `noise/*` | every step | — |
| `val/*` | every `--point_val_interval` steps, and step 1 | `--val_dir` set |
| `prior/*`, `diag/sigma` | every `--dist_eval_interval` steps, and step 1 | `--dist_eval_dir` set |

`--dist_eval_interval` defaults to `-1`, which resolves once at `Trainer.__init__` to
`--point_val_interval`; `0` disables. Every rank must reach `dist_validate()` on the same steps —
the episodes are sharded round-robin and the metrics all-reduced, so a rank that skips it deadlocks
the group.

### Scale

Everything on every surface is in **standardized outcome units**. On the train and val surfaces that
is the per-episode context standardization. On the prior surface it is the frozen set's
`y_context_std` at `n_max`, taken from `frozen_meta.pt` — per-episode standardization is random and
correlated with `n_ctx`, so scoring in each episode's own moving scale manufactures trends. See
`_dist_eval.py`'s module docstring.

---

## 2. Reading a run, in order

Each entry conditions the ones below it. This is the same order `python -m posterior_eval.score`
prints its verdict block.

1. **`prior/dist/out_of_grid_frac`** — if this is not ≈ 0, `--cate_vmin`/`--cate_vmax` are too tight
   and every number below it is scored against targets no forecast on this grid can cover. Fix the
   grid first. (`target/cate_out_of_grid_frac` is the training-side twin.)
2. **`prior/dist/sd_over_rmse`** — 1.0 is calibrated, below 1 is over-confident. **Never read a width
   or a width trend without it.**
3. **`prior/dist/pit_ks` vs `prior/dist/pit_ks_band`**, then `pit_ks_episode_p90` vs
   `pit_ks_episode_band`. The pooled statistic passing while the per-episode tail fails is
   cancellation, not calibration.
4. **`prior/dist/crps` against `prior/dist/crps_over_oracle`** — as the ratio approaches 1 the
   remaining CRPS is the bin grid, and the remedy is more `--nbins`, not more steps.
5. **`diag/sigma` against `prior/dist/mean_sd`** — if the width is sitting on the schedule's floor you
   are measuring the schedule.
6. **`prior/point/*` against `val/point/*`** — in-prior point accuracy is the guard that a
   distributional objective has not traded away the point estimate.

---

## 3. Training surface

Logged every step. All training keys are **means over the step's micro-batches**: each micro-batch
contributes `value / num_micro_batches` and `run_batch` sums them, rescaling by
`num_micro_batches / successful_micro_batches` if any micro-batch was skipped for OOM.

### `loss/*` — the objective and its parts

Which keys exist depends on `--loss_type`. `loss/total` is the only one common to all four.

| Key | Loss types | What it is |
|---|---|---|
| `loss/total` | all | The scalar actually backpropagated, per step. |
| `loss/y0`, `loss/y1` | `bins_po` | Per-arm binned cross-entropy. |
| `loss/cate` | `bins_po`, `hl`, `cate_single*` | The CATE-consistency term, **already multiplied by `--lambda_cate`**. A zero here means the weight is zero, not that the term is satisfied. |
| `loss/ate_nll`, `loss/he_crps` | `cate_decomposition` | The ATE-NLL and heterogeneity-CRPS parts, each already multiplied by its lambda. |
| `loss/y0_hl`, `loss/y1_hl` | `hl` | Per-arm HL-Gaussian NLL. |
| `loss/cate_dist` | `cate_single*` | The single-head distributional NLL over `tau`. Under `cate_single_noise_aware` this is the deconvolved form; the key does not change between the two, which is deliberate — checkpoints load across them untouched. |
| `loss/hl_sigma` | `hl`, `cate_single*` | The current HL-Gaussian sigma from `get_hl_sigma()`. **A schedule, not a result.** See [`diag/sigma`](#diagsigma). |

Direction: all `loss/*` lower is better, except `loss/hl_sigma`, which is not a quality metric at
all — it follows `--hl_start_sigma` → `--hl_final_sigma` over
`--hl_warmup_sigma_steps` / `--hl_linear_decay_steps` regardless of the model.

### `target/*` — what the batch actually contained

Diagnostics of the generated targets, not of the model. They catch a broken prior or a mis-scaled
grid before you spend a day blaming the architecture.

| Key | What it is | Want |
|---|---|---|
| `target/cate_mean` | Mean of the standardized MC CEPO target `mu1 − mu0`. | ≈ 0, stable |
| `target/cate_std` | Its SD. | O(1), stable |
| `target/sampled_ite_mean` | Mean of the *sampled* ITE `y1 − y0` (one draw, not the MC mean). | ≈ `target/cate_mean` |
| `target/ite_minus_cate_mse` | `E[(ITE − CATE)²]`, the irreducible outcome-noise gap between the two estimands. | — |
| `target/cate_out_of_grid_frac` | Share of targets outside `[--cate_vmin, --cate_vmax]`. `cate_single_noise_aware` only. | ≈ 0 |

`target/ite_minus_cate_mse` is the number that makes the estimand trap concrete: it is exactly the
excess MSE any ITE-scored evaluation carries over a CATE-scored one, and therefore the size of the
error [§7](#7-why-valdist-does-not-exist) is about. It does not go to zero and should not be expected
to.

A rising `target/cate_out_of_grid_frac` means the loss is being driven by rows it cannot represent —
widen the grid rather than tuning anything else.

### `noise/*` — the noise-aware objective

`--loss_type cate_single_noise_aware` only. Everything here is in **head units** (the model's own
standardized grid).

| Key | What it is | Want |
|---|---|---|
| `noise/mean_s` | `E[√s²]`, the mean per-row sampling SD of the MC target (`S2` from the batch tuple). | small vs `mean_sd_pred` |
| `noise/mean_sd_pred` | Mean predicted SD, as the **discrete** moment `Σ pᵢ(cᵢ−μ)²`. | — |
| `noise/s_over_sd_pred` | `mean_s / mean_sd_pred`. | < 0.2 |
| `noise/frac_fallback` | Share of rows with no replicate variance (`K = 1`, context rows, or data predating `S2`), which fall back to the plain sigma-smoothed NLL. | ≈ 0 |
| `noise/mean_K` | Mean realized `K_eff` after `--prior_mc_mem_budget` capping. | = `--mcmc_cate` |

**`noise/mean_sd_pred` is not comparable to `prior/dist/mean_sd`.** It is the discrete moment over
bin centres; `mean_sd` includes the within-bin `w²/12` term. The two series differ by that term by
construction. Plotting them together is a mistake the `_dist_metrics` docstring calls out explicitly.

`noise/s_over_sd_pred` is a *series*, not a one-off check, because `sd_pred` shrinks as the posterior
sharpens: a replicate count that was comfortably sufficient early can become the binding constraint
later in the same run. Its prior-surface counterpart is `prior/dist/s_over_sd`.

---

## 4. Val surface — `val/point/*`

One full pass over `--val_dir` (the 480 hand-written realistic mechanisms), scored by
`CausalEvaluator` with prefix `val/point/`.

**Per-episode metrics are summarized by their median**, with `/p25` and `/p75` companions — not by a
mean. The R²-style metrics are unbounded below, so a handful of near-homogeneous episodes can drag a
mean arbitrarily negative while the model is fine.

**Near-homogeneous episodes are dropped** from every scale-free metric: an episode with
`Var(τ) < 1e-6` (`CausalEvaluator.homogeneous_var_threshold`) contributes to `cate/rmse`,
`cate/sign_agreement` and the ATE keys but not to `cate/npehe`, `cate/r2`, `cate/r2_shape` or
`ate/bias_norm`. This matches `benchmarks/bench_metrics.py`'s `ITE_VAR_THRESHOLD`. It also means
those four keys are computed on a *different episode subset* than the others — do not algebraically
combine them.

| Key | Formula (per episode, then median) | Want |
|---|---|---|
| `val/point/cate/rmse` | `√ mean_i (τ̂ᵢ − τᵢ)²` | ↓ |
| `val/point/cate/npehe` | `√ (MSE / Var(τ))` | ↓; **1.0 is the do-nothing baseline** (predicting the episode's constant ATE) |
| `val/point/cate/r2` | `1 − MSE / Var(τ)` = `1 − npehe²` | ↑, max 1 |
| `val/point/cate/r2_shape` | `1 − Var(τ̂ − τ) / Var(τ)` | ↑, max 1 |
| `val/point/cate/sign_agreement` | share of rows with `sign(τ̂) = sign(τ)` | ↑ |
| `val/point/ate/bias_norm` | `(mean τ̂ − mean τ) / SD(τ)`, signed | 0 |
| `val/point/po/rmse_y0`, `po/rmse_y1` | per-arm RMSE | ↓ |

Each of the above also emits `<key>/p25` and `<key>/p75`.

| Key | Formula (over the whole pass) | Want |
|---|---|---|
| `val/point/ate/rmse` | `√ mean_episodes (ATÊ − ATE)²` | ↓ |
| `val/point/ate/r2` | `1 − MSE_ATE / Var_episodes(ATE)` | ↑ |
| `val/point/eval/episodes` | episodes accumulated in the pass | constant across passes |
| `val/time` | wall-clock seconds for the pass (flat under `val/`, not `val/point/` — a pass diagnostic, not a point-vs-dist metric) | — |

Caveats worth knowing:

- **`cate/r2_shape` minus `cate/r2` is the bias share.** `r2_shape` removes the constant offset, so a
  large gap between them means the model has the shape right and the level wrong — a different repair
  from a low `r2_shape`.
- **`cate/sign_agreement` is not accuracy near zero.** Where the true effect is ≈ 0 the sign is noise,
  so the achievable ceiling depends on the episode's effect distribution, and 0.5 is chance only if
  effects are sign-balanced.
- **`po/*` exists only for `--loss_type bins_po` and `hl`.** The `cate_*` heads predict the effect
  directly and have nothing honest to pass, so the keys are simply absent rather than zero-filled
  (step 2).
- **`ate/*` is a point estimate with no credible interval**, here and everywhere. See
  [§6](#why-there-is-no-ate-credible-interval).

---

## 5. Prior surface — `prior/*`

One pass over `--dist_eval_dir`, the frozen set built by `posterior_eval/`. Produced by
`score_frozen_set` and split in two: point metrics from the same `CausalEvaluator`, distributional
metrics from `PosteriorEvaluator`.

Preconditions, both enforced at `Trainer.__init__` rather than hundreds of steps in:

- **`--loss_type` must be `cate_single` or `cate_single_noise_aware`.** `bins_po` and `hl` raise
  `UnsupportedLossTypeError`: the posterior of a difference is not identified by its two per-arm
  marginals, so there is nothing to score. `cate_decomposition` is refused too — its grid is anchored
  on a per-row predicted ATE, so there is no single shared grid.
- **`frozen_meta.pt` must be present and well-formed** in the frozen set's root. The flag points at
  the root, holding both `val/` and the sidecar — *not* at `val/`.

Scoring is **forward-only, single pass — never `predict_ensemble`**. Averaging feature permutations
yields a mixture: strictly wider and differently calibrated. It is a legitimate object to evaluate,
but a different one, and mixing the two makes every width number meaningless.

### `prior/point/*`

Identical definitions to [§4](#4-val-surface--valpoint), same evaluator, same
median-over-episodes and same homogeneous-episode exclusion — on the frozen in-prior episodes
instead: `cate/rmse`, `cate/npehe`, `cate/r2`, `cate/r2_shape`, `cate/sign_agreement`,
`ate/bias_norm` (each with `/p25`, `/p75`), `ate/rmse`, `ate/r2`, `eval/episodes`. No `po/*` — the
frozen pass only ever feeds `pred_cate`/`true_cate`.

One key exists here and nowhere else:

| Key | What it is | Want |
|---|---|---|
| `prior/point/cate/rmse_pooled` | `√( Σ_rows (τ̂ − τ*)² / Σ rows )` — pooled over **rows**, accumulated in the driver outside either evaluator | ↓ |

`cate/rmse` is a *median over episodes*; `cate/rmse_pooled` is a mean over rows. They are different
numbers and neither is wrong. `rmse_pooled` exists because it is exactly what
`posterior_eval/report.py` prints at `n = n_max`, so a mismatch between the two is a scale bug and
this is the cheapest place to catch one (`test_pooled_rmse_matches_the_report_path`).

### `prior/dist/*`

Definitions and closed forms are in [`_dist_metrics.py`](../proximalfm/train/_dist_metrics.py). All
of these read the histogram as a **piecewise-uniform density**, not as point masses at bin centres.
Everything is pooled over rows (row-weighted sums), except the PIT-derived keys.

**The headline four.**

| Key | Formula | Want | The trap |
|---|---|---|---|
| `crps` | `∫ (F(t) − 1{t ≥ y})² dt`, exact closed form | ↓ | Proper, so it ranks models — but it mixes calibration and sharpness and therefore never *diagnoses* one. Reduces to \|error\| for a point forecast, so it is on the scale of an absolute error. |
| `sd_over_rmse` | `√( E[Var P] / (MSE_obs − s²) )` | **1.0** | `< 1` is over-confidence. See [§6](#6-the-traps-in-full). |
| `pit_ks` | KS distance of `u = F(y)` from `Uniform(0,1)` | `< pit_ks_band` | Shrinks with `n`, so the bare value is not comparable across evaluation sets — always read it against the band. |
| `crps_over_oracle` | `crps / oracle_discretized_crps` | ↓, **but see below** | Once it approaches 1 the remaining CRPS is the *bin grid*, not the model. |

**Coverage and width.** Reported in pairs, and to be read in pairs: coverage alone is trivially
achieved by widening, width alone by being confidently wrong.

| Key | What it is | Want |
|---|---|---|
| `cov50`, `cov90` | Empirical coverage of the central 50 % / 90 % interval `[F⁻¹((1−L)/2), F⁻¹((1+L)/2)]` | 0.50 / 0.90 |
| `width50`, `width90` | Mean width of that interval | ↓ *at fixed coverage* |

**PIT, pooled and per-episode.**

| Key | What it is | Want |
|---|---|---|
| `pit_ks` | KS statistic of the pooled PIT sample | `< pit_ks_band` |
| `pit_ks_band` | Asymptotic Kolmogorov critical value at α = 0.05, `√(−½ ln(α/2)) / √n` | — |
| `pit_ks_episode_median` | Median of the per-episode KS statistics | `< pit_ks_episode_band` |
| `pit_ks_episode_p90` | Their 90th percentile | `< pit_ks_episode_band` |
| `pit_ks_episode_band` | Median of the per-episode bands | — |

PIT *shape* reads the failure mode, and the KS statistic alone does not: U-shaped (mass at 0 and 1)
is too narrow, dome-shaped is too wide, tilted is biased.

The bands are a **guide, not an exact test**, on two counts: they are asymptotic, and PIT values from
the same episode are dependent, so the effective sample size is below `n`. Treat a marginal excursion
as marginal.

The pooled PIT sample is capped at `max_pit` (200 000) by uniform subsampling *at compute time*, so
it stays an unbiased draw across all episodes rather than a prefix. Under DDP the cap is applied per
rank as `max_pit // world_size` before the gather, which means **the pooled `pit_ks` is not bitwise
reproducible across world sizes** — it is the same statistic on a different subsample. The
per-episode keys are exact.

**The variance identity and its parts.** All from one pass; `sd_over_rmse` is the one to report and
the rest are there to debug it.

| Key | What it is |
|---|---|
| `mean_var` | `E[Var P]`, the mean predicted variance (includes the `w²/12` within-bin term) |
| `mean_sd` | `E[SD P]`, the mean predicted SD |
| `mse_obs` | `E[(μ_P − y)²]`, the observed MSE against the frozen target |
| `mean_s2` | `E[s²]`, the mean per-row sampling variance of that target, from the sidecar's `s2_std` |
| `mse_corrected` (internal) | `max(mse_obs − mean_s2, 0)` — clamped, because with a good model and a noisy target the empirical MSE can fall below `s²` and a negative denominator is not a ratio |
| `var_ratio` | `mean_var / mse_corrected`; 1 when calibrated |
| `sd_over_rmse` | `√var_ratio` — the form to report, because it is in SD units, so 0.8 reads directly as "intervals 20 % too narrow" |
| `s_over_sd` | `√(mean_s2) / √(mean_var)` — the gate on whether the target noise is worth correcting at all |

The `s²` subtraction is **unconditional**, and its sign matters: omit it and the denominator is
inflated, making the model read as *more* over-confident than it is.

`var_ratio` is a **global** identity, not a per-row one — `Var P` is not an estimate of that row's
squared error. There is no meaningful per-row version of this ratio; do not construct one.

**Health and bookkeeping.**

| Key | What it is | Want |
|---|---|---|
| `out_of_grid_frac` | Share of targets outside `[bin_edges[0], bin_edges[-1]]` | ≈ 0 — read this **before** anything else |
| `crps_oracle` | The oracle CRPS itself (denominator of `crps_over_oracle`) | — |
| `eval/episodes`, `eval/rows` | Counts accumulated in the pass | constant across passes |

Out-of-grid targets are scored **honestly, not clipped**: CRPS is charged the distance to the nearest
edge and the PIT is pinned to exactly 0 or 1. Clipping instead would make an unrepresentable target
look merely extreme.

### `prior/*` flat keys

| Key | What it is |
|---|---|
| `prior/time` | Wall-clock seconds for the pass |
| `prior/n_ctx` | Context prefix length scored. Equals `n_max` in training; only differs on an `n`-sweep |
| `prior/grid_scale` | The factor `episode_predictive` applied to the head's own grid to land in the frozen space (`y_std_predict / y_std_frozen`), pooled row-weighted. **Exactly 1.0 at `n_ctx == n_max`**, i.e. on every in-training pass |

`prior/grid_scale` looks redundant today precisely because it is 1. It is published anyway because
`diag/sigma` is built from it, and a caller that hardcoded 1.0 would be wrong exactly where an
`n`-sweep at shorter prefixes gets interesting.

### `diag/sigma`

| Key | Formula | Emitted |
|---|---|---|
| `diag/sigma` | `get_hl_sigma() * prior/grid_scale` | from `dist_validate()`, alongside `prior/dist/mean_sd` |

The same number as `loss/hl_sigma`, put on the width axis. `loss/hl_sigma` is logged from the
micro-batch path, in head units, at the training cadence — three separate reasons a plot of it
against `prior/dist/mean_sd` is not a comparison. `diag/sigma` is in `mean_sd`'s units at `mean_sd`'s
cadence. Both keys are kept; `loss/hl_sigma` is correct on the training surface.

### What is *not* on this surface

Step 7 of the plan — the target-noise convolution — **is not built**. There is no
`convolve_predictive`, no `crps_noise_floor`, and no `--dist_eval_noise_threshold`. What *did* land
is the part that matters at current noise levels: the `s²` subtraction inside the variance identity,
which is unconditional, and `prior/dist/s_over_sd`, which is the measured ratio that decides when the
convolution becomes worth building. The induced width inflation is `√(1 + (s/σ)²) − 1`, i.e. under
2 % below `s_over_sd = 0.2`. Watch that key: it drifts *up* as the posterior sharpens, so a fine-tune
that succeeds is what eventually makes step 7 necessary.

---

## 6. The traps in full

### `sd_over_rmse` — never report a width without it

Target 1.0. Below 1 is over-confidence, above 1 is a wasted posterior.

Every width-like number on the prior surface (`mean_sd`, `width50`, `width90`, and any trend in them)
is uninterpretable on its own, because narrowing is trivially achievable and looks like progress. A
beautifully shaped shrinkage curve at `sd_over_rmse = 0.6` is a model that is confidently wrong. Pair
them, always, in the same plot if possible.

`cov50`/`cov90` and the PIT are the other half of the diagnosis: `sd_over_rmse` says *whether* the
scale is wrong, the PIT shape says *how*.

### `crps_over_oracle` — the difference between converged and grid-limited

`oracle_discretized_crps` is the CRPS of a forecaster that knows the target exactly and puts all its
mass in the target's bin. It still cannot express a point mass on a finite grid, so it pays
`wᵢ(v³ + (1−v)³)/3 ∈ [wᵢ/12, wᵢ/3]`.

As `crps_over_oracle → 1`, the remaining CRPS is the **bin grid**, not the model: the remedy is more
`--nbins`, not more steps. Without this key a converged run and a grid-limited run produce identical
curves — `crps` flat and low in both cases.

It is a diagnostic ratio, not a normalized score. Committing to one bin is the natural floor, not a
proven infimum over all grid-representable distributions. Read it as "how much of my CRPS is grid,
not model."

### `diag/sigma` vs `prior/dist/mean_sd` — the schedule floors the width

The HL-Gaussian target is a Gaussian of width `sigma` discretized onto the grid, so a head that fits
it *perfectly* reports `√(sigma² + w²/12)` and no less. `sigma` and the grid's `w/√12` are the two
floors under `mean_sd`, and they add in quadrature.

The consequence: a posterior sitting on that floor stops narrowing because the **schedule** stopped,
not because the model converged. The sd curve flattens at exactly the wrong moment and reads as
convergence. This is the most likely misreading of the whole fine-tune, which is why `diag/sigma` is
emitted on `mean_sd`'s axis at `mean_sd`'s cadence rather than left as `loss/hl_sigma`.

### The `w²/12` term — two SD series that are not the same series

`prior/dist/mean_sd` reads the histogram as a density and picks up the within-bin variance `w²/12`.
`noise/mean_sd_pred` is the discrete moment over bin centres and does not. A head with all its mass
in one bin reports `w/√12` for the first and 0 for the second.

Both are correct for what they are — `mean_sd_pred` is a diagnostic of the *loss*, `mean_sd` a
calibration metric — but they differ by a constant floor and must never be plotted together or
compared.

### Pooled vs per-episode KS — pooling hides cancellation

Over-confident on half the episodes and under-confident on the other half is **uniform overall**. The
pooled `pit_ks` passes its band and the posterior is badly miscalibrated everywhere.

`pit_ks_episode_p90 ≥ pit_ks_episode_band` while `pit_ks < pit_ks_band` is precisely that signature.
`posterior_eval.score` prints it as an explicit warning; there is a test constructing two
opposite-direction episode groups and asserting the pooled statistic passes while the p90 fails.

### Median vs pooled RMSE

`prior/point/cate/rmse` is a median over episodes. `prior/point/cate/rmse_pooled` is a mean over
rows. Comparing either to a number from elsewhere requires knowing which one you have — the pooled
key exists solely so there is a quantity that must match `posterior_eval/report.py` exactly.

### The homogeneous-episode exclusion

`cate/npehe`, `cate/r2`, `cate/r2_shape` and `ate/bias_norm` are computed on a strict subset of
episodes (those with `Var(τ) ≥ 1e-6`). If the prior's heterogeneity distribution shifts, that subset
changes size and these four metrics move for reasons that have nothing to do with the model. Their
episode count is not `eval/episodes`, and it is not logged.

### Why there is no ATE credible interval

`ate/*` is a point estimate on every surface, deliberately. The head predicts a distribution over
`tau` for each row **independently**; there is no cross-row joint, so the posterior of a row-average
is not identified — any dependence structure between rows is unconstrained, and the interval you
would get from treating them as independent is an arbitrary choice of that structure, not an
inference. `ate/rmse`, `ate/r2` and `ate/bias_norm` are all point accuracy. (Notes §2d.)

---

## 7. Why `val/dist/*` does not exist

**The obvious improvement, and the one that would be silently wrong.**

The `val/` prefix reserves a `dist` segment that has no producer. This is not an oversight and adding
one against the current `--val_dir` would produce a plausible, wrong number rather than an error.

The realistic mechanisms in `synthetic/` carry **no Monte-Carlo CEPO target**. Each row has a single
sampled potential-outcome pair, so `y1 − y0` is a draw of the *ITE*, not an estimate of the CATE.
Scoring a predictive distribution against it is a proper scoring rule — for the **ITE predictive**,
which is a strictly wider object than the CATE posterior by exactly the outcome noise. The size of
that gap is logged: it is `target/ite_minus_cate_mse`.

So a CRPS or a PIT computed there would be *minimized by an over-wide posterior*. Every calibration
metric would reward the exact failure the fine-tune exists to avoid, and nothing would look broken:
`sd_over_rmse` would sit near 1 for a model whose CATE posterior is far too wide.

The frozen prior set exists for this reason and no other: high `K` gives it a low-noise `tau_star`,
and `frozen_meta.pt` carries the residual `s²` so what noise remains is subtracted rather than
ignored.

This changes only when the realistic generator grows replicates — plan step 10, not before. Until
then, distributional scoring happens on `prior/*` only.

---

## 8. Everything else

### Flat per-step keys

| Key | What it is |
|---|---|
| `lr` | Current learning rate from the scheduler |
| `train_time` | Wall-clock seconds for the step's forward/backward |
| `prior_time` | Prior generation time, **amortized**: the block's total generation time divided by the steps it feeds. A diagnostic of the generator, not a component of `train_time` |

These predate the namespace scheme and are flat by inertia rather than by design.

### Run summary (wandb `run.summary`, once at startup)

`trainable_params`, `frozen_params`, `total_params`, `loaded_checkpoint_params`, `pct_params_loaded`.

### `sys/*`

Reserved for wandb's automatic system metrics (GPU utilization, memory, …). ProximalFM writes nothing
under it.

---

## 9. `--best_metric`

Default `val/point/cate/npehe`, with `--best_metric_mode min`. Any key from the val or prior surface
works; `_check_best_metric_config` warns at startup about three configurations that silently no-op:

- `--save_best` with no `--val_dir`;
- a `prior/dist/*` `best_metric` with no `--dist_eval_dir`;
- a name whose direction contradicts `--best_metric_mode`. The heuristic is textual: `r2` and
  `sign_agreement` are treated as higher-is-better, and `rmse`, `npehe`, `mse`, `bias`, `loss`,
  `crps` as lower-is-better. A key matching neither list (e.g. `prior/dist/cov90` or
  `prior/dist/sd_over_rmse`, both of which have a *target* rather than a direction) gets no warning
  in either mode — and `sd_over_rmse` is genuinely unsuitable as a `best_metric`, since 1.0 is the
  target and both `min` and `max` walk away from it.

`_maybe_save_best` only fires on a step where the surface producing the monitored key actually ran,
so a `prior/*` `best_metric` with `--dist_eval_interval` coarser than `--point_val_interval` does not
warn on the in-between validation passes.

---

## 10. Adding a metric

1. **Pick the surface first.** If it cannot be computed honestly on that surface's data, the answer
   is a different surface, not a caveat in the docstring. [§7](#7-why-valdist-does-not-exist) is the
   worked example.
2. **Pure function in [`_dist_metrics.py`](../proximalfm/train/_dist_metrics.py)** if it is
   distributional, tested against a closed form in `tests/test_dist_metrics.py`. Point metrics stay
   fused into `CausalEvaluator`'s accumulator — three lines of arithmetic each, nothing to factor out.
3. **Accumulate as a row- or episode-weighted sum** in the evaluator, and add the key to
   `PosteriorEvaluator._sum_keys()` — that list is built from `self.levels`, not from
   `sorted(self._sums)`, because a rank with an empty shard has no keys at all and a mismatched
   all-reduce payload is a hang, not an error.
4. **Anything accumulated in the driver** (`score_frozen_set`'s `mse_sum` / `scale_sum` / `rows`)
   needs its own explicit reduce. This is the easiest of the three to forget, and the failure is
   silent: it reports rank 0's shard.
5. **Add it here**, with its target value and its caveat.
