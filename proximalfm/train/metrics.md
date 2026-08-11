# Metrics catalogue

Every key ProximalFM logs, what it means, what value to want, and the caveat that makes it misleading.

---

## 1. Three surfaces

Each answers a different question, on its own data, in its own namespace. Nothing is shared between
them except the metric *functions*.

| Surface | Prefix | Data | Question | Kinds |
|---|---|---|---|---|
| **Train** | `loss/`, `target/`, `noise/` | on-the-fly prior | is the optimization healthy? | loss & batch diagnostics **only** |
| **Val** | `synt_point/`, `synt_dist/` | `data/synt_eval/` realistic mechanisms (out-of-prior) | does it generalize off the prior? | point, **and** distributional on a replicate set |
| **Prior** | `prior_point/`, `prior_dist/` | `prior_eval/` frozen high-`K` set (in-prior) | is the posterior any good? | point **and** distributional |

Plus two non-surfaces: `diag/*` (schedules and derived quantities put on another surface's axis) and
a handful of flat keys (`lr`, `prior_time`, `train_time`, `Charts/*`).

The boundaries are load-bearing, and each is a thing someone will try to "fix":

- **Train carries no point metrics.** They would be computed on the same distribution as the loss, at
  a batch size chosen for throughput rather than estimator variance. They tell you nothing the loss
  does not. Removed in step 1.
- **Val depends on the *set*, not the surface, for both kinds.** `synt_point/*` and `synt_dist/*` are
  both a side effect of one pass, `score_frozen_set` over `--synt_eval_dir`, and both require a
  replicate-bearing set (`--mc_cate K`, `K > 1`); on anything else the pass is refused rather than
  computed. See [§7](#7-val_dist-and-the-set-it-may-be-scored-on) — the failure it describes is still
  the single most likely wrong contribution to this codebase. There is no separate point-only pass or
  `--val_dir` any more: a plain, non-replicate set cannot be validated against at all.
- **Prior carries both, in-prior.** Its point metrics are worth having *separately* from Val's
  precisely because they are in-prior, which makes them the sensitive guard against a distributional
  loss quietly trading away point accuracy.

`synt_point/*` and `synt_dist/*` always describe the same dataset by construction — one flag,
`--synt_eval_dir`, one pass.

### Cadence

| Surface | When | Gate |
|---|---|---|
| `loss/*`, `target/*`, `noise/*` | every step | — |
| `synt_point/*` + `synt_dist/*` | every `--synt_eval_interval` steps, and step 1 | `--synt_eval_dir` set |
| `prior_point/*` + `prior_dist/*`, `diag/sigma` | every `--prior_eval_interval` steps, and step 1 | `--prior_eval_dir` set |

`--prior_eval_interval` and `--synt_eval_interval` both default to `50`; `0` disables. Training is
single-process (DDP was removed); `prior_validate()` and `synt_validate()` are two calls into one
`Trainer._validate_surface(surface)`, so the two surfaces share their loading, scoring and logging
path by construction.

### Scale

Everything on every surface is in **standardized outcome units**, but not the same standardization:

- **Train** — the per-episode context standardization.
- **`synt_point/*`, `synt_dist/*`, `prior_point/*`, `prior_dist/*`** — the set's frozen `y_context_std`
  at `n_max`, taken from `frozen_meta.pt`. All four now come from the same evaluator machinery
  (`score_episode`, inside `score_frozen_set`), which reads its target off `record["tau_star_std"]` —
  already in frozen units — so this is not a choice made per surface, it is what the shared code path
  produces. Per-episode standardization is random and correlated with `n_ctx`, so scoring in each
  episode's own moving scale would manufacture trends. See `_dist_eval.py`'s module docstring.

So `synt_point/cate/rmse` and `synt_dist/*`'s widths are **not** in the same units even on the same
data, which is why `sd_over_rmse` is computed inside the dist pass against its own RMSE rather than
against the point surface's.

---

## 2. Reading a run, in order

Each entry conditions the ones below it. This is the same order `python -m prior_eval.score`
prints its verdict block.

1. **`prior_dist/out_of_grid_frac`** — if this is not ≈ 0, `--cate_vmin`/`--cate_vmax` are too tight
   and every number below it is scored against targets no forecast on this grid can cover. Fix the
   grid first. (`target/cate_out_of_grid_frac` is the training-side twin.)
2. **`prior_dist/sd_over_rmse`** — 1.0 is calibrated, below 1 is over-confident. **Never read a width
   or a width trend without it.**
3. **`prior_dist/pit_ks_pooled` vs `prior_dist/pit_ks_pooled_band`**, then `pit_ks/p90` vs
   `pit_ks/band`. The pooled statistic passing while the per-episode tail fails is
   cancellation, not calibration.
4. **`prior_dist/crps` against `prior_dist/crps_over_oracle`** — as the ratio approaches 1 the
   remaining CRPS is the bin grid, and the remedy is more `--nbins`, not more steps.
5. **`diag/sigma` against `prior_dist/mean_sd`** — if the width is sitting on the schedule's floor you
   are measuring the schedule.
6. **`prior_point/*` against `synt_point/*`** — in-prior point accuracy is the guard that a
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
| `loss/cate_mse` | `bins_po`, `hl`, `cate_single*` | The CATE-consistency term as a plain MSE, **unweighted**: `mean((E_P[tau] - tau_target)^2)` on the head's grid scale. This is the interpretable one — comparable across `--lambda_cate` settings and across runs. |
| `loss/cate_mse_weighted` | `bins_po`, `hl`, `cate_single*` | The same term **already multiplied by `--lambda_cate`**, i.e. its actual contribution to `loss/total`. A zero here means the weight is zero, not that the term is satisfied. Was called `loss/cate` before 2026-08-06; runs older than that log the weighted value under the old name. |
| `loss/ate_nll`, `loss/he_crps` | `cate_decomposition` | The ATE-NLL and heterogeneity-CRPS parts, each already multiplied by its lambda. |
| `loss/y0_hl`, `loss/y1_hl` | `hl` | Per-arm HL-Gaussian NLL. |
| `loss/cate_dist` | `cate_single*` | The single-head distributional NLL over `tau`. Under `cate_single_noise_aware` this is the deconvolved form; the key does not change between the two, which is deliberate — checkpoints load across them untouched. |
| `loss/hl_sigma` | `hl`, `cate_single*` | The current HL-Gaussian sigma from `get_hl_sigma()`. **A schedule, not a result.** See [`diag/sigma`](#diagsigma). |

Direction: all `loss/*` lower is better, except `loss/hl_sigma`, which is not a quality metric at
all — it follows `--hl_start_sigma` → `--hl_final_sigma` over
`--hl_warmup_sigma_steps` / `--hl_linear_decay_steps` regardless of the model.

The endpoint is set in **bin widths**, not absolute units: `--hl_final_sigma_bins` (default `2`)
times the width of the grid the loss smooths over — tau's (`--cate_vmin/--cate_vmax`) for
`cate_single*`, the potential-outcome grid (`--vmin/--vmax`) for `hl`, both over `--nbins`
(`_train_config.resolve_hl_final_sigma`). `--hl_final_sigma` still takes an absolute value and
overrides it, but only its ratio to the bin width is meaningful: sigma is a floor on the learnable
width under `cate_single` and a *bias* (too narrow) under `cate_single_noise_aware`, and a sigma
chosen independently of the grid is how it once ended up at 8 bin widths — a width deficit at the
objective's optimum, which no amount of further training removes. The resolved value is what lands
in the run's wandb config.

### `target/*` — what the batch actually contained

Diagnostics of the generated targets, not of the model. They catch a broken prior or a mis-scaled
grid before you spend a day blaming the architecture.

| Key | What it is | Want |
|---|---|---|
| `target/cate_mean` | Mean of the standardized MC CEPO target `mu1 − mu0`. | ≈ 0, stable |
| `target/cate_std` | Its SD. The denominator the other standardized-scale keys are read against. | O(1), stable |
| `target/sampled_ite_mean` | Mean of the *sampled* ITE `y1 − y0` (one draw, not the MC mean). | ≈ `target/cate_mean` |
| `target/ite_minus_cate_mse` | `E[(ITE − CATE)²]`, the irreducible outcome-noise gap between the two estimands. | — |
| `target/cate_out_of_grid_frac` | Share of targets outside `[--cate_vmin, --cate_vmax]`. `cate_single_noise_aware` only. | ≈ 0 |

`target/ite_minus_cate_mse` is the number that makes the estimand trap concrete: it is exactly the
excess MSE any ITE-scored evaluation carries over a CATE-scored one, and therefore the size of the
error [§7](#7-val_dist-and-the-set-it-may-be-scored-on) is about. It does not go to zero and should not be expected
to.

A rising `target/cate_out_of_grid_frac` means the loss is being driven by rows it cannot represent —
widen the grid rather than tuning anything else.

`target/cate_std` is not a quality metric; it is the scale the other standardized-scale numbers are
meaningless without. Four ratios use it as the denominator:

- **`loss/cate_mse` / `cate_std²`** — the null-model line. A model that always predicts the batch
  mean scores exactly `cate_std²`, so the ratio is `1 − R²`.
- **`cate_std` vs `[--cate_vmin, --cate_vmax]`** — the *leading* indicator for
  `target/cate_out_of_grid_frac`, which only moves once the grid is already too tight. `O(1)` against
  the recommended `±5` leaves ~5σ of headroom.
- **`noise/mean_s` / `cate_std`** — whether the noise-aware deconvolution has anything to act on; on the frozen set at `K=4000` it is 0.008, i.e. negligible,
  but it grows as `1/√K` and a training run at low `--mc_cate` sits far higher).
- **`loss/hl_sigma` / `cate_std`** — whether the histogram smoothing is swamping the signal. Sigma is
  a floor on learned width, so a small target spread against a fixed sigma flattens the sd curve at
  the floor; that is the `diag/sigma` trap in [§6](#6-the-traps-in-full) seen from the target side.

Secondary use: it is pooled over all query rows of all episodes in the batch, so it mixes
within-episode heterogeneity with between-episode ATE spread and is **not** a measure of effect
heterogeneity. What that pooling does buy is a batch-composition alarm — if `loss/cate_mse` jumps on
the same step `cate_std` jumps, the batch mix changed and there is nothing to debug in the model.

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

**`noise/mean_sd_pred` is not comparable to `prior_dist/mean_sd`.** It is the discrete moment over
bin centres; `mean_sd` includes the within-bin `w²/12` term. The two series differ by that term by
construction. Plotting them together is a mistake the `_dist_metrics` docstring calls out explicitly.

`noise/s_over_sd_pred` is a *series*, not a one-off check, because `sd_pred` shrinks as the posterior
sharpens: a replicate count that was comfortably sufficient early can become the binding constraint
later in the same run. Its prior-surface counterpart is `prior_dist/s_over_sd`.

**`noise/s_over_sd_pred` and `prior_dist/s_over_sd` answer the same question on different data and
are not the same number.** Four differences, in decreasing order of size:

1. **`K`, and it dominates.** The training key measures the *training batch*, whose targets carry
   `--mc_cate` replicates; the prior key measures the *frozen set*, built at `K = 4000`. Since
   `s ∝ 1/√K`, a run at `--mc_cate 64` sits `√(4000/64) ≈ 7.9×` higher on the numerator alone. This
   is why `prior_dist/s_over_sd ≈ 0.008` can be negligible while the training-side ratio is the
   binding reason `cate_single_noise_aware` exists at all.
2. **Different functional.** `noise/s_over_sd_pred = E[√s²] / E[√Var]` — a ratio of mean *SDs*.
   `prior_dist/s_over_sd = √(E[s²]) / √(E[Var])` — a ratio of root mean *variances*. By Jensen these
   disagree whenever the per-row spread is heterogeneous, which it always is.
3. **Different denominator by the `w²/12` term**, the same trap as `mean_sd_pred` vs `mean_sd`: the
   training key's `var_pred` is the discrete moment over bin centres, the prior key's `mean_var`
   includes the within-bin term. The prior-side denominator is therefore strictly larger.
4. **Different rows and a dilution.** The prior key covers query rows only, every one with a real
   `s²`. The training key covers whatever rows the loss sees, and rows with no replicate variance
   enter `mean_s` as **zeros** rather than being excluded — so a large `noise/frac_fallback` drags
   `noise/mean_s` down and makes the ratio look safer than it is. Read the two keys together.

Scale is *not* on the list: both are standardized, the training side per-episode and the prior side
against the frozen `y_context_std`. Same `s²` definition on both (`Var_k(y1 − y0)/K`, paired), just
computed at a different `K` and normalized differently.

---

## 4. Val surface — `synt_point/*` and `synt_dist/*`

One pass over `--synt_eval_dir` (a realistic replicate set's **root**, not its `val/`), scored by
`score_frozen_set` — the *same* function and the same per-episode forward pass the prior surface uses
([§5](#5-prior-surface)), just pointed at a different set. Point metrics (`CausalEvaluator`) and
distributional metrics (`PosteriorEvaluator`) are both side effects of that one pass, not two separate
ones: `score_episode`'s evaluator fan-out reads both off the same `probs`/`edges`/`tau_star_std`, so
`synt_point/*` costs nothing beyond what `synt_dist/*` already pays for encoding the context.

**Only a replicate-bearing set (`--mc_cate K`, `K > 1`) can be scored here.** The gate is the
sidecar's absence — a plain `--mc_cate 1` set has no `frozen_meta.pt`, and `score_frozen_set` refuses
it (`FileNotFoundError`) at `Trainer.__init__` rather than silently scoring either metric kind against
sampled ITEs. There is no fallback point-only pass over an unreplicated set: `synt_point/*` once
came from a separate batched pass (`point_validate()`) over `--val_dir` that would happily score
against `Y1 − Y0` when `MU0`/`MU1` were absent — that path is gone, along with the inflated
`cate/rmse ≈ √(PEHE² + Var(noise))` numbers it produced. See
[§7](#7-val_dist-and-the-set-it-may-be-scored-on) for the full argument, and for the one number that
does *not* carry over from the prior set: `synt_dist/s_over_sd` is not negligible at realistic `K`.

### `synt_point/*`

Identical definitions to the table below, same evaluator as `prior_point/*` ([§5](#5-prior-surface)),
same median-over-episodes and homogeneous-episode exclusion — scored on the realistic set instead of
the frozen prior set.

**Per-episode metrics are summarized by their median**, with `/p25` and `/p75` companions — not by a
mean. The R²-style metrics are unbounded below, so a handful of near-homogeneous episodes can drag a
mean arbitrarily negative while the model is fine.

**Near-homogeneous episodes are dropped** from every scale-free metric: an episode with
`Var(τ) < 1e-6` (`CausalEvaluator.homogeneous_var_threshold`) contributes to `cate/rmse`,
`cate/sign_agreement` and the ATE keys but not to `cate/npehe`, `cate/r2`, `cate/r2_shape` or
`ate/bias_norm`. This matches `benchmarks/metrics.py`'s `TARGET_VAR_THRESHOLD`. It also means
those four keys are computed on a *different episode subset* than the others — do not algebraically
combine them.

| Key | Formula (per episode, then median) | Want |
|---|---|---|
| `synt_point/cate/rmse` | `√ mean_i (τ̂ᵢ − τᵢ)²` | ↓ |
| `synt_point/cate/npehe` | `√ (MSE / Var(τ))` | ↓; **1.0 is the do-nothing baseline** (predicting the episode's constant ATE) |
| `synt_point/cate/r2` | `1 − MSE / Var(τ)` = `1 − npehe²` | ↑, max 1 |
| `synt_point/cate/r2_shape` | `1 − Var(τ̂ − τ) / Var(τ)` | ↑, max 1 |
| `synt_point/cate/sign_agreement` | share of rows with `sign(τ̂) = sign(τ)` | ↑ |
| `synt_point/ate/bias_norm` | `(mean τ̂ − mean τ) / SD(τ)`, signed | 0 |
| `synt_point/po/rmse_y0`, `po/rmse_y1` | per-arm RMSE | ↓ |

Each of the above also emits `<key>/p25` and `<key>/p75`.

| Key | Formula (over the whole pass) | Want |
|---|---|---|
| `synt_point/ate/rmse` | `√ mean_episodes (ATÊ − ATE)²` | ↓ |
| `synt_point/ate/r2` | `1 − MSE_ATE / Var_episodes(ATE)` | ↑ |
| `synt_point/eval_episodes` | episodes accumulated in the pass | constant across passes |
| `Charts/val_point_time` | wall-clock seconds for the pass (filed under `Charts/` beside `Charts/prior_eval_time` — a pass diagnostic, not a metric about the model) | — |

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

### `synt_dist/*`

Same keys as `prior_dist/*` ([§5](#5-prior-surface)), same frozen scale.

---

## 5. Prior surface — `prior_point/*` + `prior_dist/*`

One pass over `--prior_eval_dir`, the frozen set built by `prior_eval/`. Produced by
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

### `prior_point/*`

Identical definitions to [§4](#4-val-surface--val_point-and-val_dist), same evaluator, same
median-over-episodes and same homogeneous-episode exclusion — on the frozen in-prior episodes
instead: `cate/rmse`, `cate/npehe`, `cate/r2`, `cate/r2_shape`, `cate/sign_agreement`,
`ate/bias_norm` (each with `/p25`, `/p75`), `ate/rmse`, `ate/r2`, `eval_episodes`. No `po/*` — the
frozen pass only ever feeds `pred_cate`/`true_cate`.

One key exists here and nowhere else:

| Key | What it is | Want |
|---|---|---|
| `prior_point/cate/rmse_pooled` | `√( Σ_rows (τ̂ − τ*)² / Σ rows )` — pooled over **rows**, accumulated in the driver outside either evaluator | ↓ |

`cate/rmse` is a *median over episodes*; `cate/rmse_pooled` is a mean over rows. They are different
numbers and neither is wrong. `rmse_pooled` exists because it is exactly what
`prior_eval/report.py` prints at `n = n_max`, so a mismatch between the two is a scale bug and
this is the cheapest place to catch one (`test_pooled_rmse_matches_the_report_path`).

### `prior_dist/*`

Definitions and closed forms are in [`_dist_metrics.py`](../proximalfm/train/_dist_metrics.py). All
of these read the histogram as a **piecewise-uniform density**, not as point masses at bin centres.
Everything is pooled over rows (row-weighted sums), except the PIT-derived keys.

**The headline four.**

| Key | Formula | Want | The trap |
|---|---|---|---|
| `crps` | `∫ (F(t) − 1{t ≥ y})² dt`, exact closed form | ↓ | Proper, so it ranks models — but it mixes calibration and sharpness and therefore never *diagnoses* one. Reduces to \|error\| for a point forecast, so it is on the scale of an absolute error. |
| `sd_over_rmse` | `√( E[Var P] / (MSE_obs − s²) )` | **1.0** | `< 1` is over-confidence. See [§6](#6-the-traps-in-full). |
| `pit_ks_pooled` | KS distance of `u = F(y)` from `Uniform(0,1)` | `< pit_ks_pooled_band` | Shrinks with `n`, so the bare value is not comparable across evaluation sets — always read it against the band. |
| `crps_over_oracle` | `crps / oracle_discretized_crps` | ↓, **but see below** | Once it approaches 1 the remaining CRPS is the *bin grid*, not the model. |

**Coverage and width.** Reported in pairs, and to be read in pairs: coverage alone is trivially
achieved by widening, width alone by being confidently wrong.

| Key | What it is | Want |
|---|---|---|
| `cov50_pooled`, `cov90_pooled` | Empirical coverage of the central 50 % / 90 % interval `[F⁻¹((1−L)/2), F⁻¹((1+L)/2)]`, **row-weighted over the whole pass** | 0.50 / 0.90 |
| `cov50/p25`, `/p50`, `/p75` | Quantiles of the **per-episode** 50 % coverages | p50 ≈ 0.50, IQR ≈ the binomial floor |
| `cov90/p25`, `/p50`, `/p75` | Same for the 90 % interval | p50 ≈ 0.90, IQR ≈ the binomial floor |
| `width50`, `width90` | Mean width of that interval | ↓ *at fixed coverage* |

`cov*_pooled` was called `cov50`/`cov90` before 2026-08-06; the bare names no longer exist.

**Both are means over rows, not over episodes.** `coverage()` reduces over every element it is handed
and `PosteriorEvaluator.update` accumulates `value * n_rows`, so the pass-level number is a
row-weighted mean — an episode with more query rows counts more. On a frozen set with a constant
`n_q` (`frozen_prod_v1`: 128) row-weighted and episode-weighted coincide, but under a *row-level*
strata cut they do not. Note the asymmetry: coverage also gets per-episode quantiles, **width does
not** — there is only the pooled number.

**The interval is central (equal-tailed), so it is contiguous by construction — it is not an HPD
region.** It is defined by two quantiles of the CDF, `[F⁻¹((1−L)/2), F⁻¹((1+L)/2)]`, evaluated by
`predictive_quantile` as the exact inverse of the piecewise-linear `F`. That makes `hi − lo` a single
well-defined number for *any* histogram, including a multimodal one. What it is not is the *shortest*
interval carrying mass `L`: for a bimodal posterior the central interval spans the low-density gap
between the modes, and `width{L}` correspondingly overstates how much of the axis the posterior
actually occupies. The highest-density region is the object that can break into disjoint pieces —
that is precisely why it is not what is reported: it needs a threshold search, its "width" is a total
measure rather than an interval, and it is a nuisance to accumulate and compare across episodes. The
central interval is monotone in `L`, exact, and its coverage is unambiguous. The cost is the
multimodal case, and `width90 / width50` is the free diagnostic for it: a Gaussian gives 3.29/1.35 =
2.44, and a materially larger ratio means heavy tails or separated modes.

**A width floor comes with the sigma schedule and the grid.** `sigma` and `w/√12` floor `mean_sd`
(see [§6](#diagsigma-vs-prior_distmean_sd--the-schedule-floors-the-width)) and therefore floor the
widths too — early in a run with a large `--hl_start_sigma`, a flat `width50`/`width90` is the
schedule, not the model. Check `diag/sigma` before reading any width trend. A nonzero
`out_of_grid_frac` pushes the other way: truncated tails shrink the width while hurting coverage, so
the pair moves in opposite directions for a reason that is not calibration.

**Pooled coverage cancels, and the per-episode quantiles are what catch it.** Over-covering on half
the episodes and under-covering on the other half is a perfect `cov90_pooled = 0.90` with `cov90/p25`
and `cov90/p75` straddling it. This is the coverage-side twin of
[§6](#pooled-vs-per-episode-ks--pooling-hides-cancellation), which makes the same argument for PIT.

**The IQR has a binomial floor — never read it against zero.** A perfectly calibrated model still
scores `Binomial(n, L)/n` per episode, so the p25–p75 gap is `≈ 1.35·√(L(1−L)/n)` from sampling alone:
at `n_q = 128` that is **0.060 for `cov50`** and **0.036 for `cov90`**. Only spread beyond that is
miscalibration. Unlike `pit_ks_pooled`, no band key is emitted — the floor depends only on `n_q` and the
level, both of which you know. Under a *row-level* strata cut the effective `n` is the rows kept per
episode, not `n_q`, so the floor widens accordingly.

Only the pooled keys are additive across strata cuts; the quantiles are sample statistics, exactly
like `pit_ks_pooled`. `tests/test_prior_eval_strata.py` pins both halves of that.

**PIT, pooled and per-episode.**

| Key | What it is | Want |
|---|---|---|
| `pit_ks_pooled` | KS statistic of the pooled PIT sample | `< pit_ks_pooled_band` |
| `pit_ks_pooled_band` | Asymptotic Kolmogorov critical value at α = 0.05, `√(−½ ln(α/2)) / √n` | — |
| `pit_ks/p50` | Median of the per-episode KS statistics | `< pit_ks/band` |
| `pit_ks/p90` | Their 90th percentile | `< pit_ks/band` |
| `pit_ks/band` | Median of the per-episode bands | — |

PIT *shape* reads the failure mode, and the KS statistic alone does not: U-shaped (mass at 0 and 1)
is too narrow, dome-shaped is too wide, tilted is biased.

The bands are a **guide, not an exact test**, on two counts: they are asymptotic, and PIT values from
the same episode are dependent, so the effective sample size is below `n`. Treat a marginal excursion
as marginal.

The pooled PIT sample is capped at `max_pit` (200 000) by uniform subsampling *at compute time*, so
it stays an unbiased draw across all episodes rather than a prefix. The per-episode keys are exact.

**The variance identity and its parts.** All from one pass; `sd_over_rmse` is the one to report and
the rest are there to debug it.

| Key | What it is |
|---|---|
| `mean_var` | `E[Var P]`, the mean predicted variance (includes the `w²/12` within-bin term) |
| `mean_sd` | `E[SD P]`, the mean predicted SD |
| `mse_obs` | `E[(μ_P − τ*)²]`, the uncorrected MSE of the predictive **mean** against the frozen target |
| `mean_s2` | `E[s²]`, the mean per-row sampling variance of that target, from the sidecar's `s2_std` |
| `mse_corrected` (internal) | `max(mse_obs − mean_s2, 0)` — clamped, because with a good model and a noisy target the empirical MSE can fall below `s²` and a negative denominator is not a ratio |
| `var_ratio` | `mean_var / mse_corrected`; 1 when calibrated |
| `sd_over_rmse` | `√var_ratio` — the form to report, because it is in SD units, so 0.8 reads directly as "intervals 20 % too narrow" |
| `s_over_sd` | `√(mean_s2) / √(mean_var)` — the gate on whether the target noise is worth correcting at all |

**`mse_obs` is not an outcome MSE, and "obs" does not mean "observational".** No observed `Y` enters
this path at any point. The `y` argument of `variance_identity` is bound to `record["tau_star_std"]` —
the frozen set's high-`K` Monte-Carlo CEPO target — over the episode's **query rows only** (`probs`
has shape `(n_q, B)`; context rows are the conditioning and are never scored). So it is
`mean_rows (E_P[tau] − τ*)²`: a CATE prediction against a CATE target, not an ITE and not an outcome.
The contrast the name draws is `mse_obs` (raw) vs `mse_corrected` (after subtracting the residual MC
noise `mean_s2`) — *uncorrected*, not *observational*. Scoring against an ITE draw instead is exactly
the [§7](#7-val_dist-and-the-set-it-may-be-scored-on) error, and it would inflate this denominator by
`target/ite_minus_cate_mse`.

**`mse_obs` equals `prior_point/cate/rmse_pooled²`.** Both are the row-weighted mean of the same
`(μ_P − τ*)²` over the same rows — one accumulated inside `PosteriorEvaluator`, the other in the
driver at `score_episode`. They must agree to float precision; a gap means a shard reduction is
wrong. That makes `sd_over_rmse` readable in one line: the width the model *claims* (`mean_var`)
against the error it *makes* (`rmse_pooled²`, denoised).

**The ratio is recomputed from pooled sums, not averaged.** Only the four linear pieces (`mean_var`,
`mean_sd`, `mse_obs`, `mean_s2`) are accumulated row-weighted and all-reduced; `var_ratio` and
`sd_over_rmse` are formed once at the end. It is a **ratio of means, never a mean of ratios** — the
latter is unbounded and dominated by easy episodes with a near-zero denominator. Note also that
`sd_over_rmse = √(E[Var]/…)` is *not* `E[SD]/RMSE`: by Jensen `mean_sd ≤ √mean_var`, so do not
reconstruct one from the other.

The `s²` subtraction is **unconditional**, and its sign matters: omit it and the denominator is
inflated, making the model read as *more* over-confident than it is.

`var_ratio` is a **global** identity, not a per-row one — `Var P` is not an estimate of that row's
squared error. There is no meaningful per-row version of this ratio; do not construct one. It is also
**necessary but not sufficient**: one scalar over the whole pass, so a model 2× too wide on half the
rows and 2× too narrow on the other half scores exactly 1.0. That is the same cancellation as pooled
`pit_ks_pooled`, with the same companions — PIT *shape* says how the scale is wrong, the per-episode
coverage quantiles say whether it is wrong in opposite directions in different places.

A `nan` here means `mse_corrected` collapsed to ≈ 0, i.e. `mse_obs ≤ mean_s2`. At `K = 4000`
(`s_over_sd ≈ 0.008`) that is not a plausible model outcome — suspect a raw-vs-head unit mixup in
`s2_std` first.

**Health and bookkeeping.**

| Key | What it is | Want |
|---|---|---|
| `out_of_grid_frac` | Share of targets outside `[bin_edges[0], bin_edges[-1]]` | ≈ 0 — read this **before** anything else |
| `crps_oracle` | The oracle CRPS itself (denominator of `crps_over_oracle`) | — |
| `eval_episodes`, `eval_rows` | Counts accumulated in the pass | constant across passes |

Out-of-grid targets are scored **honestly, not clipped**: CRPS is charged the distance to the nearest
edge and the PIT is pinned to exactly 0 or 1. Clipping instead would make an unrepresentable target
look merely extreme.

### Stratified cuts — offline only, **no wandb producer**

The same `prior_dist/*` keys, recomputed per stratum of the frozen set, by
`python -m prior_eval.score_strata`. Cuts are defined in `prior_eval/strata.py` and stored as
a `strata.pt` manifest of per-episode (or per-query-row) membership beside `frozen_meta.pt`; nothing
about the 365 MB set is duplicated or rebuilt.

**These keys are never logged**, and the reason is a key budget rather than a correctness one:
~20 keys × 7 cuts × 2 sides is not a dashboard. It is a milestone-checkpoint tool. If a stratum curve
later earns a place in wandb, that is a deliberate decision, not a default. (Contrast
[§7](#7-val_dist-and-the-set-it-may-be-scored-on), where scoring the wrong *set* is refused outright
because the number would be wrong rather than merely numerous.)

Why it is cheap: `PosteriorEvaluator` accumulates row-weighted **sums**, so a stratum's metrics are
exactly what a separate pass over only that stratum's rows would produce. Every cut is scored from
one forward pass per episode, fanned out through `score_episode`'s `dist_evaluators` argument.
`tests/test_prior_eval_strata.py` pins this to `rel=1e-9`.

The reading rule, which is the whole point of the tool:

| Comparable across strata | Not comparable |
|---|---|
| `sd_over_rmse`, `cov50_pooled`, `cov90_pooled`, `cov90/p25`–`p75`, `crps_over_oracle`, `pit_ks_pooled` (vs **its own** band) | `crps`, `mean_sd`, `width50`, `width90`, `cate/rmse` |

Strata differ in intrinsic difficulty *and* in the scale of `tau`, so a worse `crps` under weak
proxies is the expected consequence of a harder problem and is not a finding. Two further traps:
compare a cut against **its complement**, never against the pool that contains it; and `pit_ks_pooled_band`
scales as `1/√n`, so a smaller stratum earns a wider band and reading it against the pooled band
manufactures failures. The headline is `sd_over_rmse` on `weak_proxy` vs its complement — pooled ≈ 1
with `weak_proxy` well below 1 is a posterior that fails to widen where proximal identification is
fragile, which is worse than merely being inaccurate there.

### The `n`-sweep — offline only, **no wandb producer**

The same `prior_dist/*` keys at nested context prefixes, by
`python -m prior_eval.sweep`. Same key-budget reasoning as the stratified pass: one row per `n`
per metric is a table, not a dashboard. The set needs no rebuild — prefix `n` is the first `n`
context rows of an episode that already exists, and `prior_dist/grid_scale` (above) is what puts
every prefix on one scale.

Reported per `n` beyond the standard keys:

| Key | What it is |
|---|---|
| `mean_sd_geo` | Geometric mean over episodes of the per-episode mean sd, **frozen units**. The geometric mean, not the row-weighted one, because episodes differ in `tau` scale by orders of magnitude and the pooled mean is whatever the largest-scale episodes do |
| `mean_sd_head_geo` | Same, divided by each episode's `grid_scale`, i.e. in the model's **own** units |
| `rmse_pooled` | `√(mse_obs − mean_s2)` — the target-noise-corrected error, i.e. the *denominator of `sd_over_rmse`*, not `prior_point/cate/rmse_pooled` |
| `sd_floor` | `√(σ² + w²/12) × grid_scale`. **`σ` comes from `--hl_sigma` (head units, = `loss/hl_sigma` at that step); a checkpoint does not carry it.** Omit the flag and this is the grid term alone, typically ~1e-3 and two orders of magnitude under the real floor — at which point a schedule-pinned flat width curve is indistinguishable from a converged one. `sd_floor_is_grid_only` flags it |
| `within_frac` | `E[Var per row] / Var(mixture)` — what the model still does not know, over the total. Runs **1 at the prior limit → 0 as context accumulates**. This is the key that discriminates the prior limit |
| `between_sd` | `√Var(row means)` — the complement: variation the model has resolved |
| `mixture_sd` | Sd of the pooled mixture, by the law of total variance — **not** the mean of the per-row sds. At the shortest prefix it should approach `prior.tau_sd` |
| `prior_ks`, `prior_w1` | Distance from the pooled mixture's CDF to the empirical CDF of `tau_star_std`. **Marginal calibration, not a prior-limit check** — see below |

Three traps, all specific to this pass:

- **`prior_ks` is not evidence about the prior limit.** Averaging a calibrated posterior over the
  data-generating distribution returns the prior marginal *at every* `n` — an identity, not a
  property of short prefixes. So small everywhere is correct, including at `n_max`, and it says
  nothing about whether the width responds to context. It is close kin to the pooled PIT. The
  statistic that does discriminate is `within_frac`: flat across the sweep means the model has
  learned a *marginal*, not a posterior, however well the CDFs overlay.

- **Two width curves, one claim each.** `grid_scale` is 1 at `n_ctx == n_max` and *systematically
  below it* at short prefixes, because a context sd from few rows of a heavy-tailed `y`
  under-estimates. So the frozen-unit width shrinks at small `n` for a reason that has nothing to do
  with the head. Frozen units are what the posterior *claims* (and the only units commensurable with
  the RMSE, hence with `sd_over_rmse`); head units are what the head *learned*. `β(sd) − β(sd_head)`
  is the size of the standardization effect, and it is not noise.
- **Do not fit through either end.** The upper end lands on `sd_floor` (that is the schedule, not
  convergence)
  and the lower end plateaus at the prior — which is the *other* question, not an input to it.
  `fit_window` drops prefixes within `3×` the floor automatically; the small-`n` plateau is left in
  and excluded by hand with `--fit_min` once the table shows where it starts.

The done-when is a pair, never a single number: `mean_sd` falls monotonically **while**
`sd_over_rmse` stays ≈ 1, i.e. `β(sd) ≈ β(rmse)`. A width contracting faster than the error is a
posterior getting over-confident with context; a width that does not move at all is a point estimate
wearing a histogram.

### Flat keys

| Key | What it is |
|---|---|
| `Charts/prior_eval_time` | Wall-clock seconds for the pass |
| `prior_dist/n_ctx` | Context prefix length scored. Equals `n_max` in training; only differs on an `n`-sweep |
| `prior_dist/grid_scale` | The factor `episode_predictive` applied to the head's own grid to land in the frozen space (`y_std_predict / y_std_frozen`), pooled row-weighted. **Exactly 1.0 at `n_ctx == n_max`**, i.e. on every in-training pass |

`prior_dist/grid_scale` looks redundant today precisely because it is 1. It is published anyway because
`diag/sigma` is built from it, and a caller that hardcoded 1.0 would be wrong exactly where an
`n`-sweep at shorter prefixes gets interesting.

### `diag/sigma`

| Key | Formula | Emitted |
|---|---|---|
| `diag/sigma` | `get_hl_sigma() * prior_dist/grid_scale` | from `prior_validate()`, alongside `prior_dist/mean_sd` |

The same number as `loss/hl_sigma`, put on the width axis. `loss/hl_sigma` is logged from the
micro-batch path, in head units, at the training cadence — three separate reasons a plot of it
against `prior_dist/mean_sd` is not a comparison. `diag/sigma` is in `mean_sd`'s units at `mean_sd`'s
cadence. Both keys are kept; `loss/hl_sigma` is correct on the training surface.

### What is *not* on this surface

Step 7 of the plan — the target-noise convolution — **is not built**. There is no
`convolve_predictive`, no `crps_noise_floor`, and no `--prior_eval_noise_threshold`. What *did* land
is the part that matters at current noise levels: the `s²` subtraction inside the variance identity,
which is unconditional, and `prior_dist/s_over_sd`, which is the measured ratio that decides when the
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

`cov50_pooled`/`cov90_pooled` and the PIT are the other half of the diagnosis: `sd_over_rmse` says *whether* the
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

### `diag/sigma` vs `prior_dist/mean_sd` — the schedule floors the width

The HL-Gaussian target is a Gaussian of width `sigma` discretized onto the grid, so a head that fits
it *perfectly* reports `√(sigma² + w²/12)` and no less. `sigma` and the grid's `w/√12` are the two
floors under `mean_sd`, and they add in quadrature.

The consequence: a posterior sitting on that floor stops narrowing because the **schedule** stopped,
not because the model converged. The sd curve flattens at exactly the wrong moment and reads as
convergence. This is the most likely misreading of the whole fine-tune, which is why `diag/sigma` is
emitted on `mean_sd`'s axis at `mean_sd`'s cadence rather than left as `loss/hl_sigma`.

### The `w²/12` term — two SD series that are not the same series

`prior_dist/mean_sd` reads the histogram as a density and picks up the within-bin variance `w²/12`.
`noise/mean_sd_pred` is the discrete moment over bin centres and does not. A head with all its mass
in one bin reports `w/√12` for the first and 0 for the second.

Both are correct for what they are — `mean_sd_pred` is a diagnostic of the *loss*, `mean_sd` a
calibration metric — but they differ by a constant floor and must never be plotted together or
compared.

### Pooled vs per-episode KS — pooling hides cancellation

Over-confident on half the episodes and under-confident on the other half is **uniform overall**. The
`pit_ks_pooled` passes its band and the posterior is badly miscalibrated everywhere.

`pit_ks/p90 ≥ pit_ks/band` while `pit_ks_pooled < pit_ks_pooled_band` is precisely that signature.
`prior_eval.score` prints it as an explicit warning; there is a test constructing two
opposite-direction episode groups and asserting the pooled statistic passes while the p90 fails.

### Median vs pooled RMSE

`prior_point/cate/rmse` is a median over episodes. `prior_point/cate/rmse_pooled` is a mean over
rows. Comparing either to a number from elsewhere requires knowing which one you have — the pooled
key exists solely so there is a quantity that must match `prior_eval/report.py` exactly.

### The homogeneous-episode exclusion

`cate/npehe`, `cate/r2`, `cate/r2_shape` and `ate/bias_norm` are computed on a strict subset of
episodes (those with `Var(τ) ≥ 1e-6`). If the prior's heterogeneity distribution shifts, that subset
changes size and these four metrics move for reasons that have nothing to do with the model. Their
episode count is not `eval_episodes`, and it is not logged.

### Why there is no ATE credible interval

`ate/*` is a point estimate on every surface, deliberately. The head predicts a distribution over
`tau` for each row **independently**; there is no cross-row joint, so the posterior of a row-average
is not identified — any dependence structure between rows is unconstrained, and the interval you
would get from treating them as independent is an arbitrary choice of that structure, not an
inference. `ate/rmse`, `ate/r2` and `ate/bias_norm` are all point accuracy. (Notes §2d.)

---

## 7. `synt_dist/*` and the set it may be scored on

**The line is drawn at the dataset, not at the surface.** This section used to say `val/dist/*`
cannot exist at all; replicate mode in the realistic generator changed what is true, and the
reasoning is worth keeping because
the failure it describes is still one command away.

### The error

A realistic set built **without** `--mc_cate` carries no Monte-Carlo CEPO target. Each row has a
single sampled potential-outcome pair, so `y1 − y0` is a draw of the *ITE*, not an estimate of the
CATE. Scoring a predictive distribution against it is a proper scoring rule — for the **ITE
predictive**, an object strictly wider than the CATE posterior by exactly the outcome noise. The size
of that gap is logged: it is `target/ite_minus_cate_mse`.

A CRPS or PIT computed there is *minimized by an over-wide posterior*. Every calibration metric would
reward the exact failure the fine-tune exists to avoid, and nothing would look broken: `sd_over_rmse`
would sit near 1 for a model whose CATE posterior is far too wide.

### Why it now errors instead

`synt_eval/build_proximal_datasets.py --mc_cate K` (`K > 1`) replicates each query row through the
latent/noise draw and emits `MU0`/`MU1` with their paired sampling variance `S2`, plus a
`frozen_meta.pt` sidecar carrying the frozen comparison scale and `tau_star_std`/`s2_std`. That set is
scorable on exactly the terms the frozen prior set is.

**The sidecar's absence is the gate.** `--synt_eval_dir` loads it, unguarded, at `Trainer.__init__`, so
pointing it at a `mc_cate=1` set — or at a plain `val/` subdirectory — raises `FileNotFoundError`
before training starts. The wrong number is not reachable; you get a stack trace instead.

Two consequences worth internalising:

- The estimand is `E[Y1 − Y0 | X]`, chosen to match `SCMPrior.generate_dataset` exactly. Scoring a
  posterior against a *different* conditional makes a calibrated model read as systematically
  over-wide, which is the same class of error as scoring the ITE, just subtler.
- `K` there is moderate, not the prior set's 4000, so `synt_dist/s_over_sd` is **not** negligible the
  way `prior_dist/s_over_sd` is. It is the series that decides whether the predictive-side
  noise convolution is needed; the trigger is a sustained value above ~0.15.

### The point surface is gated by exactly the same thing

Before the two were wired together, `synt_point/*` came from a separate batched pass
(`point_validate()`) over `--val_dir`, which would run on *any* set — `LoadPriorDataset` substitutes
`Y0`/`Y1` for `MU0`/`MU1` only when the latter are absent, so a `mc_cate=1` set still produced a number,
just one inflated by outcome noise (`cate/rmse` estimating `√(PEHE² + Var(noise))` rather than `PEHE`).

That fallback is gone. `synt_point/*` is now a side effect of the same `score_frozen_set` pass that
produces `synt_dist/*`, over `--synt_eval_dir`, so it is gated by the same sidecar-presence check: no
`frozen_meta.pt` means no pass at all, point or distributional. There is no longer a way to get a
point-only validation number, inflated or otherwise, from a non-replicate set.

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

Default `synt_point/cate/npehe`, with `--best_metric_mode min`. Any key from the val or prior surface
works, and the comparison is gated on the step where *that surface's pass* actually ran, so
`--prior_eval_interval` and `--synt_eval_interval` need not coincide. This surface was renamed from
`val/point/*`, and the old spelling is simply gone — a run resumed with it monitors a key nothing
emits, which `_check_best_metric_config` reports at startup along with two other configurations that
silently no-op:

- `--save_best` with neither `--prior_eval_dir` nor `--synt_eval_dir` set;
- a `best_metric` on a surface whose pass is disabled: `prior_dist/*` or `prior_point/*` with no
  `--prior_eval_dir`, or `synt_dist/*` or `synt_point/*` with no `--synt_eval_dir`;
- a name whose direction contradicts `--best_metric_mode`. The heuristic is textual: `r2` and
  `sign_agreement` are treated as higher-is-better, and `rmse`, `npehe`, `mse`, `bias`, `loss`,
  `crps` as lower-is-better. A key matching neither list (e.g. `prior_dist/cov90_pooled` or
  `prior_dist/sd_over_rmse`, both of which have a *target* rather than a direction) gets no warning
  in either mode — and `sd_over_rmse` is genuinely unsuitable as a `best_metric`, since 1.0 is the
  target and both `min` and `max` walk away from it.

`_maybe_save_best` only fires on a step where the surface producing the monitored key actually ran,
so a `prior_dist/*` `best_metric` evaluated less frequently does not
warn on the in-between validation passes.

---

## 10. Adding a metric

1. **Pick the surface first.** If it cannot be computed honestly on that surface's data, the answer
   is a different surface, not a caveat in the docstring. [§7](#7-val_dist-and-the-set-it-may-be-scored-on) is the
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
