# Appendix B reproduction commands

Maps every table/figure currently used (or planned) in `docs/latex/appendices/B_prior.tex`
to the exact `analysis_prior` CLI invocation that produces it. See
[`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md) for the full status/tracking doc —
this file is just the "how do I regenerate artifact X" cheat sheet.

All commands run from the repo root:

```
uv run python -m analysis_prior.main <args>
```

Add `--force` to regenerate the cache instead of reusing it, and `--n-jobs -1` to parallelise
generation across all cores.

---

## `PriorSpec`, `SpecFamily`, and how caching works

`config.py` defines two dataclasses that drive everything above:

- **`PriorSpec`** is the single source of truth for "which datasets are we analysing" —
  dimension ranges, `seq_len`, `prior_type`, `mcmc_cate`, HP overrides, etc. It's JSON-serialisable
  and hashes to a stable cache key. `--spec <name>` looks one up from `config.PRESETS` (e.g.
  `general_ite`, `smoke`, `no_u_augment`).
- **`SpecFamily`** is a base `PriorSpec` plus a dict of named interventions (small field overrides)
  used for comparison studies — "what does changing one knob do to the prior?" `--family <name>`
  looks one up from `config.FAMILIES` (e.g. `dim_ablation`, which compares `triangular` /
  `uniform` / `unenforced` dimension sampling against the same `general_ite` base). Each
  intervention resolves to its own `PriorSpec` via `SpecFamily.resolve()`.

**Cache keys are content-addressed, not name-addressed.** A spec's `name` and `n_datasets` are
cosmetic — they don't affect what any individual dataset looks like — so they're excluded from the
cache key. `PriorSpec.cache_key` is the hash of the *generative* fields only (`generative_digest()`
in `config.py`). Practically:

- Two specs that differ only by `name` or `n_datasets` share the same cache file under
  `cache/<hash>.pt` (+ a `.json` sidecar with the generative spec, for `--list-caches`).
- Each dataset's seed is derived from `(generative_digest, index)`, not from `n_datasets`, so
  dataset `i` is bit-identical regardless of how many datasets were requested. Asking for fewer
  just takes a prefix; asking for more (`load_or_generate` in `generate.py`) generates only the
  missing indices and appends them — existing datasets are never touched or regenerated.
- Bumping `CACHE_FORMAT_VERSION` in `config.py` invalidates every existing cache file (it's
  folded into the hash), for when a generation-code change would make an old cache wrong instead
  of just differently-labelled.

This is why `--family dim_ablation`'s `triangular` variant (the identity intervention, `{}`) and
`--spec general_ite` resolve to the *same* cache file: they're generatively identical specs that
only differ in name.

### Presets and families (current)

| preset | n | seq_len | K (`mcmc_cate`) | notes |
|---|---|---|---|---|
| `default` | 64 | 2048 | 100 | mirrors training runs |
| `mc_targets` | 100 | 2048 | **100** | B.2.6 Monte Carlo targets; the only `K>1` preset in use. Precision comes from `T` paired within-row contrasts, so `seq_len` (not `n_datasets`) is the knob that tightens the variance self-check |
| `general_ite` | **1024** | 2048 | **1** | ITE-phase workhorse; base of `dim_ablation` and `cat_ablation` |
| `smoke` | 4 | 100 | 8 | fast CI/dev |
| `no_u_augment` | 64 | 2048 | 100 | ablation: `confounding_dim_control=False` (f_A/f_Y see raw full U) |
| `no_categorization` | 64 | 2048 | 100 | ablation: `cat_col_prob` pinned to 0 |

| family | base | interventions |
|---|---|---|
| `dim_ablation` | `general_ite` | `triangular` (identity), `uniform` (`dim_U_distribution=uniform`), `unenforced` (`enforce_completeness_heuristic=False`) |
| `cat_ablation` | `general_ite` | `full` (identity), `ablation` (`enforce_categorical_completeness=False`), `baseline` (`corrupt_U=False`, `enforce_categorical_completeness=False`) |
| `smoke_family` | `smoke` | `triangular`, `uniform` |

---

## Implemented and wired into the tex

Thirteen analyses are registered today (`analyses/__init__.py`): `dimensions`,
`dimension_comparison`, `hyperparameters`, `categorical_completeness`, `mechanism_inputs`,
`mechanism_outputs`, `treatment_walkthrough`, `treatment_overlap`, `heterogeneity`, `mc_targets`,
`prior_overview`, `exclusion`, `confounding_strength`. The
subsections below match the *current* `B_prior.tex` numbering (B.2 = "Generative Model, Step-by-Step").

### B.2.1 Hyperparameters — `tab:prior_fixed_hp`, `tab:prior_sampled_hp`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters` |
| **Config group** | preset `general_ite` (`config.PRESETS["general_ite"]`) |
| **Analysis module** | `analyses/hyperparameters.py` → `HyperparametersAnalysis` (`name="hyperparameters"`) |
| **Output dir** | `results/general_ite/hyperparameters/` |
| **Table outputs** | `sampled_hyperparameters.tex` → `tab:prior_sampled_hp`; `fixed_hyperparameters.tex` → `tab:prior_fixed_hp` |
| **Appendix location** | §B.2.1 "Hyperparameters: Fixed vs Sampled" (`app:prior_hyperparams`) |

This analysis is spec-only (reads the resolved HP config, not generated data), so any `--spec`
works and no dataset cache is needed.

### B.2.2 Dimensions & Dimension Heuristic — `fig:prior_dim_marginals`, `tab:prior_dim_sampling`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (base preset `general_ite`, variants `triangular` / `uniform` / `unenforced`) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_marginals.pdf` → `fig:prior_dim_marginals` |
| **Table output** | `dimension_sampling.tex` → `tab:prior_dim_sampling` |
| **Appendix location** | §B.2.2 "Dimensions & Dimension Heuristic" (`app:prior_dimensions`) |

### B.2.3 Mechanism input composition — `fig:prior_mechanism_input_composition`, `tab:prior_mechanism_inputs`

Off the prior's own statistics (no regression): for `f_A` and `f_Y`, each parent block's share of the
mechanism input, measured two ways — *wiring* (column-count share, equal to the input-variance share
since parent blocks are standardised before concatenation) and *first-layer* injected variance (exact
from the realised weights) — plotted against the sampled knob steering each channel.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis mechanism_inputs` |
| **Config group** | preset `general_ite` (reads the mechanism-diagnostics payload in `sampled_hp` / `mechanisms`) |
| **Analysis module** | `analyses/mechanism_inputs.py` → `MechanismInputsAnalysis` (`name="mechanism_inputs"`) |
| **Output dir** | `results/general_ite/mechanism_inputs/` |
| **Figure output** | `mechanism_input_composition.pdf` → `fig:prior_mechanism_input_composition` |
| **Table output** | `mechanism_input_composition_table.tex` → `tab:prior_mechanism_inputs` |
| **Appendix location** | §B.2.3 "MLP Generation, Noise & Edge Sampling" (`app:prior_mechanisms`), "Delivered input composition" paragraph |

Needs the per-dataset mechanism diagnostics (`sampled_hp`, `mechanisms`); a cache written before they
existed must be regenerated once with `--force` (data is bit-identical, the diagnostics are simply
recorded alongside).

### B.2.3 Mechanism output dependence — `fig:prior_mechanism_output_dependence`, `tab:prior_mechanism_outputs`

The output-side companion: what each mechanism's output *depends on*, via a cross-fitted TabICL
regression on the delivered data. Leave-one-block-out ΔR² gives each parent block's unique
contribution — `f_A` predicts `logit(π)` from `{U, X, Z}`, `f_Y` predicts `Y0` from `{U, X, W}`; the
exogenous-noise channel, having no block to drop, is read as the unexplained share `1 − R²`. A
per-dataset tqdm progress bar tracks the run.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --n-datasets 256 --analysis mechanism_outputs` |
| **Config group** | preset `general_ite`. **Uses every dataset in the cache**, so the count is set by `--n-datasets` (the default 1024 ≈ 8 h of TabICL — cap it; `--n-datasets 256` is served as a prefix of the existing cache, no regeneration) |
| **Analysis module** | `analyses/mechanism_outputs.py` → `MechanismOutputsAnalysis` (`name="mechanism_outputs"`) |
| **Regression helper** | `analysis_prior/regression.py` — cross-fitted out-of-fold, model `linear`/`rf`/`tabicl` (attr `MechanismOutputsAnalysis.model`, default `tabicl`); predictions cached under `cache/regression/`, so re-renders and resumed runs are instant |
| **Output dir** | `results/general_ite/mechanism_outputs/` |
| **Figure output** | `mechanism_output_dependence.pdf` → `fig:prior_mechanism_output_dependence` |
| **Table output** | `mechanism_output_dependence_table.tex` → `tab:prior_mechanism_outputs` |
| **Fast demo** | `PYTHONPATH=. uv run python -m analysis_prior.examples.mechanism_outputs_small` (a few datasets, random forest, ~seconds; writes to `results/general_ite/mechanism_outputs_small/`, not the paper path; `--model tabicl --n 16` for a quick TabICL taste) |
| **Appendix location** | §B.2.3 "MLP Generation, Noise & Edge Sampling" (`app:prior_mechanisms`), "Output dependence" paragraph |

### B.2.4 Post-process (numeric completeness) — `fig:prior_dimension_joint_heatmaps`

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison` |
| **Config group** | family `dim_ablation` (same run as B.2.2 — one invocation produces both) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionComparisonAnalysis` (`name="dimension_comparison"`) |
| **Output dir** | `results/dim_ablation/dimension_comparison/` |
| **Figure output** | `dimension_joint_heatmaps_comparison.pdf` → `fig:prior_dimension_joint_heatmaps` |
| **Appendix location** | §B.2.4 "Post-Process" (`app:prior_postprocess`), "Numeric completeness" paragraph |

> **Removed:** the numeric survival table (`completeness_survival_unenforced.tex` →
> `tab:prior_completeness_survival`) is **no longer rendered** — its rendering was deleted in
> commit `9bcd8db`, and the tex no longer `\input`s it. Any `completeness_survival_unenforced.tex`
> / `.csv` still sitting under `results/dim_ablation/dimension_comparison/` are stale leftovers and
> can be deleted; a fresh run will not regenerate them.

### B.2.4 Post-process (categorical completeness) — `fig:prior_category_count_marginals`, `fig:prior_proxy_enforcement_vs_u_demand`, `tab:prior_categorical_completeness_survival`

Discrete analogue of the dimension heuristic: a proxy block is *categorically complete* w.r.t.
`U` when every column of `U` can be matched one-to-one to a proxy column that is at least as
informative (continuous dominates any finite categorical; a categorical column dominates another only
if its category count is `>=`). Since the categorisation metadata isn't persisted in the cache, each
block's profile is reconstructed from realised values (a column with `<= max_categories` distinct
values counts as categorical) — see the module docstring in `analyses/categorical.py` for why that's
unambiguous at the shipped cap.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness` |
| **Config group** | family `cat_ablation` (base preset `general_ite`, variants `full` / `ablation` / `baseline` — enforcement and `U`-categorisation toggled via `enforce_categorical_completeness` / `corrupt_U`) |
| **Analysis module** | `analyses/categorical.py` → `CategoricalCompletenessAnalysis` (`name="categorical_completeness"`) |
| **Output dir** | `results/cat_ablation/categorical_completeness/` |
| **Figure outputs** | `category_count_marginals.pdf` → `fig:prior_category_count_marginals`; `proxy_enforcement_vs_u_demand.pdf` → `fig:prior_proxy_enforcement_vs_u_demand` |
| **Table output** | `categorical_completeness_survival.tex` → `tab:prior_categorical_completeness_survival` |
| **Appendix location** | §B.2.4 "Post-Process" (`app:prior_postprocess`), paragraphs "Categorical completeness" / "Enforcement scales with `U`'s demand" / "Survival" |

`proxy_enforcement_vs_u_demand` is a single three-panel figure: (a, b) condition on one `U` column's
own cardinality (proxy continuous fraction, proxy mean category count), (c) conditions at the dataset
level on `|cats(U)|` (proxy categorical fraction). It replaces the former standalone
`proxy_stats_by_u_cardinality` (panels a, b) and `proxy_composition_vs_u_demand` (panel c) — any
leftovers of those two under `results/cat_ablation/categorical_completeness/` are stale and will not
regenerate. All three panels plot only the categorised-`U` arms (`full`, `ablation`) — `baseline` has
a continuous `U` by construction and is dropped, though it still appears in the category-marginals
figure and the survival table.

### B.2.5 Treatment assignment walkthrough — `fig:prior_treatment_walkthrough`

A single-SCM illustration of `sample_A`'s pipeline: raw `f_A` logit → standardise (`eq:A_normalise`)
→ shift/scale (`eq:A_affine`) → sigmoid+clamp (`eq:A_sigmoid`/`eq:A_clamp`) → Bernoulli draw, tracking
one highlighted unit (red) through all five stages. Pure function of the raw logits now retained in
the cache (`record.a_logits`, plus the population `a_logit_mean`/`a_logit_std` that `sample_A`
standardises by) and the per-dataset propensity knobs; it re-applies the exact transform, so the
reconstructed `π` matches the stored `A_probs` to ~1e-7 (asserted in the test and reported as
`reconstruction_residual` in `summary.json`). The illustrated dataset is auto-selected for a legible,
symmetric clamp (balanced prevalence, moderate scale); its index/seed land in `summary.json`. Panel (d)
draws the clamp at an enlarged `DISPLAY_EPSILON` (0.05) — the production `propensity_epsilon` (1e-3)
sits flush against 0/1 and is invisible on a `[0,1]` axis; only the display bound is enlarged, and the
`A_probs` validation still uses the true 1e-3.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_walkthrough` |
| **Config group** | preset `general_ite` (`K=1`, so the observed slice is the whole population and the standardisation is exact) |
| **Analysis module** | `analyses/treatment.py` → `TreatmentWalkthroughAnalysis` (`name="treatment_walkthrough"`) |
| **Output dir** | `results/general_ite/treatment_walkthrough/` |
| **Figure output** | `treatment_walkthrough.pdf` → `fig:prior_treatment_walkthrough` |
| **Appendix location** | §B.2.5 "Treatment Assignment" (`app:prior_treatment`) |

The raw logits are collected only when the cache carries mechanism diagnostics. A cache written before
they existed (i.e. before `a_logits` was added) must be regenerated once with `--force` — the data is
bit-identical, the logits are simply recorded alongside. The B.3.1 overlap audit (`app:prior_overlap`,
`treatment_overlap`) lives in this same `treatment.py` module — same subject, different scale — and
needs no logits.

### B.2.6 Heterogeneity dial — `fig:prior_heterogeneity_spread`

> **Rendered but not yet `\input` into `B_prior.tex`.** The figure exists; the `\cm` at
> `app:prior_outcome` also asks for `fig:prior_cate_vs_ite_K` (still planned, see below), so the
> marker stays until both land.

Does the remix dial `gamma` control the spread of the delivered effect? Substituting `eq:remix`
collapses the per-row weights to `tau = Y1 - Y0 = gamma * tau_raw + (1 - gamma) * tau_bar` — the
per-row `alpha` cancels — so `sd(tau) = gamma * sd(tau_raw)` exactly.

**`gamma` cannot be pinned from a `PriorSpec`**: it is drawn inside the generator
(`torch.rand(1)` in `_dataset.py`), not from the sampled-HP config, so there is no way to hold an
SCM fixed and turn the dial. It *is* recorded per dataset (`record.gamma_hetero`) and drawn
uniformly, so one cache is already a sweep — the analysis reads the delivered `gamma` and asks
whether dispersion tracks it **across** the prior. Two consequences worth knowing before reading
the figure:

- the ATE-preservation half of the claim is **not** testable this way (the ATE varies by orders of
  magnitude across SCMs, swamping any `gamma` effect) — only the spread claim is;
- `Y` is not standardised by `Corrupt`, so every statistic is a ratio. Dispersion is
  `sd(tau)/sd([Y0;Y1])`; `summary.json` reports `spearman_gamma_vs_scale` (0.04 on `general_ite`)
  as the check that this denominator is `gamma`-neutral and the panel-(c) overlay compares like
  with like.

Three panels: **(a)** two effect densities on a shared outcome-scaled axis, one gathered
(`gamma ≈ 0.1`) and one spread (`gamma ≈ 0.9`) — the pair is chosen for *matched* SCM dispersion
`c = dispersion/gamma`, so the contrast is about `gamma` rather than SCM luck; **(b)**
`corr(Y0, Y1)` vs `gamma`, whose `gamma → 0` endpoint is pinned at exactly 1 by the algebra
(`Y1 = Y0 + tau_bar`) and needs no scale reference; **(c)** dispersion violins per `gamma` bin
against the proportionality `gamma * median(c)`.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis heterogeneity` |
| **Config group** | preset `general_ite` (reuses the same cache as `mechanism_inputs` / `treatment_walkthrough` / `prior_overview`) |
| **Analysis module** | `analyses/outcome.py` → `HeterogeneitySpreadAnalysis` (`name="heterogeneity"`) |
| **Output dir** | `results/general_ite/heterogeneity/` |
| **Figure output** | `heterogeneity_spread.pdf` → `fig:prior_heterogeneity_spread` |
| **Appendix location** | §B.2.6 "Potential Outcomes, Heterogeneity Remix & CATE Targets" (`app:prior_outcome`) |

Statistics use the ITE `Y1 - Y0` of the realised replicate, which satisfies the collapse above at
any `K`. At `K = 1` the Monte Carlo CATE target `MU1 - MU0` **is** that same quantity, so on a
`general_ite` cache the panels read equally as a CATE view. Needs only `gamma_hetero` from the
diagnostics payload (a cache predating it must be regenerated once with `--force`), no regression.

### B.2.6 Monte Carlo CATE targets — `fig:prior_cate_vs_ite`, `tab:prior_mc_signal_share`

> **Rendered but not yet `\input` into `B_prior.tex`.** Together with
> `fig:prior_heterogeneity_spread` above, this closes the `\cm` at `app:prior_outcome`.
> Note the label is `fig:prior_cate_vs_ite`, not the planned `..._K`: there is no measured `K`
> sweep in it (see below).

How much of the training target is real? The target `Q_hat_K = MU1 - MU0` is unbiased at every
`K`, so a small `K` costs variance, not bias. Writing `sigma^2(x) = Var(tau|x)` for the
within-unit replicate variance and `Var_x(Q)` for the between-unit CATE variation,

```
Var_rows(Q_hat_K) = Var_x(Q) + E[sigma^2]/K      →      S(K) = 1/(1 + NSR/K)
```

with `NSR = E[sigma^2]/Var_x(Q)`. `S(K)` is the fraction of the target's variance that is signal —
equivalently its R² against the true CATE.

**The one trick that makes this free.** The retained replicate is one of the `K` that went into
`MU`, so the mean of the *other* `K-1` is recoverable exactly and is statistically **independent**
of the retained one: `Q_loo = (K*Q_hat_K - ITE)/(K-1)`. That gives two independent measurements of
the same `Q(x)` — parallel-forms reliability — hence two subtraction-free estimators:
`Var_x(Q) = Cov(ITE, Q_loo)` and `E[sigma^2] = mean[(ITE - Q_loo)^2]*(K-1)/K`. Neither differences
near-equal variances, which is why a short `seq_len` works: precision comes from `T` paired
contrasts.

**Measured vs extrapolated.** Only `NSR` is measured, once per dataset. `S(K)` then follows
analytically (it is the variance of a sample mean, exact under the i.i.d. replicate structure the
generator guarantees) — it is **not** a measured sweep, and cannot be: the cache pins measured
anchors only at `K=1`, `K-1` and `K`, and intermediate `K` would need the per-replicate matrix.
That is why `S(K)` is reported as a *table* rather than drawn as a band of curves. `gamma` cancels
from `NSR` (both variances carry `gamma^2`), so no faceting is needed; datasets with
`gamma < 0.05` are dropped only because both variances then vanish numerically.

Two panels: **(a)** one dataset's `K=1` target against the held-out `Q_loo`, with the identity and
the fitted regression. The informative feature is the **slope**, not the diagonal cloud — a
"CATE target" that were merely the ITE plus noise would regress with slope 1, whereas replicate
averaging shrinks it to `Var(Q)/Var(ITE) = S(1)`; the gap between the two lines is what the Monte
Carlo integration removed. **(b)** signal share across the prior at each `K` in `SHARE_PANEL_K`
(currently `1, 10, 100`), on a sequential ramp; the mass sweeps from the left edge to the right as
`K` grows.

`SHARE_PANEL_K` is deliberately **independent of the cache's own `mcmc_cate`**: `S(K)` follows
analytically from the measured `NSR`, so any `K` can be displayed off any `K>1` cache. Retargeting
the panel at a different production setting is a one-line edit, no regeneration. Keep it a subset of
`K_GRID` so every plotted value stays quotable from the table and `summary.json` — a test enforces
that.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec mc_targets --analysis mc_targets --n-jobs -1` |
| **Config group** | preset `mc_targets` (`n=100`, `seq_len=2048`, **`K=100`** — the only preset with `K>1`; raising `n_datasets` appends to the same cache, changing `seq_len` or `mcmc_cate` starts a new one) |
| **Analysis module** | `analyses/mc_targets.py` → `McTargetsAnalysis` (`name="mc_targets"`) |
| **Output dir** | `results/mc_targets/mc_targets/` |
| **Figure output** | `cate_vs_ite.pdf` → `fig:prior_cate_vs_ite` |
| **Table output** | `mc_signal_share.tex` → `tab:prior_mc_signal_share` |
| **Appendix location** | §B.2.6 "Potential Outcomes, Heterogeneity Remix & CATE Targets" (`app:prior_outcome`) |

Rejects a `K=1` cache outright (there is no replicate structure to measure). `summary.json` reports
two implied identities as `*_residual`; they are not equally informative — `var_ite_residual` is
near-algebraic given how the estimators are built, while `var_cate_residual` is the real check, and
is floored by the sampling error of a variance over `T` rows (`~sqrt(2/T)`, so ~3% at the current
`seq_len=2048`, against a delivered median of 5%). **That floor is the reason to raise `seq_len`**,
not the number of datasets.

### B.2.x Visualize the prior — `fig:prior_feature_distributions`, `fig:prior_feature_pairs`, `fig:prior_feature_pairs_cross`

Three galleries of the delivered data, all off the prior's own statistics (no training):

- **`prior_feature_distributions`** — one column per block (`X, U, W, Z`, `π` for `A`, `Y0`/`Y1` for `Y`),
  `k` rows, each the marginal of one feature column sampled from a random dataset (solid = continuous,
  dashed = categorical; the `Y` column overlays both potential outcomes in two colours).
- **`prior_feature_pairs`** — within-block dependence: one column per block, `k` rows, each a scatter of
  two same-block features (`A`: `π` vs the draw; `Y`: `Y0` vs `Y1`).
- **`prior_feature_pairs_cross`** — cross-block dependence: one column per chosen block pair
  (`U`–`W`, `U`–`Z`, `W`–`Z`, `U`–`π`), `k` rows, each a scatter of one feature from each block.

All three sample datasets with a fixed seed (`PriorOverviewAnalysis.seed`), so every figure is
reproducible.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis prior_overview` |
| **Config group** | preset `general_ite` (reuses the same cache as `mechanism_inputs` / `treatment_walkthrough`) |
| **Analysis module** | `analyses/overview.py` → `PriorOverviewAnalysis` (`name="prior_overview"`) |
| **Output dir** | `results/general_ite/prior_overview/` |
| **Figure outputs** | `prior_feature_distributions.pdf` → `fig:prior_feature_distributions`; `prior_feature_pairs.pdf` → `fig:prior_feature_pairs`; `prior_feature_pairs_cross.pdf` → `fig:prior_feature_pairs_cross` |
| **Appendix location** | §B.2.x "Visualize the prior" (`app:prior_visualize`) |

Uses only the cached feature blocks and the `max_categories` field of `sampled_hp` (for the
continuous/categorical split); no regression or mechanism-diagnostic payload, so any `general_ite` cache
works without `--force`.

### B.3.1 Overlap audit — `fig:prior_overlap`

> **Rendered but not yet `\input` into `B_prior.tex`.** The figure exists; §B.3.1 has no prose yet.

Audits `as:positivity`. The key point is that **no regression is needed**: the cached `A_probs` *is*
the ground-truth `P(A=1 | X, U)` the assumption is stated over, so the audit is a pure summary of
cached values. Fitting a propensity model on `(X, U)` would only re-estimate the same quantity with
error; the genuinely different object — the *observable* propensity `P(A=1 | X, W, Z)`, which
marginalises `U` away and therefore looks better-behaved — is deliberately left to the confounding
ladder (T2.2), not measured here.

Positivity also cannot fail: `sample_A` clamps `π ∈ [ε, 1-ε]` with `ε = 1e-3`, so every dataset
satisfies it by fiat. The figure therefore reports **severity of practical overlap**, not pass/fail.

Two panels:

- **(a) pooled overlap margin** — histogram of `min(π, 1-π)` over every row of every dataset, on a
  log axis (the mass that matters piles against `ε`, and a linear axis hides it entirely). Folding
  the two tails into one statistic makes "distance to a degenerate assignment" a single axis. Bars
  are per-bin *shares of all rows* on a log grid, so bin width grows to the right — read the panel
  for where mass sits, not as a density.
- **(b) per-dataset quantile curves** — each dataset's quantile function of `π`, summarised across
  datasets as a median curve with 25–75% and 10–90% bands. Panel (a) pools rows and so cannot tell
  "every dataset is mildly bad" from "a few are degenerate"; the band width is exactly that
  distinction. Reference lines at `π = 0.05, 0.95`.

`summary.json` carries the numbers for the prose: pooled and per-dataset shares outside
`[0.05, 0.95]`, the share of datasets that are *materially* poor (>10% of rows outside), the
clamp-boundary fraction (how much work "positivity by fiat" is doing), and median prevalence.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_overlap` |
| **Config group** | preset `general_ite` (same cache as `treatment_walkthrough`; no `--force`) |
| **Analysis module** | `analyses/treatment.py` → `OverlapAnalysis` (`name="treatment_overlap"`) |
| **Output dir** | `results/general_ite/treatment_overlap/` |
| **Figure output** | `overlap.pdf` → `fig:prior_overlap` |
| **Appendix location** | §B.3.1 "Overlap" (`app:prior_overlap`) |

Reads only `A_probs`, `A` and `sampled_hp` — no mechanism-diagnostic payload, so any `general_ite`
cache works. The plan's overlap **table** (T2.1) is not built yet; the statistics it needs are
already in `summary.json`.

### B.3.2 Exclusion restrictions — `fig:prior_exclusion`, `tab:prior_exclusion`

Audits `as:proxy_exclusion` (`Y ⟂ Z | (A,X,U)` and `W ⟂ (A,Z) | (X,U)`), which holds **by
construction** — `_dataset.py` never wires `Z` into `f_W`/`f_Y` nor `W` into `f_Z`/`f_A`. The audit
therefore has two halves, and the second is what makes the first mean anything:

- **The check** — cross-fitted incremental `ΔR²` of the excluded block. Y-side: predict `Y_obs` from
  `{A, X, U, Z}`, report `R²(all) − R²(all \ Z)`. A-side: predict `logit(A_probs)` (the
  `mechanism_outputs` convention) from `{X, U, W}`, report the drop for `W`. Both expected `≈ 0`.
- **The positive control** — the same two checks with `U` removed (`{A,X,Z}`; `{X,W}`). `U` is the
  only path linking `Z` to `Y` and `W` to `A`, so the control must light up: it measures whether the
  test *could* have caught a leak.

Two design points worth knowing before changing the module. `Z` is deliberately **not** in the
A-side conditioning set: the clause conditions on `(X,U)` only, and `Z` is itself a proxy of `U`, so
leaving it in lets it absorb the confounder and blunts the control (measured: the A-side control
median collapses from clearly-positive to ~0.008 with `Z` present). Datasets with `d_U = 0` are
skipped — with no confounder the control has no path to open — and counted in `summary.json`.

**The control's power is only as large as the confounding**, and that turns out to matter: over half
the datasets route almost no `U` into the target (`confounding_share_of_U_in_*` is `Beta(1.2,2.8)`),
so the *pooled* control median is small (0.004 Y-side, 0.011 A-side at n=200) and reads as "no
power" if left unconditioned. The figure therefore has a second column, and the summary/table a
second subset:

- **`u_dependence`** — the unique contribution of `U` to the target, `R²(with U) − R²(without U)`,
  i.e. the difference of the two conditions' *full*-set fits. Free: both terms are already computed.
- **Top tercile by `u_dependence`** — the datasets where the control is entitled to fire. There the
  control median rises to 0.031 (Y) / 0.060 (A) with ~58–60% above threshold, while the check stays
  at −0.006 / −0.004 with 0% / 3% above threshold. Spearman(`u_dependence`, control) ≈ 0.38 / 0.43.

**The four fits form a 2×2 lattice, and the figure reads two perpendicular edges of it.** Y-side
(A-side is the same with `{X,U,W}` and `W` for `Z`):

| | includes `Z` | drops `Z` |
|---|---|---|
| **includes `U`** | ① `R²(Y \| A,X,U,Z)` | ② `R²(Y \| A,X,U)` |
| **drops `U`** | ③ `R²(Y \| A,X,Z)` | ④ `R²(Y \| A,X)` |

check = ① − ②, control = ③ − ④ (both: *drop the proxy* → the **y-axis**); `u_dependence` = ① − ③
(*drop `U`* → the **colour** and the point ordering). Control and `u_dependence` share ③ with
opposite signs, so that shared term biases their correlation *downward* — the association the figure
shows is not an artifact of reusing fits.

Figure: two panels, one per clause, each a pair of columns ("conditioning on `U`" vs "control: `U`
dropped") with per-dataset points, a median tick and a `ΔR² = 0` reference. Both scales name their
dropped block — y-axis "`ΔR²` of the excluded proxy", colorbar "confounding: `ΔR²` of `U`" — since
two unlabelled "incremental `R²`" axes are indistinguishable. **Two encodings, deliberately
different.** The median tick carries the condition's colour — target-block colour for the check
(`Y` sky blue, `A` purple), `U`'s vermillion for the control. The *points* carry `u_dependence`:
colour (viridis, shared linear normalisation by raw value across panels) **and**
horizontal position, which is the dataset's *rank* by confounding strength rather than random jitter
(`SPREAD` sets the width). Jitter would spend the column's only spare dimension on nothing; the rank
spends it on the variable that explains the control's shape, so the control column visibly fans
upward to the right while the check column stays flat — the association is readable from the point
cloud, with colour as a redundant cue for the absolute scale.

Three things were tried and dropped at the author's request, each self-contained enough not to be
redone. A second column of scatter panels (`ΔR²` vs `u_dependence`, with a moving-average trend) —
the rank layout inside the same column carries the same read in half the space, and the trend
survives numerically in the table and `summary.json`. The violin body that used to sit under the
points — its KDE shape competes with the points' own x-axis (confounding rank), which the violin
knows nothing about, so the two no longer agreed on what the x-axis meant; briefly replaced by a
plain IQR box, then that too was cut (barely visible at this scale, and the point cloud already shows
spread directly) down to just the median tick. And two in-panel text annotations — the median-value
label (redundant with the tick) and the per-column detectable-leak share against `LEAK_THRESHOLD`
(redundant with the table, and its 0.02 cutoff is a judgment call — roughly the 95th percentile of
the check column's own null distribution, not a derived quantity — that doesn't need to be visible on
every render).

Do **not** switch the colour/ordering to the `confounding_share_of_U_in_*` knob. Tested twice — at
n=200 and again at the full n=1024, on the suspicion the first result was small-sample noise. It was
not: bootstrapped Spearman against the control's `ΔR²` is **0.120** [0.052, 0.188] (Y) and **0.124**
[0.053, 0.195] (A) for the knob, versus **0.391** [0.322, 0.454] and **0.401** [0.339, 0.458] for the
measured `u_dependence` — non-overlapping CIs, a real but ~3× weaker association that more data will
not improve. On the figure's own visual claim (share of top-decile control points landing in the
right half of the ordering, 0.50 = no information): **0.86/0.86** measured vs **0.61/0.65** knob. The
knob is an input-*width* share; the delivered dependence also rides on `d_U`, the MLP draw, the noise
share and how much of `U` the proxies already absorb.

The knob did surface one real defect: it *rendered* better only because `Beta(1.2,2.8)` spreads
evenly over [0,1] while the measured dependence is heavy-tailed, so on a linear ramp ~70% of datasets
land in the darkest fifth (vs 26% for the knob) and the colour channel is mostly wasted. A
decile-spaced ramp (`BoundaryNorm` over percentiles) was tried to spend the ramp evenly, then dropped
at the author's request: the value of the raw `ΔR²` matters more than the visual balance, even
though most datasets end up in the darker bands. The ramp (`BoundaryNorm`, `N_COLOUR_BINS`) is
linear over the raw value, capped at the 99.5th percentile so one near-fully-confounded draw does not
compress the whole scale into a single band; the colourbar ticks the raw edges directly.

Table: median / 90th-percentile `ΔR²` and the share above `LEAK_THRESHOLD` (0.02), for each
(clause, condition) × (all datasets, top tercile).

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis exclusion` |
| **Config group** | preset `general_ite` (same cache as the other B.3 audits; no `--force`) |
| **Analysis module** | `analyses/exclusion.py` → `ExclusionAnalysis` (`name="exclusion"`) |
| **Output dir** | `results/general_ite/exclusion/` |
| **Outputs** | `exclusion.pdf` → `fig:prior_exclusion`; `exclusion_table.tex` → `tab:prior_exclusion` |
| **Appendix location** | §B.3.2 "Exclusion Restrictions" (`app:prior_exclusion`) |

Regression settings are class attributes (`model="rf"`, `n_splits=3`, `max_rows=1024`) — set
`analysis.model = "tabicl"` for the robustness pass. Runs ~1.5 s/dataset with `rf` on the full 1024
rows (≈25 min for the whole `general_ite` cache), then re-renders instantly off `cache/regression/`.
Predictor sets are tagged so they are shared with the other leave-one-out audits (T2.3) wherever the
block combination coincides.

### B.3.3 Confounding strength — `fig:prior_confounding`, `tab:prior_confounding`

> **`\input` into `B_prior.tex`.** A concise paragraph plus both figures (`fig:prior_confounding`,
> `fig:prior_confounding_share`) and the table are now in §B.3.3. Asset sync (copying the rendered
> `.pdf`/`.tex` into `docs/latex/assets/prior/`) is still a separate, un-run step for this doc.

Audits `as:latent_ignorability`, and with it the empirical premise the paper rests on: that the
synthetic confounding is *real* (an estimator adjusting only on measured covariates is biased) and
that conditioning on the true `U` restores unbiasedness. A four-rung ladder, every rung scored
against the same exact ground truth `mean(MU1 − MU0)`:

| rung | adjustment | what it is |
|---|---|---|
| `naive` | none | `mean(Y\|A=1) − mean(Y\|A=0)` |
| `backdoor_X` | `X` | the correct adjustment set *if* there were no latent confounder |
| `backdoor_XWZ` | `(X, W, Z)` | the DAG-blind move: every measured column in the adjustment set |
| `oracle_XU` | `(X, U)` | **identification** oracle — the minimal set `as:latent_ignorability` names |
| `oracle_XUW` | `(X, U, W)` | **efficiency** oracle — adds `W`, a genuine parent of `f_Y` |

**Why two oracles.** `W` is redundant for identification (conditioning on `U` already blocks the
backdoor path) but it is *not* redundant for estimation: `_dataset.py:854` feeds `W_direct` — a
per-dataset column subsample of `W` — straight into `f_Y`, so an oracle that omits it is running a
**misspecified outcome model** and its residual error overstates the floor. The `XU`→`XUW` gap is
therefore a pure regression-quality quantity, which matters because the oracle rung *is* how the
figure's floor is read. Adding `W` cannot open a path: it is pre-treatment (`f_W` never sees `A`) and
its only parents, `U` and `X`, are both already conditioned on.

`Z` is in neither oracle. It is a parent of `f_A` only and excluded from `f_Y` by construction
(`as:proxy_exclusion`), so in an outcome-regression estimator it can add noise but no signal — and an
instrument-like variable in an adjustment set is the textbook bias-amplification risk. Throwing in
everything measured is rung 3's job, not the oracle's.

Errors are `|ATE_rung − ATE_true| / sd(Y_obs)` — normalised because outcome scales are arbitrary
per SCM, the same reasoning `fig:prior_heterogeneity_spread` uses. **With `K=1` the ground truth is
exact for the rows in hand**: `MU0/MU1` *are* the realised potential outcomes, so `mean(MU1 − MU0)`
is the in-sample ATE and no rung is measured against a noisy target. The oracle rung's error is
therefore pure regression estimation error, which is exactly why it belongs in the ladder — it is the
floor the other rungs should be read against.

Every rung but `naive` goes through `regression.t_learner_mu` (T2.0). Its one subtlety: the
opposite-arm rows are a genuine counterfactual and are predicted from a single fit on the whole arm,
while the *same*-arm rows are predicted out-of-fold within that arm. Without that split the factual
half would carry an in-sample-inflated fit, biasing `mean(mu1 − mu0)` by an amount that depends on
the treated fraction.

**Second figure: error vs. delivered confounding share.** `confounding_share_of_U_in_{A,Y}` is a
*sampled* HP (`Beta(1.2, 2.8)`), so it cannot be swept while holding the SCM fixed — pinning it
changes `generative_digest()` and hence every dataset seed, the same trap documented for the
`mcmc_cate` sweep (T3.1) and `seq_len` (T2.6). It does not need to be: 1024 draws *are* the sweep,
and the realised share is recorded per dataset. Datasets are binned on the **delivered** `U` share
(first-layer injected variance, reusing `mechanism_inputs.mechanism_frame` so B.2.3 and this audit
report the same number — not the target share), on a 5×5 quantile grid of `(s_Y, s_A)`.

The grid is 2-D on purpose: confounding bias needs `U` in **both** mechanisms, so a marginal view in
either share alone averages away the interaction. Panel (b) bins the *oracle* rung on the same grid
and **shares the colour scale** — that is the whole point of the pair. An oracle error that grows
with confounding strength would be finite-sample regression difficulty, not a failure of
`as:latent_ignorability`.

Datasets with `d_U = 0` are **kept** (unlike in the exclusion audit, where they made the positive
control vacuous): here they are the unconfounded end of the very axis being measured.

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis confounding_strength` |
| **Config group** | preset `general_ite` (same cache as the other B.3 audits; no `--force`) |
| **Analysis module** | `analyses/confounding.py` → `ConfoundingStrengthAnalysis` (`name="confounding_strength"`) |
| **Output dir** | `results/general_ite/confounding_strength/` |
| **Outputs** | `confounding_ladder.pdf` → `fig:prior_confounding`; `confounding_vs_share.pdf`; `confounding_table.tex` → `tab:prior_confounding` |
| **Appendix location** | §B.3.3 "Confounding Strength" (`app:prior_confounding`) |

**Headline (n=1024, full `general_ite` cache, `rf`).** Median normalised error: naive 0.041,
backdoor-`X` 0.038, backdoor-`(X,W,Z)` 0.030, oracle-`(X,U)` 0.028, oracle-`(X,U,W)` **0.025** — at
the median the ladder is nearly flat, and adjusting on `X` alone is marginally *worse* than not
adjusting (estimation variance without removing the `U` path). Backdoor-`X` is 1.34× the `(X,U)`
oracle and 1.47× the `(X,U,W)` one; adding `W` tightens the floor by 1.10×.

The result is in the tail, not the median: p90 error 0.195 / 0.163 / 0.121 / 0.104 / 0.105, share
materially biased (>10% of `sd(Y)`) **24.1% / 21.1% / 14.8% / 10.8% / 10.7%**. So ~1 dataset in 4 is
genuinely confounded and either oracle roughly halves that rate — a real effect, but a **minority
regime**, not the typical draw. Read this as *coverage*, not "the proxies are decorative": the median
prior draw is only mildly confounded because `Beta(1.2, 2.8)` sits low.

**Conditioning on the delivered share is a much weaker signal than it first looked** (an earlier,
smaller-sample pass over-read it — see git history if curious). At the full n=1024, Spearman of
`share_min` vs. backdoor-`X` error is **0.047** (was 0.11 at n=250); the quartile contrast is 0.033
(bottom) → 0.039 (top), and the grid's top-right cell (both shares in their top tercile, n=109) is
0.045 against an oracle floor of 0.028 — a real but modest ~1.6× gap, not the ~2× the smaller sample
suggested. The oracle control's own quartile move (0.030 → 0.024, Spearman −0.035) is if anything
backwards, i.e. noise, which is the right scale to judge the backdoor-`X` movement against. Bottom
line: the grid shows the interaction is directionally present but small relative to cell-to-cell
noise at this n; don't oversell the conditioning story.

**Why the grid's columns (`share_Y`) look flatter than its rows (`share_A`) — a normalisation
artifact, not a missing effect.** `sd(Y_obs)` is computed from the very outcome whose `U`-composition
`share_Y` describes, so the two are mechanically entangled (Spearman `sd_y` vs `share_Y` = 0.056, vs.
`sd_y` vs `share_A` = 0.041). Concretely: the *unnormalised* naive/backdoor bias correlates with
`share_Y` (Spearman 0.069 / 0.072) at least as strongly as with `share_A` (0.059 / 0.062) — if
anything a bit more. Dividing by `sd_y` — necessary for cross-SCM comparability, the same reasoning
`fig:prior_heterogeneity_spread` uses — then partially cancels the `share_Y` signal specifically (an
SCM where `U` dominates `f_Y` tends to have a larger `sd_y` too, shrinking the ratio), dropping its
normalised correlation to 0.032 / 0.042, while `share_A`'s survives mostly intact (0.053 / 0.057).
Both `raw_error` and the plotted `abs_error` are kept in `confounding_frame`'s output, and the
`spearman_*_vs_{raw,abs}_error_*` / `spearman_sd_y_vs_share_*` keys are in `summary.json` as the
evidence trail.

**`W`'s benefit as an oracle addition is not uniform across the confounding-strength grid.** Pooled,
`(X,U,W)` beats `(X,U)` (0.025 vs. 0.028 median); but in the grid's top-right (high-confounding) cell
the two are essentially tied (0.0283 vs. 0.0280, n=109 each) rather than `(X,U,W)` extending its lead.
`W` is a proxy of `U`, which drives assignment, so where `U` dominates it can cost cross-arm overlap
in predictor space about as fast as it sharpens the outcome model — the pooled advantage comes from
elsewhere on the grid, not from the hard-confounding corner. Worth a finer bin count or a direct
per-cell `(X,U)` vs `(X,U,W)` comparison before leaning on this in the prose; at 3×3/n=1024 it reads
as "no clear winner in the hard cell," not a reversal.

Needs the **mechanism-diagnostics payload** (`record.mechanisms`) for the share grid — a cache
predating it must be regenerated once with `--force` (same data, diagnostics added). Regression
settings are class attributes (`model="rf"`, `n_splits=3`, `max_rows=1024`); set
`analysis.model = "tabicl"` for the robustness pass. ~1.4 s/dataset with `rf`, then re-renders
instantly off `cache/regression/`. The T-learner cache is keyed separately from `cached_oof`'s, so
it never collides with the exclusion/relevance entries.

---

## Implemented but not currently referenced in the tex

### Single-spec dimension audit (no comparison)

| | |
|---|---|
| **Command** | `PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis dimensions` |
| **Config group** | preset `general_ite` (any single preset works — swap `--spec`) |
| **Analysis module** | `analyses/dimensions.py` → `DimensionsAnalysis` (`name="dimensions"`) |
| **Output dir** | `results/<spec-name>/dimensions/` |
| **Outputs** | `dimension_histograms.*`, `dimension_joint_heatmaps.*` (single-spec marginal/joint dim plots), `completeness_heuristic.tex` (numeric-survival-by-`d_U` table) |
| **Appendix location** | none — superseded by the `dimension_comparison` family run above |

---

## Planned, not yet implemented (Tracks 1–3 of the appendix plan)

No CLI command exists yet — the modules below are named in
[`docs/prior_appendix_plan.md`](../docs/prior_appendix_plan.md) §3/§4 as future `analyses/`
additions, not currently in `analyses/__init__.py`'s registry. Listed here so the mapping is complete once they land.

| Appendix § | Label | Planned module (`name=`) | Planned config group | Target artifact | Plan track |
|---|---|---|---|---|---|
| B.3.4 Proxy relevance | `app:prior_relevance` | `proxy_relevance` | `general_ite` | Fig audit(d) | T2.3 |
| B.3.5 Ill-posedness | `app:prior_illposed` | `illposedness` | `general_ite` | `fig:prior_illposedness` | T2.5 |
| B.3.6 Identification stress test | `app:prior_identification` | `identification` | `general_ite` | `fig:prior_identification` | T2.6 |
| B.3 assembly | `app:prior_audit` | `audit_summary` | `general_ite` (K=1) | 4-panel audit fig, `tab:prior_audit_summary` | T2.7 |
| B.5 MC error | `app:prior_extensions` | `mc_error` | K-sweep family, K∈{1,5,25,100,200,5000} | `fig:mc_error` | T3.1 |
| B.5 structural coverage | `app:prior_extensions` | *(hand-written, no module)* | — | `tab:prior_structural_coverage` | T3.2 |
| B.5 memory budget | `app:prior_extensions` | `mem_budget` | `default` (T=2048) | `tab:prior_mem_budget` | T3.3 |
| B.4 downstream ablation | `app:prior_downstream` | *(needs training runs)* | — | `tab:prior_downstream_ablation` | T3.4 (deferred) |

The regression-based audits (B.3.2/B.3.3/B.3.4) share a **cross-fitted regression helper** (plan
T0.1), `analysis_prior/regression.py` — built for the B.2.3 `mechanism_outputs` analysis
(fold-stable out-of-fold predictions, pluggable `linear`/`rf`/`tabicl` model, disk-cached
predictions under `cache/regression/`). Reuse it for those modules.

> **Removed module.** `analyses/confounding.py` (`confounding_dim`, added in `4801ef3`) was **deleted**
> in `a77c467`. It has no replacement in the registry; the old analytic `eq:conf_dim` /
> `tab:prior_conf_dim` / `fig:prior_conf_dim` deliverables are dropped from the plan and re-scoped as
> the empirical `mechanism_inputs` analysis (T1.1). Any `results/*/confounding_dim/` and
> `results/*/input_sampler/` directories on disk are **orphans** (no module produces them) and can be
> deleted.

**Note on `general_ite` vs. final data.** All the `general_ite`-based commands above (K=1) are fine
for drafting the ITE-phase (B.2/B.3) artifacts, and `heterogeneity` needs nothing more (the
`gamma` collapse is `K`-independent). Only the Monte Carlo artifacts (`mc_error`,
`fig:prior_cate_vs_ite_K`) need `K>1`.

> **A `K` sweep across specs does not hold the SCM fixed.** Changing `mcmc_cate` resizes the
> `T*K` tensors the generator draws, which shifts the RNG stream, so the mechanism weights differ
> between arms — per-dataset error curves across `K` specs would be comparing different SCMs.
> A single `K>1` cache avoids this entirely: it stores both the retained replicate
> (`Y1 - Y0`, variance `Var(Q) + E[sigma^2]`) and the `K`-average (`MU1 - MU0`, variance
> `Var(Q) + E[sigma^2]/K`), two equations that solve per dataset for the CATE signal and the
> within-`x` replicate noise — hence the whole signal-share curve `1/(1 + NSR/K)` with no gold
> standard and no matched SCMs. The MC analysis wants many datasets, not long ones, so a shorter
> `seq_len` at `K=100` is much cheaper than reusing `general_ite`'s `seq_len=2048`.

---

## Reproducing the appendix: full command list

Everything below regenerates every table/figure currently wired into `B_prior.tex` (see the
per-artifact sections above for what each command produces and where it lands). Run from the repo
root:

```
# B.2.1 Hyperparameters — tab:prior_fixed_hp, tab:prior_sampled_hp
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis hyperparameters

# B.2.2 Dimensions + B.2.4 numeric completeness — fig:prior_dim_marginals, tab:prior_dim_sampling,
# fig:prior_dimension_joint_heatmaps
PYTHONPATH=. uv run python -m analysis_prior.main --family dim_ablation --analysis dimension_comparison

# B.2.4 Categorical completeness — fig:prior_category_count_marginals,
# fig:prior_proxy_stats_by_u_cardinality, fig:prior_proxy_composition_vs_u_demand,
# tab:prior_categorical_completeness_survival
PYTHONPATH=. uv run python -m analysis_prior.main --family cat_ablation --analysis categorical_completeness

# B.2.3 Mechanism input composition — fig:prior_mechanism_input_composition,
# tab:prior_mechanism_inputs.
# Needs the per-dataset mechanism diagnostics: a cache written before they
# existed must be regenerated once with --force (same data, diagnostics added).
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis mechanism_inputs

# B.2.3 Mechanism output dependence — fig:prior_mechanism_output_dependence,
# tab:prior_mechanism_outputs.
# Cross-fitted TabICL leave-one-block-out R2 on the delivered data (slow: a few
# thousand in-context regressions the first time; predictions are cached under
# cache/regression/ so re-renders are instant). Uses ALL datasets in the cache,
# so pass --n-datasets to control the count: general_ite's default is 1024
# (~8h of TabICL), so cap it. --n-datasets 256 is served as a prefix of the
# existing 1024-dataset cache — no regeneration.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --n-datasets 256 --analysis mechanism_outputs

# ^ Fast demo of the same figure (a few datasets, random forest, ~seconds) for
# development — writes to results/general_ite/mechanism_outputs_small/, not the
# paper path. Pass --model tabicl --n 16 for a quick TabICL taste.
PYTHONPATH=. uv run python -m analysis_prior.examples.mechanism_outputs_small

# B.2.5 Treatment assignment walkthrough — fig:prior_treatment_walkthrough,
# tab:prior_treatment_walkthrough.
# Reuses the same general_ite cache as mechanism_inputs (both need the raw f_A
# logits, added alongside the mechanism diagnostics — regenerate once with
# --force if the cache predates them).
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_walkthrough

# B.2.6 Heterogeneity dial — fig:prior_heterogeneity_spread.
# Reads the delivered gamma (record.gamma_hetero) off the same general_ite
# cache; gamma cannot be pinned from a spec, so the sweep is the prior's own
# uniform draw. K=1 is fine — the gamma collapse is K-independent.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis heterogeneity

# B.2.6 Monte Carlo CATE targets — fig:prior_cate_vs_ite,
# tab:prior_mc_signal_share.
# The only command needing a K>1 cache (preset mc_targets, K=100), which it
# generates on first run. Everything is arithmetic on MU0/MU1 and the retained
# replicate -- no per-replicate matrix, no regression.
PYTHONPATH=. uv run python -m analysis_prior.main --spec mc_targets --analysis mc_targets --n-jobs -1

# B.2.x Visualize the prior — fig:prior_feature_distributions,
# fig:prior_feature_pairs, fig:prior_feature_pairs_cross.
# Reuses the same general_ite cache; needs no diagnostics payload (only the
# feature blocks and the max_categories field), so no --force is required.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis prior_overview

# B.3.1 Overlap audit — fig:prior_overlap.
# Pure summary of the cached A_probs (the true P(A=1|X,U)) — no regression, no
# diagnostics payload, seconds on the same general_ite cache.
PYTHONPATH=. uv run python -m analysis_prior.main --spec general_ite --analysis treatment_overlap
```

Add `--force` to any of these to regenerate the cache instead of reusing it, and `--n-jobs -1`
to parallelise generation across all cores.

The `hyperparameters`, `dim_ablation`, and `cat_ablation` outputs land under the paths the tex
`\input`s / `\includegraphics` from `assets/prior/`. If your build reads figures straight from
`results/`, no copying is needed; otherwise copy/symlink the PDFs into the appendix figures tree
before building LaTeX.

Everything in "Planned, not yet implemented" above has no CLI command yet — those modules don't
exist in `analyses/__init__.py`'s registry.

## Discovery commands

```
uv run python -m analysis_prior.main --list-analyses   # registered analyses
uv run python -m analysis_prior.main --list-caches     # cached specs on disk
uv run python -m analysis_prior.main --list-families   # registered comparison families
```
