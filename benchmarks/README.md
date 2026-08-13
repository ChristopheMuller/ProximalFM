# `benchmarks/` — method comparison

Runs a fixed set of datasets through every CATE estimator (baselines, proximal methods,
foundation-model competitors, ProximalFM checkpoints) and reports how well each one recovers the
treatment effect. One YAML drives the whole thing.

```bash
uv run benchmarks/run.py --config benchmarks/config_val.yaml
uv run benchmarks/plots.py --path data/synt_eval/<name>/bench
```

---

## 1. What is being scored

Each dataset ("episode") is split into a **context** the methods may fit on and **query** rows they
must predict for. There are two different ground truths on those query rows, and the difference is
the single most important thing to understand before reading any number here:

| target | what it is | why it matters |
|---|---|---|
| **`cate`** *(primary)* | `tau*(x) = E[Y1 − Y0 \| X = x]`, estimated by the generator's Monte-Carlo average `MU1 − MU0` over `K_eff` replicates | the estimand ProximalFM is trained on, and the best any method seeing only `X` can attain |
| **`ite`** *(secondary)* | the realized draw `Y1 − Y0` for that row, carrying its own `U` and noise | irreducibly noisy: `E[Var(Y1−Y0 \| X)]` is typically ~95% of `Var(ITE)`, which flattens every method onto nPEHE ≈ 1 |

The CATE target only exists when the set was built with replicates (`--mc_cate K`, `K > 1`).
Without them the run falls back to the ITE track and says so on the console.

### The three tracks

Methods do not all answer the same question, so they are not all ranked in the same table. Every
prediction output carries an **estimand tag** that spells out what it answers — a *kind* and the
variables the prediction reads at **predict** time, in the canonical order `x,w,z,u`:

```
cate(x)   cate(x,w)   cate(x,z)   cate(x,w,z)   cate(x,u)   cate(x,w,z,u)   ate(x)   ate(x,w,z)
```

`cate` means the prediction varies per row, `ate` that it is one scalar per episode. Only
*predict-time* use counts: `PLearner` fits nuisances on both proxies and is still `cate(x)`, because
its prediction is a function of `x` alone. `u` appears when `is_oracle: true` concatenates the true
confounder onto `X`. Tags are derived from the config by
[`methods.method_estimand`](methods.py) (override with e.g. `estimand: cate(x)` on a method or an
`outputs[]` entry; the legacy `x`/`xw`/`ate` spellings still parse, as `cate(x)`/`cate(x,w)`/`ate(x)`).

Which table a tag lands in is then two checks and nothing more — the kind, and whether `w` or `z` is
in the parenthesis:

| tag | appears in |
|---|---|
| `cate(…)` without `w`/`z` — e.g. `cate(x)`, `cate(x,u)` | **`summary_cate.csv`** (headline), `summary_ite.csv`, `summary_ate.csv` |
| `cate(…)` with `w` or `z` — e.g. `cate(x,w)`, `cate(x,w,z)` | `summary_ite.csv`, `summary_ate.csv` |
| `ate(…)` | `summary_ate.csv` only |

`u` is deliberately *not* disqualifying: an oracle row is in the headline table as an upper bound,
which is the whole reason it is computed.

`cate(x,w)` is a genuinely finer conditional, not a better estimate of `tau(x)`. On the ITE target that
extra information is a real advantage; on the CATE target it is answering a different question. It
therefore gets its own row in the ITE table and none in the CATE ranking — the "they had the
advantage and still lost" comparison.

**Reference rows** are computed with no fitting and appear in every table, shown but never
ranked (and never eligible for a bolded best-of cell):

- **`Oracle-CATE(X)`** — the true `tau*(x)`. Scores nPEHE ≈ 0 on the cate track (a self-check on the
  whole pipeline; if it does not, something is wrong with the target plumbing) and the **irreducible
  noise floor** on the ite track. That floor is what makes the `cate(x,w)` methods' advantage readable.
- **`Oracle-ATE`** — the best per-episode constant. The nPEHE = 1 / Shape R² = 0 anchor.
- **`Oracle-ITE`** — the realized `Y1 − Y0` itself, the mirror image of `Oracle-CATE(X)`. Perfect on
  the ite track (nPEHE = 0) and, on the cate track, the **price of the noise**: it predicts
  `tau*(x)` plus that row's own `U` and noise, so its nPEHE is `SD(ITE − tau*)/SD(tau*)`, normally
  well above 1, with a strongly negative Shape R². Read beside `Oracle-ATE` it says how much of
  `Var(ITE)` no `tau(x)` method should be trying to explain — and it is the reason a good method can
  beat a *perfect ITE predictor* on the CATE table without anything being wrong. At the ATE level it
  is near-unbiased (only the row-average of the noise survives), so its `ATE RMSE` is the
  per-episode sampling floor for that column. Only emitted when the set carries a Monte-Carlo CATE
  target; without one it would duplicate `Oracle-CATE(X)` exactly.

### Metrics

Per episode, everything relative to `SD(tau)` so episodes can be pooled
([`metrics.py`](metrics.py) has the formulas and the caveats):

- `npehe` = `sqrt(mean((tau_hat − tau)²) / Var(tau))` — 1.0 is the constant-ATE baseline.
- `cate_r2` = `1 − npehe²`; `shape_r2` = `1 − Var(resid)/Var(tau)` (heterogeneity recovery, level
  error removed — the most discriminative one).
- `ate_bias_norm`, `ate_rmse` — the level component.
- `s2_over_var` — the share of the CATE target's variance that is Monte-Carlo noise *in the target
  itself*. **Reported, never subtracted**: ~0.004 at `K=1000`, ~0.2 at `K=8`. A warning fires past
  5%; past that, every method scores worse than it is and the set wants a larger `--mc_cate`.

Episodes with `Var(tau) < 1e-6` are dropped from the scale-free metrics (same cutoff as the
trainer's `CausalEvaluator`, pinned by a test — so the benchmark's cate-track nPEHE and wandb's
`synt_point/cate/npehe` are the same quantity).

---

## 2. Files

| file | what it does |
|---|---|
| **[`run.py`](run.py)** | The entry point. Loads the data, fits every method over every episode in a process pool, runs the ProximalFM checkpoints, scores all three tracks, writes `results/`. |
| **[`data.py`](data.py)** | Builds the standardized benchmark dict from the raw sparse `val/` batches (`batch_*.pt`). Standardization uses **context-only statistics** — preserving that no-test-leakage property is the point of the module. Emits `true_ITEs` always and `true_CATEs`/`cate_s2`/`cate_K_eff` when the set has replicates. Imports nothing from `proximalfm` on purpose (`sparse2dense` is vendored, kept in sync by a test) so the benchmark runs where proximalfm is not installed. |
| **[`data_csv.py`](data_csv.py)** | The same dict from a single real-world CSV (Causal Chambers), given a column→variable-block mapping. Always one episode. No CATE target exists here, so this path is ITE-only by construction. |
| **[`methods.py`](methods.py)** | Every estimator behind one `CATEEstimator` interface (`fit(X, A, Y, W=, Z=)` / `predict(X, W=, Z=)` / `predict_variants`), instantiated by `build_method` from the config's `cls:`. Also `method_estimand`, the estimand derivation. |
| **[`metrics.py`](metrics.py)** | Pure numpy/pandas scoring: per-episode metrics, track selection, the summary tables, the rich console table. No torch, no model — that isolation is why the point metrics are duplicated rather than shared with `train/_eval.py`. |
| **[`tables.py`](tables.py)** | A `# %%` script: reads `summary_{cate,ite,ate}.csv` and prints one LaTeX table each, top-3 bolded (direction-aware, oracles excluded). Set `experiment_dir` at the top. |
| **[`plots.py`](plots.py)** | Figures from `results.csv`: nPEHE and Shape R² boxplots, ATE scatter, bias-vs-nPEHE, critical-difference diagram, and per-episode true-vs-predicted scatters. `--target {cate,ite}` picks the track; outputs are suffixed with it. |
| **[`tuning.py`](tuning.py)** | Method-agnostic hyperparameter search scaffolding. A method supplies `fit_score(hparams) -> float` (lower better); this owns the split, the grid/random enumeration and the selection. Used by NMMR's `tune:` block. |
| **[`baselines/`](baselines/)** | The heavy proximal implementations: `kpv.py`, `pmmr.py`, `nmmr.py`, `dfpv.py`, `plearner.py`, plus the shared `kernels.py`. `methods.py` wraps them; the estimator maths lives here. |
| **[`causal_chambers/`](causal_chambers/)** | Real-data benchmark (Causal Chambers), notebook/Rmd driven, separate from the main run. `prepare_chamber_csv.py` turns a raw export into the canonical CSV. |
| `config_val.yaml` | The main config: a fixed synthetic build, the full method list, ProximalFM checkpoints. |
| `config_chamber.yaml` | The single-CSV real-data path. |

---

## 3. Config

```yaml
data:
  source: load                                  # load | csv
  val_dir: data/synt_eval/<name>/val            # a build's val/ directory
  output_dir: data/synt_eval/<name>/bench       # where data.pt, preds/, results/ go

run:
  seed: 999999999
  n_workers: 6
  device: mps                                   # for the neural models only
  force_rerun: false

methods:                                        # baselines; comment one out to disable
  - name: S-OLS
    cls: S_Learner
    base: linear
    use_W: false
    use_Z: false

models:                                         # ProximalFM checkpoints
  - name: ProximalFM5-last
    checkpoint: mc_run_5/step-70000.ckpt
    all: false                                  # true: score every ckpt in the directory
```

Build a set to point `val_dir` at with **`--mc_cate K` (K > 1)**, or there is no CATE track:

```bash
uv run python -m synt_eval.build_proximal_datasets --name proximal_val_v1 \
  --n_datasets 480 --n_max 2048 --n_q 128 --mc_cate 200 --seed 123
```

`prior_eval.build` sets work too — same on-disk format, and they always carry replicates.

**Keys the `data:` block takes per `source:`.**

| `source:` | keys |
|---|---|
| `load` *(default)* | `val_dir` |
| `csv` | `csv_path`, `columns`, `prop_train`, `shuffle`, `normalize_treatment` |

The benchmark scores **pre-built sets only**. There is no on-the-fly prior sampling: a set that
exists only inside one run is one nobody else can score against, so the row split, the replicate
count and the seed belong to a builder (`prior_eval.build`, `synt_eval/build_proximal_datasets.py`,
`lit_eval.build`) whose output is on disk and has a `content_digest`. `source: generate` and its
`n_context`/`n_query`/`n_batch`/`prior:` keys are gone; a config still asking for it is **refused
with an error** rather than quietly falling back to a cache. `prop_train` survives only under `csv`,
where the row count is a property of the file and a ratio is the only thing there is to give.

**Common method keys.** `name` (display + output filename), `cls` (dispatch), `base`
(`linear` | `rf` | `tabicl`), `use_W`/`use_Z` (whether proxies enter the design matrix — **default
`True`** for the learners and the PFN wrappers), `interactions`, `is_oracle: true` (concatenates the
true `U` onto `X` — an upper bound, not a competitor), and `estimand:` to override the derived tag.

**`base: tabicl` speed.** TabICL dominates the wall clock for a reason that is entirely outside the
model: with no cache, every `predict()` re-runs the whole context (2048 rows × 8 ensemble members)
through the network, and an S-learner predicts twice off one fit. The `tabicl:` block passes options
straight to `TabICLRegressor`; the benchmark defaults to `kv_cache: repr`, which caches that context
pass at `fit()` time and reproduces the uncached predictions **bit for bit**. 480 episodes, 6 workers:

| `kv_cache`        | 480 episodes | vs uncached                                  | peak RSS / worker |
|-------------------|--------------|----------------------------------------------|-------------------|
| `false` (library) | 30.6 min     | —                                            | 1.61 GiB          |
| `repr` (default)  | 22.7 min     | bit-identical                                | 1.54 GiB          |
| `true`            | 16.6 min     | max abs Δ ≈ 3e-7 — 2 float32 ulp             | 2.44 GiB          |

```yaml
  - name: S-TabICL
    cls:   S_Learner
    base:  tabicl
    tabicl: { kv_cache: true }      # 1.8x, at float32 accumulation noise
    torch_threads: 2                # a further 1.2x; see below
```

`kv_cache: true` additionally caches the ICL layers' KV projections; the residual difference is
accumulation order, not a change of estimator. Check the memory column against `n_workers` before
using it.

`torch_threads:` (any method, default 1) raises the per-worker torch thread count that `_worker`
otherwise pins to 1. It only matters for the torch-heavy methods — the sklearn ones don't use torch
threads.

**Don't bother chasing the device.** Once the context pass is cached, every arrangement converges on
the same ceiling, because the irreducible work is one forward of 8 ensemble members over the 2048
context rows per episode. Real episodes, 480 of them:

| arrangement                          | ep/s  | 480 episodes |
|--------------------------------------|-------|--------------|
| cpu, 6 workers × 1 thread, no cache  | 0.255 | 31.3 min     |
| cpu, 6 workers × 1 thread, `kv`      | 0.467 | 17.1 min     |
| cpu, 6 workers × 2 threads, `kv`     | 0.556 | 14.4 min     |
| mps, 2–3 workers × 2 threads, `kv`   | 0.55  | 14.5 min     |
| mps, 1 process × 4 threads, `kv`     | 0.569 | 14.1 min     |

`device: mps` is ~5x faster *per episode* than one CPU worker (2.1 s vs 15.0 s), which is why a
single GPU process keeps up with six CPU ones — but it does not beat them, so it is worth choosing
only for its memory profile (one 2.4 GiB process instead of six). Breaking the ~0.57 ep/s ceiling
needs episodes batched into the model's batch dimension, the way the ProximalFM path already does
it; that means bypassing `TabICLRegressor` and padding the ragged `d_X`, so it would no longer be
the same predictions.

**Multi-output methods.** KPV / PMMR / NMMR / DFPV fit once and emit several predictions from that
one fit via an `outputs:` list — much cheaper than four separate entries:

```yaml
  - name: PMMR
    cls: PMMR
    outputs:
      - { name: PMMR-XW,      strategy: paired }       # -> xw
      - { name: PMMR-X-Wind,  strategy: independent }  # -> x
      - { name: PMMR-X-Wcond, strategy: conditional }  # -> x
      - { name: PMMR-ATE,     strategy: marginal }     # -> ate
```

---

## 4. Methods

`cls:` values accepted by `build_method`, and what each one's prediction is actually a function of:

| `cls` | prediction | estimand | notes |
|---|---|---|---|
| `S_Learner` | `f(x,·,1) − f(x,·,0)` | `cate(x…)` | one model with `A` as a feature; `interactions: true` adds `A·X` |
| `T_Learner` | one model per arm | `cate(x…)` | |
| `X_Learner` | propensity-weighted two-stage | `cate(x…)` | |
| `CausalForest` | EconML `DROrthoForest.effect(X)` | `cate(x)` | `use_W`/`use_Z` feed EconML's *nuisance controls*, not the heterogeneity features |
| `P2SLS` | a single fitted ATE, broadcast | `ate(x)` | proximal two-stage least squares |
| `PLearner` | `E[gamma \| X]` from a cross-fit proximal score | `cate(x)` | `ProximalPLearner`; W/Z used at fit, discarded at predict |
| `KernelPV` | kernel proxy-variable bridge | per strategy | `baselines/kpv.py` |
| `PMMR` | proximal maximum-moment-restriction bridge | per strategy | `baselines/pmmr.py` |
| `NMMR` | neural MMR bridge | per strategy | `baselines/nmmr.py`; supports `tune:`; no `marginal` |
| `DFPV` | deep feature proxy variable | per strategy | `baselines/dfpv.py`; forced to CPU inside the pool |
| `CausalFM`, `DoPFN`, `CausalPFN` | in-context tabular causal FMs | `cate(x…)` | imported from the git submodules via `_switch_src_context`; init them first |
| *(`models:` block)* | ProximalFM | `cate(x)` | receives the full sequence but **masks the query rows' proxies** (`_hide_query_proxies`), so it answers `tau(x)` using `W`/`Z` only through the context |

**Strategies** (the four proximal bridge methods) decide what is done with `W` at prediction time:

- `paired` → `h(1,x,w) − h(0,x,w)` at the query row's own `w`. **`cate(x,w)`.**
- `independent` → marginalizes `W` over its context empirical marginal (assumes `W ⟂ X`). **`cate(x)`.**
- `conditional` → marginalizes over `W|X` by kernel smoothing on `X`. **`cate(x)`.**
- `marginal` → averages over the empirical joint `(X, W)`: one scalar. **`ate(x)`.**

Note that `independent`/`conditional`/`marginal` marginalize over the **context** sample, so they
assume the query rows come from the same distribution.

---

## 5. Outputs

Written under `output_dir`:

```
data.pt                     the standardized dict (cached; see below)
preds/<output>_preds.npz    per-method `preds` [B, n_test] + `times` [B]
results/results.csv         long-form: one row per (method, episode, target)
results/summary_cate.csv    the headline table
results/summary_ite.csv     the ITE track (adds the `xw` methods)
results/summary_ate.csv     level only (where `ate` methods live)
results/summary.csv         a copy of the primary track's table
results/plot_*_{cate,ite}.png
```

**Everything is resumable and cached.** `data.pt` is reused if present, per-method predictions land
in `preds/` incrementally and finished methods are skipped on re-run. `--force` (or
`run.force_rerun`) recomputes. **When results look stale, suspect the cache first** — and note that a
`data.pt` cached before the CATE target existed has no `true_CATEs`, so the run silently drops to the
ITE track (it does print the banner); delete it or pass `--force`.

Predictions are per-episode and independent, so a method that raises on some episodes leaves NaN
there and is reported with a non-zero failure rate rather than taking the run down.

---

## 6. Related

- Distributional scoring of ProximalFM (CRPS, coverage, PIT) is **not** here — the benchmark's
  interface is point predictions and no baseline has a posterior. Use `python -m prior_eval.score`
  on the same set; it has the frozen comparison scale and the head's grid.
- [`../proximalfm/train/metrics.md`](../proximalfm/train/metrics.md) is the catalogue of the
  *training-side* metric namespace and the estimand traps behind it. Read it before adding a metric
  on either side.
