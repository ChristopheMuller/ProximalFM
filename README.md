# ProximalFM: Proximal Foundation Models for CATE Estimation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Status](https://github.com/ChristopheMuller/ProximalFM/actions/workflows/tests.yml/badge.svg)

ProximalFM is a tabular foundation model approach to **Proximal Causal Inference**. It uses a specialized multi-stage Transformer architecture to estimate **Conditional Average Treatment Effects (CATE)** and potential outcomes in settings plagued by unobserved confounding ($U$).

The core idea is to leverage measurement **proxies**—treatment proxies ($Z$) and outcome proxies ($W$)—to non-parametrically "bridge" and neutralize unobserved confounding, following the proximal causal framework (Miao et al., 2018; Tchetgen Tchetgen et al., 2020).

## What it Does

In standard causal inference, we assume **Unconfoundedness**: that we observe all variables ($X$) that simultaneously influence the treatment ($A$) and the outcome ($Y$). When this assumption fails due to unobserved confounders ($U$), we can establish causal identification via Proximal Causal Inference given:
- **Treatment-inducing proxy ($Z$):** Strongly associated with the treatment, but independent of the outcome conditional on $(U, X)$.
- **Outcome-inducing proxy ($W$):** Strongly associated with the outcome, but independent of the treatment conditional on $(U, X, Z)$.

**ProximalFM** processes the full causal tuple $(X, Z, W, A)$ along with context targets ($Y_{obs}$) to map tabular datasets natively to continuous potential outcomes and causal effects via **In-Context Learning (ICL)** without requiring downstream fine-tuning.

---

## The Prior: A Synthetic Prior over Proximal DGPs

ProximalFM is never trained on real data: it learns *in-context* from an infinite stream of synthetic datasets drawn from a prior $\Pi$ over structural causal models (SCMs). The prior must satisfy two competing requirements. **Validity** — whereas existing causal foundation models design their priors to *avoid* hidden confounding, ours deliberately *simulates* unmeasured confounding while structurally guaranteeing the proximal identification assumptions (latent ignorability, proxy exclusion, positivity, and a completeness heuristic), so the CATE remains identifiable from the observed proxies. **Diversity** — building on the TabICL prior, every causal mechanism is a freshly initialized random MLP (or random forest) with sampled depth, width, activations, and exogenous noise laws, covering a wide range of confounding strengths, proxy informativeness, effect heterogeneity, and realistic tabular structure. Each dataset also ships low-variance Monte-Carlo CATE labels (`MU0`/`MU1`), obtained by replicating every covariate row $K$ times and averaging the potential outcomes, so the model trains on the *identified* estimand $\mathbb{E}[Y(a)\mid X]$ rather than on noisy per-unit ITEs. Implementation: [`proximalfm/prior/_dataset.py`](proximalfm/prior/_dataset.py) (`SCMPrior.generate_dataset`), sampled hyperparameter ranges in [`_prior_config.py`](proximalfm/prior/_prior_config.py), pre-generation CLI in [`_genload.py`](proximalfm/prior/_genload.py).

<details>
<summary><b>Full details — DAG generation, assumption enforcement, batch contents, hyperparameters</b></summary>

### Why CATE labels, not the ITE

The estimand is the proximal CATE, a **difference of conditional means**:

$$Q(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x] = \underbrace{\mathbb{E}[Y(1)\mid x]}_{\mu_1(x)} - \underbrace{\mathbb{E}[Y(0)\mid x]}_{\mu_0(x)}.$$

A tempting shortcut is to train against the per-unit contrast (the **ITE**, $\tau_i = Y_i(1) - Y_i(0)$). Its mean is unbiased for $Q(x)$, but its *spread* is not a meaningful uncertainty:

$$\mathrm{Var}(\tau \mid x) = \mathrm{Var}(Y(1)\mid x) + \mathrm{Var}(Y(0)\mid x) - 2\,\mathrm{Cov}(Y(0), Y(1)\mid x),$$

and the cross-covariance term is **never identified from data** — no unit ever reveals both potential outcomes, so the coupling of the outcome noises is fixed only by arbitrary structural choices of the generator. A distributional loss on the ITE would learn exactly the part of the world the data cannot determine. Since we fully control the synthetic DGP, we instead build **low-variance targets by Monte Carlo integration conditional on $X$**: a dual-track pipeline that generates the observational context on one track and clean CATE labels on the other.

### DAG generation: one dataset, step by step

Each dataset of $T$ units (`seq_len`) is produced by `SCMPrior.generate_dataset` ([`_dataset.py`](proximalfm/prior/_dataset.py)). Once per dataset, we sample the causal mechanisms $f_X, f_U, f_W, f_Z, f_A, f_Y$ — each an independently initialized random network — together with the variable dimensions $d_U, d_X, d_W, d_Z$; mechanisms and dimensions are shared across the $K$ Monte-Carlo replicates below, so all replicates of a unit come from the *same* SCM. Then:

1. **Covariates (root).** $X = f_X(\epsilon_X) \in \mathbb{R}^{T \times d_X}$, where the exogenous noise base distribution is itself sampled per dataset (`sampling` ∈ {`normal`, `uniform`, `mixed`}). Columns are randomly discretized with probability `cat_prob = 0.2` (`process_features`), mimicking categorical tabular data. Each row is then replicated $K$ times (`mcmc_cate`, default **200**; auto-capped by `prior_mc_mem_budget` to bound memory).
2. **Latent confounder.** $U = f_U(X, \epsilon_U)$ — generated and stored, but *hidden from the model*.
3. **Proxies.** $W = f_W(U, X, \epsilon_W)$ and $Z = f_Z(U, X, \epsilon_Z)$, with amplified exogenous noise (`eps * 5` in code) so proxies are informative about $U$ without being deterministic images of $(U, X)$.
4. **Confounding amplification.** Extra nonlinear transformations of $U$ (`f_u_augment`) are appended to the parents of $A$ and $Y$; how many is drawn per dataset via `confounding_duplication_of_U_in_A` / `_in_Y` $\sim \mathrm{Beta}(1,1)$ — a per-dataset dial for confounding strength.
5. **Treatment.** Logits $\ell_A = f_A(U, X, Z, \epsilon_A)$ are standardized, rescaled by `propensity_scale` (log-scaled range ≈ 0.5–3.0), shifted by `propensity_shift` $\sim \mathcal{U}(-2.5, 2.5)$ (treatment imbalance), squashed through a sigmoid, and clamped to $[\epsilon, 1-\epsilon]$ with `propensity_epsilon = 1e-3`; then $A \sim \mathrm{Bernoulli}$.
6. **Potential outcomes.** $\bigl(Y(0), Y(1)\bigr) = f_Y(U, X, W, \epsilon_Y)$ in a **single joint two-headed forward pass** — $A$ is never an input of $f_Y$. A per-dataset coefficient $\gamma \sim \mathcal{U}(0,1)$ then interpolates each pair between the raw heterogeneous heads and a constant-shift pair $(y_0,\, y_0 + \mathrm{ATE})$, sweeping effect heterogeneity from strong to near-homogeneous. Only $f_Y$ carries the sampled label noise `noise_std` (≈ 1e-4–0.3); all upstream mechanisms receive their stochasticity through the explicit $\epsilon$ inputs, drawn freshly for every unit and every replicate.
7. **Labels and observation.** Averaging over replicates yields the Monte-Carlo conditional expected potential outcomes; one replicate $k_i \sim \mathrm{Unif}\{1,\dots,K\}$ per unit is retained as the observed row (see *Batch data* below). Constant feature columns are filtered, dimensions padded to their maxima, and degenerate draws rejected and resampled.

**Note on the edge flip ($X \to U$).** The identification-natural graph has $U \to X$; generation reverses it so that $X$ is an ancestral root. This is immaterial for identification — the proximal conditions constrain only conditional independencies given $(X, U)$, unaffected by this edge's orientation — while making the conditional-on-$X$ Monte-Carlo scheme possible: fix $X = x$, replicate, forward-sample, and obtain exact i.i.d. draws from $P(\cdot \mid X = x)$.

### Where each proximal assumption is enforced

| Assumption | How it holds | Where in code |
|---|---|---|
| **Latent ignorability** $\;(Y(0), Y(1)) \perp\!\!\!\perp A \mid (X, U)$ | *Exact by construction*: both potential outcomes are one evaluation of $f_Y$ with all inputs fixed; $A$ is not a parent of $Y$. | `input_y` excludes `A` |
| **Proxy exclusion** | *Structural*: $Z$ is omitted from the inputs of $f_W$ and $f_Y$; $W$ is omitted from the inputs of $f_Z$ and $f_A$ — holds for any realization of the random mechanisms. | `input_W`, `input_Z`, `input_A`, `input_y` |
| **Positivity** | Assignment probabilities clamped to $[10^{-3}, 1-10^{-3}]$ before sampling $A$. | `sample_A` (`propensity_epsilon`) |
| **Completeness** | *Heuristic* (not enforceable for arbitrary nonlinear mechanisms; motivated by the categorical case, where each proxy needs at least as many categories as $U$): the sampler enforces $d_W \ge d_U$ and $d_Z \ge d_U$, and both $f_W, f_Z$ receive $U$ directly with independent fresh noise, making degenerate proxies unlikely. | dimension sampling in `get_batch` |

Because $U$ confounds $A$ and $Y$ while being unobserved, naive unconfoundedness fails on these datasets *by construction* — the only route to the effect is through the proxies, which is exactly the skill the model must learn.

### Batch data: what the model sees vs. what it is trained on

Let $(w_{i,k}, z_{i,k}, a_{i,k}, y_{i,k}(0), y_{i,k}(1))$ be the $k$-th replicate of unit $i$ (covariates $x_i$ carry no replicate index — they are shared). Averaging over replicates gives the Monte-Carlo conditional expected potential outcomes (CEPOs):

$$\widehat{\mu}_a(x_i) = \frac{1}{K}\sum_{k=1}^{K} y_{i,k}(a) \;\approx\; \mathbb{E}\bigl[Y(a) \mid X = x_i\bigr], \qquad a \in \{0,1\},$$

hence the CATE target $\widehat{Q}(x_i) = \widehat{\mu}_1(x_i) - \widehat{\mu}_0(x_i)$, in which unit-level variation due to $U, W, Z$ and the outcome noise has been integrated out (with only $O(1/K)$ Monte-Carlo error). Drawing one replicate $k_i$ per unit yields the training tuple

$$\bigl(x_i,\; w_{i,k_i},\; z_{i,k_i},\; a_{i,k_i},\; y_{i,k_i},\; \widehat{\mu}_0(x_i),\; \widehat{\mu}_1(x_i)\bigr), \qquad y_{i,k_i} = y_{i,k_i}(a_{i,k_i}).$$

**The transformer receives only the observational components** $(x_i, w_{i,k_i}, z_{i,k_i}, a_{i,k_i}, y_{i,k_i})$ as context; the CEPO approximations `MU0`/`MU1` are used exclusively as training targets on the query rows and are never observed by the model. In code, `PriorDataset` yields the 17-field tuple `(U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1, d_U, d_X, d_W, d_Z, seq_len, train_size)` — `U`, `A_probs`, `Y0`, `Y1` are stored for diagnostics and validation only.

Two payoffs of training on $\widehat{Q}(x)$ instead of ITE draws: a point regression converges to the posterior mean $\mathbb{E}[Q(x)\mid \mathcal{D}_n]$; a distributional/histogram loss converges to the genuine Bayesian posterior $P(Q(x)\mid \mathcal{D}_n)$ over the *identified* estimand — expected to be tighter than the ITE spread and to contract as the context grows.

### Sampled hyperparameters (diversity knobs)

One draw per dataset (or per group of datasets sharing an SCM family); ranges from [`_prior_config.py`](proximalfm/prior/_prior_config.py):

| Parameter | Distribution / range | Controls |
|---|---|---|
| `prior_type` | `mlp_scm` | mechanism family: random MLPs |
| `num_layers` | log-scaled trunc-normal, ≥ 2 (mean up to 6) | mechanism depth |
| `hidden_dim` | log-scaled trunc-normal, ≥ 4 (mean up to 130) | mechanism width |
| `mlp_activations` | random diverse set, mixable across layers | nonlinearity of mechanisms |
| `block_wise_dropout`, `mlp_dropout_prob` | choice / scaled Beta | sparsity of the mechanism graph |
| `init_std` | log-scaled, ≈ 0.01–10 | weight scale (function roughness) |
| `noise_std` | log-scaled, ≈ 1e-4–0.3 | outcome label noise (applied in $f_Y$ only) |
| `sampling` | `normal` / `uniform` / `mixed` | exogenous noise base distributions |
| `propensity_shift`, `propensity_scale` | $\mathcal{U}(-2.5, 2.5)$; log-scaled ≈ 0.5–3 | treatment imbalance and propensity sharpness |
| `confounding_duplication_of_U_in_A` / `_in_Y` | $\mathrm{Beta}(1,1)$ | confounding strength |
| `cat_prob` | fixed 0.2 | fraction of discretized (categorical-like) columns |
| $d_U, d_X, d_W, d_Z$ | uniform in `--min_dim_*`/`--max_dim_*` (with $d_W, d_Z \ge d_U$) | variable dimensions |
| `seq_len`, `train_size` | `--min/max_seq_len`, `--min/max_train_size` | episode length and context/query split |

All of these are exposed as CLI flags of the trainer (`proximalfm/train/_train_config.py`). The prior can be consumed **on the fly** (`PriorDataset`, an infinite iterator) or **pre-generated to disk** with `python -m proximalfm.prior` ([`_genload.py`](proximalfm/prior/_genload.py): `SavePriorDataset` writes the sparse `batch_*.pt` stream, `LoadPriorDataset` replays it during training — the same format used by the fixed validation/benchmark builds below).

</details>

---

## Architecture

ProximalFM is built on top of the **TabICLv2** foundation model architecture (Qu et al., 2026), refactored strictly for continuous regression and proximal causal learning. The execution pipeline is divided into three sequential transformer stages:

```
[Input Tables: X, Z, W, A] 
           │
           ▼
┌──────────────────────────────────────┐
│ 1. Column-Wise Embedding (ColEmbed)  │ ──► Sets causal role tokens & sorts valid features
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 2. Row-Wise Interaction (RowInteract)│ ──► Aggregates feature tokens into single row-vectors via CLS
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ 3. In-Context Learning (ICLPredict)  │ ──► Bakes y_train context; outputs outcome/CATE statistics
└──────────────────────────────────────┘
```

* **Causal Role Embeddings:** Features are explicitly augmented with unique, role-specific structural tokens ($0=\text{Covariates } X$, $1=\text{Treatment Proxy } Z$, $2=\text{Outcome Proxy } W$, $3=\text{Treatment } A$) before processing.
* **Validity Packing:** Features are processed using a padding-aware stable sorting strategy (`torch.gather`). Valid feature embeddings are packed at the front of the sequence, gracefully handling datasets with varying numbers of proxies.
* **Target-Aware In-Context Learning:** During the embedding stage, the observed training targets ($Y_{train}$) are standard-scaled and encoded cleanly based on the assigned treatment track ($A=0$ vs $A=1$), acting as an anchored context for query predictions.


---

## Installation & Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and execution.

### Installation
Because this repository contains external research frameworks for benchmarking, clone using the `--recurse-submodules` flag:

```bash
git clone --recurse-submodules https://github.com/ChristopheMuller/ProximalFM.git
cd ProximalFM
uv sync
```
(If you already cloned without this flag, pull submodules manually: `git submodule update --init --recursive`).

---

## Core Training Parameters

Rather than relying entirely on a static configuration, the orchestration environment parses inputs using an advanced command-line interface. Core components are summarized below:

| Argument | Type | Description |
|---|---|---|
| `--prior_type` | `str` | Structural Causal Model strategy (`mlp_scm`, `tree_scm`, `mix_scm`). |
| `--num_batches` | `int` | Number of synthetic episode files to generate/train. |
| `--batch_size` | `int` | Global meta-batch size across DDP nodes. |
| `--micro_batch_size` | `int` | Batch slice size used for gradient accumulation. |
| `--loss_type` | `str` | Objective routing framework (`bins_po`, `cate_decomposition`, `hl`). |
| `--min_dim_X` / `--max_dim_X` | `int` | Dimensionality boundaries for baseline covariates $X$. |
| `--min_dim_Z` / `--max_dim_Z` | `int` | Dimensionality boundaries for treatment proxies $Z$. |
| `--min_dim_W` / `--max_dim_W` | `int` | Dimensionality boundaries for outcome proxies $W$. |
| `--min_train_size` / `--max_train_size` | `float` | Proportion range of sequence allocated for training context. |

---

## Usage

## 1. Training from the On-the-Fly SCM Prior

Train directly via the training loop execution manager while generating synthetic episodes dynamically on-the-fly:

```bash
python -m proximalfm.train --min_dim_U 1 --max_dim_u 5
```
Many training parameters can be added to the script by using `--variable_name variable_value`, see `train/_train_config.py`.

## 2. High-Level Predict API

Evaluate downstream datasets natively via the `predict` pipeline:
```python
from proximalfm._model.proximalfm import ProximalFM

model = ProximalFM(loss_type="bins_po", nbins=1000)
model.eval()

# Directly yields causal effect mappings matching target conditions
cate_estimates = model.predict(
    X=X_tensor, Z=Z_tensor, W=W_tensor, A=A_tensor, y_train=y_train_tensor,
    target="cate", output_type="mean"
)
```

## 3. Fixed Datasets: Validation & Benchmark

ProximalFM trains on an *infinite* synthetic stream, so model selection and method comparison need a **fixed, reproducible** set of datasets, **different from the prior**. The committed recipe [`synthetic/validation_data/config.yaml`](synthetic/validation_data/config.yaml) defines one such set: 480 realistic proximal datasets (2048 rows each, spread evenly across the 12 mechanisms of [`synthetic/realistic_proximal_mechanisms.py`](synthetic/realistic_proximal_mechanisms.py)). This data is useful for both: tuning and choosing a model, and comparing this model with other methods.

<details>
<summary><b>Full guide — rebuild the datasets, validate during training, run the benchmark</b></summary>

### Step 1 — Rebuild the datasets locally

```bash
uv run synthetic/build_proximal_datasets.py --config synthetic/validation_data/config.yaml
```

This regenerates the exact same tensors on any machine (the build is deterministic in `--seed`; the config pins every parameter, including defaults). It writes:

```
synthetic/validation_data/
├── config.yaml      # fully-resolved build recipe (the committed file)
├── manifest.json    # what was built: per-mechanism counts, seed, exact CLI
└── val/             # THE data: raw sparse batch_*.pt + metadata.json
```

`val/` is the single on-disk format for everything downstream. It is the raw, *unstandardized* layout of `proximalfm.prior._genload.SavePriorDataset` (read back by `LoadPriorDataset` for validation); the benchmark rebuilds its standardized view from these same files at load time (see [`benchmarks/bench_data.py`](benchmarks/bench_data.py) — context-only statistics, no test leakage).

Explicit flags override the config, which is how you build a **held-out test batch** with the same characteristics but never-seen DGP draws:

```bash
uv run synthetic/build_proximal_datasets.py \
    --config synthetic/validation_data/config.yaml \
    --name test_data --seed 9999
```

### Step 2 — Use it as validation during training

Point the trainer at the `val/` folder:

```bash
uv run python -m proximalfm.train \
    <your usual training flags> \
    --val_dir synthetic/validation_data/val \
    --val_interval 50 \
    --save_best true \
    --best_metric val_cate/npehe \
    --best_metric_mode min
```

| Flag | Default | Meaning |
|---|---|---|
| `--val_dir` | `None` | Directory of `batch_*.pt` files to validate on (no validation if unset). |
| `--val_interval` | `50` | Steps between validation passes. |
| `--save_best` | `true` | Keep a `best_model.ckpt`, overwritten whenever the monitored metric improves. |
| `--best_metric` | `val_cate/npehe` | Metric key to monitor (e.g. `val_cate/rmse`, `val_ate/rmse`, `val_cate/r2`). |
| `--best_metric_mode` | `min` | `min` for errors, `max` for scores like R². |
| `--val_micro_batch_size` | `--micro_batch_size` | Validation runs without backward, so it can afford a larger micro-batch. |

### Step 3 — Benchmark against other methods

[`benchmarks/bench_config_val.yaml`](benchmarks/bench_config_val.yaml) is a ready-made benchmark config pointing at this dataset. Its `data:` section is all it takes to consume a fixed build:

```yaml
data:
  source: load                                  # never regenerates
  val_dir: synthetic/validation_data/val        # the batch_*.pt directory from Step 1
  output_dir: synthetic/validation_data/bench # where preds/, results/, data.pt cache go
```

Run it (after Step 1) with:

```bash
uv run benchmarks/bench_run.py --config benchmarks/bench_config_val.yaml
```

What happens:

1. **Data** — on the first run the standardized benchmark dict is built from `val/` and cached as `<output_dir>/data.pt`; later runs reuse the cache (pass `--force` to rebuild and recompute predictions).
2. **Methods** — every entry under `methods:` (S/T/X-learners, proximal baselines, tabular causal models…) is fit per episode in parallel; predictions are saved incrementally to `<output_dir>/preds/`, so finished methods are skipped on re-runs.
3. **Models** — every checkpoint under `models:` (e.g. a `best_model.ckpt` selected in Step 2) is evaluated on the same episodes.
4. **Report** — ATE and CATE summary tables are printed and written to `<output_dir>/results/`.

Enable or disable baselines by (un)commenting entries in the `methods:` list of the config. For final held-out numbers, benchmark the *test* batch instead: copy the config and change `val_dir`/`output_dir` to the `--name test_data` build from Step 1.

</details>

---

## Testing

Validation relies on `pytest`. To run the verification suites across model dimensions and loss metrics:
```bash
uv run python -m pytest tests/
```


