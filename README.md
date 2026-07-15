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

## The Prior: A Synthetic Proximal SCM

Like other tabular foundation models, ProximalFM is never trained on real data. Instead it learns *in-context* over an infinite stream of synthetic datasets drawn from a **Structural Causal Model (SCM) prior** (see [`proximalfm/prior/_dataset.py`](proximalfm/prior/_dataset.py)). Each synthetic dataset is one draw of an SCM whose graph is deliberately built to satisfy the **proximal causal identification assumptions**, so that the model learns the mapping *"(observed proximal tuple) → CATE"* that proximal inference licenses.

### What the SCM encodes

Every dataset is generated from a randomly-parameterized DAG over $(X, U, Z, W, A, Y)$, where the mechanisms $f_U, f_W, f_Z, f_A, f_Y$ are random neural nets (`mlp_scm`) or trees (`tree_scm`). The edges encode the proximal roles directly:

- $U$ is an **unobserved confounder** feeding into $Z, W, A, Y$ (it is generated and stored, but hidden from the model at training time).
- $Z$ (**treatment-inducing proxy**) is driven by $U, X$ and feeds the treatment $A$, but *not* $Y$ directly.
- $W$ (**outcome-inducing proxy**) is driven by $U, X$ and feeds $Y$, but *not* $A$ directly.
- $A$ is drawn from a calibrated propensity (`sample_A`) with temperature/shift/clamp controls that enforce **positivity** (overlap).
- The generator also enforces the **completeness** heuristic $\dim(W) \ge \dim(U)$ and $\dim(Z) \ge \dim(U)$, a necessary condition for proximal identification.

Because $U$ influences both the treatment and the outcome while being unobserved, naive unconfoundedness fails on these datasets by construction — the only way to recover the effect is *through the proxies*, which is exactly the skill we want the model to learn.

### Sampling convenience: $X$ as a root ($X \to U$)

For *identification*, the natural graph has $U \to X$. For *generation*, we flip this edge to $X \to U$ and make $X$ an ancestral root. This flip is purely a **sampling convenience** and is believed to remain compatible with the proximal identification assumptions: with $X$ as a root we can fix $X = x$, replicate it, and ancestrally forward-sample everything downstream, obtaining **exact i.i.d. draws from $P(\cdot \mid X = x)$**. That is what makes the Monte-Carlo CATE labels below computable.

### Why Monte-Carlo CATE labels (MU0 / MU1) and not the ITE

The estimand is the proximal CATE, a **difference of conditional means**:

$$Q(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x] = \underbrace{\mathbb{E}[Y(1)\mid x]}_{\mu_1(x)} - \underbrace{\mathbb{E}[Y(0)\mid x]}_{\mu_0(x)}.$$

A tempting shortcut is to train against a single per-unit contrast (the **ITE**, $\tau_i = Y_i(1) - Y_i(0)$). Its mean is unbiased for $Q(x)$, so it is a valid target under an MSE loss — but its *spread* is not a meaningful uncertainty. Writing $Y(a) = f_a(X, W, Z) + \eta_a$,

$$\mathrm{Var}(\tau \mid x) = \mathrm{Var}(Y(1)\mid x) + \mathrm{Var}(Y(0)\mid x) - 2\,\mathrm{Cov}(Y(0), Y(1)\mid x),$$

and the cross-covariance term is **never identified from data**: no unit ever reveals both potential outcomes, so the coupling of $\eta_0, \eta_1$ is fixed only by arbitrary structural choices of the generator. Placing a *distributional* loss on the ITE therefore learns exactly the part of the world the data cannot determine.

The fix, enabled by the $X$-root flip, is to estimate the **identified** functionals directly by Monte Carlo. For each covariate row $x$ we replicate it $K$ times (`mcmc_cate`, default 200), forward-sample $K$ independent completions of $(U, W, Z, A, \varepsilon_Y)$, evaluate both potential outcomes per replicate, and average:

$$\widehat{\mu}_0(x) = \frac{1}{K}\sum_{k=1}^{K} Y^{(k)}(0), \qquad \widehat{\mu}_1(x) = \frac{1}{K}\sum_{k=1}^{K} Y^{(k)}(1), \qquad \widehat{Q}(x) = \widehat{\mu}_1(x) - \widehat{\mu}_0(x).$$

These are the `MU0` / `MU1` tensors returned by the prior. Averaging over the $K$ replicates integrates out both the outcome noise and the unidentified coupling, leaving clean per-SCM labels for the identified means (with only $O(1/K)$ Monte-Carlo error). A single observed outcome $Y_{obs}$ per row is still drawn (one replicate picked at random, consistent with its treatment $A$) to serve as the in-context training signal.

This has two payoffs:

- **Point CATE:** regressing on $\widehat{Q}(x)$ makes the model converge to the posterior mean $\mathbb{E}[Q(x)\mid \mathcal{D}_n]$.
- **Posterior over CATE:** putting a distributional/histogram loss on the per-SCM label $\widehat{Q}(x)$ — rather than on the ITE — makes the learned distribution converge to the genuine Bayesian posterior $P(Q(x)\mid \mathcal{D}_n)$ over the *identified* estimand, since finite contexts $\mathcal{D}_n$ underdetermine which SCM produced them. We expect this posterior to be tighter than the ITE spread and to contract as the context grows.

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

## Output Objectives & Loss Functions

The model bypasses traditional multi-class classification and quantile mapping. Output spacing is driven entirely by the `loss_type` flag configured during training:

### 1. Potential Outcomes Binning (`loss_type="bins_po"`)
The network outputs raw logits of dimension $\text{nbins} \times 2$. It models discrete probability distributions for both counterfactual profiles ($Y^0$ and $Y^1$) evaluated against a unified vector of histogram bin segments.
* **Loss:** Optimized via a parameter-free **Discrete Continuous Ranked Probability Score (Discrete CRPS)** loss alongside a direct MSE penalty on the unscaled CATE profile.

### 2. Additive CATE Decomposition (`loss_type="cate_decomposition"`)
The network isolates structural variations by outputting a tensor of dimension $\text{nbins} + 2$:
* **Index 0 & 1:** Maps the Average Treatment Effect ($\text{ATE}$) via a parametric Gaussian likelihood ($\mu$, $\log \sigma^2$).
* **Remaining Indices:** Maps the heterogeneous residual effect ($\text{CATE} - \text{ATE}$) via a discrete histogram distribution.
* **Loss:** Jointly optimized using standard **Gaussian Negative Log-Likelihood (NLL)** for the ATE component and **CRPS** for the heterogeneous profile.

### 3. Histogram Loss with Gaussian Soft Targets (`loss_type="hl"`)
Computes a cross-entropy loss against a smooth target distribution. For each scalar target, it constructs soft target assignments by integrating $\mathcal{N}(y, \sigma^2)$ over the specified histogram boundaries. The standard deviation $\sigma$ is managed dynamically using a linear warmup and decay schedule.

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

## Output Objectives & Loss Functions
The model bypasses traditional multi-class classification and quantile mapping. Output spacing is driven entirely by the `loss_type` flag configured during training:

### 1. Potential Outcomes Binning (`loss_type="bins_po"`)
The network outputs raw logits of dimension $\text{nbins} \times 2$. It models discrete probability distributions for both counterfactual profiles ($Y^0$ and $Y^1$) evaluated against a unified vector of histogram bin segments.
* **Loss:** Optimized via a parameter-free **Discrete Continuous Ranked Probability Score** (Discrete CRPS) loss alongside a direct MSE penalty on the unscaled CATE profile.

### 2. Additive CATE Decomposition (`loss_type="cate_decomposition"`)
The network isolates structural variations by outputting a tensor of dimension $\text{nbins} + 2$:
* **Index 0 & 1:** Maps the Average Treatment Effect ($\text{ATE}$) via a parametric Gaussian likelihood ($\mu$, $\log \sigma^2$).
* **Remaining Indices:** Maps the heterogeneous residual effect ($\text{CATE} - \text{ATE}$) via a discrete histogram distribution.
* **Loss:** Jointly optimized using standard Gaussian Negative Log-Likelihood (NLL) for the ATE component and CRPS for the heterogeneous profile.

### 3. Histogram Loss with Gaussian Soft Targets (`loss_type="hl"`)
Computes a cross-entropy loss against a smooth target distribution. For each scalar target, it constructs soft target assignments by integrating $\mathcal{N}(y, \sigma^2)$ over the specified histogram boundaries. The standard deviation $\sigma$ is managed dynamically using a linear warmup and decay schedule.

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


