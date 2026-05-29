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

---

## Testing

Validation relies on `pytest`. To run the verification suites across model dimensions and loss metrics:
```bash
uv run python -m pytest tests/
```


