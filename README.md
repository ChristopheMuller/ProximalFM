# ProximalFM: Proximal Foundation Models for CATE Estimation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Status](https://github.com/ChristopheMuller/ProximalFM/actions/workflows/tests.yml/badge.svg)

ProximalFM is a foundation model approach to **Proximal Causal Inference**. It uses a specialized Transformer architecture to estimate **Conditional Average Treatment Effects (CATE)** in settings where unobserved confounding exists.

The core idea is to leverage **proxies** ($Z$ and $W$) to "bridge" the unobserved confounding ($U$), following the proximal causal framework (Miao et al., 2018; Tchetgen Tchetgen et al., 2020).

## What it Does

In standard causal inference, we assume **Unconfoundedness**: that we observe all variables that influence both the treatment ($A$) and the outcome ($Y$). When this assumption fails due to unobserved confounders ($U$), we can use Proximal Causal Inference if we have:
- **Treatment-inducing proxy ($Z$):** Influences treatment but not the outcome (except through $U$ and other covariates).
- **Outcome-inducing proxy ($W$):** Influences outcome but not the treatment (except through $U$ and other covariates).

ProximalFM trains a **Per-Feature Transformer** on synthetic episodes generated from a flexible causal prior. The model learns to map observed contexts ($X, Z, W, A, Y$) to CATE estimates, effectively internalizing the proximal causal identification strategy during pre-training.

## Architecture

The model is built on top of the **[NanoTabICLv2](https://github.com/soda-inria/nanotabicl)** architecture (Qu, Jingang, et al. "TabICLv2: A better, faster, scalable, and open tabular foundation model." arXiv preprint arXiv:2602.11139 (2026).), using:
- **Per-Feature Encoding:** Each feature (X, Z, W, A, Y) is encoded with its own type-aware embedding.
- **Heteroscedastic Variance:** The model outputs both a mean and a log-variance for its predictions, allowing for uncertainty estimation and training via Gaussian NLL.
- **In-Context Learning:** During inference, the model takes a set of context observations and provides estimates for new query points.

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and execution.

### Prerequisites
- Python >= 3.13
- `uv` installed on your system

### Installation
Clone the repository and sync dependencies:
```bash
git clone https://github.com/ChristopheMuller/ProximalFM.git
cd ProximalFM
uv sync
```

## Configuration

All training and model parameters are defined in `config.yaml`.

| Section | Key | Description |
|---------|-----|-------------|
| **training** | `device` | Compute device (`cuda`, `mps`, or `cpu`). |
| | `epochs` | Number of training iterations over the prior. |
| | `save_dir` | Directory to save model checkpoints. |
| **model** | `embed_dim` | Embedding dimension for the model. |
| | `*_num_blocks` | Number of blocks for col, row, and icl components (e.g. `col_num_blocks`). |
| | `*_nhead` | Number of attention heads for col, row, and icl components (e.g. `col_nhead`). |
| **data** | `batch_size` | Number of episodes per batch. |
| | `samples_per_episode` | Observations per context window. |
| **prior** | `d_x`, `d_z`, `d_w` | Dimensionality of covariates and proxies. |
| | `d_u` | Dimensionality of unobserved confounders (internal to prior). |

## Usage

### Running Training
To start training the model on the causal prior defined in the configuration:
```bash
uv run python -m proximalfm.Train
```
This will:
1. Load `config.yaml`.
2. Generate synthetic episodes from the prior.
3. Train the `PerFeatureTransformerCATE` model.
4. Save checkpoints to the directory specified in the config.

## Testing

The project uses `pytest` for validation. All tests are located in the `tests/` directory.

To run the full test suite:
```bash
uv run python -m pytest tests/
```

## Directory Structure

- `proximalfm/`: Core package containing the model, training logic, and prior.
  - `Model.py`: Transformer-based CATE estimator.
  - `Prior.py`: Synthetic data generator for proximal causal structures.
  - `PriorDataLoader.py`: PyTorch DataLoader for on-the-fly episode generation.
  - `Train.py`: Main entry point for training.
- `tests/`: Unit and integration tests.
- `config.yaml`: Centralized configuration.
