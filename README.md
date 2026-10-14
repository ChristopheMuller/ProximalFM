# ProximalFM

**A foundation model for proximal causal learning**

Estimate conditional average treatment effects from observational data with hidden confounding and observed proxies, using in-context learning.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Status](https://github.com/ChristopheMuller/ProximalFM/actions/workflows/tests.yml/badge.svg)

[Paper](https://arxiv.org/abs/2610.08078) · [Model weights](https://huggingface.co/christophemuller/ProximalFM)

<details>
<summary><strong>What is proximal causal learning?</strong></summary>

**Proximal causal learning** is a framework for estimating causal effects when some confounders are unobserved, but informative proxies for them are available. If a hidden variable $U$ affects both treatment $A$ and outcome $Y$, adjusting for observed covariates $X$ alone may leave confounding bias. Proximal methods use two complementary proxy sets to recover information about that hidden confounding ([Miao et al., 2018](https://doi.org/10.1093/biomet/asy038); [Tchetgen Tchetgen et al., 2020](https://arxiv.org/abs/2009.10982)).

The observed data are partitioned by their causal roles:

| Variable | Role |
| --- | --- |
| $X$ | Observed baseline covariates defining the population or subgroup of interest. |
| $Z$ | **Treatment-confounding proxies:** informative about $U$ and potentially directly related to treatment assignment. |
| $W$ | **Outcome-confounding proxies:** informative about $U$ and potentially directly affecting the outcome. |
| $A, Y$ | Observed treatment and outcome. |

**What assumptions are needed?** In the setting considered here, identification relies on:

- **Latent exchangeability:** conditioning on $(X,U)$ would remove treatment–outcome confounding.
- **Proxy exclusion restrictions:** $Y$ is independent of $Z$ given $(A,X,U)$, and $W$ is independent of $(A,Z)$ given $(X,U)$.
- **Treatment overlap:** both treatment levels have positive probability within strata of $(X,U)$.
- **Completeness and bridge existence:** the proxies carry sufficiently rich information about hidden confounding, and a suitable confounding bridge exists under regularity conditions. Correlation with $U$ alone is insufficient.

**How do proxies identify an effect?** Classical proximal methods recover an *outcome confounding bridge* $h(W,A,X)$ satisfying the observed-data relation

```math
\mathbb{E}[Y\mid Z,A,X]
=
\mathbb{E}[h(W,A,X)\mid Z,A,X].
```

Under the identifying assumptions, the conditional average treatment effect (CATE) for binary treatment follows from the **proximal g-formula**:

```math
\tau(x)
=\mathbb{E}[Y(1)-Y(0)\mid X=x]
=\mathbb{E}[h(W,1,X)-h(W,0,X)\mid X=x].
```

The final expectation averages over $W$ conditional on $X=x$, yielding an effect conditional on $X$ alone. Proxies therefore help identify the effect without observing $U$. These assumptions require causal domain knowledge; choosing two arbitrary feature blocks as proxies does not establish identification.

Bridge functions can be estimated using parametric, kernel, or neural methods. For example, [Mastouri et al. (2021)](https://proceedings.mlr.press/v139/mastouri21a.html) develop kernel methods based on two-stage estimation and moment restrictions.

**Foundational references**

- Miao, Geng, and Tchetgen Tchetgen (2018). [*Identifying causal effects with proxy variables of an unmeasured confounder.*](https://doi.org/10.1093/biomet/asy038) Biometrika.
- Tchetgen Tchetgen, Ying, Cui, Shi, and Miao (2020). [*An Introduction to Proximal Causal Learning.*](https://arxiv.org/abs/2009.10982)

- Mastouri et al. (2021). [*Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction.*](https://proceedings.mlr.press/v139/mastouri21a.html) ICML.

</details>

<details>
<summary><strong>What is ProximalFM?</strong></summary>

**ProximalFM is a prior-data fitted network (PFN) for proximal causal inference.** PFNs learn to approximate Bayesian inference by training on datasets sampled from a specified prior. At inference time, a new dataset becomes the context for probabilistic predictions in a single forward pass, with no weight updates ([Müller et al., 2022](https://arxiv.org/abs/2112.10510)). ProximalFM applies this approach to treatment-effect estimation with hidden confounding and observed proxies.

### Goal: conditional average treatment effects

Given an observational context $\mathcal{D}=\{(x_i,z_i,w_i,a_i,y_i)\}_{i=1}^{N}$ and query covariates $x$, ProximalFM estimates

```math
\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x].
```

The context contains observed covariates, both proxy sets, binary treatment, and factual outcomes. **Queries require only $x$:** their proxies, treatment, and outcome are withheld. The model predicts the CATE directly, without fitting an explicit bridge function to each new dataset. Its distribution approximates the Bayesian posterior over CATE under the synthetic training prior; the posterior mean provides a point estimate.

### Prior: synthetic proximal causal models

ProximalFM trains on synthetic datasets with hidden confounding and observed proxies. Random neural networks generate diverse causal mechanisms, varying feature types, confounding strength, treatment overlap, and treatment effects.

The causal structure enforces latent exchangeability and proxy exclusion restrictions, while bounded treatment probabilities ensure overlap. Proxy dimension and category-count constraints encourage completeness, but do not guarantee it for arbitrary nonlinear mechanisms.

**Monte Carlo CATE targets.** For each query, we hold $x$ fixed and average treatment effects over **draws** of the downstream hidden variables, proxies, and noise. This approximates the CATE rather than a single individual effect. A noise-aware loss accounts for sampling noise in these targets. The model receives only the observed context and query covariates; hidden variables and potential outcomes are used only to generate training supervision.

### Architecture: a transformer adapted to causal roles

ProximalFM reuses the three-stage transformer backbone of [TabICLv2 (Qu et al., 2026)](https://arxiv.org/abs/2602.11139), adapting its inputs and output to the proximal causal task:

1. **Column embeddings** represent each feature value relative to its column's distribution.
2. **Row interactions** combine the feature embeddings into a representation of each unit.
3. **Dataset-level in-context learning** lets query rows attend to the observed context to predict treatment effects.

Learned **structural role embeddings** distinguish $X$, $Z$, and $W$. A **treatment-aware outcome encoder** uses separate projections for outcomes observed under treatment and control. Query proxy tokens are masked so predictions depend on query covariates alone.

A single output head represents the CATE distribution as a histogram, from which means and quantiles can be extracted. The predicted spread describes uncertainty about an average causal effect, rather than variation in individual treatment effects.

**References**

- Müller et al. (2022). [*Transformers Can Do Bayesian Inference.*](https://arxiv.org/abs/2112.10510) Introduces PFNs.
- Qu et al. (2026). [*TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model.*](https://arxiv.org/abs/2602.11139)

</details>

<details>
<summary><strong>Performance</strong></summary>

We evaluate ProximalFM against classical proximal estimators, causal foundation models, and conventional meta-learners. The main semisynthetic benchmark spans **288 instances across eight settings**, varying mechanism linearity, hidden-confounding strength, and proxy informativeness.

- **CATE accuracy:** ProximalFM is competitive across settings, with its clearest advantage at larger contexts under strong confounding and weak proxies.
- **Inference speed:** it combines strong accuracy ranks with low inference time and requires no dataset-specific training or hyperparameter selection.
- **Larger contexts:** gains plateau beyond the main training range of approximately 2,048 observations. Additional training on larger contexts brings modest improvements.
- **Uncertainty:** credible-interval coverage is close to nominal on data sampled from the training prior, but intervals become overconfident on semisynthetic data as context grows.

</details>

<details>
<summary><strong>Quick start</strong></summary>

Install with [uv](https://docs.astral.sh/uv/) and Python 3.12 or later:

```bash
git clone --recurse-submodules https://github.com/ChristopheMuller/ProximalFM.git
cd ProximalFM
uv sync
```

Save the following as `example.py` in the repository and run `uv run python example.py`. It downloads the model weights and estimates CATE on a small synthetic dataset with hidden confounding.

```python
import torch
from huggingface_hub import hf_hub_download
from proximalfm._model.proximalfm import ProximalFM

torch.set_num_threads(4)
torch.manual_seed(42)

path = hf_hub_download(
    repo_id="christophemuller/ProximalFM",
    filename="proximalfm.ckpt",
    cache_dir="checkpoints/huggingface",
)
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model = ProximalFM(**checkpoint["config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Context: hidden U affects treatment and outcome; Z and W are noisy proxies.
n = 512
x, u = torch.randn(n, 1), torch.randn(n, 1)
z, w = u + 0.3 * torch.randn(n, 1), u + 0.3 * torch.randn(n, 1)
a = torch.bernoulli(torch.sigmoid(0.4 * x[:, 0] + u[:, 0]))
y = 0.5 * x[:, 0] + u[:, 0] + a * (1 + 0.5 * x[:, 0]) + 0.3 * torch.randn(n)
query_x = torch.linspace(-2, 2, 9).reshape(-1, 1)

# Context rows first, query rows second. Query proxies are masked internally.
inputs = {
    "X": torch.cat([x, query_x]).unsqueeze(0),
    "Z": torch.cat([z, torch.zeros_like(query_x)]).unsqueeze(0),
    "W": torch.cat([w, torch.zeros_like(query_x)]).unsqueeze(0),
    "A": a.unsqueeze(0),
    "y_train": y.unsqueeze(0),
}
dims = {f"d_{name}": torch.tensor([1]) for name in ("X", "Z", "W")}
with torch.inference_mode():
    cate = model.predict(**inputs, **dims, target="cate", output_type="mean")

print("True CATE:", 1 + 0.5 * query_x[:, 0])
print("Predicted CATE:", cate[0])
```

For your own data, replace the synthetic context with observed $X$, $Z$, $W$, $A$, and $Y$. Feature tensors have shape `(1, n_context + n_query, n_features)`; treatment and outcome tensors contain context rows only, with shape `(1, n_context)`. Set `d_X`, `d_Z`, and `d_W` to the corresponding feature counts. Predictions are returned in the original outcome units.

For effect quantiles, use `output_type="quantiles", quantiles=[0.05, 0.5, 0.95]` in the same prediction call. The larger-context checkpoint is available as `proximalfm-ft.ckpt`.

</details>

## Citation

If you use ProximalFM in your research, please cite our [paper](https://arxiv.org/abs/2610.08078):

```bibtex
@misc{muller2026proximalfm,
  title         = {{ProximalFM}: Amortized Proximal Causal Inference under Hidden Confounding},
  author        = {Muller, Christophe and Kharel, Ayub and Luedtke, Alex and
                   Park, Chan and Tchetgen Tchetgen, Eric and Gamella, Juan L. and
                   Krishnan, Rahul and Silva, Ricardo and Zeitler, Jakob},
  year          = {2026},
  eprint        = {2610.08078},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url           = {https://arxiv.org/abs/2610.08078}
}
```
