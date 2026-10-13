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

$$
\mathbb{E}[Y\mid Z,A,X]
=
\mathbb{E}[h(W,A,X)\mid Z,A,X].
$$

Under the identifying assumptions, the conditional average treatment effect (CATE) for binary treatment follows from the **proximal g-formula**:

$$
\tau(x)
=\mathbb{E}[Y(1)-Y(0)\mid X=x]
=\mathbb{E}[h(W,1,X)-h(W,0,X)\mid X=x].
$$

The final expectation averages over $W$ conditional on $X=x$, yielding an effect conditional on $X$ alone. Proxies therefore help identify the effect without observing $U$. These assumptions require causal domain knowledge; choosing two arbitrary feature blocks as proxies does not establish identification.

Bridge functions can be estimated using parametric, kernel, or neural methods. For example, [Mastouri et al. (2021)](https://proceedings.mlr.press/v139/mastouri21a.html) develop kernel methods based on two-stage estimation and moment restrictions.

**Foundational references**

- Miao, Geng, and Tchetgen Tchetgen (2018). [*Identifying causal effects with proxy variables of an unmeasured confounder.*](https://doi.org/10.1093/biomet/asy038) Biometrika.
- Tchetgen Tchetgen, Ying, Cui, Shi, and Miao (2020). [*An Introduction to Proximal Causal Learning.*](https://arxiv.org/abs/2009.10982)

- Mastouri et al. (2021). [*Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction.*](https://proceedings.mlr.press/v139/mastouri21a.html) ICML.

</details>
