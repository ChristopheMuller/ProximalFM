"""Dataset adapters. Importing this package is what populates the registry.

Add a literature dataset by writing one module here, decorating its class with
``@register_dataset``, and importing it below.

Planned (see the literature catalogue in the paper's results section): the
Demand benchmark, Mastouri et al.'s synthetic proximal designs, Qiu et al.'s and
Chen et al.'s DR-proximal generators, then the semi-synthetic tier. Anything
without a ground-truth CEPO target is point-estimate only -- set
``supports_replicates = False`` on it, which makes the builder refuse
``--mc_cate > 1`` rather than write a set that would be scored
distributionally against sampled ITEs.
"""

from lit_eval.datasets.dummy_linear import DummyLinearProximal  # noqa: F401

__all__ = ["DummyLinearProximal"]
