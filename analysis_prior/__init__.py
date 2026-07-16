"""Reproducible audits of the ProximalFM synthetic prior.

This package generates datasets from the prior with all latent variables kept
(including the unmeasured confounder ``U`` and the Monte Carlo CATE targets),
caches them keyed by a hash of the generating spec, and runs the assumption
audits reported in the paper appendix.

Layout:

- :mod:`analysis_prior.config` -- :class:`PriorSpec`, the reproducible
  description of a prior configuration.
- :mod:`analysis_prior.generate` -- generation and on-disk caching.
- ``analysis_prior.analyses.*`` -- one module per analysis.
- :mod:`analysis_prior.main` -- CLI entry point.
"""

from .config import PriorSpec

__all__ = ["PriorSpec"]
