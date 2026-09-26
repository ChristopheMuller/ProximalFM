"""Ill-posedness analysis: spectral decay of the W-Z dependence operator.

Step 1 (here) is the estimation of the normalized conditional probability
matrix

    M_a[w, z] = P(W=w, Z=z | A=a, X=x) / sqrt(P(W=w | A=a, X=x) P(Z=z | A=a, X=x))

on a *discretized* version of the proxies.  Its singular values are the
canonical correlations between ``W`` and ``Z`` in the conditioning cell
(``sigma_1 = 1`` always, with singular vectors ``sqrt(p)``); how fast
``sigma_2, sigma_3, ...`` decay is the finite-dimensional stand-in for the
decay of the singular values of the conditional expectation operator
``E[. | W] -> E[. | Z]`` that governs how ill-posed the proximal inverse
problem is.
"""

from analysis_spectral._load import SpectralEpisode, load_episodes
from analysis_spectral.matrices import (
    CellMatrix,
    conditional_matrices,
    normalized_matrix,
    quantile_bins,
)

__all__ = [
    "SpectralEpisode",
    "load_episodes",
    "CellMatrix",
    "conditional_matrices",
    "normalized_matrix",
    "quantile_bins",
]
