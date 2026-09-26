"""Estimate the normalized conditional probability matrix of ``(W, Z)``.

The object of interest is, per conditioning cell ``(A=a, X=x)``,

    M[w, z] = P(W=w, Z=z | a, x) / sqrt(P(W=w | a, x) P(Z=z | a, x)),

the discrete analogue of the kernel whose singular values are the canonical
correlations between the two proxies given ``(A, X)``.  Two properties make it
the right normalization to look at:

* ``sigma_1(M) = 1`` exactly, attained at the singular vector pair
  ``(sqrt(p_W), sqrt(p_Z))`` — the trivial direction, carrying no information.
* ``sum_k sigma_k^2 - 1 = chi^2 / n``, the mean-square contingency between the
  proxies.  So the *decay* of ``sigma_2, sigma_3, ...`` is exactly how much
  signal the operator transmits beyond the constant, i.e. how ill-posed
  inverting it is.

Discretization choices, both deliberate:

* Bins are **equal-frequency (quantile) bins of the whole episode**, not of the
  cell.  Shared edges make ``M`` for ``a=0`` and ``a=1`` matrices over the same
  categories, hence comparable; computing them per cell would silently redefine
  the state space between cells.
* Marginals are the cell's own row/column sums of the estimated joint, not the
  episode-level marginals.  That is what makes ``sigma_1 = 1`` hold exactly for
  the *estimate*, so any departure in the spectrum is dependence rather than
  an artifact of mismatched normalizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from analysis_spectral._load import SpectralEpisode


def quantile_bins(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior edges of ``n_bins`` equal-frequency bins of a 1D sample.

    Duplicate edges (a discrete or heavily tied variable) are collapsed, so the
    realized number of categories can be smaller than ``n_bins``; the caller
    sees that through the shape of the resulting matrix rather than through a
    silent set of empty rows.
    """
    if v.ndim != 1:
        raise ValueError("quantile_bins expects a 1D sample")
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(v, qs)
    return np.unique(edges)


def _labels(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, v, side="right").astype(np.int64)


def _proxy_labels(V: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Categorize a proxy column block into a single label per row."""
    if V.shape[1] != 1:
        raise NotImplementedError(
            "Only scalar proxies are supported so far (the small set has "
            f"d_W = d_Z = 1); got d = {V.shape[1]}. A multivariate proxy needs "
            "a joint discretization (product bins or a vector quantizer), which "
            "changes what the spectrum means and should be an explicit choice."
        )
    edges = quantile_bins(V[:, 0], n_bins)
    return _labels(V[:, 0], edges), edges


def _x_cells(X: np.ndarray, x_bins: int) -> tuple[np.ndarray, int]:
    """Row -> covariate-cell index, and the number of cells.

    ``d_X = 0`` (the small set) gives a single cell, i.e. conditioning on ``A``
    alone.  For ``d_X > 0`` the cells are the product of per-dimension quantile
    bins; that is exponential in ``d_X`` and only usable in low dimension, which
    is the regime this analysis is built for.
    """
    n, d = X.shape
    if d == 0 or x_bins <= 1:
        return np.zeros(n, dtype=np.int64), 1
    cell = np.zeros(n, dtype=np.int64)
    n_cells = 1
    for j in range(d):
        edges = quantile_bins(X[:, j], x_bins)
        lab = _labels(X[:, j], edges)
        width = int(lab.max()) + 1
        cell = cell * width + lab
        n_cells *= width
    # Re-index to the cells that actually occur.
    uniq, cell = np.unique(cell, return_inverse=True)
    return cell.astype(np.int64), len(uniq)


@dataclass(frozen=True)
class CellMatrix:
    """The estimated matrix for one conditioning cell.

    ``M`` is ``[L_w, L_z]`` over the categories that are *occupied in this cell*;
    ``w_categories``/``z_categories`` map those back to the episode-level bin
    indices, so an empty category is dropped rather than contributing a row of
    zeros (which would add a spurious zero singular value).
    """

    a: int
    x_cell: int
    n: int
    M: np.ndarray
    joint: np.ndarray
    p_w: np.ndarray
    p_z: np.ndarray
    w_categories: np.ndarray
    z_categories: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.M.shape

    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.M, compute_uv=False)

    def chi2_over_n(self) -> float:
        """``sum_k sigma_k^2 - 1``: the mean-square contingency of the cell.

        Under exact independence its expectation is ``(L_w-1)(L_z-1)/n``, which
        is the noise floor any decay estimate has to be read against.
        """
        return float((self.M**2).sum() - 1.0)

    def independence_floor(self) -> float:
        """``E[chi^2/n]`` if ``W`` and ``Z`` were independent in this cell."""
        lw, lz = self.M.shape
        return (lw - 1) * (lz - 1) / self.n


def normalized_matrix(
    w_labels: np.ndarray,
    z_labels: np.ndarray,
    *,
    n_w: int,
    n_z: int,
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Counts -> ``(M, joint, p_w, p_z, w_kept, z_kept)`` for one cell.

    ``smoothing`` adds a Laplace pseudo-count to every cell *before*
    normalizing.  It defaults to 0 (plain MLE): smoothing shrinks the estimated
    dependence toward independence, i.e. it biases the very decay we are trying
    to measure, so it is a knob to reach for knowingly and not a default.
    """
    counts = np.zeros((n_w, n_z), dtype=np.float64)
    np.add.at(counts, (w_labels, z_labels), 1.0)
    if smoothing:
        counts += smoothing

    w_kept = np.flatnonzero(counts.sum(axis=1) > 0)
    z_kept = np.flatnonzero(counts.sum(axis=0) > 0)
    counts = counts[np.ix_(w_kept, z_kept)]

    joint = counts / counts.sum()
    p_w = joint.sum(axis=1)
    p_z = joint.sum(axis=0)
    M = joint / np.sqrt(np.outer(p_w, p_z))
    return M, joint, p_w, p_z, w_kept, z_kept


def conditional_matrices(
    ep: SpectralEpisode,
    *,
    n_bins: int = 8,
    x_bins: int = 1,
    smoothing: float = 0.0,
    min_cell_rows: Optional[int] = None,
) -> list[CellMatrix]:
    """Estimate one :class:`CellMatrix` per ``(a, x-cell)`` of an episode.

    ``min_cell_rows`` drops cells too small to estimate an ``n_bins**2`` table
    at all; it defaults to ``2 * n_bins**2`` (an average of two counts per
    cell), which is a floor on being able to compute anything, nowhere near a
    floor on the estimate being *good* — see :meth:`CellMatrix.independence_floor`.
    """
    if min_cell_rows is None:
        min_cell_rows = 2 * n_bins**2

    w_lab, w_edges = _proxy_labels(ep.W, n_bins)
    z_lab, z_edges = _proxy_labels(ep.Z, n_bins)
    n_w = len(w_edges) + 1
    n_z = len(z_edges) + 1
    x_cell, _ = _x_cells(ep.X, x_bins)

    out: list[CellMatrix] = []
    for a in np.unique(ep.A):
        for c in np.unique(x_cell):
            sel = (ep.A == a) & (x_cell == c)
            n = int(sel.sum())
            if n < min_cell_rows:
                continue
            M, joint, p_w, p_z, w_kept, z_kept = normalized_matrix(
                w_lab[sel], z_lab[sel], n_w=n_w, n_z=n_z, smoothing=smoothing
            )
            out.append(
                CellMatrix(
                    a=int(a),
                    x_cell=int(c),
                    n=n,
                    M=M,
                    joint=joint,
                    p_w=p_w,
                    p_z=p_z,
                    w_categories=w_kept,
                    z_categories=z_kept,
                )
            )
    return out
