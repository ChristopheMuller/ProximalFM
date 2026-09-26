"""How fast the estimated spectrum decays -- and whether that is measurable.

The severity of an ill-posed inverse problem is conventionally read off the
decay of the singular values of the operator being inverted: ``sigma_j ~ j^-p``
is *mildly* ill-posed (polynomial rates), ``sigma_j ~ exp(-c j)`` is *severely*
ill-posed (only logarithmic rates).  Which of the two holds is the question, so
this module fits both and reports which describes the surviving spectrum
better, rather than assuming one.

Everything here is indexed by ``j = k - 1``: ``sigma_1 = 1`` is the trivial
direction (the constant function, present by construction and carrying no
information), so the informative spectrum is ``sigma_2, sigma_3, ...`` and the
natural abscissa for a decay law starts that sequence at ``j = 1``.

The load-bearing part is not the fit but the **null**.  An ``L x L`` table from
``n`` rows has singular values well above zero even when ``W`` and ``Z`` are
independent, and the size of that floor grows with ``L`` (measured: median
``sigma_2`` of 0.07 at ``L=4`` rising to 0.21 at ``L=16``, on independent data).
A decay fitted through the floor is a fit to sampling noise and reads as slow
decay no matter what the truth is.  So the null is estimated per cell by
permutation -- shuffling ``Z``'s labels within the cell, which preserves both
marginals exactly and is therefore the *conditional* test matching how ``M`` is
normalized -- and only the ``sigma_j`` above their own null quantile enter the
fit.  When fewer than ``min_points`` survive, the rate is ``nan``: the honest
answer at that ``n`` and ``L``, not a number to be averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from analysis_spectral._load import SpectralEpisode
from analysis_spectral.matrices import (
    CellMatrix,
    _proxy_labels,
    _x_cells,
    normalized_matrix,
)

Model = Literal["exp", "poly"]


def null_spectra(
    w_labels: np.ndarray,
    z_labels: np.ndarray,
    *,
    n_w: int,
    n_z: int,
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Permutation null of the whole spectrum: ``[n_perm, min(L_w, L_z)]``.

    Shuffling ``z_labels`` breaks the dependence while holding both marginals
    and ``n`` fixed, so the resulting singular values are exactly the floor that
    this cell's table size and sample size produce on their own.
    """
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(n_perm):
        M, *_ = normalized_matrix(w_labels, rng.permutation(z_labels), n_w=n_w, n_z=n_z)
        out.append(np.linalg.svd(M, compute_uv=False))
    width = min(len(s) for s in out)
    return np.stack([s[:width] for s in out])


@dataclass(frozen=True)
class DecayFit:
    """One decay law fitted to the surviving part of a spectrum."""

    model: Model
    rate: float
    intercept: float
    r2: float
    n_points: int

    def predict(self, j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=np.float64)
        if self.model == "exp":
            return np.exp(self.intercept - self.rate * j)
        return np.exp(self.intercept) * j**-self.rate


def fit_decay(sigma: np.ndarray, j: np.ndarray, model: Model) -> DecayFit:
    """OLS of ``log sigma`` on ``j`` (exp) or ``log j`` (poly).

    ``rate`` is signed so that larger always means faster decay: the slope of
    ``log sigma`` against the model's abscissa, negated.  ``r2`` is computed in
    the same log space for both models, which is what makes them comparable --
    they are two link functions on one response, not two different responses.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    if len(sigma) < 2:
        return DecayFit(model, np.nan, np.nan, np.nan, len(sigma))
    x = j if model == "exp" else np.log(j)
    y = np.log(sigma)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return DecayFit(model, -float(slope), float(intercept), r2, len(sigma))


@dataclass(frozen=True)
class CellSpectrum:
    """The spectrum of one conditioning cell, with its own noise floor.

    ``sigma`` is the raw estimate including the trivial ``sigma_1 = 1``;
    ``sigma_corrected`` has the null's mean square removed from each squared
    singular value (clipped at 0), which is the first-order deconvolution of the
    sampling noise from the estimate.  ``above_floor`` marks the ``j`` whose raw
    value exceeds the null's ``q``-quantile -- the ones a fit may use.
    """

    a: int
    x_cell: int
    n: int
    shape: tuple[int, int]
    sigma: np.ndarray
    null_mean: np.ndarray
    null_q: np.ndarray
    sigma_corrected: np.ndarray
    above_floor: np.ndarray

    @property
    def sigma2(self) -> float:
        """The leading canonical correlation -- max corr(f(W), g(Z)) in-cell.

        A rank-1 spectrum is 0, not undefined: it means one of the proxies is
        constant within the cell (the prior does emit categorical proxies, and a
        binary one can collapse to a single level under conditioning), so the
        operator transmits nothing beyond the trivial direction.  That is the
        maximally ill-posed case, and it has to enter the table as such rather
        than dropping out as a missing value.
        """
        return float(self.sigma[1]) if len(self.sigma) > 1 else 0.0

    def effective_rank(self) -> int:
        """Leading directions above the noise floor -- a *contiguous prefix*.

        Not ``above_floor.sum()``.  The test is applied per direction at level
        ``q``, so a fraction ``1 - q`` of the null directions clear it by
        chance; with ``L = 32`` that is about 1.5 spurious crossings per cell,
        scattered anywhere in the tail.  Counting them inflates the rank, and
        it inflates it *more at larger ``L``*, which would make the summary
        drift with the resolution for a purely combinatorial reason.

        The spectrum is sorted, so the signal-carrying part is a prefix by
        construction: once a direction is indistinguishable from noise, so is
        everything after it.  Taking the leading run instead is both the honest
        reading and the one that stops the false positives accumulating.
        """
        stop = np.flatnonzero(~self.above_floor)
        return int(stop[0]) if len(stop) else int(len(self.above_floor))

    def fit(self, model: Model, *, min_points: int = 3) -> DecayFit:
        # The leading run only -- see effective_rank. A fit over a scattered
        # set of survivors runs partly through the noise floor, which is flat,
        # and can even come back negative (a spectrum that "grows" with j).
        idx = np.arange(self.effective_rank())
        if len(idx) < min_points:
            return DecayFit(model, np.nan, np.nan, np.nan, len(idx))
        j = idx + 1.0
        sig = self.sigma_corrected[1:][idx]
        keep = sig > 0
        if keep.sum() < min_points:
            return DecayFit(model, np.nan, np.nan, np.nan, int(keep.sum()))
        return fit_decay(sig[keep], j[keep], model)


def cell_spectrum(
    cm: CellMatrix,
    w_labels: np.ndarray,
    z_labels: np.ndarray,
    *,
    n_w: int,
    n_z: int,
    n_perm: int = 200,
    quantile: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> CellSpectrum:
    """Attach a permutation null to an estimated :class:`CellMatrix`."""
    sigma = cm.singular_values()
    null = null_spectra(w_labels, z_labels, n_w=n_w, n_z=n_z, n_perm=n_perm, rng=rng)
    width = min(len(sigma), null.shape[1])
    sigma = sigma[:width]
    null_mean = null[:, :width].mean(axis=0)
    null_q = np.quantile(null[:, :width], quantile, axis=0)
    corrected = np.sqrt(np.clip(sigma**2 - null_mean**2, 0.0, None))
    above = sigma[1:] > null_q[1:]
    return CellSpectrum(
        a=cm.a,
        x_cell=cm.x_cell,
        n=cm.n,
        shape=cm.shape,
        sigma=sigma,
        null_mean=null_mean,
        null_q=null_q,
        sigma_corrected=corrected,
        above_floor=above,
    )


def episode_spectra(
    ep: SpectralEpisode,
    *,
    n_bins: int = 8,
    x_bins: int = 1,
    n_perm: int = 200,
    quantile: float = 0.95,
    min_cell_rows: Optional[int] = None,
    seed: int = 0,
) -> list[CellSpectrum]:
    """Every cell of one episode, each with its own null.

    Re-derives the labels rather than calling ``conditional_matrices``, because
    the permutation null has to shuffle the *same* labels the matrix was built
    from -- re-binning per permutation would resample the discretization too.
    """
    if min_cell_rows is None:
        min_cell_rows = 2 * n_bins**2
    rng = np.random.default_rng(seed)

    w_lab, w_edges = _proxy_labels(ep.W, n_bins)
    z_lab, z_edges = _proxy_labels(ep.Z, n_bins)
    n_w, n_z = len(w_edges) + 1, len(z_edges) + 1
    x_cell, _ = _x_cells(ep.X, x_bins)

    out: list[CellSpectrum] = []
    for a in np.unique(ep.A):
        for c in np.unique(x_cell):
            sel = (ep.A == a) & (x_cell == c)
            n = int(sel.sum())
            if n < min_cell_rows:
                continue
            M, joint, p_w, p_z, w_kept, z_kept = normalized_matrix(
                w_lab[sel], z_lab[sel], n_w=n_w, n_z=n_z
            )
            cm = CellMatrix(
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
            out.append(
                cell_spectrum(
                    cm,
                    w_lab[sel],
                    z_lab[sel],
                    n_w=n_w,
                    n_z=n_z,
                    n_perm=n_perm,
                    quantile=quantile,
                    rng=rng,
                )
            )
    return out


def episode_summary(cells: list[CellSpectrum], *, min_points: int = 3) -> dict:
    """Collapse an episode's cells into one row.

    There is deliberately no participation-ratio ("entropy rank") column.  One
    used to be emitted, as the smooth companion to :meth:`effective_rank`, and
    it measured the wrong thing: computed over the whole corrected spectrum
    rather than the leading run, it scored how *evenly spread* the residue was,
    so flat noise came out high and a genuinely dominant ``sigma_2`` came out
    low.  Measured on the 20k set it ran at ``rho = -0.19`` against
    ``eff_rank`` and ``-0.31`` against ``sigma2`` -- anti-correlated with both
    quantities it was supposed to track -- and the 132 episodes with no signal
    at all scored a median of 4.20 where 0 was the only correct answer.

    Cells are combined by their **sample share**, i.e. by ``P(A=a, X in cell)``:
    the difficulty an estimator actually faces is the difficulty of the cells it
    has rows in.  ``*_min`` is kept beside it because a single degenerate cell
    can dominate the inverse problem regardless of how rare it is, and the two
    disagreeing is itself informative.
    """
    if not cells:
        return {}
    n = np.array([c.n for c in cells], dtype=np.float64)
    w = n / n.sum()
    s2 = np.array([c.sigma2 for c in cells])
    er = np.array([c.effective_rank() for c in cells], dtype=np.float64)
    out = {
        "n_cells": len(cells),
        "n_rows": float(n.sum()),
        "sigma2": float((w * s2).sum()),
        "sigma2_min": float(s2.min()),
        "eff_rank": float((w * er).sum()),
        "eff_rank_min": float(er.min()),
        "measurable": bool((er >= min_points).all()),
    }
    for model in ("exp", "poly"):
        fits = [c.fit(model, min_points=min_points) for c in cells]
        rate = np.array([f.rate for f in fits])
        r2 = np.array([f.r2 for f in fits])
        ok = np.isfinite(rate)
        out[f"{model}_rate"] = (
            float((w[ok] * rate[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        )
        out[f"{model}_r2"] = (
            float((w[ok] * r2[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        )
    return out
