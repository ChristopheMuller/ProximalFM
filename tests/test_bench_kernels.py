"""The shared kernels behind PMMR and KPV.

`ColumnWiseGaussianKernel` evaluates its per-column product as one exponential
over pre-scaled coordinates. That is an algebraic identity, not an
approximation, so the tests below pin it against the per-column form written out
longhand — the definition in the class docstring — rather than against recorded
values. Both estimators read the same numbers out of this class, so a change
here moves every proximal-kernel result in the benchmark at once.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from benchmarks.methods.kernels import BinaryKernel, ColumnWiseGaussianKernel, as_2d


def product_form(k: ColumnWiseGaussianKernel, x, y) -> np.ndarray:
    """K(x, y) as the literal product over columns, from the fitted sigmas."""
    K = np.ones((x.shape[0], y.shape[0]))
    for d, sigma in enumerate(k._sigmas):
        K *= np.exp(-cdist(x[:, [d]], y[:, [d]], "sqeuclidean") / sigma)
    return K


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ─────────────────────────────────────────────────────────────────────────────
# ColumnWiseGaussianKernel — the identity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d", [1, 2, 5, 10])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.7778])
def test_single_exponential_equals_the_per_column_product(rng, d, scale):
    """exp(-sum_d s_d/sigma_d) == prod_d exp(-s_d/sigma_d), to rounding.

    Rounding here really is rounding: the two differ only in the order the same
    exponent is accumulated, so 1e-13 relative is generous by two decades.
    """
    X = rng.normal(size=(120, d))
    Y = rng.normal(size=(40, d)) * 2.0  # off-support rows probe the tails

    k = ColumnWiseGaussianKernel().fit(X, scale=scale)
    got, want = k(X, Y), product_form(k, X, Y)

    assert np.allclose(got, want, rtol=1e-13, atol=0)


def test_bandwidths_are_the_median_heuristic_per_column(rng):
    """sigma_d = 2 * median(pairwise |x_d - x'_d|)^2 * scale, column by column.

    Pinned because `scale` is a tuned parameter in both estimators: if this
    formula moves, every `kernel_scale` value in every config silently changes
    meaning.
    """
    X = rng.normal(size=(80, 3)) * np.array([1.0, 10.0, 0.1])
    k = ColumnWiseGaussianKernel().fit(X, scale=2.0)

    for d in range(3):
        col = X[:, [d]]
        expected = 2.0 * np.median(cdist(col, col, "euclidean")) ** 2 * 2.0
        assert k._sigmas[d] == pytest.approx(expected)

    # Per-column, not shared: a column scaled by 10 gets a ~100x wider sigma.
    assert k._sigmas[1] / k._sigmas[0] == pytest.approx(100.0, rel=0.5)


def test_scale_widens_every_column_proportionally(rng):
    X = rng.normal(size=(60, 4))
    base = ColumnWiseGaussianKernel().fit(X, scale=1.0)
    wide = ColumnWiseGaussianKernel().fit(X, scale=3.0)
    assert np.allclose(wide._sigmas, 3.0 * np.asarray(base._sigmas))


# ─────────────────────────────────────────────────────────────────────────────
# ColumnWiseGaussianKernel — properties any kernel must have
# ─────────────────────────────────────────────────────────────────────────────


def test_gram_is_symmetric_psd_with_unit_diagonal(rng):
    X = rng.normal(size=(50, 3))
    K = ColumnWiseGaussianKernel().fit(X)(X, X)

    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)  # the `w**2/12`-free case: k(x, x) = 1
    assert np.all(K > 0) and np.all(K <= 1.0 + 1e-12)
    assert np.linalg.eigvalsh((K + K.T) / 2).min() > -1e-8


def test_cross_gram_shape_and_consistency(rng):
    X = rng.normal(size=(30, 2))
    Y = rng.normal(size=(7, 2))
    k = ColumnWiseGaussianKernel().fit(X)

    assert k(X, Y).shape == (30, 7)
    assert np.allclose(k(X, Y), k(Y, X).T)


# ─────────────────────────────────────────────────────────────────────────────
# ColumnWiseGaussianKernel — degenerate inputs
# ─────────────────────────────────────────────────────────────────────────────


def test_constant_column_is_floored_not_divided_by_zero():
    """A zero-variance column gives median 0; the floor keeps sigma positive."""
    X = np.ones((20, 1))
    k = ColumnWiseGaussianKernel().fit(X)

    assert k._sigmas[0] == 1e-3
    assert np.allclose(k(X, X), 1.0)  # every point identical -> kernel is 1


def test_zero_column_input_is_the_empty_product():
    """d = 0 means an empty product, i.e. the all-ones kernel.

    The per-column loop returned ones by construction but the unfitted-guard
    tripped first, so this used to raise. It is reachable through
    `PMMREstimator.score` with covariate-free data, where the caller currently
    branches around it.
    """
    Z = np.zeros((5, 0))
    K = ColumnWiseGaussianKernel().fit(Z)(Z, Z)

    assert K.shape == (5, 5)
    assert np.allclose(K, 1.0)


def test_calling_before_fit_raises():
    with pytest.raises(RuntimeError, match="Call fit"):
        ColumnWiseGaussianKernel()(np.zeros((2, 1)), np.zeros((2, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# BinaryKernel — the indicator PMMR's per-arm decomposition depends on
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_kernel_is_the_indicator():
    A = np.array([[0.0], [1.0], [1.0], [0.0]])
    K = BinaryKernel().fit(A)(A, A)

    same = A.ravel()[:, None] == A.ravel()[None, :]
    assert np.array_equal(K, same.astype(float))
    # Exactly zero off-arm — `pmmr.PMMREstimator.prepare` decomposes on this.
    assert np.all(K[np.ix_([0, 3], [1, 2])] == 0.0)


@pytest.mark.parametrize("bad", [np.array([[0.0], [2.0]]), np.array([[-1.0], [1.0]])])
def test_binary_kernel_refuses_non_01_encodings(bad):
    with pytest.raises(ValueError, match=r"values in"):
        BinaryKernel().fit(bad)


def test_binary_kernel_refuses_multicolumn_treatment():
    with pytest.raises(ValueError, match="scalar binary treatment"):
        BinaryKernel().fit(np.zeros((4, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# as_2d
# ─────────────────────────────────────────────────────────────────────────────


def test_as_2d_promotes_a_vector_to_a_column():
    assert as_2d(np.arange(3)).shape == (3, 1)
    assert as_2d(np.zeros((3, 2))).shape == (3, 2)
    assert as_2d([1, 2, 3]).dtype == float
