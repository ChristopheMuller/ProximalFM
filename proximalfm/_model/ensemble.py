####
# Ensemble prediction for ProximalFM.
#
# Idea ported from TabICL's sklearn `EnsembleGenerator` (see `_sklearn/`):
# create several equivalent "views" of the same table by permuting feature
# columns, run the model on each view, then average the predictions.
#
# Why it helps here: `RowInteraction` applies rotary position embeddings (RoPE)
# over feature positions, so ProximalFM is only *approximately* invariant to the
# ordering of feature columns. Different orderings therefore yield slightly
# different (but equally valid) CATE estimates. Averaging over a handful of
# permutations reduces this order-sensitivity variance, usually improving the
# prediction.
#
# Unlike the sklearn version we do NOT ensemble over normalization methods:
# ProximalFM standardizes its inputs internally (`_standardize_forward_inputs`),
# so feature-column permutation is the natural transformation to ensemble over.
####

from __future__ import annotations

import sys
import math
import random
import itertools
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch import Tensor

# Bijective, information-preserving per-column value transforms usable at
# inference time. Each is applied identically to context + query rows so the
# transformed table is an *equivalent view* of the same data (see the module
# docstring). Lossy maps (x**2, categorization) are intentionally excluded:
# they make the query unit ambiguous and would bias the ensemble.
VALUE_TRANSFORMS = ("none", "sign", "signed_log", "arcsinh", "quantile")


class RecursionLimitManager:
    """Temporarily raise the recursion limit (needed for Latin-square generation)."""

    def __init__(self, limit: int):
        self.limit = limit
        self.original_limit = None

    def __enter__(self):
        self.original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.limit)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.setrecursionlimit(self.original_limit)
        return False  # propagate exceptions


class FeatureShuffler:
    """Generate feature-column permutations for ensemble diversity.

    Ported from the TabICL sklearn ``Shuffler``. Given a number of feature
    columns, it produces a list of index permutations according to the selected
    method. Each permutation can be used to reorder the columns of a feature
    block (X, Z, or W) to create an alternative view of the same data.

    Parameters
    ----------
    n_elements : int
        Number of feature columns to shuffle.

    method : str, default="latin"
        Permutation strategy:
        - ``"none"``: No shuffling (returns the identity ordering).
        - ``"shift"``: All circular shifts of the columns.
        - ``"random"``: Random permutations.
        - ``"latin"``: Latin-square permutations (systematic coverage).

    max_elements_for_latin : int, default=4000
        Above this many columns, ``"latin"`` falls back to ``"random"`` to avoid
        prohibitively expensive Latin-square generation.

    random_state : int or None, default=None
        Seed for reproducible permutations.
    """

    def __init__(
        self,
        n_elements: int,
        method: str = "latin",
        max_elements_for_latin: int = 4000,
        random_state: Optional[int] = None,
    ):
        self.n_elements = n_elements
        self.method = method
        self.max_elements_for_latin = max_elements_for_latin
        self.random_state = random_state

    def shuffle(self, n_estimators: int) -> List[List[int]]:
        """Return up to ``n_estimators`` column-index permutations."""
        self.rng_ = random.Random(self.random_state)
        indices = list(range(self.n_elements))

        # Latin squares are only feasible for a limited number of elements.
        if self.n_elements > self.max_elements_for_latin and self.method == "latin":
            method = "random"
        else:
            method = self.method

        # Degenerate cases: nothing to permute.
        if method == "none" or n_estimators == 1 or self.n_elements <= 1:
            return [indices]

        if method == "shift":
            patterns = [indices[-i:] + indices[:-i] for i in range(self.n_elements)]
        elif method == "random":
            if self.n_elements <= 5:
                all_perms = [list(p) for p in itertools.permutations(indices)]
                patterns = self.rng_.sample(
                    all_perms, min(n_estimators, len(all_perms))
                )
            else:
                patterns = [
                    self.rng_.sample(indices, self.n_elements)
                    for _ in range(n_estimators)
                ]
        elif method == "latin":
            with RecursionLimitManager(100000):
                patterns = self._latin_squares()
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'none', 'shift', 'random', or 'latin'."
            )

        return patterns

    def _latin_squares(self) -> List[List[int]]:
        """Generate a randomized Latin square of permutations."""

        def _shuffle_transpose_shuffle(matrix):
            square = deepcopy(matrix)
            self.rng_.shuffle(square)
            trans = list(zip(*square))
            self.rng_.shuffle(trans)
            return trans

        def _rls(symbols):
            n = len(symbols)
            if n == 1:
                return [symbols]
            sym = self.rng_.choice(symbols)
            symbols.remove(sym)
            square = _rls(symbols)
            square.append(square[0].copy())
            for i in range(n):
                square[i].insert(i, sym)
            return square

        symbols = list(range(self.n_elements))
        square = _rls(symbols)
        shuffles = _shuffle_transpose_shuffle(square)

        return [list(shuffle) for shuffle in shuffles]


def apply_permutation(
    x: Tensor, perm: Optional[List[int]], d: Optional[Tensor] = None
) -> Tensor:
    """Reorder the feature columns of a block tensor by ``perm``.

    Only the *valid* columns are permuted; padding columns (those at or beyond
    ``d`` for a given batch element) are kept in place so that the model's
    padding masks and per-batch feature counts (``d_X``/``d_Z``/``d_W``) remain
    correct. Because only valid columns are shuffled amongst themselves, the
    valid-feature count is unchanged and the original ``d`` tensor can be passed
    through untouched.

    Parameters
    ----------
    x : Tensor
        Feature block of shape ``(B, T, F)``.

    perm : list of int or None
        A permutation of ``range(F)``. ``None`` (or the identity) leaves ``x``
        unchanged.

    d : Tensor of shape ``(B,)`` or None, default=None
        Per-batch count of valid feature columns. When ``None`` all ``F``
        columns are treated as valid.

    Returns
    -------
    Tensor
        Column-permuted copy of ``x`` (same shape).
    """
    if perm is None:
        return x

    B, T, F = x.shape
    device = x.device
    base = torch.as_tensor(perm, dtype=torch.long, device=device)

    if base.numel() != F:
        raise ValueError(
            f"Permutation length {base.numel()} does not match feature width {F}."
        )

    # Fast path: identity permutation.
    if torch.equal(base, torch.arange(F, device=device)):
        return x

    if d is None:
        gather_idx = base.view(1, F).expand(B, F)
    else:
        d = d.to(device=device)
        rows = []
        for b in range(B):
            db = int(d[b].item())
            db = max(0, min(db, F))
            # Keep valid columns (those < db) in permuted order, append padding as-is.
            valid = base[base < db]
            pad = torch.arange(db, F, device=device)
            rows.append(torch.cat([valid, pad]))
        gather_idx = torch.stack(rows, dim=0)

    gather_idx = gather_idx.view(B, 1, F).expand(B, T, F)
    return torch.gather(x, dim=2, index=gather_idx)


def _context_standardize(x: Tensor, train_size: int) -> Tensor:
    """Per-column standardize using context (first ``train_size``) rows only.

    No-leak: statistics come only from the context rows, then applied to all
    rows. Constant columns (std ~ 0, e.g. padding) are left unscaled.
    """
    ctx = x[:, :train_size, :]
    mu = ctx.mean(dim=1, keepdim=True)
    std = ctx.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std


def _quantile_to_normal(z: Tensor, train_size: int, eps: float = 1e-6) -> Tensor:
    """Rank-based Gaussianization fit on context rows (bijective on support).

    For each column, map every value to its empirical rank among the context
    values, then through the inverse normal CDF. Query values are ranked against
    the same context, so the transform is consistent across context and query.
    """
    ctx = z[:, :train_size, :]  # (B, train_size, F)
    ctx_sorted = ctx.sort(dim=1).values  # sorted along the sequence axis
    # Put the feature axis first and the (sorted) sequence axis last so that
    # searchsorted batches over (B, F) and searches along the context axis.
    boundaries = ctx_sorted.permute(0, 2, 1).contiguous()  # (B, F, train_size)
    vals = z.permute(0, 2, 1).contiguous()  # (B, F, T)

    idx = torch.searchsorted(
        boundaries, vals
    )  # counts of context < val, in [0, train_size]
    ranks = (idx.to(z.dtype) + 0.5) / boundaries.shape[-1]
    ranks = ranks.clamp(eps, 1.0 - eps)
    out = math.sqrt(2.0) * torch.erfinv(2.0 * ranks - 1.0)
    return out.permute(0, 2, 1).contiguous()  # (B, T, F)


def apply_value_transform(x: Tensor, name: Optional[str], train_size: int) -> Tensor:
    """Apply a bijective per-column value transform to a feature block.

    The block is first context-standardized (so the nonlinear maps operate on a
    consistent scale), then warped. ``"none"`` returns ``x`` untouched so the
    canonical view exactly reproduces a plain ``predict``. The model's own
    internal standardization runs afterwards regardless.

    Parameters
    ----------
    x : Tensor
        Feature block of shape ``(B, T, F)``.
    name : str or None
        One of :data:`VALUE_TRANSFORMS`. ``None``/``"none"`` is the identity.
    train_size : int
        Number of context rows (used for no-leak standardization / ranking).
    """
    if name is None or name == "none":
        return x

    z = _context_standardize(x, train_size)

    if name == "sign":
        return -z
    if name == "signed_log":
        return torch.sign(z) * torch.log1p(z.abs())
    if name == "arcsinh":
        return torch.asinh(z)
    if name == "quantile":
        return _quantile_to_normal(z, train_size)

    raise ValueError(
        f"Unknown value transform {name!r}. Expected one of {VALUE_TRANSFORMS}."
    )


class FeatureEnsemble:
    """Generate feature-transformation views for ensemble CATE prediction.

    Produces up to ``n_estimators`` "views" of the same table. Each view combines
    two independent, information-preserving levers:

    1. **Column permutation** of the X/Z/W blocks (RoPE-order diversity).
    2. **A bijective per-column value transform** (see :data:`VALUE_TRANSFORMS`)
       applied to the enabled blocks.

    Both are label-preserving, so every view is an equally-valid encoding of the
    same data and averaging over views reduces variance. The first view is always
    the canonical one (identity permutation + ``"none"`` transform), so an
    ensemble reduces exactly to a plain ``predict`` when there is a single view.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members (views) to generate.

    feat_shuffle_method : str, default="latin"
        Permutation strategy passed to :class:`FeatureShuffler`
        (``"none"``, ``"shift"``, ``"random"``, or ``"latin"``).

    value_transforms : list of str or None, default=None
        Bijective value transforms to cycle across views, drawn from
        :data:`VALUE_TRANSFORMS`. ``None`` (default) means permutation-only
        (equivalent to ``["none"]``). ``"none"`` is always used for the first
        view and deduplicated into the front of the list otherwise.

    shuffle_X, shuffle_Z, shuffle_W : bool, default=(True, False, False)
        Whether to permute the covariate / treatment-proxy / outcome-proxy
        columns across views.

    transform_X, transform_Z, transform_W : bool, default=(True, False, False)
        Which blocks the value transform is applied to. Per-column bijections
        preserve the proximal conditional-independence structure, so warping Z/W
        is valid, but the default warps only the covariates X.

    random_state : int or None, default=42
        Seed for reproducible permutations.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        feat_shuffle_method: str = "latin",
        value_transforms: Optional[List[str]] = None,
        shuffle_X: bool = True,
        shuffle_Z: bool = False,
        shuffle_W: bool = False,
        transform_X: bool = True,
        transform_Z: bool = False,
        transform_W: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = n_estimators
        self.feat_shuffle_method = feat_shuffle_method
        self.value_transforms = self._normalize_transforms(value_transforms)
        self.shuffle_X = shuffle_X
        self.shuffle_Z = shuffle_Z
        self.shuffle_W = shuffle_W
        self.transform_X = transform_X
        self.transform_Z = transform_Z
        self.transform_W = transform_W
        self.random_state = random_state

    @staticmethod
    def _normalize_transforms(transforms: Optional[List[str]]) -> List[str]:
        """Validate transforms and ensure ``"none"`` is first (canonical view)."""
        if not transforms:
            return ["none"]
        for t in transforms:
            if t not in VALUE_TRANSFORMS:
                raise ValueError(
                    f"Unknown value transform {t!r}. Expected one of {VALUE_TRANSFORMS}."
                )
        # "none" first, then the rest in given order without duplicates.
        ordered = ["none"] + [t for t in transforms if t != "none"]
        seen: set = set()
        return [t for t in ordered if not (t in seen or seen.add(t))]

    def _block_perms(
        self, width: int, enabled: bool, seed_offset: int
    ) -> List[Optional[List[int]]]:
        """Return one permutation (or None) per ensemble member for a single block."""
        if not enabled or width <= 1:
            return [None] * self.n_estimators

        rs = None if self.random_state is None else self.random_state + seed_offset
        shuffler = FeatureShuffler(
            n_elements=width, method=self.feat_shuffle_method, random_state=rs
        )
        patterns = shuffler.shuffle(self.n_estimators)

        identity = list(range(width))
        perms: List[Optional[List[int]]] = [identity]  # first member: canonical view
        # Cycle through the generated patterns to fill the remaining members.
        i = 0
        while len(perms) < self.n_estimators:
            perms.append(patterns[i % len(patterns)])
            i += 1
        return perms[: self.n_estimators]

    def generate(
        self, d_X_width: int, d_Z_width: int, d_W_width: int
    ) -> List[Dict[str, object]]:
        """Build the list of ensemble views.

        Parameters
        ----------
        d_X_width, d_Z_width, d_W_width : int
            The number of feature columns (last-dim size) of the X, Z, and W
            tensors respectively.

        Returns
        -------
        list of dict
            One dict per ensemble member with keys ``"X"``, ``"Z"``, ``"W"``
            (column permutation or ``None``) and ``"transform"`` (the value
            transform name for that view). The first view is always the
            canonical one: identity permutations + ``"none"``.
        """
        x_perms = self._block_perms(d_X_width, self.shuffle_X, seed_offset=0)
        z_perms = self._block_perms(d_Z_width, self.shuffle_Z, seed_offset=1)
        w_perms = self._block_perms(d_W_width, self.shuffle_W, seed_offset=2)

        views: List[Dict[str, object]] = []
        for i in range(self.n_estimators):
            # transforms[0] == "none" and perms[0] == identity, so view 0 is canonical.
            transform = self.value_transforms[i % len(self.value_transforms)]
            views.append(
                {
                    "X": x_perms[i],
                    "Z": z_perms[i],
                    "W": w_perms[i],
                    "transform": transform,
                }
            )
        return views
