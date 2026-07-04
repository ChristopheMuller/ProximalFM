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
import random
import itertools
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from torch import Tensor


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


class FeatureEnsemble:
    """Generate feature-permutation views for ensemble CATE prediction.

    Produces up to ``n_estimators`` "views", each a set of independent column
    permutations for the X, Z, and W feature blocks. The first view is always
    the identity (the original, unshuffled data) so the ensemble includes the
    canonical prediction.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members (views) to generate.

    feat_shuffle_method : str, default="latin"
        Permutation strategy passed to :class:`FeatureShuffler`
        (``"none"``, ``"shift"``, ``"random"``, or ``"latin"``).

    shuffle_X : bool, default=True
        Whether to permute the covariate (X) columns across views.

    shuffle_Z : bool, default=False
        Whether to permute the treatment-proxy (Z) columns across views.

    shuffle_W : bool, default=False
        Whether to permute the outcome-proxy (W) columns across views.

    random_state : int or None, default=42
        Seed for reproducible permutations.
    """

    def __init__(
        self,
        n_estimators: int = 8,
        feat_shuffle_method: str = "latin",
        shuffle_X: bool = True,
        shuffle_Z: bool = False,
        shuffle_W: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = n_estimators
        self.feat_shuffle_method = feat_shuffle_method
        self.shuffle_X = shuffle_X
        self.shuffle_Z = shuffle_Z
        self.shuffle_W = shuffle_W
        self.random_state = random_state

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
    ) -> List[Dict[str, Optional[List[int]]]]:
        """Build the list of ensemble views.

        Parameters
        ----------
        d_X_width, d_Z_width, d_W_width : int
            The number of feature columns (last-dim size) of the X, Z, and W
            tensors respectively.

        Returns
        -------
        list of dict
            One dict per ensemble member with keys ``"X"``, ``"Z"``, ``"W"``,
            each mapping to a column permutation (list of int) or ``None``.
        """
        x_perms = self._block_perms(d_X_width, self.shuffle_X, seed_offset=0)
        z_perms = self._block_perms(d_Z_width, self.shuffle_Z, seed_offset=1)
        w_perms = self._block_perms(d_W_width, self.shuffle_W, seed_offset=2)

        return [
            {"X": x_perms[i], "Z": z_perms[i], "W": w_perms[i]}
            for i in range(self.n_estimators)
        ]
