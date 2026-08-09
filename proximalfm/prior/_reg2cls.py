from __future__ import annotations

import random
import warnings
from collections import namedtuple

import numpy as np
import torch
from torch import nn, Tensor
from proximalfm.prior._utils import torch_nanstd, standard_scaling, outlier_removing


# ---------------------------------------------------------------------------
# Categorical completeness heuristic (Track F)
# ---------------------------------------------------------------------------
#
# A processed feature block (U, W, or Z) has a *profile* summarising its
# discrete structure:
#
#   * ``continuous`` -- the number of columns left continuous.
#   * ``categorical`` -- the multiset of category-counts of its categorical
#     columns, stored sorted descending for a canonical form.
#
# A continuous column carries strictly more information than any finite
# categorical column, so a proxy ``P`` is *complete* w.r.t. ``U`` iff every
# column of ``U`` can be matched one-to-one to a column of ``P`` that is at
# least as informative (see :func:`dominates`). This is the discrete analogue
# of the dimension heuristic ``min(d_W, d_Z) >= d_U`` and collapses to it when
# nothing is categorised.
BlockProfile = namedtuple("BlockProfile", ["continuous", "categorical"])


def block_profile(n_total_cols: int, meta: dict[int, int]) -> BlockProfile:
    """Summarise a processed block as a :class:`BlockProfile`.

    Parameters
    ----------
    n_total_cols : int
        Total number of columns in the block.

    meta : dict[int, int]
        Mapping ``{column index -> number of categories}`` for the columns that
        were categorised (as returned by :func:`num2cat`). Columns absent from
        ``meta`` are treated as continuous.

    Returns
    -------
    BlockProfile
        ``continuous`` = ``n_total_cols - len(meta)``; ``categorical`` = the
        category counts sorted descending.
    """
    cats = tuple(sorted(meta.values(), reverse=True))
    return BlockProfile(continuous=n_total_cols - len(meta), categorical=cats)


def dominates(profile_p: BlockProfile, profile_u: BlockProfile) -> bool:
    """Whether block ``P`` is categorically complete w.r.t. block ``U``.

    Implements the two-condition heuristic:

    1. **Continuous slots.** ``P`` needs at least as many continuous columns as
       ``U`` (a categorical column can never cover a continuous requirement):
       ``profile_p.continuous >= profile_u.continuous``.
    2. **Categorical slots.** The ``profile_p.continuous - profile_u.continuous``
       leftover continuous columns of ``P`` act as wildcards, each dominating
       any categorical requirement. Sorting the augmented capacity
       ``{wildcards} ∪ cats(P)`` and ``cats(U)`` descending, every one of the
       top ``|cats(U)|`` entries must dominate element-wise.

    Returns ``True`` iff both hold. When ``U`` has no categorical columns this
    reduces to ``profile_p.continuous >= profile_u.continuous``.
    """
    if profile_p.continuous < profile_u.continuous:
        return False

    wildcards = profile_p.continuous - profile_u.continuous
    capacity = sorted(
        [float("inf")] * wildcards + list(profile_p.categorical), reverse=True
    )
    required = sorted(profile_u.categorical, reverse=True)

    if len(capacity) < len(required):
        return False
    return all(capacity[i] >= required[i] for i in range(len(required)))


def permute_classes(input: Tensor) -> Tensor:
    """Label-encode and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape ``(T,)`` containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels of shape ``(T,)``.
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, input)]

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device)
    permuted = perm[mapped]

    return permuted


class MulticlassAssigner(nn.Module):
    """Transform the input into discrete classes using rank-based or value-based thresholding.

    Input shape: ``(T,)`` -> Output shape: ``(T,)``

    Parameters
    ----------
    num_classes : int
        The target number of discrete classes to output.

    mode : str, default="rank"
        The method used to determine class boundaries:
        - "rank": Boundaries are randomly sampled from the input.
        - "value": Boundaries are randomly sampled from a normal distribution.

    ordered_prob : float, default=0.2
        Probability of keeping the natural class order.

    Raises
    ------
    ValueError
        If ``num_classes < 2``, or ``mode`` is not one of :attr:`MODES`.
    """

    #: The boundary-sampling strategies :meth:`forward` implements.
    MODES = ("rank", "value")

    def __init__(self, num_classes: int, mode: str = "rank", ordered_prob: float = 0.2):
        """Initialize the MulticlassAssigner."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "The number of classes must be at least 2 for MulticlassAssigner."
            )
        if mode not in self.MODES:
            raise ValueError(
                f"Unknown MulticlassAssigner mode {mode!r} "
                f"(expected one of {self.MODES})."
            )

        self.num_classes = num_classes
        self.ordered_prob = ordered_prob
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        """Assign class labels to continuous input values.

        Parameters
        ----------
        input : Tensor
            Input of shape ``(T,)``.

        Returns
        -------
        Tensor
            Class labels of shape ``(T,)`` with integer values
            ``[0, num_classes-1]``.
        """

        T = input.shape[0]
        device = input.device

        if self.mode == "rank":
            boundary_indices = torch.randint(
                0, T, (self.num_classes - 1,), device=device
            )
            boundaries = input[boundary_indices]
        elif self.mode == "value":
            boundaries = torch.randn(self.num_classes - 1, device=device)
        else:
            # Unreachable via __init__, which validates `mode`; reachable by
            # assigning to `.mode` afterwards. Without this branch `boundaries`
            # is simply never bound and the next line raises UnboundLocalError,
            # which says nothing about what went wrong.
            raise ValueError(
                f"Unknown MulticlassAssigner mode {self.mode!r} "
                f"(expected one of {self.MODES})."
            )

        # Compare input tensor with boundaries and sum across the boundary dimension to get classes
        classes = (input.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        # Permute classes
        if random.random() > self.ordered_prob:
            classes = permute_classes(classes)

        # Reverse classes
        if random.random() > 0.5:
            classes = self.num_classes - 1 - classes

        return classes


def _categorize_column(X: Tensor, col: int, num_cats: int) -> None:
    """Discretize column ``col`` of ``X`` in place into ``num_cats`` classes."""
    assigner = MulticlassAssigner(num_cats, mode="rank", ordered_prob=0.3)
    X[:, col] = assigner(X[:, col]).float()


def _sample_num_cats(max_categories: int, min_cats: int = 2) -> int:
    """Draw a category count in ``[min_cats, max_categories]`` (gamma-based)."""
    return min(max(round(random.gammavariate(1, 10)), min_cats), max_categories)


def num2cat(
    X: Tensor, hp: dict, require_profile: BlockProfile | None = None
) -> tuple[Tensor, dict[int, int]]:
    """Convert some feature columns to categorical based on hyperparameters.

    Two modes:

    * **Unconstrained** (``require_profile is None``): each column is
      independently discretized with probability ``hp["cat_col_prob"]`` (a
      per-dataset value shared across columns and blocks) into a random number
      of categories via :class:`MulticlassAssigner`. When it is low the block
      stays fully numeric, so no separate eligibility gate is needed.

    * **Construct-by-design** (``require_profile`` given): categorise ``X`` so
      its resulting profile is guaranteed to :func:`dominates` ``require_profile``
      *before* any downstream constant-column filtering. One categorical column
      is placed per category-count required by ``require_profile``, each drawn to
      dominate that requirement; the remaining columns are then optionally
      categorised as in the unconstrained mode, capped so at least
      ``require_profile.continuous`` continuous columns survive.

    The input tensor is modified in place. Feasibility is guaranteed upstream by
    the dimension heuristic (a proxy always has ``>= d_U`` columns), so an
    infeasible request (fewer columns than the requirement needs) is a caller
    bug and raises ``ValueError`` rather than being silently truncated.

    Parameters
    ----------
    X : Tensor
        Feature tensor of shape ``(T, H)``.

    hp : dict
        Hyperparameters. Uses ``cat_col_prob`` (default 0.25) as the per-column
        categorisation probability, and ``max_categories`` (default 10) as the
        upper bound on categories per column.

    require_profile : BlockProfile, optional
        When given, the profile (typically ``U``'s) that the categorisation of
        ``X`` must dominate.

    Returns
    -------
    tuple[Tensor, dict[int, int]]
        The (in-place mutated) feature tensor and metadata mapping
        ``{column index -> number of categories}`` for every categorised column.
    """
    max_categories = hp.get("max_categories", 10)

    if require_profile is not None:
        return X, _num2cat_enforced(X, hp, require_profile, max_categories)

    meta: dict[int, int] = {}
    col_prob = hp.get("cat_col_prob", 0.25)  # per-column categorisation probability
    for col in range(X.shape[1]):
        if random.random() < col_prob:
            num_cats = _sample_num_cats(max_categories)
            _categorize_column(X, col, num_cats)
            meta[col] = num_cats
    return X, meta


def _num2cat_enforced(
    X: Tensor, hp: dict, require_profile: BlockProfile, max_categories: int
) -> dict[int, int]:
    """Categorise ``X`` so its profile dominates ``require_profile`` by design.

    The ``n`` columns are partitioned (after a random shuffle) into three roles:

    * ``cols[:c_u]`` -- reserved continuous, guaranteeing ``c_P >= c_u`` (the
      heuristic's continuous-slot condition) no matter what follows.
    * ``cols[c_u : c_u + k]`` -- one slot per categorical requirement of ``U``.
      Each is covered *either* by a categorical column with dominating
      cardinality *or* (with probability ``1 - proxy_cat_cover_prob``) left as a
      continuous wildcard, which strictly dominates any categorical requirement.
      Covering with continuous keeps proxies from being biased toward
      categorical and yields more informative proxies.
    * ``cols[c_u + k :]`` -- extra columns, optionally categorised as in the
      unconstrained mode. These only *add* categorical capacity (the reserved
      and wildcard-covered continuous columns are never touched here), so
      domination is preserved.
    """
    n = X.shape[1]
    c_u = require_profile.continuous
    required = sorted(require_profile.categorical, reverse=True)
    k = len(required)

    if n < c_u + k:
        raise ValueError(
            f"Cannot enforce categorical completeness: block has {n} columns "
            f"but the requirement needs at least {c_u + k} "
            f"({c_u} continuous + {k} categorical). This should be "
            "guaranteed by the dimension heuristic upstream."
        )

    cols = list(range(n))
    random.shuffle(cols)

    meta: dict[int, int] = {}

    # cols[:c_u] stay continuous (reserved).

    # Requirement slots: cover each with a categorical column or a continuous
    # wildcard. `U`'s counts are themselves capped at `max_categories`, so the
    # categorical floor is always feasible.
    cat_cover_prob = hp.get("proxy_cat_cover_prob", 0.5)
    for col, min_cats in zip(cols[c_u : c_u + k], required):
        if random.random() < cat_cover_prob:
            num_cats = _sample_num_cats(max_categories, min_cats=min_cats)
            _categorize_column(X, col, num_cats)
            meta[col] = num_cats
        # else: leave continuous -> wildcard covering this requirement.

    # Extra optional categorisation on the leftover columns, as in the
    # unconstrained mode. Only adds capacity; the reserved/wildcard continuous
    # columns above are untouched, so the profile still dominates.
    col_prob = hp.get("cat_col_prob", 0.25)
    for col in cols[c_u + k :]:
        if random.random() < col_prob:
            num_cats = _sample_num_cats(max_categories)
            _categorize_column(X, col, num_cats)
            meta[col] = num_cats

    return meta


def process_features(
    X: Tensor, hp: dict, require_profile: BlockProfile | None = None
) -> tuple[Tensor, dict[int, int]]:
    """Apply the prior's feature corruptions: categorization, outlier clamping, scaling.

    This is the feature-only pipeline shared by the proximal prior. Unlike the
    removed ``Reg2Cls`` pipeline, it does **not** pad, permute, or scale by the
    max-feature count, and it never touches the target ``y`` — padding and
    active-dimension tracking are handled separately by the dataset via
    ``filter_constant_features_and_pad``.

    Parameters
    ----------
    X : Tensor
        Feature tensor of shape ``(T, H)``.

    hp : dict
        Hyperparameters (see :func:`num2cat`).

    require_profile : BlockProfile, optional
        When given, categorisation is constrained so the block's profile is
        guaranteed to :func:`dominates` ``require_profile`` (before any
        downstream constant-column filtering). See :func:`num2cat`.

    Returns
    -------
    tuple[Tensor, dict[int, int]]
        The processed feature tensor of shape ``(T, H)`` and the categorisation
        metadata ``{column index -> number of categories}``. The metadata
        reflects the categorisation applied here; downstream constant-column
        filtering may drop columns, so callers that enforce completeness must
        re-derive the profile post-filter (see :func:`observed_block_profile`).
        Use :func:`block_profile` to summarise this (pre-filter) metadata.
    """
    X, meta = num2cat(X, hp, require_profile=require_profile)
    X = outlier_removing(X, threshold=4)
    X = standard_scaling(X)
    return X, meta


def observed_block_profile(block: Tensor, meta: dict[int, int]) -> BlockProfile:
    """Profile of a block *after* constant-column filtering, from realised values.

    Mirrors :meth:`SCMPrior.filter_constant_features_and_pad`: a column survives
    iff it is non-constant (``amax != amin``). Surviving columns listed in
    ``meta`` are categorical; their cardinality is measured from the realised
    distinct values (not the intended count in ``meta``) so the profile reflects
    what actually survives observation subsampling and filtering. This is the
    profile to feed :func:`dominates` when enforcing completeness post-filter.

    Known blind spot (deliberately not fixed; changing it would change which
    datasets the prior accepts). A column absent from ``meta`` is counted as
    continuous unconditionally, i.e. an unbounded wildcard in :func:`dominates`,
    without checking how many distinct values it actually realises. A mechanism
    output that was never discretised can still land on a small lattice -- when
    the mechanism reads it close to an affine function of a categorised input --
    and is then treated as more informative than it is. Measured on the shipped
    prior this affects ~0.3% of never-discretised columns (~3% of datasets carry
    one) and lets roughly 1 dataset in 1024 pass enforcement while failing an
    audit that types every column from its realised values. The error is
    one-sided: applied to ``U`` the same convention demands a *continuous* proxy
    column in return, which is stricter, so this can only admit a dataset via the
    proxy blocks, never reject a valid one. See appendix B.2.4 ("Survival") and
    ``analysis_prior/analyses/categorical.py``.

    Parameters
    ----------
    block : Tensor
        Realised (pre-padding) feature block of shape ``(T, H)`` -- e.g. the
        observed ``U``/``W``/``Z`` before ``filter_constant_features_and_pad``.

    meta : dict[int, int]
        Categorisation metadata from :func:`process_features` for this block
        (column indices align, since row subsampling preserves columns).

    Returns
    -------
    BlockProfile
        Continuous count and category-counts (sorted descending) of the columns
        that survive filtering.
    """
    keep = block.amax(dim=0) != block.amin(dim=0)  # non-constant survive filtering
    cats: list[int] = []
    continuous = 0
    for col in range(block.shape[1]):
        if not bool(keep[col]):
            continue
        if col in meta:
            cats.append(int(block[:, col].unique().numel()))
        else:
            continuous += 1
    return BlockProfile(
        continuous=continuous, categorical=tuple(sorted(cats, reverse=True))
    )
