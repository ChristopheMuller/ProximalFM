from __future__ import annotations

import random
import warnings
from collections import namedtuple

import numpy as np
import torch
from torch import nn, Tensor


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


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculate the standard deviation of a tensor, ignoring NaNs.

    Pure-torch implementation that stays on-device. The previous version routed
    through ``input.cpu().numpy()`` -> ``np.nanstd`` -> ``torch.from_numpy``,
    which forces a host copy (and dtype churn) on every call. This is a hot path:
    ``process_features`` runs it 3x per proxy on ``(T * K)``-row tensors, so the
    roundtrip dominated the feature-corruption cost.

    A fully-masked-out column yields 0 here (rather than NumPy's NaN); every
    caller either operates on finite data or already remaps that NaN to 0, so the
    downstream result is unchanged.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple of int, optional
        The dimension or dimensions to reduce. Defaults to None (reduce all
        dimensions).

    keepdim : bool, default=False
        Whether the output tensor has `dim` retained or not.

    ddof : int, default=0
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        Accumulation/return dtype. Defaults to float.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    if dtype is not None:
        input = input.to(dtype)

    reduce_dim = tuple(range(input.dim())) if dim is None else dim

    mask = ~torch.isnan(input)
    count = mask.sum(dim=reduce_dim, keepdim=True)
    total = torch.nansum(input, dim=reduce_dim, keepdim=True)
    mean = total / count.clamp(min=1)

    centered = torch.where(mask, input - mean, torch.zeros((), device=input.device))
    sq_sum = (centered * centered).sum(dim=reduce_dim, keepdim=True)
    var = sq_sum / (count - ddof).clamp(min=1)
    std = var.sqrt()

    if not keepdim:
        std = std.reshape(()) if dim is None else std.squeeze(dim)

    return std.to(torch.float)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardize features by removing the mean and scaling to unit variance.

    Computes :math:`(x - \\mu) / \\sigma` where NaNs are ignored in the
    :math:`\\mu` and :math:`\\sigma` calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(T, H)``, where T is sequence length, H is features.

    clip_value : float, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between ``-clip_value`` and ``clip_value``.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamp outliers in the input tensor based on a specified number of standard deviations.

    Values outside :math:`[\\mu - k\\sigma,\\; \\mu + k\\sigma]` are clamped,
    where :math:`k` is the threshold.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(T, H)``.

    threshold : float, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


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
    """

    def __init__(self, num_classes: int, mode: str = "rank", ordered_prob: float = 0.2):
        """Initialize the MulticlassAssigner."""
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                "The number of classes must be at least 2 for MulticlassAssigner."
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

    * **Unconstrained** (``require_profile is None``): with probability
      ``hp["cat_prob"]`` the tensor becomes eligible; each column is then
      independently discretized (with a shared per-call probability) into a
      random number of categories via :class:`MulticlassAssigner`.

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
        Hyperparameters. Uses ``cat_prob`` (default 0.2) as the probability the
        tensor is eligible, and ``max_categories`` (default 10) as the upper
        bound on categories per column.

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
    if random.random() < hp.get("cat_prob", 0.2):
        col_prob = random.random()  # Probability for converting a specific column
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
    extra = cols[c_u + k :]
    if extra and random.random() < hp.get("cat_prob", 0.2):
        col_prob = random.random()
        for col in extra:
            if random.random() < col_prob:
                num_cats = _sample_num_cats(max_categories)
                _categorize_column(X, col, num_cats)
                meta[col] = num_cats

    return meta


def process_features(
    X: Tensor, hp: dict, require_profile: BlockProfile | None = None
) -> tuple[Tensor, BlockProfile]:
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
        When given, categorisation is constrained so the returned profile is
        guaranteed to :func:`dominates` ``require_profile`` (before any
        downstream constant-column filtering). See :func:`num2cat`.

    Returns
    -------
    tuple[Tensor, BlockProfile]
        The processed feature tensor of shape ``(T, H)`` and its
        :class:`BlockProfile`. The profile reflects the categorisation applied
        here; downstream constant-column filtering may reduce it, so callers
        that enforce completeness must re-check post-filter.
    """
    X, meta = num2cat(X, hp, require_profile=require_profile)
    X = outlier_removing(X, threshold=4)
    X = standard_scaling(X)
    return X, block_profile(X.shape[1], meta)
