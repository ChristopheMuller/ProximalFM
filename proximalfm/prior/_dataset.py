"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

####
# Changes vs TabICL:
# - Proximal DAG Structure: Implemented specific causal relationships (U -> X, W, Z, A, Y)
#   to replace generic i.i.d. feature sampling.
# - Feature Partitioning: Replaced 'max_features' with specific ranges for Latent U,
#   Covariates X, Outcome Proxies W, and Treatment Proxies Z.
# - Potential Outcomes: Added counterfactual generation for Y(0) and Y(1) via joint
#   forward passes with intervened treatment values.
# - Completeness Heuristic: Enforced dim(W) >= dim(U) and dim(Z) >= dim(U) during
#   hyperparameter sampling to support proximal identification.
# - Positivity Control: Introduced 'sample_A' with temperature scaling and epsilon
#   clamping to ensure treatment overlap (P(A|X,U) > 0).
# - Robust Padding: Replaced dynamic filtering with 'filter_constant_features_and_pad'
#   to maintain fixed block sizes for X, W, and Z across datasets.
# - Regression Alignment: Dropped the target-side reg-to-cls transform to keep Y
#   continuous for CATE estimation. Only the feature-side corruptions of Reg2Cls
#   (categorization / outlier clamping / scaling via process_features) are reused,
#   applied to X, W, Z (and optionally U) at their generation step so the observed
#   proxies stay consistent with the mechanisms that produced Y and A.
# - Adaptive Scaling: Updated 'adjust_max_features' to scale the entire proximal
#   feature set (X+W+Z) proportionally for large sequence lengths to prevent OOM.
# - Target-Leaf Mechanisms: f_Z and f_A run on the T observed rows only, not on
#   the N Monte-Carlo rows. Z and A are leaves w.r.t. MU0/MU1 (Z never feeds
#   f_W/f_Y, A never feeds f_Y), so their replicates were computed only to be
#   discarded at 'obs_idx'. Distribution-preserving; see generate_dataset.
# - Query-Only MC Targets: 'mc_targets_on_context=False' replicates only the
#   query rows [train_size:] when estimating MU0/MU1, leaving the context prefix
#   NaN. The trainer reads MU only on that slice, so this is a pure compute win
#   there; analysis/visualisation code uses all T rows, hence the opt-in default.
# - Exogenous-Noise Model: All meaningful randomness enters each mechanism as
#   explicit eps input columns whose count is proportional to the mechanism's
#   signal input width (sampled 'noise_share_of_input_in_*' converted via
#   confounding_input_dim). Within-layer GaussianNoise ('noise_std') is kept at
#   a tiny fixed dither (1e-3, see DEFAULT_FIXED_HP) purely so continuous
#   columns realise distinct values under saturating/discretising activations;
#   it is a tie-breaker for the distinct-count profile classification, not part
#   of the noise model.
####


#########
# TODO:
# - Generate not only U -> X but also X -> U or X <-> U relationships
# - Generate not only U -> Z but also Z -> U
#########

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from ._mlp_scm import MLPSCM
from ._tree_scm import TreeSCM
from ._utils import XSampler

from ._diagnostics import mechanism_diagnostics

from ._hp_sampling import HpSamplerList
from ._reg2cls import (
    process_features,
    block_profile,
    observed_block_profile,
    dominates,
)
from ._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)

# Emit the Monte Carlo memory-budget warning at most once per process to avoid
# flooding logs when many datasets get capped.
_MC_BUDGET_WARNED = False


def _scalar_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON-friendly scalars of a params dict (sampled + fixed HPs, dims).

    Drops tensors, classes and other rich objects; converts numpy scalars so
    the result survives serialisation unchanged.
    """
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (bool, str)):
            out[key] = value
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = float(value)
    return out


def _mc_mean_var(replicates: Tensor, K: int) -> Tensor:
    """Sampling variance of the Monte Carlo mean, per row.

    ``replicates`` is ``(rows, K)`` holding the per-replicate CATE draws
    ``y1_mc - y0_mc``; the supervised target is their mean over ``K``, so its
    sampling variance is ``Var_k(.) / K`` (unbiased, ``K - 1`` dof).

    Note this is the *paired* variance of the difference, not
    ``[Var(y1_mc) + Var(y0_mc)] / K``. Both arms come out of a single ``f_y``
    forward pass over the same replicate rows (shared ``U_mc`` and ``alphas``),
    so they are strongly positively correlated and summing their variances
    would materially over-state the target noise.

    ``K < 2`` admits no variance estimate; returns zeros rather than NaN so
    consumers can tell "no replicates" apart from the NaN used to mark context
    rows that were never Monte Carlo replicated at all.
    """
    if K < 2:
        return torch.zeros(
            replicates.shape[0], device=replicates.device, dtype=replicates.dtype
        )
    return replicates.var(dim=1, unbiased=True) / K


class Prior:
    """Abstract base class for dataset prior generators (Proximal Causal Learning).

    Defines the interface and common functionality for generating synthetic
    observational datasets with unmeasured confounding and proxies.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    min_dim_U : int, default=3
        Minimum dimensionality of the unmeasured latent confounder U.

    max_dim_U : int, default=10
        Maximum dimensionality of the unmeasured latent confounder U.

    min_dim_W : int, default=3
        Minimum dimensionality of the outcome-inducing proxy W.

    max_dim_W : int, default=10
        Maximum dimensionality of the outcome-inducing proxy W.

    min_dim_Z : int, default=3
        Minimum dimensionality of the treatment-inducing proxy Z.

    max_dim_Z : int, default=10
        Maximum dimensionality of the treatment-inducing proxy Z.

    min_dim_X : int, default=5
        Minimum dimensionality of the observed baseline covariates X.

    max_dim_X : int, default=20
        Maximum dimensionality of the observed baseline covariates X.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_dim_U: int = 3,
        max_dim_U: int = 10,
        min_dim_W: int = 3,
        max_dim_W: int = 10,
        min_dim_Z: int = 3,
        max_dim_Z: int = 10,
        min_dim_X: int = 5,
        max_dim_X: int = 20,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_dim_U <= max_dim_U, "Invalid U dimension range"
        assert min_dim_W <= max_dim_W, "Invalid W dimension range"
        assert min_dim_Z <= max_dim_Z, "Invalid Z dimension range"
        assert min_dim_X <= max_dim_X, "Invalid X dimension range"

        self.min_dim_U = min_dim_U
        self.max_dim_U = max_dim_U
        self.min_dim_W = min_dim_W
        self.max_dim_W = max_dim_W
        self.min_dim_Z = min_dim_Z
        self.max_dim_Z = max_dim_Z
        self.min_dim_X = min_dim_X
        self.max_dim_X = max_dim_X

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(
        min_train_size: Union[int, float], max_train_size: Union[int, float]
    ) -> None:
        """Check if the training size range is valid.

        Parameters
        ----------
        min_train_size : int or float
            Minimum training size (position or ratio).

        max_train_size : int or float
            Maximum training size (position or ratio).

        Raises
        ------
        AssertionError
            If training size range is invalid.

        ValueError
            If training size types are mismatched or invalid.
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(
            max_train_size, (int, float)
        ):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert (
                0 < min_train_size < max_train_size
            ), "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert (
                0 < min_train_size < max_train_size < 1
            ), "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError(
                "Both training sizes must be of the same type (int or float)"
            )

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int],
        max_seq_len: int,
        log: bool = False,
        replay_small: bool = False,
    ) -> int:
        """Select a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length.

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes.

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets.

        Returns
        -------
        int
            The sampled sequence length.
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(
        min_train_size: Union[int, float],
        max_train_size: Union[int, float],
        seq_len: int,
    ) -> int:
        """Select a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int or float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int or float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length.

        Returns
        -------
        int
            The sampled training size position.

        Raises
        ------
        ValueError
            If training size range has incompatible types.
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(
        seq_len: int, max_x: int, max_w: int, max_z: int
    ) -> Tuple[int, int, int]:
        """Adjust the maximum dimensions proportionally based on the sequence length."""
        total_features = max_x + max_w + max_z + 1  # +1 for Treatment A

        if seq_len <= 10240:
            limit = 100
        elif 10240 < seq_len <= 20000:
            limit = 80
        elif 20000 < seq_len <= 30000:
            limit = 60
        elif 30000 < seq_len <= 40000:
            limit = 40
        elif 40000 < seq_len <= 50000:
            limit = 30
        elif 50000 < seq_len <= 60000:
            limit = 20
        elif 60000 < seq_len <= 65000:
            limit = 15
        else:
            limit = 10

        if total_features <= limit:
            return max_x, max_w, max_z

        scale = limit / total_features
        adj_x = max(1, int(max_x * scale))
        adj_w = max(1, int(max_w * scale))
        adj_z = max(1, int(max_z * scale))
        return adj_x, adj_w, adj_z

    @staticmethod
    def filter_constant_features_and_pad(
        X: Tensor, d: Tensor, max_d: int
    ) -> Tuple[Tensor, Tensor]:
        """Remove features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape ``(B, T, H)`` where B is batch size,
            T is sequence length, and H is feature dimensionality.

        d : Tensor
            Number of features per dataset of shape ``(B,)``, indicating how many
            features are actually used in each dataset (rest is padding).

        Returns
        -------
        X_new : Tensor
            The filtered tensor with non-informative features removed.

        d_new : Tensor
            The updated feature count per dataset.
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            if xi.ndim == 1:
                xi = xi.unsqueeze(0)

            # Treat di as an integer, but bound it safely by the actual tensor size
            actual_di = min(int(di), xi.shape[-1])

            # Slice down to the true active feature width to prevent indexing out of bounds
            xi_active = xi[:, :actual_di]
            num_features = xi.shape[-1]
            # Identify non-constant features. A column has >1 unique value iff its
            # max differs from its min -- a single vectorized reduction instead of a
            # Python loop calling torch.unique (which sorts) once per column.
            if actual_di > 0:
                unique_mask = xi_active.amax(dim=0) != xi_active.amin(dim=0)
            else:
                unique_mask = torch.zeros(0, dtype=torch.bool, device=xi.device)
            di_new = int(unique_mask.sum())

            # Extract the non-constant features and pad back up to the original size
            xi_unique = xi_active[:, unique_mask]
            xi_new = F.pad(
                xi_unique,
                pad=(0, num_features - di_new),
                mode="constant",
                value=0,
            )
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        # Padding to max_d if needed
        if X_new.shape[-1] < max_d:
            X_new = F.pad(X_new, (0, max_d - X_new.shape[-1]), mode="constant", value=0)
        d_new = torch.tensor([min(int(di), max_d) for di in d_new], device=X_new.device)

        return X_new, d_new


class SCMPrior(Prior):
    """Generate synthetic datasets using Structural Causal Models (SCM).

    The data generation process follows a hierarchical structure:

    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary
       transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_dim_U, max_dim_U : int
        Minimum and maximum dimensionality for the latent confounder U.

    min_dim_W, max_dim_W : int
        Minimum and maximum dimensionality for outcome proxies W.

    min_dim_Z, max_dim_Z : int
        Minimum and maximum dimensionality for treatment proxies Z.

    min_dim_X, max_dim_X : int
        Minimum and maximum dimensionality for observed covariates X.

    min_seq_len, max_seq_len : int or None
        Minimum and maximum samples per dataset. If ``min_seq_len`` is None
        the generator will use ``max_seq_len`` directly.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'.
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on
        probabilities.

    mcmc_cate : int, default=200
        Number of Monte Carlo samples used to estimate the CATE targets
        (MU0/MU1) per row. Higher values reduce the noise of the estimate at
        the cost of more memory/compute.

    mc_targets_on_context : bool, default=True
        If True (the default), every row gets ``mcmc_cate`` replicates and
        MU0/MU1 are estimated for the whole sequence. If False, only the query
        rows ``[train_size:]`` are replicated and MU0/MU1 are ``NaN`` on the
        context prefix. Training reads MU only on the query slice, so False is
        the cheaper equivalent there; see :meth:`generate_dataset` for the
        implications.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters.

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_dim_U: int = 3,
        max_dim_U: int = 5,
        min_dim_W: int = 3,
        max_dim_W: int = 5,
        min_dim_Z: int = 3,
        max_dim_Z: int = 5,
        min_dim_X: int = 5,
        max_dim_X: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        mcmc_cate: int = 200,
        mc_targets_on_context: bool = True,
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp

        self.min_dim_U = min_dim_U
        self.max_dim_U = max_dim_U
        self.min_dim_W = min_dim_W
        self.max_dim_W = max_dim_W
        self.min_dim_Z = min_dim_Z
        self.max_dim_Z = max_dim_Z
        self.min_dim_X = min_dim_X
        self.max_dim_X = max_dim_X

        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.mcmc_cate = mcmc_cate
        self.mc_targets_on_context = mc_targets_on_context
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    def hp_sampling(self) -> Dict[str, Any]:
        """Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones.
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @staticmethod
    def _per_row_estimate(params: Dict[str, Any]) -> int:
        """Estimated live tensor elements per *replicated* (``N``-row) row.

        The single widest block is the MLP-SCM activation pool of width
        ``num_layers * hidden_dim``; alongside it live the raw + normalized
        proxy and confounder blocks and the exogenous-noise inputs.

        Counts only what the replicate block actually carries. ``f_Z`` and
        ``f_A`` are leaves w.r.t. MU0/MU1 and run on the ``T`` observed rows
        (see :meth:`generate_dataset`), so ``Z``, ``A`` and their covariate /
        noise inputs contribute nothing here: the ``N``-row mechanisms are
        ``f_U``, ``f_W`` and ``f_y`` alone.
        """
        num_layers = max(2, int(params.get("num_layers", 4)))
        hidden_dim = int(params.get("hidden_dim", 20))

        # The eps blocks widen each mechanism's input by up to
        # share / (1 - share) times its signal width (share clamped at 0.75 by
        # confounding_input_dim, i.e. worst case 4x). Use the largest sampled
        # share as a coarse over-estimate; exact widths depend on binomial
        # subsampling draws not known at cap time. Z/A are excluded: their eps
        # blocks are T rows.
        max_noise_share = max(
            min(float(params.get(f"noise_share_of_input_in_{k}", 0.3)), 0.75)
            for k in ("U", "W", "Y")
        )
        noise_mult = 1.0 + max_noise_share / (1.0 - max_noise_share)

        return (
            num_layers * hidden_dim
            + params["dim_X"]  # X_rep_
            # raw + normalized U and W (Z is T rows)
            + 2 * (params["dim_U"] + params["dim_W"])
            # eps_U, eps_W, eps_Y, sized off the widest N-row signal input
            + math.ceil(noise_mult * 3 * (params["dim_X"] + params["dim_W"] + 1))
            # the X_U_ / X_W_ / X_Y_ covariate subsamples
            + 3 * params["dim_X"]
            # y_joint_raw (2), alphas, cate_raw, y0_mc, y1_mc
            + 6
        )

    def _cap_mc_samples(
        self, K: int, T_ctx: int, T_q: int, params: Dict[str, Any]
    ) -> int:
        """Cap the Monte Carlo sample count ``K`` to respect the memory budget.

        The generator materializes several ``N``-row tensors at once, where
        ``N = T_ctx + T_q * K`` is the replicated row count (``T_ctx`` context
        rows carry no replicates, see :meth:`generate_dataset`). Given the
        per-row width from :meth:`_per_row_estimate`, we choose the largest
        ``K`` for which ``N * per_row`` stays under ``prior_mc_mem_budget``
        (counted in tensor elements / floats).

        Returns a value in ``[1, K]``. A budget of ``None`` or ``<= 0`` disables
        capping and returns ``K`` unchanged.

        ``prior_mc_mem_budget`` is a calibration dial, not a hard ceiling: it
        counts only the replicate-block tensors :meth:`_per_row_estimate`
        models, so measured peak RSS runs ~1.5x above it (allocator overhead,
        the transient copies inside ``process_features``, the T-row Z/A branch,
        and the interpreter baseline are all outside the model). Tune it
        against observed RSS rather than reading it as bytes.

        With ``T_ctx = 0, T_q = T`` this reduces to the historical
        ``budget // (T * per_row)``: floor division satisfies
        ``(a // b) // c == a // (b * c)`` for positive integers.
        """
        budget = params.get("prior_mc_mem_budget", None)
        if budget is None or budget <= 0:
            return K

        per_row = self._per_row_estimate(params)
        max_rows = int(budget // max(1, per_row))
        K_eff = max(1, min(K, (max_rows - T_ctx) // max(1, T_q)))

        global _MC_BUDGET_WARNED
        if K_eff < K and not _MC_BUDGET_WARNED:
            _MC_BUDGET_WARNED = True
            warnings.warn(
                f"prior_mc_mem_budget={budget:,} elements caps mcmc_cate from {K} "
                f"to {K_eff} for T_ctx={T_ctx}, T_q={T_q} (estimated {per_row} "
                "cols/row). MU0/MU1 will use fewer Monte Carlo samples (still "
                "unbiased, just noisier). Raise prior_mc_mem_budget if you have "
                "more RAM, or lower max_seq_len / hidden_dim / num_layers to keep "
                "K high.",
                RuntimeWarning,
                stacklevel=2,
            )

        return K_eff

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate a single valid dataset based on the provided parameters.

        Monte Carlo replicates
        ----------------------
        Each covariate row is replicated ``K = mcmc_cate`` times so MU0/MU1 can
        be estimated as the mean of the potential outcomes over the replicates.
        ``params["mc_targets_on_context"]`` controls which rows get replicated:

        * ``True`` (default) -- all ``T`` rows; the mechanisms see ``T * K`` rows.
        * ``False`` -- only the ``T_q = T - train_size`` query rows; the
          mechanisms see ``T_ctx + T_q * K`` rows and MU0/MU1 are ``NaN`` on the
          context prefix. At the default ``train_size ~ U(0.1, 0.9) * T`` this
          roughly halves generation compute and peak memory.

        With ``T_ctx > 0`` the pooled row set is dominated by the query
        replicates, so the statistics computed over it -- ``safe_normalize`` on
        U/W, the rank-mode bin boundaries in ``process_features`` and
        ``ate_raw`` -- are estimated from a query-weighted rather than a
        balanced sample. All remain consistent estimators of the same
        population quantities over ``N`` i.i.d. rows, so the DGP is unchanged
        *in distribution*; the generated data is however not bit-identical to a
        ``mc_targets_on_context=True`` run under the same seed. Equality is
        exact only at ``T_ctx == 0``.

        Which mechanisms see the replicates
        -----------------------------------
        Only ``f_U``, ``f_W`` and ``f_y`` run on the ``N`` replicated rows,
        because ``MU0``/``MU1`` are means over the replicate worlds of ``Y``.
        ``f_Z`` and ``f_A`` are *leaves* with respect to those targets -- the
        proximal DAG keeps ``Z`` out of ``f_W``/``f_Y`` and ``A`` out of
        ``f_Y`` -- so their non-observed replicates could only ever be
        discarded. They are therefore evaluated on the ``T`` observed rows
        alone, reading the confounder in its realised world (``U_obs_``). This
        is the same joint law by construction (``X_rep_[obs_idx] == X_``) and
        removes ~40% of generation compute; the branch's own statistics
        (``safe_normalize`` on Z, its bin boundaries, ``sample_A``'s logit
        standardisation) are consequently estimated from ``T`` rows.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            ``(U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1, d_U, d_X, d_W,
            d_Z, S2, K_eff, diagnostics)`` for a single dataset, all tensors
            un-batched. Row-shaped entries are ``(seq_len,)`` or
            ``(seq_len, max_dim_*)``; ``d_*`` and ``K_eff`` are 0-d ``long``.

            ``S2`` is the per-row sampling variance of the CATE target
            ``MU1 - MU0`` (see :func:`_mc_mean_var`), NaN on the context prefix
            wherever ``MU0``/``MU1`` are. ``K_eff`` is the realised replicate
            count after :meth:`_cap_mc_samples`, which may be below the
            requested ``mcmc_cate``. ``diagnostics`` is ``None`` unless the
            ``collect_mechanism_diagnostics`` fixed HP is set.
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        def safe_normalize(tensor: Tensor) -> Tensor:
            """Normalize a tensor while avoiding division by zero."""
            # One fused reduction for both stats instead of two separate scans;
            # runs on the N-row U/W/Z blocks.
            std, mean = torch.std_mean(tensor, dim=0, keepdim=True)
            return (tensor - mean) / (std + 1e-8)

        T = params["seq_len"]
        K = int(params.get("mcmc_cate", 200))
        if K < 1:
            raise ValueError("mcmc_cate must be >= 1")

        # Only the query rows need Monte Carlo replicates: MU0/MU1 are read
        # exclusively on the ``[train_size:]`` slice downstream, and the model
        # must not see the CEPOs of its own context. Context rows therefore run
        # a single replicate, and the mechanisms process
        # ``N = T_ctx + T_q * K`` rows instead of ``T * K``.
        if params.get("mc_targets_on_context", True):
            T_ctx = 0
        else:
            # train_size is only metadata elsewhere, so it is not guaranteed to
            # be < seq_len (int-valued ranges are allowed). Clamp so at least
            # one query row survives.
            T_ctx = max(0, min(int(params.get("train_size", 0)), T - 1))
        T_q = T - T_ctx

        # Memory safeguard
        K = self._cap_mc_samples(K, T_ctx, T_q, params)
        N = T_ctx + T_q * K

        # Counts resamples triggered by the completeness check below.
        attempt = 0

        while True:
            attempt += 1

            # Row index of the realised (observed) replicate of every row. Drawn
            # up front because the Z -> A branch is evaluated on *these rows
            # only*: both are leaves w.r.t. the MC targets (Z never enters
            # f_W/f_Y, A never enters f_Y), so their non-observed replicates
            # were only ever computed to be discarded here.
            if T_ctx == 0:
                obs_k = torch.randint(K, size=(T,), device=self.device)
                obs_idx = torch.arange(T, device=self.device) * K + obs_k
            else:
                obs_k = torch.randint(K, size=(T_q,), device=self.device)
                obs_idx = torch.cat(
                    [
                        torch.arange(T_ctx, device=self.device),
                        T_ctx + torch.arange(T_q, device=self.device) * K + obs_k,
                    ]
                )

            eps_X_sampler = XSampler(
                seq_len=T,
                num_features=params["dim_X"],
                pre_stats=params.get("pre_sample_cause_stats", False),
                sampling=params.get("sampling", "mixed"),
                device=self.device,
            )
            eps_X = eps_X_sampler.sample()

            f_X = prior_cls(
                in_dim=eps_X.shape[-1],
                num_outputs=params["dim_X"],
                **params,
            ).to(self.device)

            X = f_X(eps_X)
            X, _ = process_features(X, params)
            X_ = safe_normalize(X)

            categorize_U = params.get("corrupt_U", False)
            enforce_cat = params.get("enforce_categorical_completeness", False)
            if enforce_cat and not categorize_U:
                raise ValueError(
                    "enforce_categorical_completeness=True requires corrupt_U=True: "
                    "categorical completeness is defined relative to a categorised U."
                )

            # Row layout of every N-row tensor below:
            #   [ context rows 0..T_ctx-1 ] [ query row 0 x K ] [ query row 1 x K ] ...
            # i.e. the replicate block is a contiguous suffix, so the MU
            # reduction is a single view(T_q, K).mean(1) and no gather is needed.
            # Built by broadcasting straight into one preallocated buffer: an
            # expand().reshape() plus a cat would copy the query block twice.
            X_rep_ = X_.new_empty((N, params["dim_X"]))
            X_rep_[:T_ctx] = X_[:T_ctx]
            X_rep_[T_ctx:].view(T_q, K, params["dim_X"])[:] = X_[T_ctx:, None, :]

            if params.get("covariate_edge_subsampling", True):

                def _cov_subsample_cols(key: str) -> Tensor:
                    p = float(params.get(f"covariate_share_of_X_in_{key}", 0.5))
                    m = int(
                        torch.distributions.binomial.Binomial(
                            total_count=params["dim_X"], probs=p
                        ).sample()
                    )
                    return torch.randperm(params["dim_X"])[:m]

                # X_rep_[obs_idx] == X_ by construction of the replicate layout,
                # so the Z/A mechanisms read their covariates straight off the
                # T-row X_ rather than the N-row replicate buffer.
                X_U_ = X_rep_[:, _cov_subsample_cols("U")]
                X_W_ = X_rep_[:, _cov_subsample_cols("W")]
                X_Z_ = X_[:, _cov_subsample_cols("Z")]
                X_A_ = X_[:, _cov_subsample_cols("A")]
                X_Y_ = X_rep_[:, _cov_subsample_cols("Y")]
            else:
                X_U_ = X_W_ = X_Y_ = X_rep_
                X_Z_ = X_A_ = X_

            def sample_eps(
                key: str, n_signal: int, rows: int, floor: int = 1
            ) -> Tensor:
                share = float(params.get(f"noise_share_of_input_in_{key}", 0.3))
                k = self.confounding_input_dim(share, n_signal, floor=floor)
                return XSampler(
                    seq_len=rows,
                    num_features=k,
                    pre_stats=params.get("pre_sample_cause_stats", False),
                    sampling=params.get("sampling", "mixed"),
                    device=self.device,
                ).sample()

            eps_U = sample_eps("U", X_U_.shape[-1], N, floor=params["dim_U"])

            input_U = torch.cat([X_U_, eps_U], dim=-1)

            f_U = prior_cls(
                in_dim=input_U.shape[-1],
                num_outputs=params["dim_U"],
                **params,
            ).to(self.device)

            U_mc = f_U(input_U)
            u_meta: dict[int, int] = {}
            if categorize_U:
                U_mc, u_meta = process_features(U_mc, params)
            U_mc_ = safe_normalize(U_mc)
            # The confounder as realised in the *observed* world of each row --
            # the only latent state the Z/A branch is ever read in.
            U_obs_ = U_mc_[obs_idx]

            # Proxies must dominate U's pre-filter profile when enforcing.
            require_profile = (
                block_profile(params["dim_U"], u_meta) if enforce_cat else None
            )

            eps_W = sample_eps("W", params["dim_U"] + X_W_.shape[-1], N)
            input_W = torch.cat([U_mc_, X_W_, eps_W], dim=-1)

            f_W = prior_cls(
                in_dim=input_W.shape[-1],
                num_outputs=params["dim_W"],
                **params,
            ).to(self.device)

            W_mc = f_W(input_W)
            W_mc, w_meta = process_features(
                W_mc, params, require_profile=require_profile
            )
            W_mc_ = safe_normalize(W_mc)

            # Z is a leaf w.r.t. MU0/MU1, so it is generated on the T observed
            # rows only -- hence ``Z_obs`` rather than the ``_mc`` (N-row)
            # naming used for U and W above.
            eps_Z = sample_eps("Z", params["dim_U"] + X_Z_.shape[-1], T)
            input_Z = torch.cat([U_obs_, X_Z_, eps_Z], dim=-1)

            f_Z = prior_cls(
                in_dim=input_Z.shape[-1],
                num_outputs=params["dim_Z"],
                **params,
            ).to(self.device)

            Z_obs = f_Z(input_Z)
            Z_obs, z_meta = process_features(
                Z_obs, params, require_profile=require_profile
            )
            Z_obs_ = safe_normalize(Z_obs)

            if params.get("proxy_edge_subsampling", True):
                m_W = int(
                    torch.distributions.binomial.Binomial(
                        total_count=params["dim_W"],
                        probs=float(params.get("direct_edge_share_of_W_in_Y", 1.0)),
                    ).sample()
                )
                m_Z = int(
                    torch.distributions.binomial.Binomial(
                        total_count=params["dim_Z"],
                        probs=float(params.get("direct_edge_share_of_Z_in_A", 1.0)),
                    ).sample()
                )
                W_direct = W_mc_[:, torch.randperm(params["dim_W"])[:m_W]]
                Z_direct = Z_obs_[:, torch.randperm(params["dim_Z"])[:m_Z]]
            else:
                W_direct = W_mc_
                Z_direct = Z_obs_

            if params.get("confounding_dim_control", True):
                # Shares are defined relative to the non-noise (signal) input
                # width: confounding widths first, then the eps widths below
                # are computed from the resulting full signal width. This
                # avoids a circular dependency between the two share knobs.
                k_Y = self.confounding_input_dim(
                    params.get("confounding_share_of_U_in_Y", 0.3),
                    X_Y_.shape[-1] + W_direct.shape[-1],
                )
                k_A = self.confounding_input_dim(
                    params.get("confounding_share_of_U_in_A", 0.3),
                    X_A_.shape[-1] + Z_direct.shape[-1],
                )
                perm = tuple(torch.randperm(params["dim_U"]).tolist())
                idx_y, idx_a, aug_dim_y, aug_dim_a = self.split_confounding_inputs(
                    k_Y, k_A, params["dim_U"], perm
                )
                U_sel_y = U_mc_[:, idx_y]
                U_sel_a = U_obs_[:, idx_a]

                aug_out_dim = max(aug_dim_y, aug_dim_a)
                if aug_out_dim > 0:
                    f_u_augment = prior_cls(
                        in_dim=U_mc_.shape[-1],
                        num_outputs=aug_out_dim,
                        num_layers=3,
                        hidden_dim=max(10, aug_out_dim),
                    ).to(self.device)
                    # Both sides must read the *same* network. Only the Y side
                    # needs all N rows, so when it takes no augmentation columns
                    # the net runs on the T observed rows alone.
                    if aug_dim_y > 0:
                        u_aug = f_u_augment(U_mc_)
                        u_aug_y = u_aug[:, :aug_dim_y]
                        u_aug_a = u_aug[obs_idx][:, :aug_dim_a]
                    else:
                        u_aug_y = U_mc_[:, :0]
                        u_aug_a = f_u_augment(U_obs_)[:, :aug_dim_a]
                else:
                    u_aug_y = U_mc_[:, :0]
                    u_aug_a = U_obs_[:, :0]
            else:
                U_sel_y = U_mc_
                U_sel_a = U_obs_
                u_aug_y = U_mc_[:, :0]
                u_aug_a = U_obs_[:, :0]

            # A inherits Z's row set: every input above is already T rows, so
            # the treatment is drawn directly in its observed world.
            eps_A = sample_eps(
                "A",
                U_sel_a.shape[-1]
                + X_A_.shape[-1]
                + Z_direct.shape[-1]
                + u_aug_a.shape[-1],
                T,
            )
            input_A = torch.cat([U_sel_a, X_A_, Z_direct, eps_A, u_aug_a], dim=-1)

            f_A = prior_cls(
                in_dim=input_A.shape[-1],
                num_outputs=1,
                u_skip=0,
                **params,
            ).to(self.device)

            A_logits = f_A(input_A).squeeze(-1)

            A, A_probs = self.sample_A(
                A_logits,
                shift=params.get("propensity_shift", 0.0),
                scale=params.get("propensity_scale", 1.0),
                epsilon=params.get("propensity_epsilon", 1e-3),
            )

            eps_Y = sample_eps(
                "Y",
                U_sel_y.shape[-1]
                + X_Y_.shape[-1]
                + W_direct.shape[-1]
                + u_aug_y.shape[-1],
                N,
            )
            input_y = torch.cat([U_sel_y, X_Y_, W_direct, eps_Y, u_aug_y], dim=-1)

            # MU0/MU1 below are Monte Carlo means over the K replicates of each
            # row. Their validity requires the observed rows and the replicates
            # to share the *same* forward pass per mechanism: MLPSCM.forward
            # draws a fresh output-column permutation per call, so splitting
            # generation into separate passes would silently change the
            # mechanism between the targets and the observed data.
            f_y = prior_cls(
                in_dim=input_y.shape[-1],
                num_outputs=2,
                u_skip=0,
                **params,
            ).to(self.device)

            y_joint_raw = f_y(input_y)
            y0_raw = y_joint_raw[:, 0]
            y1_raw = y_joint_raw[:, 1]

            gamma_hetero = torch.rand(1, device=self.device)
            alphas = torch.rand(N, device=self.device)

            cate_raw = y1_raw - y0_raw
            ate_raw = cate_raw.mean()

            y1_mc = (alphas + (1 - alphas) * gamma_hetero) * y1_raw + (
                1 - gamma_hetero
            ) * (1 - alphas) * (y0_raw + ate_raw)

            y0_mc = ((1 - alphas) + alphas * gamma_hetero) * y0_raw + (
                1 - gamma_hetero
            ) * alphas * (y1_raw - ate_raw)

            # Per-replicate CATE draws. The supervised target MU1 - MU0 is their
            # mean, so S2 = Var_k(.) / K is the variance of the noise that target
            # carries around the true CEPO contrast -- what the noise-aware loss
            # deconvolves. See :func:`_mc_mean_var` for why this is paired.
            cate_mc = y1_mc - y0_mc

            # The query block is contiguous by construction, so ``view`` (which
            # would raise on a non-contiguous layout) doubles as an assertion.
            if T_ctx == 0:
                MU0 = y0_mc.view(T, K).mean(dim=1)
                MU1 = y1_mc.view(T, K).mean(dim=1)
                S2 = _mc_mean_var(cate_mc.view(T, K), K)
            else:
                # Context rows have no K-average to report. NaN rather than the
                # single realised draw: any consumer that reads MU outside
                # ``[train_size:]`` is looking at something that was never
                # computed, and should fail loudly instead of silently getting
                # ITE-quality values.
                nan_pad = torch.full(
                    (T_ctx,), float("nan"), device=self.device, dtype=y0_mc.dtype
                )
                MU0 = torch.cat([nan_pad, y0_mc[T_ctx:].view(T_q, K).mean(dim=1)])
                MU1 = torch.cat([nan_pad, y1_mc[T_ctx:].view(T_q, K).mean(dim=1)])
                S2 = torch.cat([nan_pad, _mc_mean_var(cate_mc[T_ctx:].view(T_q, K), K)])

            U = U_mc[obs_idx]
            W = W_mc[obs_idx]
            # Z, A and A_probs were generated on the observed rows directly.
            Z = Z_obs
            y0 = y0_mc[obs_idx]
            y1 = y1_mc[obs_idx]

            y = torch.where(A == 1, y1, y0)

            # Categorical-completeness profiles are computed on the realised
            # (observed, pre-padding) blocks so they reflect exactly what
            # survives the constant-column filtering below.
            if enforce_cat:
                u_profile = observed_block_profile(U, u_meta)
                w_profile = observed_block_profile(W, w_meta)
                z_profile = observed_block_profile(Z, z_meta)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            U, X, W, Z, A, A_probs, y, y0, y1, MU0, MU1, S2 = (
                U.unsqueeze(0),
                X.unsqueeze(0),
                W.unsqueeze(0),
                Z.unsqueeze(0),
                A.unsqueeze(0),
                A_probs.unsqueeze(0),
                y.unsqueeze(0),
                y0.unsqueeze(0),
                y1.unsqueeze(0),
                MU0.unsqueeze(0),
                MU1.unsqueeze(0),
                S2.unsqueeze(0),
            )
            d_U = torch.tensor([params["dim_U"]], device=self.device, dtype=torch.long)
            d_X = torch.tensor([params["dim_X"]], device=self.device, dtype=torch.long)
            d_W = torch.tensor([params["dim_W"]], device=self.device, dtype=torch.long)
            d_Z = torch.tensor([params["dim_Z"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            U, d_U = self.filter_constant_features_and_pad(U, d_U, params["max_dim_U"])
            X, d_X = self.filter_constant_features_and_pad(X, d_X, params["max_dim_X"])
            W, d_W = self.filter_constant_features_and_pad(W, d_W, params["max_dim_W"])
            Z, d_Z = self.filter_constant_features_and_pad(Z, d_Z, params["max_dim_Z"])

            # Basic validity: every observed block needs at least one active
            # (non-constant) column, else the dataset is unusable.
            basic_ok = (d_X > 0).all() and (d_W > 0).all() and (d_Z > 0).all()

            # Proximal validity: the completeness heuristic must survive column
            # filtering. dim_W, dim_Z >= dim_U holds at sampling time, but the
            # filtering above acts on each block independently, so a proxy can
            # drop below d_U afterwards. Reject and resample those. Requiring
            # d_U > 0 also rejects the rare unconfounded draws where U collapsed
            # to constants.
            if params.get("enforce_completeness_heuristic", True):
                proximal_ok = bool(
                    (d_U > 0).all() and (torch.minimum(d_W, d_Z) >= d_U).all()
                )
            else:
                proximal_ok = True

            if enforce_cat:
                proximal_ok = proximal_ok and (
                    dominates(w_profile, u_profile) and dominates(z_profile, u_profile)
                )

            give_up = attempt >= int(params.get("completeness_max_attempts", 50))

            if basic_ok and (proximal_ok or give_up):
                # Optional audit payload: how each parent block feeds f_A / f_Y.
                # Collected only on the accepted attempt and under fork_rng, so
                # enabling it leaves the generated data (and the RNG stream any
                # later dataset sees) bit-identical to a run without it.
                diagnostics = None
                if params.get("collect_mechanism_diagnostics", False) and isinstance(
                    f_A, MLPSCM
                ):
                    widths_A = {
                        "U_sel": U_sel_a.shape[-1],
                        "X": X_A_.shape[-1],
                        "proxy": Z_direct.shape[-1],
                        "eps": eps_A.shape[-1],
                        "U_aug": u_aug_a.shape[-1],
                    }
                    widths_Y = {
                        "U_sel": U_sel_y.shape[-1],
                        "X": X_Y_.shape[-1],
                        "proxy": W_direct.shape[-1],
                        "eps": eps_Y.shape[-1],
                        "U_aug": u_aug_y.shape[-1],
                    }
                    with torch.random.fork_rng(devices=[]):
                        diagnostics = {
                            "params": _scalar_params(params),
                            "gamma_hetero": float(gamma_hetero),
                            "attempts": attempt,
                            "A_logits": A_logits.detach().cpu(),
                            "A_logit_mean": float(A_logits.mean()),
                            "A_logit_std": float(A_logits.std()),
                            "mechanisms": {
                                "A": mechanism_diagnostics(f_A, input_A, widths_A),
                                "Y": mechanism_diagnostics(f_y, input_y, widths_Y),
                            },
                        }
                U, X, W, Z, A, A_probs, y, y0, y1, MU0, MU1, S2, d_U, d_X, d_W, d_Z = (
                    U.squeeze(0),
                    X.squeeze(0),
                    W.squeeze(0),
                    Z.squeeze(0),
                    A.squeeze(0),
                    A_probs.squeeze(0),
                    y.squeeze(0),
                    y0.squeeze(0),
                    y1.squeeze(0),
                    MU0.squeeze(0),
                    MU1.squeeze(0),
                    S2.squeeze(0),
                    d_U.squeeze(0),
                    d_X.squeeze(0),
                    d_W.squeeze(0),
                    d_Z.squeeze(0),
                )
                # The *realised* K, i.e. after ``_cap_mc_samples``. The nominal
                # ``mcmc_cate`` is not a substitute: prior_mc_mem_budget caps it
                # per dataset behind a once-per-process warning, so this is the
                # only place downstream code can see how many replicates the
                # targets were actually averaged over.
                K_eff = torch.tensor(K, device=self.device, dtype=torch.long)
                return (
                    U,
                    X,
                    W,
                    Z,
                    A,
                    A_probs,
                    y,
                    y0,
                    y1,
                    MU0,
                    MU1,
                    d_U,
                    d_X,
                    d_W,
                    d_Z,
                    S2,
                    K_eff,
                    diagnostics,
                )

    def sample_A(
        self,
        logits: Tensor,
        shift: float = 0.0,
        scale: float = 1.0,
        epsilon: float = 1e-3,
    ) -> Tensor:
        logits = (logits - logits.mean()) / (logits.std() + 1e-8)
        logits = logits * scale + shift

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=epsilon, max=1.0 - epsilon)
        return torch.bernoulli(probs).long(), probs

    @staticmethod
    def confounding_input_dim(
        share: float, n_other: int, floor: int = 2, max_share: float = 0.75
    ) -> int:
        """Number of U-derived input columns targeting a given input ``share``.

        The confounding strength of ``U`` in a random MLP mechanism scales with
        the fraction of the mechanism's input width that is U-derived. Given a
        target ``share`` s and the width ``n_other`` of the competing (non-U)
        inputs, the count k solving k / (k + n_other) = s is
        ``k = s / (1 - s) * n_other``. ``dim_U`` is deliberately absent: the
        knob controls U's *total* effective input width (subsampling U below
        ``dim_U``, augmenting above it), decoupling confounding strength from
        the confounder's dimension.

        ``share`` is clamped to ``max_share`` (bounding k at 3 * n_other for
        the default 0.75, a memory cap on the augmentation network) and the
        result floored at ``floor`` so every mechanism sees at least that many
        U-derived columns.

        The same share -> width conversion also sizes the exogenous-noise
        (eps) input blocks in ``generate_dataset`` (via the sampled
        ``noise_share_of_input_in_*`` hyperparameters), with ``n_other`` set
        to the mechanism's non-noise signal width.
        """
        s = min(float(share), max_share)
        return max(floor, round(s / (1.0 - s) * n_other))

    @staticmethod
    def split_confounding_inputs(
        k_y: int, k_a: int, dim_U: int, perm: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
        """Split per-side U-input budgets into raw-U columns and aug widths.

        Each side (Y, A) consumes the first ``min(k, dim_U)`` columns of the
        shared permutation ``perm`` of U's columns, plus ``max(0, k - dim_U)``
        columns of the shared augmentation network's output. Taking prefixes of
        the *same* permutation (and the same aug output) makes the two sides'
        U-derived inputs nested: the smaller side's columns are all shared with
        the larger side, and only the difference is side-exclusive -- so the
        common confounding channel has width ``min(k_y, k_a)`` by construction.

        Returns ``(idx_y, idx_a, aug_y, aug_a)``: the raw-U column indices per
        side and the number of augmentation columns per side.
        """
        idx_y = tuple(perm[: min(k_y, dim_U)])
        idx_a = tuple(perm[: min(k_a, dim_U)])
        return idx_y, idx_a, max(0, k_y - dim_U), max(0, k_a - dim_U)

    @staticmethod
    def triangular_dim_pmf(lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray]:
        """Linearly-decreasing integer pmf on ``[lo, hi]``: ``p(d) ∝ (hi+1-d)``.

        Skews the sampled dimension toward smaller values. Paired with the
        conditional-uniform proxy sampling (``dim_W ~ U{dim_U..max}``), a
        triangular ``dim_U`` makes the induced ``dim_W``/``dim_Z`` marginals a
        gentle linear ramp rather than piling up at ``max_dim``. In particular,
        for ``lo=1`` the proxy marginal is exactly ``p(w) = w / sum(1..hi)``.

        Returns ``(values, probs)`` with ``probs`` summing to 1.
        """
        vals = np.arange(lo, hi + 1)
        weights = (hi + 1 - vals).astype(float)
        return vals, weights / weights.sum()

    @classmethod
    def sample_dim(cls, lo: int, hi: int, distribution: str) -> int:
        """Sample an integer dimension in ``[lo, hi]``.

        ``"uniform"`` draws flat over the range (the original behaviour);
        ``"triangular"`` uses :meth:`triangular_dim_pmf` (Option B).
        """
        if distribution == "uniform":
            return int(np.random.randint(lo, hi + 1))
        if distribution == "triangular":
            vals, probs = cls.triangular_dim_pmf(lo, hi)
            return int(np.random.choice(vals, p=probs))
        raise ValueError(
            f"Unknown dim distribution {distribution!r} "
            "(expected 'uniform' or 'triangular')"
        )

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, ...]:
        """Generate a batch of proximal datasets.
        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size.

        Returns
        -------
        Tuple containing:
            X, W, Z, A, Y_obs, Y0, Y1 : Tensors (or NestedTensors if seq_len_per_gp=True)
            d_X, d_W, d_Z : Active dimension trackers for the proxies and covariates
            seq_lens, train_sizes : Metadata tensors
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len,
                self.max_seq_len,
                log=self.log_seq_len,
                replay_small=self.replay_small,
            )
            global_train_size = self.sample_train_size(
                self.min_train_size, self.max_train_size, global_seq_len
            )

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()

            # Sequence Length & Max Dimension Scaling
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len,
                    self.max_seq_len,
                    log=self.log_seq_len,
                    replay_small=self.replay_small,
                )
                gp_train_size = self.sample_train_size(
                    self.min_train_size, self.max_train_size, gp_seq_len
                )
                # Adjust maximum dimensions proportionally if sequence is huge to avoid OOM
                gp_max_X, gp_max_W, gp_max_Z = self.adjust_max_features(
                    gp_seq_len, self.max_dim_X, self.max_dim_W, self.max_dim_Z
                )
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_X, gp_max_W, gp_max_Z = (
                    self.max_dim_X,
                    self.max_dim_W,
                    self.max_dim_Z,
                )
            gp_max_W = max(gp_max_W, self.max_dim_U)
            gp_max_Z = max(gp_max_Z, self.max_dim_U)

            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(
                    size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp
                )
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                subgp_sampled_hp = {
                    k: v() if callable(v) else v for k, v in group_sampled_hp.items()
                }

                # 1. Sample Latent Confounder Dimension. A triangular pmf
                # (Option B) skews d_U small so the conditional-uniform proxies
                # below get a gentle linear marginal instead of piling up at
                # max_dim; "uniform" restores the original flat d_U for A/B.
                dim_U = self.sample_dim(
                    self.min_dim_U,
                    self.max_dim_U,
                    self.fixed_hp.get("dim_U_distribution", "triangular"),
                )

                # 2. Enforce Completeness: dim(W) >= dim(U) and dim(Z) >= dim(U)
                # We use max() to ensure the lower bound satisfies both the user's config and the DAG requirement
                min_W_req = max(self.min_dim_W, dim_U)
                max_W_req = max(gp_max_W, min_W_req)
                dim_W = np.random.randint(min_W_req, max_W_req + 1)

                min_Z_req = max(self.min_dim_Z, dim_U)
                max_Z_req = max(gp_max_Z, min_Z_req)
                dim_Z = np.random.randint(min_Z_req, max_Z_req + 1)

                # 3. Sample Covariates (No completeness requirement relative to U)
                max_X_req = max(self.min_dim_X, gp_max_X)
                dim_X = np.random.randint(self.min_dim_X, max_X_req + 1)

                for ds_idx in range(actual_subgp_size):
                    params = {
                        **self.fixed_hp,
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        "mcmc_cate": self.mcmc_cate,
                        "mc_targets_on_context": self.mc_targets_on_context,
                        # Pass the memory-safe maximums so the PyTorch module knows what to pad up to
                        "max_dim_U": self.max_dim_U,
                        "max_dim_X": gp_max_X,
                        "max_dim_W": gp_max_W,
                        "max_dim_Z": gp_max_Z,
                        **subgp_sampled_hp,
                        "prior_type": subgp_prior_type,
                        # Pass the exact active dimensions for this specific graph
                        "dim_U": dim_U,
                        "dim_W": dim_W,
                        "dim_Z": dim_Z,
                        "dim_X": dim_X,
                        "device": self.device,
                    }
                    param_list.append(params)

        # Parallel Generation
        if self.n_jobs != 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs,
                backend="loky",
                inner_max_num_threads=self.num_threads_per_generate,
            ):
                results = joblib.Parallel()(
                    joblib.delayed(self.generate_dataset)(params)
                    for params in param_list
                )
        else:
            results = [self.generate_dataset(params) for params in param_list]

        (
            U_list,
            X_list,
            W_list,
            Z_list,
            A_list,
            A_probs_list,
            Y_obs_list,
            Y0_list,
            Y1_list,
            MU0_list,
            MU1_list,
            d_U_list,
            d_X_list,
            d_W_list,
            d_Z_list,
            S2_list,
            K_eff_list,
            diagnostics_list,
        ) = zip(*results)

        # Per-dataset mechanism diagnostics (None unless the fixed HP
        # collect_mechanism_diagnostics is set). Exposed as an attribute rather
        # than a return value to keep the batch tuple stable for training code.
        self.last_diagnostics = list(diagnostics_list)

        # Combine Results
        if self.seq_len_per_gp:
            U = nested_tensor([t.to(self.device) for t in U_list], device=self.device)
            X = nested_tensor([t.to(self.device) for t in X_list], device=self.device)
            W = nested_tensor([t.to(self.device) for t in W_list], device=self.device)
            Z = nested_tensor([t.to(self.device) for t in Z_list], device=self.device)
            A = nested_tensor([t.to(self.device) for t in A_list], device=self.device)
            A_probs = nested_tensor(
                [t.to(self.device) for t in A_probs_list], device=self.device
            )
            Y_obs = nested_tensor(
                [t.to(self.device) for t in Y_obs_list], device=self.device
            )
            Y0 = nested_tensor([t.to(self.device) for t in Y0_list], device=self.device)
            Y1 = nested_tensor([t.to(self.device) for t in Y1_list], device=self.device)
            MU0 = nested_tensor(
                [t.to(self.device) for t in MU0_list], device=self.device
            )
            MU1 = nested_tensor(
                [t.to(self.device) for t in MU1_list], device=self.device
            )
            S2 = nested_tensor([t.to(self.device) for t in S2_list], device=self.device)
        else:
            U = torch.stack(U_list).to(self.device)
            X = torch.stack(X_list).to(self.device)
            W = torch.stack(W_list).to(self.device)
            Z = torch.stack(Z_list).to(self.device)
            A = torch.stack(A_list).to(self.device)
            A_probs = torch.stack(A_probs_list).to(self.device)
            Y_obs = torch.stack(Y_obs_list).to(self.device)
            Y0 = torch.stack(Y0_list).to(self.device)
            Y1 = torch.stack(Y1_list).to(self.device)
            MU0 = torch.stack(MU0_list).to(self.device)
            MU1 = torch.stack(MU1_list).to(self.device)
            S2 = torch.stack(S2_list).to(self.device)

        # Metadata
        d_U = torch.stack(d_U_list).to(self.device)
        d_X = torch.stack(d_X_list).to(self.device)
        d_W = torch.stack(d_W_list).to(self.device)
        d_Z = torch.stack(d_Z_list).to(self.device)
        K_eff = torch.stack(K_eff_list).to(self.device)

        seq_lens = torch.tensor(
            [params["seq_len"] for params in param_list],
            device=self.device,
            dtype=torch.long,
        )
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list],
            device=self.device,
            dtype=torch.long,
        )

        return (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            MU0,
            MU1,
            d_U,
            d_X,
            d_W,
            d_Z,
            seq_lens,
            train_sizes,
            S2,
            K_eff,
        )

    def get_prior(self) -> str:
        """Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name.
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(
                ["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probs", [0.7, 0.3])
            )
        else:
            return self.prior_type


class PriorDataset(IterableDataset):
    """Main dataset class that provides an infinite iterator over synthetic tabular datasets
    designed for Proximal Causal Learning.

    Generates datasets adhering to a specific Directed Acyclic Graph (DAG) structure
    where an unmeasured confounder U is addressed using outcome-inducing (W) and
    treatment-inducing (Z) proxies, alongside observed baseline covariates (X).

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_dim_U : int, default=3
        Minimum dimensionality of the unmeasured latent confounder U.

    max_dim_U : int, default=10
        Maximum dimensionality of the unmeasured latent confounder U.

    min_dim_W : int, default=3
        Minimum dimensionality of the outcome-inducing proxy W. To satisfy the
        completeness assumption, the sampled dimension should ideally satisfy
        $dim(W) \\ge dim(U)$.

    max_dim_W : int, default=10
        Maximum dimensionality of the outcome-inducing proxy W.

    min_dim_Z : int, default=3
        Minimum dimensionality of the treatment-inducing proxy Z. To satisfy the
        completeness assumption, the sampled dimension should ideally satisfy
        $dim(Z) \\ge dim(U)$.

    max_dim_Z : int, default=10
        Maximum dimensionality of the treatment-inducing proxy Z.

    min_dim_X : int, default=5
        Minimum dimensionality of the observed baseline covariates X.

    max_dim_X : int, default=20
        Maximum dimensionality of the observed baseline covariates X.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    prior_type : str, default="mlp_scm"
        Type of prior structural causal model to use for the DAG mechanisms:
         - 'mlp_scm': MLP-based non-linear mechanisms.
         - 'tree_scm': Tree-based mechanisms (e.g., Random Forest, XGBoost).
         - 'mix_scm': Probabilistic mix of the above models.

    mcmc_cate : int, default=200
        Number of Monte Carlo samples used to estimate the CATE targets
        (MU0/MU1) per row.

    mc_targets_on_context : bool, default=True
        If False, replicate only the query rows ``[train_size:]`` and leave
        MU0/MU1 as ``NaN`` on the context prefix. Roughly halves generation
        cost at the default ``train_size ~ U(0.1, 0.9) * seq_len``. Training
        never reads context MU; analysis/visualisation code does, so the
        default keeps the full-sequence behaviour.

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters for the SCM.

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation (e.g., dropout, noise scale).

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_dim_U: int = 3,
        max_dim_U: int = 10,
        min_dim_W: int = 3,
        max_dim_W: int = 10,
        min_dim_Z: int = 3,
        max_dim_Z: int = 10,
        min_dim_X: int = 5,
        max_dim_X: int = 20,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        mcmc_cate: int = 200,
        mc_targets_on_context: bool = True,
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()

        self.prior = SCMPrior(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            batch_size_per_subgp=batch_size_per_subgp,
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            seq_len_per_gp=seq_len_per_gp,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
            prior_type=prior_type,
            mcmc_cate=mcmc_cate,
            mc_targets_on_context=mc_targets_on_context,
            fixed_hp=scm_fixed_hp,
            sampled_hp=scm_sampled_hp,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_dim_U = min_dim_U
        self.max_dim_U = max_dim_U
        self.min_dim_W = min_dim_W
        self.max_dim_W = max_dim_W
        self.min_dim_Z = min_dim_Z
        self.max_dim_Z = max_dim_Z
        self.min_dim_X = min_dim_X
        self.max_dim_X = max_dim_X
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type
        self.mcmc_cate = mcmc_cate
        self.mc_targets_on_context = mc_targets_on_context

    @property
    def last_diagnostics(self):
        """Per-dataset mechanism diagnostics of the most recent batch.

        ``None`` entries unless ``collect_mechanism_diagnostics`` is set in
        ``scm_fixed_hp``; ``None`` before any batch was generated.
        """
        return getattr(self.prior, "last_diagnostics", None)

    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call.

        Returns
        -------
        tuple
            The 19-tuple ``(U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1,
            d_U, d_X, d_W, d_Z, seq_lens, train_sizes, S2, K_eff)``.

            Row-shaped entries are ``(batch_size, seq_len)`` or
            ``(batch_size, seq_len, max_dim_*)``, or a NestedTensor when
            ``seq_len_per_gp=True``. Per-dataset metadata (``d_*``,
            ``seq_lens``, ``train_sizes``, ``K_eff``) is ``(batch_size,)``.

            ``S2`` and ``K_eff`` are appended last deliberately: existing
            positional readers index 0..16, so growing the tuple at the end
            leaves them valid. See :meth:`SCMPrior.generate_dataset` for their
            meaning.
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "PriorDataset":
        """Return an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator.
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return the next batch from the iterator.

        Since this is an infinite iterator, it never raises StopIteration
        and instead continuously generates new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters.
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features U: {self.min_dim_U} - {self.max_dim_U}\n"
            f"  features W: {self.min_dim_W} - {self.max_dim_W}\n"
            f"  features Z: {self.min_dim_Z} - {self.max_dim_Z}\n"
            f"  features X: {self.min_dim_X} - {self.max_dim_X}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  mcmc_cate: {self.mcmc_cate}\n"
            f"  mc_targets_on_context: {self.mc_targets_on_context}\n"
            f"  device: {self.device}\n"
            f")"
        )


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
