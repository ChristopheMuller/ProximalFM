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
# - Regression Alignment: Removed classification logic (Reg2Cls) and class-balance
#   sanity checks to facilitate continuous CATE estimation.
# - Adaptive Scaling: Updated 'adjust_max_features' to scale the entire proximal
#   feature set (X+W+Z) proportionally for large sequence lengths to prevent OOM.
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

from ._hp_sampling import HpSamplerList
from ._reg2cls import Reg2Cls
from ._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


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
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(
                xi[:, unique_mask],
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
        d_new = torch.tensor([min(di, max_d) for di in d_new], device=X_new.device)

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

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        X : Tensor
            Features tensor of shape ``(seq_len, max_features)``.

        y : Tensor
            Labels tensor of shape ``(seq_len,)``.

        d : Tensor
            Number of active features after filtering (scalar Tensor).
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            u_sampler = XSampler(
                seq_len=params["seq_len"],
                num_features=params["dim_U"],
                pre_stats=params.get("pre_sample_cause_stats", False),
                sampling=params.get("sampling", "mixed"),
                device=self.device,
            )
            U = u_sampler.sample()

            f_X = prior_cls(
                in_dim=params["dim_U"], num_outputs=params["dim_X"], **params
            ).to(self.device)
            X = f_X(U)

            input_WZ = torch.cat([U, X], dim=-1)
            f_W = prior_cls(
                in_dim=params["dim_U"] + params["dim_X"],
                num_outputs=params["dim_W"],
                **params,
            ).to(self.device)
            W = f_W(input_WZ)

            f_Z = prior_cls(
                in_dim=params["dim_U"] + params["dim_X"],
                num_outputs=params["dim_Z"],
                **params,
            ).to(self.device)
            Z = f_Z(input_WZ)

            input_A = torch.cat([U, X, Z], dim=-1)
            f_A = prior_cls(
                in_dim=params["dim_U"] + params["dim_X"] + params["dim_Z"],
                num_outputs=1,
                **params,
            ).to(self.device)
            A_logits = f_A(input_A).squeeze(-1)
            A = self.sample_A(
                A_logits,
                epsilon=params["propensity_epsilon"],
                temperature=params["propensity_temperature"],
            )

            input_y = torch.cat([U, X, W], dim=-1)
            input_joint = self.prepare_input(type="y", input=input_y, **params)
            f_y = prior_cls(
                in_dim=params["dim_U"] + params["dim_X"] + params["dim_W"] + 1,
                num_outputs=1,
                **params,
            ).to(self.device)
            y_joint = f_y(input_joint).squeeze(-1)

            n = input_y.shape[0]
            y0, y1 = y_joint[:n], y_joint[n:]

            y = torch.where(A == 1, y1, y0)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, W, Z, y, y0, y1 = (
                X.unsqueeze(0),
                W.unsqueeze(0),
                Z.unsqueeze(0),
                y.unsqueeze(0),
                y0.unsqueeze(0),
                y1.unsqueeze(0),
            )
            d_X = torch.tensor([params["dim_X"]], device=self.device, dtype=torch.long)
            d_W = torch.tensor([params["dim_W"]], device=self.device, dtype=torch.long)
            d_Z = torch.tensor([params["dim_Z"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d_X = self.filter_constant_features_and_pad(X, d_X, params["max_dim_X"])
            W, d_W = self.filter_constant_features_and_pad(W, d_W, params["max_dim_W"])
            Z, d_Z = self.filter_constant_features_and_pad(Z, d_Z, params["max_dim_Z"])
            if (d_X > 0).all() and (d_W > 0).all() and (d_Z > 0).all():
                X, W, Z, A, y, y0, y1, d_X, d_W, d_Z = (
                    X.squeeze(0),
                    W.squeeze(0),
                    Z.squeeze(0),
                    A.squeeze(0),
                    y.squeeze(0),
                    y0.squeeze(0),
                    y1.squeeze(0),
                    d_X.squeeze(0),
                    d_W.squeeze(0),
                    d_Z.squeeze(0),
                )
                return (
                    X,
                    W,
                    Z,
                    A,
                    y,
                    y0,
                    y1,
                    d_X,
                    d_W,
                    d_Z,
                )

    def prepare_input(self, type: str, input: Tensor, **params) -> Tensor:
        if type == "A":
            return input

        elif type == "y":
            n = input.shape[0]

            A0 = torch.zeros((n, 1), device=input.device)
            A1 = torch.ones((n, 1), device=input.device)

            input_y0 = torch.cat([input, A0], dim=-1)
            input_y1 = torch.cat([input, A1], dim=-1)

            input_joint = torch.cat([input_y0, input_y1], dim=0)
            return input_joint

    def sample_A(
        self, logits: Tensor, epsilon: float = 0.05, temperature: float = 1.0
    ) -> Tensor:
        probs = torch.sigmoid(logits / temperature)
        probs = torch.clamp(probs, min=epsilon, max=1.0 - epsilon)
        return torch.bernoulli(probs).long()

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

                # 1. Sample Latent Confounder Dimension
                dim_U = np.random.randint(self.min_dim_U, self.max_dim_U + 1)

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
                        # Pass the memory-safe maximums so the PyTorch module knows what to pad up to
                        "max_dim_X": gp_max_X
                        if self.seq_len_per_gp
                        else self.max_dim_X,
                        "max_dim_W": gp_max_W
                        if self.seq_len_per_gp
                        else self.max_dim_W,
                        "max_dim_Z": gp_max_Z
                        if self.seq_len_per_gp
                        else self.max_dim_Z,
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
        if self.n_jobs > 1 and self.device == "cpu":
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
            X_list,
            W_list,
            Z_list,
            A_list,
            Y_obs_list,
            Y0_list,
            Y1_list,
            d_X_list,
            d_W_list,
            d_Z_list,
        ) = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            X = nested_tensor([t.to(self.device) for t in X_list], device=self.device)
            W = nested_tensor([t.to(self.device) for t in W_list], device=self.device)
            Z = nested_tensor([t.to(self.device) for t in Z_list], device=self.device)
            A = nested_tensor([t.to(self.device) for t in A_list], device=self.device)
            Y_obs = nested_tensor(
                [t.to(self.device) for t in Y_obs_list], device=self.device
            )
            Y0 = nested_tensor([t.to(self.device) for t in Y0_list], device=self.device)
            Y1 = nested_tensor([t.to(self.device) for t in Y1_list], device=self.device)
        else:
            X = torch.stack(X_list).to(self.device)
            W = torch.stack(W_list).to(self.device)
            Z = torch.stack(Z_list).to(self.device)
            A = torch.stack(A_list).to(self.device)
            Y_obs = torch.stack(Y_obs_list).to(self.device)
            Y0 = torch.stack(Y0_list).to(self.device)
            Y1 = torch.stack(Y1_list).to(self.device)

        # Metadata
        d_X = torch.stack(d_X_list).to(self.device)
        d_W = torch.stack(d_W_list).to(self.device)
        d_Z = torch.stack(d_Z_list).to(self.device)

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

        return X, W, Z, A, Y_obs, Y0, Y1, d_X, d_W, d_Z, seq_lens, train_sizes

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
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len, max_features)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of
            ``(batch_size, seq_len, max_features)``.

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of ``(batch_size, seq_len)``.

        d : Tensor
            Number of active features per dataset of shape ``(batch_size,)``.

        seq_lens : Tensor
            Sequence length for each dataset of shape ``(batch_size,)``.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape ``(batch_size,)``.
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
