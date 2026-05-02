####
# Changes vs TabICL:
# - Modular Transformation: Converted from a standalone dataset generator into a
#   reusable DAG component that maps parent nodes to child nodes.
# - Externalized Input: Removed internal 'XSampler' and 'handle_outputs' logic; the
#   module now processes an explicit 'input' tensor in the forward pass.
# - Parameter Flexibility: Removed the hardcoded overrides for 'is_causal',
#   'num_layers', and 'hidden_dim', allowing these to be controlled via the
#   global SCMPrior configuration.
# - Dimension Refactoring: Replaced implicit causal selection parameters with
#   explicit 'in_dim' and 'num_outputs' to align with the proximal DAG structure.
# - API Alignment: Renamed lambda parameters (e.g., 'max_depth_lambda') to match
#   the updated naming convention in the hyperparameter sampler.
# - Error Handling: Switched to 'torch.nan_to_num' for cleaner and more efficient
#   recovery from tree-induced numerical instabilities.
# - Sequential Logic: Streamlined the forward pass to a standard sequential flow,
#   removing intermediate state tracking and causal indexing.
####

#####
# TODO:
# - cpu to gpu transfer: can be improved?
# - XGB does not work -> crashes
#####

from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from ._utils import GaussianNoise, XSampler


class TreeLayer(nn.Module):
    """A layer that transforms input features using a tree-based model.

    This layer fits a specified tree-based regression model (Decision Tree,
    Extra Trees, Random Forest, or XGBoost) to the input features using
    randomly generated target values. It then uses the trained model
    to predict the outputs.

    Parameters
    ----------
    tree_model : str
        The type of tree-based model to use. Options are "decision_tree",
        "extra_trees", "random_forest", "xgboost".

    max_depth : int
        The maximum depth allowed for the individual trees in the model.

    n_estimators : int
        The number of trees in the ensemble.

    out_dim : int
        The desired output dimension for the transformed features. This determines
        the number of target variables (``y_fake``) generated for fitting the
        multi-output regressor.

    device : str or torch.device
        The device ('cpu' or 'cuda') on which to place the output tensor.
    """

    def __init__(
        self,
        tree_model: str,
        max_depth: int,
        n_estimators: int,
        out_dim: int,
        device: str,
    ):
        super(TreeLayer, self).__init__()
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.out_dim = out_dim
        self.device = device

        if tree_model == "decision_tree":
            self.model = MultiOutputRegressor(
                DecisionTreeRegressor(max_depth=max_depth, splitter="random"), n_jobs=-1
            )
        elif tree_model == "extra_trees":
            self.model = MultiOutputRegressor(
                ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth),
                n_jobs=-1,
            )
        elif tree_model == "random_forest":
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth),
                n_jobs=-1,
            )
        elif tree_model == "xgboost":
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                tree_method="hist",
                multi_strategy="multi_output_tree",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Invalid tree model: {tree_model}")

    def forward(self, X):
        """Applies the fitted tree-based transformation to the input features.

        Fits the internal tree model using X and random targets, then predicts
        and returns the outputs.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Transformed features tensor of shape ``(n_samples, out_dim)``.
        """
        X = X.nan_to_num(0.0).cpu()
        y_fake = np.random.randn(X.shape[0], self.out_dim)
        self.model.fit(X, y_fake)
        y = self.model.predict(X)
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        if self.out_dim == 1:
            y = y.view(-1, 1)

        return y


class TreeSCM(nn.Module):
    """A Tree-based Structural Causal Model for generating synthetic datasets.

    Similar to MLP-based SCM but uses tree-based models (like Random Forests or XGBoost)
    for potentially non-linear feature transformations instead of linear layers.

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    num_outputs : int, default=1
        The number of outputs.

    is_causal : bool, default=True
        - If `True`, simulates a causal graph: `X` and `y` are sampled from the
          intermediate outputs of the tree transformations applied to initial causes.
          The `num_causes` parameter controls the number of initial root variables.
        - If `False`, simulates a direct predictive mapping: Initial causes are used
          directly as `X`, and the final output of the tree layers becomes `y`. `num_causes`
          is effectively ignored and set equal to `num_features`.

    num_causes : int, default=10
        The number of initial root 'cause' variables sampled by `XSampler`.
        Only relevant when `is_causal=True`. If `is_causal=False`, this is internally
        set to `num_features`.

    y_is_effect : bool, default=True
        Specifies how the target `y` is selected when `is_causal=True`.
        - If `True`, `y` is sampled from the outputs of the final tree layer(s),
          representing terminal effects in the causal chain.
        - If `False`, `y` is sampled from the earlier intermediate outputs (after
          permutation), representing variables closer to the initial causes.

    in_clique : bool, default=False
        Controls how features `X` and targets `y` are sampled from the flattened
        intermediate tree outputs when `is_causal=True`.
        - If `True`, `X` and `y` are selected from a contiguous block of the
          intermediate outputs, potentially creating denser dependencies among them.
        - If `False`, `X` and `y` indices are chosen randomly and independently
          from all available intermediate outputs.

    sort_features : bool, default=True
        Determines whether to sort the features based on their original indices from
        the intermediate tree outputs. Only relevant when `is_causal=True`.

    num_layers : int, default=5
        Number of tree transformation layers.

    hidden_dim : int, default=10
        Output dimension size for intermediate tree transformations.

    tree_model : str, default="xgboost"
        Type of tree model to use ("decision_tree", "extra_trees", "random_forest",
        "xgboost"). XGBoost is favored for performance as it supports multi-output
        regression natively.

    max_depth_lambda : float, default=0.5
        Lambda parameter for sampling the max depth for tree models from an
        exponential distribution.

    n_estimators_lambda : float, default=0.5
        Lambda parameter for sampling the number of estimators (trees) per layer
        from an exponential distribution.

    sampling : str, default="normal"
        The method used by `XSampler` to generate the initial 'cause' variables.
        Options:
        - "normal": Standard normal distribution (potentially with pre-sampled stats).
        - "uniform": Uniform distribution between 0 and 1.
        - "mixed": A random combination of normal, multinomial (categorical),
          Zipf (power-law), and uniform distributions across different cause variables.

    pre_sample_cause_stats : bool, default=False
        If `True` and `sampling="normal"`, the mean and standard deviation for
        each initial cause variable are pre-sampled. Passed to `XSampler`.

    noise_std : float, default=0.01
        The base standard deviation for the Gaussian noise added after each tree
        layer's transformation.

    pre_sample_noise_std : bool, default=False
        Controls how the standard deviation for the `GaussianNoise` layers is determined.
        If `True`, the noise standard deviation for each output dimension of a layer
        is sampled from a normal distribution centered at 0 with `noise_std`.
        If `False`, a fixed `noise_std` is used for all dimensions.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        in_dim: int,
        num_outputs: int,
        num_layers: int = 2,
        hidden_dim: int = 10,
        tree_model: str = "xgboost",
        max_depth_lambda: float = 0.5,
        n_estimators_lambda: float = 0.5,
        noise_std: float = 0.01,
        pre_sample_noise_std: bool = False,
        device: str = "cpu",
        **kwargs: dict,
    ):
        super(TreeSCM, self).__init__()

        self.in_dim = in_dim
        self.num_outputs = num_outputs

        # Tree models are computationally heavy, so we keep layers and dims small by default.
        # Overriding kwargs if necessary, but using the passed parameters as a base.
        self.num_layers = max(1, num_layers)
        self.hidden_dim = hidden_dim

        self.tree_model = tree_model
        self.tree_depth_lambda = max_depth_lambda
        self.tree_n_estimators_lambda = n_estimators_lambda
        self.noise_std = noise_std
        self.pre_sample_noise_std = pre_sample_noise_std
        self.device = device

        # Build layers sequentially
        layers = []

        if self.num_layers == 1:
            # If only 1 layer, it goes straight from input to output
            layers.append(self.generate_layer_modules(is_output_layer=True))
        else:
            # First layer outputs to hidden_dim
            layers.append(self.generate_layer_modules(is_output_layer=False))

            # Intermediate layers
            for _ in range(self.num_layers - 2):
                layers.append(self.generate_layer_modules(is_output_layer=False))

            layers.append(self.generate_layer_modules(is_output_layer=True))
        self.layers = nn.Sequential(*layers).to(device)

    def generate_layer_modules(self, is_output_layer=False):
        """Generates a layer module with activation, tree-based transformation, and noise."""
        out_dim = self.num_outputs if is_output_layer else self.hidden_dim

        max_depth = 2 + int(np.random.exponential(1 / self.tree_depth_lambda))
        n_estimators = 1 + int(np.random.exponential(1 / self.tree_n_estimators_lambda))
        tree_layer = TreeLayer(
            tree_model=self.tree_model,
            max_depth=min(max_depth, 4),
            n_estimators=min(n_estimators, 4),
            out_dim=out_dim,
            device=self.device,
        )

        if self.pre_sample_noise_std:
            noise_std = torch.abs(
                torch.normal(
                    torch.zeros(size=(1, out_dim), device=self.device),
                    float(self.noise_std),
                )
            )
        else:
            noise_std = self.noise_std
        noise_layer = GaussianNoise(noise_std)

        return nn.Sequential(tree_layer, noise_layer)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Passes the input tensor through the tree-based mechanism block.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor containing the concatenated parent variables.
            Shape: (seq_len, in_dim)

        Returns
        -------
        torch.Tensor
            The generated child variables.
            Shape: (seq_len, num_outputs)
        """
        out = input

        # Simple sequential pass through the tree layers and noise
        for layer in self.layers:
            out = layer(out)

        # Check for NaNs (which can occasionally happen with extreme tree splits or noise)
        # and handle them safely
        if torch.any(torch.isnan(out)):
            out = torch.nan_to_num(out, nan=0.0)

        return out
