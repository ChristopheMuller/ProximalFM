####
# Changes vs TabICL:
# - Proximal Input Protocol: Expanded 'forward', '_train_forward', and
#   '_inference_forward' to accept the full causal quadruple: X (covariates),
#   Z (treatment proxies), W (outcome proxies), and A (treatment).
# - Causal Dimension Aggregation: Added '_get_d_total' to compute the unified
#   feature count ($d_{total} = d_X + d_Z + d_W + 1$) required for
#   position-aware row interactions.
# - Target Task Simplification: Removed 'max_classes' and 'num_quantiles'
#   logic, effectively deprecating the 'QuantileToDistribution' post-processing
#   to focus on direct proximal learning.
# - Parameter Pruning: Removed feature grouping configurations
#   (col_feature_group, col_feature_group_size) from the top-level module
#   to align with the new 'ColEmbedding' implementation.
# - Deprecated Stat Methods: Excised 'predict_stats' and
#   'predict_stats_with_cache', as the current iteration prioritizes raw
#   proximal output over distribution-based regression stats.
# - Interface Streamlining: Removed 'return_logits' and 'softmax_temperature'
#   logic from inference calls, reflecting the shift from classification
#   to continuous proximal estimation.
####

####
# TODO:
# - bring back forward with cache
####


from __future__ import annotations
from typing import Optional, Union, Literal

from torch import nn, Tensor
import torch

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ICLearning
from .kv_cache import TabICLCache
from .inference_config import InferenceConfig


class TabICL(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabICL is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    This class is the underlying raw PyTorch module for TabICL. It is not
    intended to be used directly. Instead, use the classes from the top-level
    `tabicl` package such as :class:`tabicl.TabICLClassifier` or
    :class:`tabicl.TabICLRegressor` that wrap this class to include the
    necessary preprocessing of input features and postprocessing of
    predictions.

    Parameters
    ----------
    max_classes : int, default=10
        Determines the task type and output behavior:
        - If max_classes=0: The model performs regression using quantile prediction.
        - If max_classes>0: The model performs classification. This value specifies
          the number of classes the model supports natively. If the number of classes
          in the dataset exceeds this value, mixed-radix ensembling is used during
          column-wise embedding and hierarchical classification is used during in-context learning.

    num_quantiles : int, default=999
        Number of quantiles to predict for regression tasks. Only used when max_classes=0.
        The model directly predicts these quantile values.

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the in-context
        learning transformer, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer.

    col_nhead : int, default=8
        Number of attention heads in the column embedding transformer.

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer.

    col_affine : bool, default=False
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    col_feature_group : bool or Literal["same", "valid"], default="same"
        Feature grouping mode:
        - False: No grouping
        - True or "same": Group through circular permutation (output has same number of groups as features)
        - "valid": Group through padding and reshaping (output may have fewer groups)

    col_feature_group_size : int, default=3
        Number of features per group when feature grouping is enabled.

    col_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the column embedding transformer. Note that only the first
        attention layer of the induced self-attention blocks uses SSMax.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer.

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer.

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row.

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer.

    row_rope_interleaved : bool, default=False
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    icl_num_blocks : int, default=12
        Number of transformer blocks in the in-context learning transformer.

    icl_nhead : int, default=8
        Number of attention heads in the in-context learning transformer.

    icl_ssmax : bool or str, default="qassmax-mlp-elementwise"
        Type of scalable softmax to use in the in-context learning transformer.
        If True, equivalent to "qassmax-mlp-elementwise".
        If False, equivalent to "none".
        If a string, uses the specified scalable softmax type.
        Options include:
            - "none": No scaling applied
            - "ssmax": :math:`q_{\\text{scaled}} = q \\cdot (s \\cdot \\log n)` where s is learnable per-head parameter
            - "ssmax-mlp": Uses MLP to compute scaling factors based on sequence length
            - "ssmax-mlp-elementwise": Elementwise scaling per head dimension using MLP
            - "qassmax-mlp": Query-aware scaling: :math:`\\text{scale} = \\text{base\\_mlp}(\\log n) \\cdot (1 + \\tanh(\\text{query\\_mlp}(q)))`
            - "qassmax-mlp-elementwise": Elementwise query-aware scaling

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components.

    dropout : float, default=0.0
        Dropout probability across all components.

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model.

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components.

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers (sets bias=False in nn.LayerNorm).

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        nbins: int = 1000,
        vmin: float = -10.0,
        vmax: float = 10.0,
        joint_quantiles: bool = False,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 8,
        col_num_inds: int = 128,
        col_affine: bool = False,
        col_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        row_rope_interleaved: bool = False,
        icl_num_blocks: int = 12,
        icl_nhead: int = 8,
        icl_ssmax: Union[
            bool,
            Literal[
                "none",
                "ssmax",
                "ssmax-mlp",
                "ssmax-mlp-elementwise",
                "qassmax-mlp",
                "qassmax-mlp-elementwise",
            ],
        ] = "qassmax-mlp-elementwise",
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        **kwargs,  # capture old parameters that are no longer used to avoid breaking old configs
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL

        self.nbins = nbins
        self.vmin = vmin
        self.vmax = vmax
        self.joint_quantiles = joint_quantiles

        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.col_affine = col_affine
        self.col_ssmax = col_ssmax
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_rope_base = row_rope_base
        self.row_rope_interleaved = row_rope_interleaved
        self.icl_num_blocks = icl_num_blocks
        self.icl_nhead = icl_nhead
        self.icl_ssmax = icl_ssmax
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.bias_free_ln = bias_free_ln

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            affine=col_affine,
            reserve_cls_tokens=row_num_cls,
            ssmax=col_ssmax,
            recompute=recompute,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            dim_feedforward=embed_dim * ff_factor,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            rope_interleaved=row_rope_interleaved,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            recompute=recompute,
        )

        bin_edges = torch.linspace(vmin, vmax, nbins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + 0.5 * bin_width

        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_width", bin_width)
        self.register_buffer("bin_centers", bin_centers)

        self.icl_predictor = ICLearning(
            d_model=icl_dim,
            num_blocks=icl_num_blocks,
            nhead=icl_nhead,
            dim_feedforward=icl_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=icl_ssmax,
            recompute=recompute,
            out_dim=nbins * 2,
        )

        # KV cache for efficient inference
        self._cache: Optional[TabICLCache] = None

    @property
    def has_cache(self) -> bool:
        """Check if a valid cache is stored."""
        return self._cache is not None and not self._cache.is_empty()

    def clear_cache(self) -> None:
        """Clear the stored cache."""
        self._cache = None

    def _get_d_total(self, d_X, d_Z, d_W):
        if d_X is not None and d_Z is not None and d_W is not None:
            return d_X + d_Z + d_W + 1  # RowInteraction adds num_cls itself
        return None

    def _get_y0_y1_shift_scale(self, y_context: Tensor, A_context: Tensor):
        """Calculates standardization stats separately for treated and control."""
        # Ensure A is squeezed to (B, train_size) to match y_context
        if A_context.dim() > 2:
            A_context = A_context.squeeze()

        # 1. Compute joint stats for all batches (used as fallback or if strictly enforced)
        joint_mean = y_context.mean(dim=1, keepdim=True)
        # unbiased=False matches the manual N-division variance calculation below
        joint_std = y_context.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        # If explicitly forced to use joint quantiles for ALL batches
        if getattr(self, "joint_quantiles", False):
            return joint_mean, joint_std, joint_mean, joint_std

        # 2. Compute separate stats
        treated_mask = (A_context == 1).float()
        control_mask = (A_context == 0).float()

        treated_counts = treated_mask.sum(dim=1, keepdim=True)
        control_counts = control_mask.sum(dim=1, keepdim=True)

        # 3. Create a fallback mask: True if EITHER group has fewer than 5 samples in that specific episode
        fallback_mask = (treated_counts < 5) | (control_counts < 5)

        # Clamp counts to avoid division by zero during the manual calculation
        safe_treated_counts = treated_counts.clamp(min=1.0)
        safe_control_counts = control_counts.clamp(min=1.0)

        y1_mean = (y_context * treated_mask).sum(
            dim=1, keepdim=True
        ) / safe_treated_counts
        y0_mean = (y_context * control_mask).sum(
            dim=1, keepdim=True
        ) / safe_control_counts

        y1_std = (
            torch.sqrt(
                ((y_context - y1_mean) ** 2 * treated_mask).sum(dim=1, keepdim=True)
                / safe_treated_counts
            )
            + 1e-8
        )
        y0_std = (
            torch.sqrt(
                ((y_context - y0_mean) ** 2 * control_mask).sum(dim=1, keepdim=True)
                / safe_control_counts
            )
            + 1e-8
        )

        # 4. Apply the fallback: Use joint stats if fallback_mask is True, else use separate stats
        y0_shift = torch.where(fallback_mask, joint_mean, y0_mean)
        y1_shift = torch.where(fallback_mask, joint_mean, y1_mean)

        y0_scale = torch.where(fallback_mask, joint_std, y0_std)
        y1_scale = torch.where(fallback_mask, joint_std, y1_std)

        return y0_shift, y0_scale, y1_shift, y1_scale

    def _train_forward(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A: Tensor,
        y_train: Tensor,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        Returns
        -------
        Tensor
            Predictions of shape (B, test_size, out_dim):

            - For regression (max_classes=0): out_dim = num_quantiles
            - For classification (max_classes>0): out_dim = max_classes
        """
        d_total = self._get_d_total(d_X, d_Z, d_W)

        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Column-wise embedding -> Row-wise interaction
        representations = self.row_interactor(
            self.col_embedder(
                X=X,
                Z=Z,
                W=W,
                A=A,
                y_train=y_train,
                d_X=d_X,
                d_Z=d_Z,
                d_W=d_W,
                embed_with_test=embed_with_test,
            ),
            d=d_total,
        )

        # Dataset-wise in-context learning
        return self.icl_predictor(representations, y_train=y_train)

    def _inference_forward(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A: Tensor,
        y_train: Tensor,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        embed_with_test: bool = False,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration.

        Returns
        -------
        Tensor
            For regression (max_classes=0):
                Predictions of shape (B, test_size, num_quantiles), where test_size = T - train_size

            For classification (max_classes>0):
                If return_logits=True: Logits of shape (B, test_size, num_classes)
                If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        train_size = y_train.shape[1]
        assert (
            train_size <= X.shape[1]
        ), "Number of training samples exceeds total samples"

        if inference_config is None:
            inference_config = InferenceConfig()

        # Column-wise embedding -> Row-wise interaction
        d_total = self._get_d_total(d_X, d_Z, d_W)
        representations = self.row_interactor(
            self.col_embedder(
                X=X,
                Z=Z,
                W=W,
                A=A,
                y_train=y_train,
                d_X=d_X,
                d_Z=d_Z,
                d_W=d_W,
                embed_with_test=embed_with_test,
                mgr_config=inference_config.COL_CONFIG,
            ),
            d=d_total,
            mgr_config=inference_config.ROW_CONFIG,
        )

        # Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            mgr_config=inference_config.ICL_CONFIG,
        )

        return out

    def forward(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A: Tensor,
        y_train: Tensor,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        embed_with_test: bool = False,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        embed_with_test : bool, default=False
            If True, allow training samples to attend to test samples during embedding.

        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns for each table in the batch. Used only in inference mode.
            When provided, indicates that X contains the same table with different feature orders.
            In this case, column-wise embeddings are computed once and then shuffled accordingly.

        return_logits : bool, default=True
            If True, return raw logits instead of probabilities. Used only in inference mode.

        softmax_temperature : float, default=0.9
            Temperature for the softmax function. Used only in inference mode.

        inference_config : Optional[InferenceConfig], default=None
            Inference configuration. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):

                - For regression (max_classes=0): out_dim = num_quantiles
                - For classification (max_classes>0): out_dim = max_classes

            For inference mode:
                For regression (max_classes=0):
                    Predictions of shape (B, test_size, num_quantiles)

                For classification (max_classes>0):
                    If return_logits=True: Logits of shape (B, test_size, num_classes)
                    If return_logits=False: Probabilities of shape (B, test_size, num_classes)
        """

        if self.training:
            out = self._train_forward(
                X, Z, W, A, y_train, d_X, d_Z, d_W, embed_with_test
            )
        else:
            out = self._inference_forward(
                X, Z, W, A, y_train, d_X, d_Z, d_W, embed_with_test, inference_config
            )

        return out

    @torch.no_grad()
    def predict(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A: Tensor,
        y_train: Tensor,
        target: Literal["y0", "y1", "cate"] = "cate",
        output_type: Literal["mean", "quantiles", "distribution"] = "mean",
        quantiles: Optional[list[float]] = None,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Union[Tensor, dict]:
        """
        Predicts potential outcomes or CATE given query features alongside context.
        Handles both batched sequences and single instances.

        Parameters
        ----------
        X, Z, W, A: Tensor
            The full sequence tensors (context + queries). Shapes should be (B, T, D).
        y_train: Tensor
            The context labels of shape (B, train_size).
        target: Literal["y0", "y1", "cate"]
            The causal estimand to extract.
        output_type: Literal["mean", "quantiles", "distribution"]
            The type of prediction output.
        quantiles: Optional[list[float]]
            List of quantiles to evaluate if output_type="quantiles". Default is [0.1, 0.5, 0.9].

        Returns
        -------
        Tensor or dict
            Depends on output_type.
        """
        # Ensure evaluation mode
        was_training = self.training
        self.eval()

        # 1. Forward pass to get logits for queries
        # Output shape: (B, test_size, nbins * 2)
        logits = self._inference_forward(
            X=X,
            Z=Z,
            W=W,
            A=A,
            y_train=y_train,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            inference_config=inference_config,
        )

        train_size = y_train.shape[1]
        A_context = A[:, :train_size]

        # 2. Extract standardisation stats from context to map bins back to the original scale
        y0_shift, y0_scale, y1_shift, y1_scale = self._get_y0_y1_shift_scale(
            y_train, A_context
        )

        # Reshape to (B, 1, 1) for broadcasting over test_size and nbins
        y0_shift, y0_scale = y0_shift.unsqueeze(-1), y0_scale.unsqueeze(-1)
        y1_shift, y1_scale = y1_shift.unsqueeze(-1), y1_scale.unsqueeze(-1)

        # 3. Separate logits and convert to probabilities
        probs_y0 = torch.softmax(logits[..., : self.nbins], dim=-1)
        probs_y1 = torch.softmax(logits[..., self.nbins :], dim=-1)

        # 4. Un-standardize bin centers: (B, 1, nbins)
        vals_y0 = self.bin_centers.view(1, 1, -1) * y0_scale + y0_shift
        vals_y1 = self.bin_centers.view(1, 1, -1) * y1_scale + y1_shift

        self.train(was_training)

        # --- Helper: Mean ---
        def _get_mean(probs, vals):
            return torch.sum(probs * vals, dim=-1)

        # --- Helper: Quantiles ---
        def _get_quantiles(probs, vals, q_list):
            cdf = torch.cumsum(probs, dim=-1)
            results = []
            for q in q_list:
                # Find bin indices where CDF crosses the quantile threshold
                idx = torch.searchsorted(cdf, q).clamp(max=self.nbins - 1)
                q_vals = torch.gather(vals, -1, idx.unsqueeze(-1)).squeeze(-1)
                results.append(q_vals)
            return torch.stack(results, dim=-1)

        # --- Output Routing ---
        if output_type == "distribution":
            if target == "y0":
                return {"values": vals_y0, "weights": probs_y0}
            elif target == "y1":
                return {"values": vals_y1, "weights": probs_y1}
            elif target == "cate":
                return {
                    "y0": {"values": vals_y0, "weights": probs_y0},
                    "y1": {"values": vals_y1, "weights": probs_y1},
                }

        elif output_type == "mean":
            mean_y0 = _get_mean(probs_y0, vals_y0)
            mean_y1 = _get_mean(probs_y1, vals_y1)

            if target == "y0":
                return mean_y0
            elif target == "y1":
                return mean_y1
            elif target == "cate":
                return mean_y1 - mean_y0

        elif output_type == "quantiles":
            if quantiles is None:
                quantiles = [0.1, 0.5, 0.9]

            q_tensor = torch.tensor(quantiles, device=logits.device, dtype=logits.dtype)
            q_y0 = _get_quantiles(probs_y0, vals_y0, q_tensor)
            q_y1 = _get_quantiles(probs_y1, vals_y1, q_tensor)

            if target == "y0":
                return q_y0
            elif target == "y1":
                return q_y1
            elif target == "cate":
                # Returns difference of marginal quantiles.
                return q_y1 - q_y0

        else:
            raise ValueError(f"Unknown output_type: {output_type}")
