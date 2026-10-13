####
# Changes vs TabICL:
# - Proximal Input Protocol: Expanded 'forward', '_train_forward', and
#   '_inference_forward' to accept the full causal quadruple: X (covariates),
#   Z (treatment-confounding proxies), W (outcome-confounding proxies), and A (treatment).
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
from .ensemble import FeatureEnsemble, apply_permutation, apply_value_transform

# Loss types sharing the single-histogram CATE head: same out_dim, same
# cate_bin_* grid, same predict path. They differ only in the training
# objective ("noise_aware" deconvolves the Monte Carlo noise on the
# target; see proximalfm.train._run.compute_cate_noise_aware_nll), which is why
# a "cate_single" checkpoint fine-tunes into the noise-aware one with no head
# surgery. Anything that branches on the head's *shape* must test membership
# here rather than equality with "cate_single".
CATE_SINGLE_LOSS_TYPES = ("cate_single", "noise_aware")

SUPPORTED_LOSS_TYPES = ("bins_po", "cate_decomposition", "hl", *CATE_SINGLE_LOSS_TYPES)

# Names this loss went by before. A checkpoint stores its `loss_type` string in
# `ckpt["config"]`, and every consumer that reconstructs a model from a
# checkpoint — the benchmark's `ProximalFM_Estimator`, `eval_datasets.prior_eval.report`,
# `visu/` — replays it through `__init__`. Renaming the loss therefore
# invalidates weights already on disk unless the old spelling still resolves;
# nothing about the architecture changed, only the label.
LEGACY_LOSS_TYPE_ALIASES = {"cate_single_noise_aware": "noise_aware"}


class ProximalFM(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    ProximalFM is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Dataset-wise in-context learning to learn patterns from labeled examples and make predictions

    Parameters
    ----------
    loss_type : Literal["bins_po", "hl", "cate_single", "noise_aware", "cate_decomposition"], default="bins_po"
        Determines the output space and learning objective:
        - "bins_po": Predicts binned distributions for both potential outcomes (y0, y1).
          Output dimension is `nbins * 2`.
        - "hl": Same output space as "bins_po" (`nbins * 2`); differs only in the training
          loss (HL-Gaussian cross-entropy instead of CRPS).
        - "cate_single": Predicts a single binned distribution directly over the treatment
          effect `tau = mu1 - mu0`, on its own bin grid (`cate_vmin`/`cate_vmax`).
          Output dimension is `nbins`. Only `target="cate"` is supported.
        - "noise_aware": Architecturally identical to "cate_single" (same
          out_dim, same grid, same predict path), so checkpoints are interchangeable.
          The training loss instead treats the supervised target as a *noisy* estimate
          of `tau` -- it is a Monte Carlo mean over `mc_cate` replicates -- and
          deconvolves that noise, so the head converges to a posterior over `tau`
          rather than to that posterior blurred by MC error.
        - "cate_decomposition": Predicts ATE as a Gaussian likelihood (mean, log-var)
          and the heterogeneous effect (CATE - ATE) as a binned distribution.
          Output dimension is `nbins + 2`.

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
        loss_type: Literal[
            "bins_po",
            "cate_decomposition",
            "hl",
            "cate_single",
            "noise_aware",
        ] = "bins_po",
        nbins: int = 1000,
        vmin: float = -10.0,
        vmax: float = 10.0,
        cate_vmin: float = -3.0,
        cate_vmax: float = 3.0,
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
        activation: Union[str, callable] = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        **kwargs,  # capture old parameters that are no longer used to avoid breaking old configs
    ):
        super().__init__()
        icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL

        loss_type = LEGACY_LOSS_TYPE_ALIASES.get(loss_type, loss_type)
        if loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type
        self.nbins = nbins
        self.vmin = vmin
        self.vmax = vmax
        self.cate_vmin = cate_vmin
        self.cate_vmax = cate_vmax
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

        if self.loss_type in CATE_SINGLE_LOSS_TYPES:
            # tau = mu1 - mu0 is much narrower than Y(a) on the standardized scale, so it
            # gets its own grid. Distinct buffer names matter: the surgical transfer loader
            # copies buffers by name+shape, so reusing "bin_edges" would let a two-head
            # checkpoint silently overwrite this grid with its own [vmin, vmax].
            cate_bin_edges = torch.linspace(cate_vmin, cate_vmax, nbins + 1)
            cate_bin_width = cate_bin_edges[1] - cate_bin_edges[0]
            cate_bin_centers = cate_bin_edges[:-1] + 0.5 * cate_bin_width

            self.register_buffer("cate_bin_edges", cate_bin_edges)
            self.register_buffer("cate_bin_width", cate_bin_width)
            self.register_buffer("cate_bin_centers", cate_bin_centers)

        if self.loss_type == "bins_po" or self.loss_type == "hl":
            out_dim = nbins * 2
        elif self.loss_type in CATE_SINGLE_LOSS_TYPES:
            out_dim = nbins
        else:  # self.loss_type == "cate_decomposition"
            out_dim = nbins + 2

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
            out_dim=out_dim,
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
            return d_X + d_Z + d_W  # RowInteraction adds num_cls itself
        return None

    def _safe_standardize(
        self,
        x: Tensor,
        train_size: int,
        fit_on_context_only: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Standardize x using context rows only by default.

        Returns:
            x_std, mean, std
        """
        fit_x = x[:, :train_size, :] if fit_on_context_only else x

        mu = fit_x.mean(dim=1, keepdim=True)
        std = fit_x.std(dim=1, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-5, torch.ones_like(std), std)

        return (x - mu) / std, mu, std

    def _hide_query_proxies(
        self,
        Z: Tensor,
        W: Tensor,
        train_size: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Hide Z/W for query rows.

        Important: call this AFTER standardization, so hidden query proxy
        values become exactly 0 in standardized space.
        """
        if Z.shape[1] > train_size:
            Z = Z.clone()
            Z[:, train_size:, :] = 0.0

        if W.shape[1] > train_size:
            W = W.clone()
            W[:, train_size:, :] = 0.0

        return Z, W

    def _standardize_forward_inputs(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        y_train: Tensor,
        train_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Standardize all model inputs in a no-leak way.

        X:
            Uses context-only stats. This is the strict no-transductive version.
            If you later decide that using all X is acceptable, change only this line.

        Z/W:
            Always use context-only stats, then hide query rows.

        y_train:
            Uses context-only stats because y only exists for context rows.
        """
        X_std, _, _ = self._safe_standardize(
            X,
            train_size=train_size,
            fit_on_context_only=True,
        )

        Z_std, _, _ = self._safe_standardize(
            Z,
            train_size=train_size,
            fit_on_context_only=True,
        )

        W_std, _, _ = self._safe_standardize(
            W,
            train_size=train_size,
            fit_on_context_only=True,
        )

        Z_std, W_std = self._hide_query_proxies(Z_std, W_std, train_size)

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
        y_train_std = (y_train - y_mean) / y_std

        return X_std, Z_std, W_std, y_train_std, y_mean, y_std

    def _train_forward(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A_train: Tensor,
        y_train: Tensor,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        embed_with_test: bool = False,
    ) -> Tensor:
        d_total = self._get_d_total(d_X, d_Z, d_W)

        B, T, H = X.shape
        train_size = y_train.shape[1]

        col_embeddings = self.col_embedder(
            X=X,
            Z=Z,
            W=W,
            A_train=A_train,
            y_train=y_train,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            embed_with_test=embed_with_test,
        )

        representations = self.row_interactor(
            col_embeddings,
            d=d_total,
            train_size=train_size,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            hide_query_proxies=True,
        )

        # Dataset-wise in-context learning
        return self.icl_predictor(representations, y_train=y_train, A_train=A_train)

    def _inference_forward(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A_train: Tensor,
        y_train: Tensor,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        embed_with_test: bool = False,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Tensor:
        """Inference forward pass handling configuration routing and KV caching."""
        if inference_config is None:
            inference_config = InferenceConfig()

        d_total = self._get_d_total(d_X, d_Z, d_W)
        train_size = y_train.shape[1]

        # 1. Column-wise embedding
        col_embeddings = self.col_embedder(
            X=X,
            Z=Z,
            W=W,
            A_train=A_train,
            y_train=y_train,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            embed_with_test=embed_with_test,
            mgr_config=inference_config.COL_CONFIG,
        )

        # 2. Row-wise interaction
        representations = self.row_interactor(
            col_embeddings,
            d=d_total,
            train_size=train_size,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            hide_query_proxies=True,
            mgr_config=inference_config.ROW_CONFIG,
        )

        # 3. Dataset-wise in-context learning
        out = self.icl_predictor(
            representations,
            y_train=y_train,
            A_train=A_train,
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
        train_size = y_train.shape[1]
        B, T, _ = X.shape

        assert train_size <= T, "Number of training samples exceeds total samples"

        # Important: all training and inference calls now go through this.
        X, Z, W, y_train, _, _ = self._standardize_forward_inputs(
            X=X,
            Z=Z,
            W=W,
            y_train=y_train,
            train_size=train_size,
        )

        if A.shape[1] == T:
            A_train = A[:, :train_size]
        else:
            A_train = A

        if self.training:
            out = self._train_forward(
                X, Z, W, A_train, y_train, d_X, d_Z, d_W, embed_with_test
            )
        else:
            out = self._inference_forward(
                X,
                Z,
                W,
                A_train,
                y_train,
                d_X,
                d_Z,
                d_W,
                embed_with_test,
                inference_config,
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
        was_training = self.training
        self.eval()

        y_context_mean = y_train.mean(dim=1, keepdim=True)
        y_context_std = y_train.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

        logits = self.forward(
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

        self.train(was_training)

        y_mean_view = y_context_mean.unsqueeze(-1)
        y_std_view = y_context_std.unsqueeze(-1)

        # --- Helper: Mean ---
        def _get_mean(probs, vals):
            return torch.sum(probs * vals, dim=-1)

        # --- Helper: Quantiles ---
        def _get_quantiles(probs, vals, q_list):
            # Assumes that vals and probs can be correctly gathered/expanded
            cdf = torch.cumsum(probs, dim=-1)
            results = []

            # Ensures vals dimension matches before gathering
            # if dynamically broadcasted from shape (B, 1, nbins)
            if vals.shape[1] == 1 and probs.shape[1] > 1:
                vals = vals.expand(-1, probs.shape[1], -1)

            for q in q_list:
                # searchsorted only accepts a scalar when the boundaries are 1-D, so
                # broadcast the level to one query per (batch, row).
                q_expanded = q.expand(*cdf.shape[:-1], 1).contiguous()
                idx = torch.searchsorted(cdf, q_expanded).clamp(max=self.nbins - 1)
                q_vals = torch.gather(vals, -1, idx).squeeze(-1)
                results.append(q_vals)
            return torch.stack(results, dim=-1)

        # ==========================================
        # ROUTING: Original architecture (Bins PO)
        # ==========================================
        if self.loss_type == "bins_po" or self.loss_type == "hl":
            probs_y0 = torch.softmax(logits[..., : self.nbins], dim=-1)
            probs_y1 = torch.softmax(logits[..., self.nbins :], dim=-1)

            vals = self.bin_centers.view(1, 1, -1) * y_std_view + y_mean_view

            if output_type == "distribution":
                if target == "y0":
                    return {"values": vals, "weights": probs_y0}
                elif target == "y1":
                    return {"values": vals, "weights": probs_y1}
                elif target == "cate":
                    return {
                        "y0": {"values": vals, "weights": probs_y0},
                        "y1": {"values": vals, "weights": probs_y1},
                    }

            elif output_type == "mean":
                mean_y0 = _get_mean(probs_y0, vals)
                mean_y1 = _get_mean(probs_y1, vals)

                if target == "y0":
                    return mean_y0
                elif target == "y1":
                    return mean_y1
                elif target == "cate":
                    return mean_y1 - mean_y0

            elif output_type == "quantiles":
                if quantiles is None:
                    quantiles = [0.1, 0.5, 0.9]
                q_tensor = torch.tensor(
                    quantiles, device=logits.device, dtype=logits.dtype
                )

                q_y0 = _get_quantiles(probs_y0, vals, q_tensor)
                q_y1 = _get_quantiles(probs_y1, vals, q_tensor)

                if target == "y0":
                    return q_y0
                elif target == "y1":
                    return q_y1
                elif target == "cate":
                    return q_y1 - q_y0

            else:
                raise ValueError(f"Unknown output_type: {output_type}")

        # ==========================================
        # ROUTING: Additive CATE Decomposition
        # ==========================================
        elif self.loss_type == "cate_decomposition":
            if target not in ["cate"]:
                raise ValueError(
                    f"loss_type 'cate_decomposition' directly estimates treatment effects "
                    f"and does not support predicting target='{target}'."
                )

            # Dim 0: ATE Mean, Dim 1: ATE log_var (used in loss, skipped here), Dim 2+: Heterogeneous Effect Logits
            ate_mu_scaled = logits[..., 0]
            probs_he = torch.softmax(logits[..., 2:], dim=-1)

            ate_mu = ate_mu_scaled * y_context_std
            vals_he = self.bin_centers.view(1, 1, -1) * y_std_view

            # CATE Distribution = Heterogeneous effect distribution anchored on the predicted ATE
            vals_cate = vals_he + ate_mu.unsqueeze(-1)

            if output_type == "distribution":
                return {"values": vals_cate, "weights": probs_he}

            elif output_type == "mean":
                return _get_mean(probs_he, vals_cate)

            elif output_type == "quantiles":
                if quantiles is None:
                    quantiles = [0.1, 0.5, 0.9]
                q_tensor = torch.tensor(
                    quantiles, device=logits.device, dtype=logits.dtype
                )

                return _get_quantiles(probs_he, vals_cate, q_tensor)

            else:
                raise ValueError(f"Unknown output_type: {output_type}")

        # ==========================================
        # ROUTING: Single CATE histogram
        # ==========================================
        elif self.loss_type in CATE_SINGLE_LOSS_TYPES:
            if target not in ["cate"]:
                raise ValueError(
                    f"loss_type '{self.loss_type}' directly estimates treatment effects "
                    f"and does not support predicting target='{target}'."
                )

            probs_cate = torch.softmax(logits, dim=-1)

            # tau is a difference, so the context mean cancels: de-standardize with the
            # context scale ONLY. Adding y_mean_view here would bias the whole CATE.
            vals_cate = self.cate_bin_centers.view(1, 1, -1) * y_std_view

            if output_type == "distribution":
                return {"values": vals_cate, "weights": probs_cate}

            elif output_type == "mean":
                return _get_mean(probs_cate, vals_cate)

            elif output_type == "quantiles":
                if quantiles is None:
                    quantiles = [0.1, 0.5, 0.9]
                q_tensor = torch.tensor(
                    quantiles, device=logits.device, dtype=logits.dtype
                )

                return _get_quantiles(probs_cate, vals_cate, q_tensor)

            else:
                raise ValueError(f"Unknown output_type: {output_type}")

        else:
            raise ValueError(f"Unhandled loss_type: {self.loss_type}")

    @staticmethod
    def _average_ensemble_outputs(outputs: list) -> Union[Tensor, dict]:
        """Average a list of ``predict`` outputs across ensemble members.

        Handles the three output shapes returned by :meth:`predict`:
        - plain ``Tensor`` (``output_type`` in {"mean", "quantiles"}): stacked
          and averaged over ensemble members.
        - a ``{"values", "weights"}`` distribution dict: the bin ``values`` are
          identical across feature permutations (they depend only on the y-context
          statistics and bin centers, not on the ordering of X columns), so the
          ``weights`` (probabilities) are averaged and the ``values`` of the first
          member are kept.
        - a nested ``{"y0": {...}, "y1": {...}}`` dict: averaged recursively.
        """
        first = outputs[0]

        if isinstance(first, Tensor):
            return torch.stack(outputs, dim=0).mean(dim=0)

        if isinstance(first, dict):
            # Distribution outputs: recurse if nested (y0/y1), otherwise average weights.
            if "weights" in first:
                weights = torch.stack([o["weights"] for o in outputs], dim=0).mean(
                    dim=0
                )
                return {"values": first["values"], "weights": weights}
            return {
                key: ProximalFM._average_ensemble_outputs([o[key] for o in outputs])
                for key in first
            }

        raise TypeError(f"Cannot average ensemble output of type {type(first)}.")

    @torch.no_grad()
    def predict_ensemble(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A: Tensor,
        y_train: Tensor,
        target: Literal["y0", "y1", "cate"] = "cate",
        output_type: Literal["mean", "quantiles", "distribution"] = "mean",
        quantiles: Optional[list[float]] = None,
        n_estimators: int = 8,
        feat_shuffle_method: Literal["none", "shift", "random", "latin"] = "latin",
        value_transforms: Optional[list[str]] = None,
        shuffle_X: bool = True,
        shuffle_Z: bool = False,
        shuffle_W: bool = False,
        transform_X: bool = True,
        transform_Z: bool = False,
        transform_W: bool = False,
        random_state: Optional[int] = 42,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> Union[Tensor, dict]:
        """Ensemble prediction by averaging over feature-transformed views.

        Creates ``n_estimators`` alternative views of the inputs, runs
        :meth:`predict` on each, and averages the resulting estimates. Each view
        combines two information-preserving levers:

        1. **Column permutation** of the X/Z/W blocks. Since
           :class:`RowInteraction` uses rotary position embeddings over feature
           positions, the model is only approximately invariant to column
           ordering, so different orderings give diverse-but-valid estimates.
        2. **A bijective per-column value transform** (``value_transforms``:
           ``"sign"``, ``"signed_log"``, ``"arcsinh"``, ``"quantile"``), applied
           consistently to context and query rows. These re-express the features
           in a warped basis without losing information, adding a stronger form
           of diversity than permutation alone.

        Both levers are label-preserving, so averaging reduces variance. Only
        valid feature columns are permuted (padding beyond ``d_X``/``d_Z``/
        ``d_W`` is left in place), so per-batch feature counts pass through
        unchanged.

        Parameters
        ----------
        X, Z, W, A, y_train : Tensor
            Same inputs as :meth:`predict`.

        target, output_type, quantiles : see :meth:`predict`.

        n_estimators : int, default=8
            Number of views to average over. ``n_estimators=1`` reproduces the
            plain :meth:`predict` (canonical view only).

        feat_shuffle_method : {"none", "shift", "random", "latin"}, default="latin"
            Column-permutation strategy (see :class:`FeatureEnsemble`).

        value_transforms : list of str or None, default=None
            Bijective value transforms to cycle across views (see
            :data:`~proximalfm._model.ensemble.VALUE_TRANSFORMS`). ``None`` means
            permutation-only.

        shuffle_X, shuffle_Z, shuffle_W : bool
            Which blocks to permute across views (default: X only).

        transform_X, transform_Z, transform_W : bool
            Which blocks the value transform is applied to (default: X only).

        random_state : int or None, default=42
            Seed for reproducible permutations.

        d_X, d_Z, d_W, inference_config : see :meth:`predict`.

        Returns
        -------
        Tensor or dict
            The ensemble-averaged prediction, with the same structure as the
            corresponding single-view :meth:`predict` output.
        """
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}.")

        ensembler = FeatureEnsemble(
            n_estimators=n_estimators,
            feat_shuffle_method=feat_shuffle_method,
            value_transforms=value_transforms,
            shuffle_X=shuffle_X,
            shuffle_Z=shuffle_Z,
            shuffle_W=shuffle_W,
            transform_X=transform_X,
            transform_Z=transform_Z,
            transform_W=transform_W,
            random_state=random_state,
        )
        views = ensembler.generate(X.shape[2], Z.shape[2], W.shape[2])
        train_size = y_train.shape[1]

        outputs = []
        for view in views:
            transform = view["transform"]
            # Value transform (bijective, per-column) on the enabled blocks, then
            # permute columns. The two commute (transform is per-column), so the
            # order is immaterial.
            X_v = (
                X
                if not transform_X
                else apply_value_transform(X, transform, train_size)
            )
            Z_v = (
                Z
                if not transform_Z
                else apply_value_transform(Z, transform, train_size)
            )
            W_v = (
                W
                if not transform_W
                else apply_value_transform(W, transform, train_size)
            )

            X_v = apply_permutation(X_v, view["X"], d_X)
            Z_v = apply_permutation(Z_v, view["Z"], d_Z)
            W_v = apply_permutation(W_v, view["W"], d_W)

            out = self.predict(
                X=X_v,
                Z=Z_v,
                W=W_v,
                A=A,
                y_train=y_train,
                target=target,
                output_type=output_type,
                quantiles=quantiles,
                d_X=d_X,
                d_Z=d_Z,
                d_W=d_W,
                inference_config=inference_config,
            )
            outputs.append(out)

        return self._average_ensemble_outputs(outputs)
