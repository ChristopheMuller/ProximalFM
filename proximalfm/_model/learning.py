####
# Changes vs TabICL:
# - Regression Specialization: Excised all classification-specific logic,
#   including hierarchical classification ('ClassNode'), grouping, and
#   label encoding.
# - Unified Target Encoder: Standardized 'y_encoder' as 'nn.Linear(1, d_model)',
#   removing the 'OneHotAndLinear' path previously used for classification.
# - Inference Streamlining: Simplified '_predict_standard' and '_inference_forward'
#   by removing 'softmax_temperature', 'return_logits', and multi-class
#   slicing logic.
# - Changed the y_encoder to take in both y and A, and use A to determine which linear layer to
#    use for encoding the target values. This allows the model to learn separate embeddings for
#    treated vs control samples, which is important for causal inference tasks.
# - Cleaned API: Removed 'max_classes' and 'out_dim' from the constructor,
#   as the task type is now strictly continuous proximal learning.
# - Caching Logic: Simplified 'forward_with_cache' and 'forward_with_repr_cache'
#   to handle only regression outputs without verifying class counts or
#   hierarchical depth.
####


from __future__ import annotations

from typing import Optional, Union
from collections import OrderedDict
from torch import nn, Tensor, where

from .encoders import Encoder
from .kv_cache import KVCache
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ICLearning(nn.Module):
    """Dataset-wise in-context learning.

    Parameters
    ----------
    d_model : int
        Model dimension.

    num_blocks : int
        Number of blocks used in the ICL encoder.

    nhead : int
        Number of attention heads of the ICL encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the ICL encoder.

    dropout : float, default=0.0
        Dropout probability.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in the ICL encoder.
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

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
        out_dim: int = 999,
    ):
        super().__init__()

        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            recompute=recompute,
        )
        if self.norm_first:
            self.ln = nn.LayerNorm(d_model, bias=not bias_free_ln)

        self.y_encoder = nn.Linear(1, d_model)
        self.a_encoder = nn.Linear(1, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, out_dim)
        )
        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=out_dim)

    def _icl_predictions(self, R: Tensor, y_train: Tensor, A: Tensor) -> Tensor:
        """In-context learning predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        Returns
        -------
        Tensor
            Predictions of shape (B, T, out_dim) #out-dim = 4

        """

        train_size = y_train.shape[1]

        # 1. Shape normalization for Treatment A
        if A.dim() == 2:
            A = A.unsqueeze(-1)  # (B, T) -> (B, T, 1)

        # 2. Apply A-encoder across the ENTIRE sequence length (Context + Test)
        a_emb = self.a_encoder(A.float())
        src = R + a_emb

        # 3. Add universal Target outcome embeddings strictly to the context rows
        y_emb = self.y_encoder(y_train.unsqueeze(-1))
        src[:, :train_size] = src[:, :train_size] + y_emb

        # 4. Dataset-wide context parsing
        src = self.tf_icl(src, train_size=train_size)
        if self.norm_first:
            src = self.ln(src)
        out = self.decoder(src)

        return out

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        A: Tensor,
        auto_batch: bool = True,
    ) -> Tensor:
        """Generate predictions.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches.

        Returns
        -------
        Tensor
            Predictions of shape (B, test_size, 4), where test_size = T - train_size; 4 = dim of output (mean + log-var for two potential outcomes)

        """

        out = self.inference_mgr(
            self._icl_predictions,
            inputs=OrderedDict([("R", R), ("y_train", y_train), ("A", A)]),
            auto_batch=auto_batch,
        )

        train_size = y_train.shape[1]
        out = out[:, train_size:]

        return out

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        A: Tensor,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager.

        Returns
        -------
        Tensor
            For regression:
                Predictions of shape (B, test_size, 4), where test_size = T - train_size; 4 = dim of output (mean + log-var for two potential outcomes)

        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = InferenceConfig().ICL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        out = self._predict_standard(R, y_train, A=A)
        return out

    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        A: Tensor,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        """In-context learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        y_train : Tensor
            Training targets of shape (B, train_size), where train_size is the position
            to split the input into training and test data.

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.

        Returns
        -------
        Tensor
            For training mode:
                Predictions of shape (B, test_size, out_dim):
                - For regression: out_dim = 4

            For inference mode:
                Predictions of shape (B, test_size, 4); 4 = dim of output (mean + log-var for two potential outcomes)

        """

        if self.training:
            train_size = y_train.shape[1]
            out = self._icl_predictions(R, y_train, A=A)
            out = out[:, train_size:]
        else:
            out = self._inference_forward(R, y_train, A=A, mgr_config=mgr_config)

        return out
