from __future__ import annotations

from typing import Optional, Union
from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layers import SkippableLinear
from .encoders import SetTransformer
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class ColEmbedding(nn.Module):
    """Distribution-aware column-wise embedding.

    This module maps each scalar cell in a column to a high-dimensional embedding while
    capturing statistical regularities within the column. Unlike traditional approaches
    that use separate embedding layers per column, it employs a shared set transformer
    to process all features.

    ColEmbedding operates in two modes depending on the `affine` parameter:

    When ``affine=True``:

    1. Each scalar cell is first linearly projected into the embedding dimension
    2. The set transformer generates distribution-aware weights and biases for each column
    3. The final column embeddings are computed as: :math:`\\text{column} \\times W + b`

    When ``affine=False``:

    1. Each scalar cell is first linearly projected into the embedding dimension
    2. The set transformer processes the projected features
    3. The final column embeddings are directly the set transformer's output

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of induced self-attention blocks used in the set transformer.

    nhead : int
        Number of attention heads of the set transformer.

    dim_feedforward : int
        Dimension of the feedforward network of the set transformer.

    num_inds : int
        Number of inducing points used in self-attention blocks of the set transformer.

    dropout : float, default=0.0
        Dropout probability used in the set transformer.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    affine : bool, default=True
        If True, computes embeddings as: :math:`\\text{features} \\times W + b`.
        If False, directly uses the set transformer output as embeddings.

    reserve_cls_tokens : int, default=4
        Number of slots to reserve for CLS tokens to avoid concatenation.

    ssmax : bool or str, default=False
        Type of scalable softmax to use in attention. Note that only the first attention layer of
        the induced self-attention blocks uses SSMax.
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

        When enabled and num_classes > max_classes, labels are decomposed into D digits
        via mixed-radix representation with balanced bases :math:`[k_0, \\ldots, k_{D-1}]` where each
        :math:`k_i \\leq` max_classes and :math:`\\prod_i k_i \\geq` num_classes. Each digit defines a coarser
        grouping of the original classes. The set transformer is run once per digit, and the
        final outputs are averaged across all digits.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        affine: bool = True,
        reserve_cls_tokens: int = 4,
        ssmax: Union[bool, str] = False,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.reserve_cls_tokens = reserve_cls_tokens
        self.affine = affine

        self.in_linear = SkippableLinear(1, embed_dim)

        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            recompute=recompute,
        )

        self.y0_encoder = nn.Linear(1, embed_dim)
        self.y1_encoder = nn.Linear(1, embed_dim)

        if affine:
            self.out_w = SkippableLinear(embed_dim, embed_dim)
            self.ln_w = (
                nn.LayerNorm(embed_dim, bias=not bias_free_ln)
                if norm_first
                else nn.Identity()
            )

            self.out_b = SkippableLinear(embed_dim, embed_dim)
            self.ln_b = (
                nn.LayerNorm(embed_dim, bias=not bias_free_ln)
                if norm_first
                else nn.Identity()
            )

        self.inference_mgr = InferenceManager(enc_name="tf_col", out_dim=embed_dim)

        self.role_embedding = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)

    def y_encoder(self, y: Tensor, A: Tensor) -> Tensor:
        """Encodes the target variable y and treatment A into embeddings."""
        y0_emb = self.y0_encoder(y.unsqueeze(-1))
        y1_emb = self.y1_encoder(y.unsqueeze(-1))
        y_emb = torch.where(A.unsqueeze(-1) == 0, y0_emb, y1_emb)
        return y_emb

    def _get_invalid_mask(
        self, max_len: int, lengths: Optional[Tensor], B: int, device: torch.device
    ) -> Tensor:
        """Returns a boolean mask of shape (B, max_len) where True = Invalid Padding."""
        if lengths is None:
            return torch.zeros(B, max_len, dtype=torch.bool, device=device)
        indices = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)
        return indices >= lengths.unsqueeze(1)

    def _compute_embeddings(
        self,
        X: Tensor,
        Z: Tensor,
        W: Tensor,
        A_train: Tensor,
        train_size: int,
        y_train: Optional[Tensor],
        inv_concat: Tensor,
        embed_with_test: bool = False,
    ) -> Tensor:
        """Feature embedding using a shared set transformer with Causal Roles.

        Assumes X, Z, W, A have been transposed/unsqueezed to shape:
        (B, num_features, T, 1)
        """
        device = X.device

        src_X = self.in_linear(X) + self.role_embedding(torch.tensor(0, device=device))
        src_Z = self.in_linear(Z) + self.role_embedding(torch.tensor(1, device=device))
        src_W = self.in_linear(W) + self.role_embedding(torch.tensor(2, device=device))

        src = torch.cat([src_X, src_Z, src_W], dim=1)

        raw_features = torch.cat([X, Z, W], dim=1)

        assert y_train is not None, "y_train must be provided."

        A_context = A_train[:, 0, :train_size, 0]
        y_emb = self.y_encoder(y_train, A_context)
        y_emb_broadcast = y_emb.unsqueeze(1)

        src[..., :train_size, :] = src[..., :train_size, :] + y_emb_broadcast

        inv_mask_exp = inv_concat.unsqueeze(-1).unsqueeze(-1).expand_as(src)
        src[inv_mask_exp] = -100.0

        src = self.tf_col(src, train_size=None if embed_with_test else train_size)

        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = raw_features * weights + biases
        else:
            embeddings = src

        embeddings[inv_mask_exp] = -100.0

        valid_mask = ~inv_concat
        sort_indices = torch.argsort(
            valid_mask.int(), dim=1, descending=True, stable=True
        )

        gather_indices = sort_indices.unsqueeze(-1).unsqueeze(-1).expand_as(embeddings)
        embeddings_sorted = torch.gather(embeddings, dim=1, index=gather_indices)

        return embeddings_sorted

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
        """Transform input table into embeddings for training.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)

        Z : Tensor
            Tensor of the same shape as X representing the "Z" variables.

        W : Tensor
            Tensor of the same shape as X representing the "W" variables.

        A : Tensor
            Tensor of the same shape as X representing the "A" variables.

        y_train : Tensor
            Target values for training samples of shape (B, train_size).
            Used only for target-aware embedding.

        d : Optional[Tensor], default=None
            The number of features per dataset of shape (B,).
            Only supported when feature grouping is disabled.

        embed_with_test : bool, default=False
            If True, inducing points attend to all samples (train + test).
            If False, inducing points only attend to training samples.

        Returns
        -------
        Tensor
            Embeddings of shape (B, T, G+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - G is the number of feature groups
             - C is the number of class tokens
             - E is embedding dimension.
        """

        train_size = y_train.shape[1]
        B = X.shape[0]

        X, Z, W, A_train = X.float(), Z.float(), W.float(), A_train.float()

        if A_train.dim() == 2:
            A_train = A_train.unsqueeze(-1)

        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)
            if d_X is not None:
                d_X = d_X + self.reserve_cls_tokens

        inv_X = self._get_invalid_mask(X.shape[2], d_X, B, X.device)
        inv_Z = self._get_invalid_mask(Z.shape[2], d_Z, B, Z.device)
        inv_W = self._get_invalid_mask(W.shape[2], d_W, B, W.device)
        inv_concat = torch.cat([inv_X, inv_Z, inv_W], dim=1)

        X_feat = X.transpose(1, 2).unsqueeze(-1)
        Z_feat = Z.transpose(1, 2).unsqueeze(-1)
        W_feat = W.transpose(1, 2).unsqueeze(-1)
        A_feat = A_train.transpose(1, 2).unsqueeze(-1)

        embeddings = self._compute_embeddings(
            X_feat,
            Z_feat,
            W_feat,
            A_feat,
            train_size,
            y_train,
            inv_concat,
            embed_with_test,
        )

        return embeddings.transpose(1, 2)

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
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = InferenceConfig().COL_CONFIG
        self.inference_mgr.configure(**mgr_config)

        train_size = y_train.shape[1]
        B = X.shape[0]

        X, Z, W, A_train = X.float(), Z.float(), W.float(), A_train.float()

        if A_train.dim() == 2:
            A_train = A_train.unsqueeze(-1)

        if self.reserve_cls_tokens > 0:
            X = F.pad(X, (self.reserve_cls_tokens, 0), value=-100.0)
            if d_X is not None:
                d_X = d_X + self.reserve_cls_tokens

        inv_X = self._get_invalid_mask(X.shape[2], d_X, B, X.device)
        inv_Z = self._get_invalid_mask(Z.shape[2], d_Z, B, Z.device)
        inv_W = self._get_invalid_mask(W.shape[2], d_W, B, W.device)
        inv_concat = torch.cat([inv_X, inv_Z, inv_W], dim=1)

        X_feat = X.transpose(1, 2).unsqueeze(-1)
        Z_feat = Z.transpose(1, 2).unsqueeze(-1)
        W_feat = W.transpose(1, 2).unsqueeze(-1)
        A_feat = A_train.transpose(1, 2).unsqueeze(-1)

        embeddings = self.inference_mgr(
            self._compute_embeddings,
            inputs=OrderedDict(
                [
                    ("X", X_feat),
                    ("Z", Z_feat),
                    ("W", W_feat),
                    ("A_train", A_feat),
                    ("train_size", train_size),
                    ("y_train", y_train),
                    ("inv_concat", inv_concat),
                    ("embed_with_test", embed_with_test),
                ]
            ),
        )

        return embeddings.transpose(1, 2)

    def forward(
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
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if self.training:
            embeddings = self._train_forward(
                X, Z, W, A_train, y_train, d_X, d_Z, d_W, embed_with_test
            )
        else:
            embeddings = self._inference_forward(
                X, Z, W, A_train, y_train, d_X, d_Z, d_W, embed_with_test, mgr_config
            )

        return embeddings
