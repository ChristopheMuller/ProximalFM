from __future__ import annotations

from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of blocks used in the encoder.

    nhead : int
        Number of attention heads of the encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the encoder.

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row.

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    dropout : float, default=0.0
        Dropout probability used in the encoder.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.recompute = recompute

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            recompute=recompute,
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.out_ln = (
            nn.LayerNorm(embed_dim, bias=not bias_free_ln)
            if norm_first
            else nn.Identity()
        )
        self.inference_mgr = InferenceManager(
            enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True
        )

    def _aggregate_embeddings(
        self, embeddings: Tensor, key_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Process a batch of rows through a transformer encoder.

        This method:

        1. Processes embeddings through the transformer
        2. Extracts only the class token representations and applies normalization if pre-norm
        3. Concatenates the class tokens into a single vector per row

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings of shape (B, T, H+C, E) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features
             - C is the number of class tokens
             - E is the embedding dimension

        key_mask : Optional[Tensor], default=None
            Boolean mask of shape (B, T, H+C) where True indicates positions
            to ignore during attention (empty feature slots).

        Returns
        -------
        Tensor
            Flattened class token outputs of shape (B*T, C*E).
        """
        rope = self.tf_row.rope

        # Process all blocks except the last
        if self.recompute:
            for block in self.tf_row.blocks[:-1]:
                embeddings = checkpoint(
                    partial(block, key_padding_mask=key_mask, rope=rope),
                    embeddings,
                    use_reentrant=False,
                )
        else:
            for block in self.tf_row.blocks[:-1]:
                embeddings = block(embeddings, key_padding_mask=key_mask, rope=rope)

        # Last block: q = CLS tokens, k/v = full sequence
        last_block = self.tf_row.blocks[-1]
        if self.recompute:
            cls_outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :],
                    k=emb,
                    v=emb,
                    key_padding_mask=key_mask,
                    rope=rope,
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            cls_outputs = last_block(
                q=embeddings[..., : self.num_cls, :],
                k=embeddings,
                v=embeddings,
                key_padding_mask=key_mask,
                rope=rope,
            )
        del embeddings
        cls_outputs = self.out_ln(cls_outputs)

        return cls_outputs.flatten(-2)  # (B, T, C*E)

    def _generate_key_mask(
        self,
        HC: int,
        T: int,
        B: int,
        device: torch.device,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        hide_query_proxies: bool = False,
    ) -> Optional[Tensor]:
        """
        Build row-wise key padding mask.

        Shape:
            key_mask: (B, T, HC)

        Semantics:
            True = masked / cannot be attended to.

        Feature layout assumed after ColEmbedding sorting:
            [CLS slots, X valid, Z valid, W valid, padding]

        where the CLS slots are the first self.num_cls positions.
        """
        key_mask = torch.zeros(B, T, HC, dtype=torch.bool, device=device)

        indices = torch.arange(HC, device=device).view(1, 1, HC).expand(B, T, HC)

        # 1. Standard padding mask: mask positions beyond valid feature count.
        if d is not None:
            d = d.to(device=device)
            valid_count = d + self.num_cls
            key_mask = key_mask | (indices >= valid_count.view(B, 1, 1))

        # 2. Proximal restriction:
        #    query/test rows should not attend to Z/W feature positions.
        if hide_query_proxies:
            if train_size is None:
                raise ValueError(
                    "train_size must be provided when hide_query_proxies=True."
                )
            if d_X is None or d_Z is None or d_W is None:
                raise ValueError(
                    "d_X, d_Z, and d_W must be provided when hide_query_proxies=True."
                )

            d_X = d_X.to(device=device)
            d_Z = d_Z.to(device=device)
            d_W = d_W.to(device=device)

            # Layout after ColEmbedding:
            # [CLS, X, Z, W, padding]
            z_start = self.num_cls + d_X
            w_end = self.num_cls + d_X + d_Z + d_W

            proxy_positions = (indices >= z_start.view(B, 1, 1)) & (
                indices < w_end.view(B, 1, 1)
            )

            row_indices = torch.arange(T, device=device).view(1, T, 1)
            query_rows = row_indices >= train_size

            key_mask = key_mask | (query_rows & proxy_positions)

        # If nothing is masked, returning None is slightly cleaner.
        if not key_mask.any():
            return None

        return key_mask

    def _train_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        hide_query_proxies: bool = False,
    ) -> Tensor:
        B, T, HC, E = embeddings.shape
        device = embeddings.device

        # Safer than in-place mutation of the ColEmbedding output.
        embeddings = embeddings.clone()

        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls, :] = cls_tokens.to(device)

        key_mask = self._generate_key_mask(
            HC=HC,
            T=T,
            B=B,
            device=device,
            d=d,
            train_size=train_size,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            hide_query_proxies=hide_query_proxies,
        )

        representations = self._aggregate_embeddings(
            embeddings,
            key_mask,
        )

        return representations

    def _inference_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        hide_query_proxies: bool = False,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG
        self.inference_mgr.configure(**mgr_config)

        B, T, HC, E = embeddings.shape
        device = embeddings.device

        embeddings = embeddings.clone()

        cls_tokens = self.cls_tokens.expand(B, T, self.num_cls, self.embed_dim)
        embeddings[:, :, : self.num_cls, :] = cls_tokens.to(device)

        key_mask = self._generate_key_mask(
            HC=HC,
            T=T,
            B=B,
            device=device,
            d=d,
            train_size=train_size,
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
            hide_query_proxies=hide_query_proxies,
        )

        representations = self.inference_mgr(
            self._aggregate_embeddings,
            inputs=OrderedDict(
                [
                    ("embeddings", embeddings),
                    ("key_mask", key_mask),
                ]
            ),
        )

        return representations

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        train_size: Optional[int] = None,
        d_X: Optional[Tensor] = None,
        d_Z: Optional[Tensor] = None,
        d_W: Optional[Tensor] = None,
        hide_query_proxies: bool = False,
        mgr_config: MgrConfig = None,
    ) -> Tensor:
        if self.training:
            representations = self._train_forward(
                embeddings=embeddings,
                d=d,
                train_size=train_size,
                d_X=d_X,
                d_Z=d_Z,
                d_W=d_W,
                hide_query_proxies=hide_query_proxies,
            )
        else:
            representations = self._inference_forward(
                embeddings=embeddings,
                d=d,
                train_size=train_size,
                d_X=d_X,
                d_Z=d_Z,
                d_W=d_W,
                hide_query_proxies=hide_query_proxies,
                mgr_config=mgr_config,
            )

        return representations
