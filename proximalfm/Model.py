from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

import einops
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabpfn.model.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from tabpfn.model.layer import PerFeatureEncoderLayer

DEFAULT_EMSIZE = 32

# ---------------------------------------------------------------------------
# Variable-type token indices, used by the learnable type embedding.
# Order must be consistent with TYPE_NAMES below.
# ---------------------------------------------------------------------------
TYPE_X = 0  # baseline covariate
TYPE_Z = 1  # treatment-inducing proxy
TYPE_W = 2  # outcome-inducing proxy
TYPE_A = 3  # treatment
TYPE_Y = 4  # outcome
TYPE_MASK = 5  # unknown / masked type
NUM_TYPES = 6

TYPE_NAMES = ["X", "Z", "W", "A", "Y", "MASK"]


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


class LayerStack(nn.Module):
    """
    A dynamic container for stacking transformer layers with support for stochastic
    depth and memory-efficient training.

    Unlike nn.Sequential, this stack handles keyword arguments across all layers
    and can dynamically adjust its depth during training to improve regularization
    and inference speed.

    Attributes:
        layers (nn.ModuleList): The collection of independently initialized layers
            produced by the layer_creator.
        num_layers (int): The maximum depth of the stack.
        min_num_layers_layer_dropout (int): The minimum number of layers to execute
            during a stochastic forward pass.
        recompute_each_layer (bool): If True, enables gradient checkpointing to
            save GPU memory at the cost of compute time.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)

        return x


class PerFeatureTransformerCATE(nn.Module):

    def __init__(
        self,
        *,
        x_encoder: nn.Module | None = None,
        z_encoder: nn.Module | None = None,
        w_encoder: nn.Module | None = None,
        a_encoder: nn.Module | None = None,
        y_encoder: nn.Module | None = None,
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal["normal_rand_vec", "uni_rand_vec", "learned", "subspace"] | None
        ) = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: int | None = None,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        **layer_kwargs: Any,
    ):
        super().__init__()

        def _make_feature_encoder() -> SequentialEncoder:
            """
            Creates a per-feature linear encoder for continuous causal variables (X, Z, W).

            This factory method constructs a processing pipeline that projects a single
            input feature into the transformer's latent embedding space. It follows the
            Per-Feature-Net (PFN) paradigm where each covariate is treated as an
            independent token to allow for variable-length input sequences.

            Returns:
                SequentialEncoder: A pipeline containing a linear layer that maps
                    an input of dimension 1 to an output of dimension `ninp`.

            Note:
                num_features is set to 1 because the model rearranges the input
                tensor to process each column independently before passing it to
                the transformer's attention mechanism.
            """
            return SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=1,
                    emsize=ninp,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            )

        def _make_nan_encoder() -> SequentialEncoder:
            """
            Creates a specialized encoding pipeline for variables with intentional
            missingness (Treatment A and Outcome Y).

            This encoder uses a two-step process to handle the 'causal masking'
            required for in-context learning:
            1. It imputes NaNs (test positions) with the mean of the observed training
               data to maintain numerical stability.
            2. It generates a secondary 'indicator' feature that flags which values
               were originally NaNs, allowing the transformer to distinguish between
               real data and query positions.

            Returns:
                SequentialEncoder: A pipeline that maps a single scalar input
                    to a dual-feature representation (value + indicator),
                    projected into the embedding dimension `ninp`.
            """
            return SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # value + nan indicator
                    emsize=ninp,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main", "nan_indicators"),
                    out_keys=("output",),
                ),
            )

        self.x_encoder = x_encoder if x_encoder is not None else _make_feature_encoder()
        self.z_encoder = z_encoder if z_encoder is not None else _make_feature_encoder()
        self.w_encoder = w_encoder if w_encoder is not None else _make_feature_encoder()
        self.a_encoder = a_encoder if a_encoder is not None else _make_nan_encoder()
        self.y_encoder = y_encoder if y_encoder is not None else _make_nan_encoder()

        # ------------------------------------------------------------------
        # Learnable type embedding: one vector per variable type in TYPE_NAMES.
        # This is the core mechanism that lets the transformer distinguish X
        # from Z from W, and supports the masked-type (CFG) training scheme.
        # ------------------------------------------------------------------
        self.type_embedding = nn.Embedding(NUM_TYPES, ninp)

        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            **layer_kwargs,
        )
        if repeat_same_layer:
            # weight sharing
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            # default to 1/3 of layers in decoder, 2/3 in encoder
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            assert nlayers_decoder is not None
            self.transformer_decoder = LayerStack(
                layer_creator=layer_creator,
                num_layers=nlayers_decoder,
            )

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)

        # Predict ITE mean and log-variance for a proper scoring rule loss.
        # Output dim 2: index 0 = mean, index 1 = log_var.
        self.ite_head = nn.Linear(self.ninp, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_dict(
        self, t: torch.Tensor | dict | None, key: str = "main"
    ) -> dict | None:
        """Wrap a raw tensor in a dict if necessary; leave None as None."""
        if t is None:
            return None
        if isinstance(t, dict):
            assert key in t, f"Key '{key}' missing from input dict."
            return t
        return {key: t}

    def _encode_feature_group(
        self,
        tensor_dict: dict,
        encoder: nn.Module,
        seq_len: int,
        batch_size: int,
        single_eval_pos: int,
    ) -> torch.Tensor:
        """
        Transforms a group of tabular features into a sequence of high-dimensional
        latent tokens for the Transformer.

        This method implements the 'per-feature' encoding strategy. It
        rearranges the input tensors so that each individual column is processed
        by the encoder independently, effectively treating each feature-subject
        pair as a unique token in the Transformer's context window.

        Args:
            tensor_dict (dict): A dictionary containing the raw feature tensors
                under the 'main' key, with shape (seq_len, batch_size, num_features).
            encoder (nn.Module): The encoding pipeline (e.g., LinearInputEncoder)
                used to project scalar features into the embedding space.
            seq_len (int): The number of subjects (rows) in the dataset.
            batch_size (int): The number of independent datasets in the batch.
            single_eval_pos (int): The index marking the boundary between
                observational training data and interventional test queries.

        Returns:
            torch.Tensor: An encoded tensor of shape (batch, seq, num_features, ninp),
                where 'ninp' is the model's internal embedding dimension.

        Note:
            The function uses einops to temporarily merge the batch and feature
            dimensions during encoding to leverage vectorized operations across
            all variables simultaneously.
        """
        # Pad to multiple of features_per_group
        for k in tensor_dict:
            nf = tensor_dict[k].shape[2]
            pad = (
                self.features_per_group - nf % self.features_per_group
            ) % self.features_per_group
            if pad > 0:
                tensor_dict[k] = torch.cat(
                    [
                        tensor_dict[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            pad,
                            device=tensor_dict[k].device,
                            dtype=tensor_dict[k].dtype,
                        ),
                    ],
                    dim=-1,
                )

        # Rearrange to (batch, seq, num_groups, group_size)
        for k in tensor_dict:
            tensor_dict[k] = einops.rearrange(
                tensor_dict[k], "s b (f n) -> b s f n", n=self.features_per_group
            )

        # Merge batch and feature dims for the encoder
        for k in tensor_dict:
            tensor_dict[k] = einops.rearrange(tensor_dict[k], "b s f n -> s (b f) n")

        embedded = encoder(
            tensor_dict,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # (seq, batch*features, ninp)

        return einops.rearrange(embedded, "s (b f) e -> b s f e", b=batch_size)

    def _encode_scalar_group(
        self,
        tensor_dict: dict,
        encoder: nn.Module,
        single_eval_pos: int,
    ) -> torch.Tensor:
        """
        Encode a scalar group (A or Y) that may contain NaNs.

        Returns shape: (batch, seq, ninp)
        """
        embedded = encoder(
            tensor_dict,
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # (seq, batch, ninp)
        return embedded.transpose(0, 1)  # (batch, seq, ninp)

    def _add_type_embedding(
        self,
        embedded: torch.Tensor,
        type_id: int,
        mask_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Add the learnable type embedding to a (batch, seq, features, ninp) tensor.

        If mask_prob > 0 (training with CFG-style masking), each feature token
        is independently replaced by TYPE_MASK with probability mask_prob.
        Z and W should be masked jointly — the caller controls this by passing
        the same mask tensor; see forward() for the joint masking logic.
        """
        b, s, f, e = embedded.shape
        device = embedded.device

        if mask_prob > 0.0 and self.training:
            # Sample mask: True means use TYPE_MASK instead of type_id
            use_mask = torch.rand(b, 1, f, device=device) < mask_prob
            type_ids = torch.where(
                use_mask,
                torch.full((b, 1, f), TYPE_MASK, device=device),
                torch.full((b, 1, f), type_id, device=device),
            )  # (b, 1, f)
            type_ids = type_ids.expand(b, s, f)  # broadcast over seq
        else:
            type_ids = torch.full((b, s, f), type_id, device=device, dtype=torch.long)

        type_emb = self.type_embedding(type_ids)  # (b, s, f, ninp)
        return embedded + type_emb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _pack_eval_io(
        self,
        X_train: torch.Tensor,
        Z_train: torch.Tensor,
        W_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        Z_test: torch.Tensor,
        W_test: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple:
        """Pack train/test tensors into the (seq, batch=1, features) format.

        At test positions A and Y are filled with NaN (unknown).
        Z and W are observed at test positions (they are pre-treatment proxies).

        Returns:
            x, z, w, a, y  — each a dict{"main": tensor[S,1,D]}
            single_eval_pos — int, first test row index
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = torch.float32

        def _to(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device, dtype=dtype)

        X_train, Z_train, W_train = _to(X_train), _to(Z_train), _to(W_train)
        A_train, Y_train = _to(A_train), _to(Y_train)
        X_test, Z_test, W_test = _to(X_test), _to(Z_test), _to(W_test)

        n_tr, n_te = X_train.shape[0], X_test.shape[0]
        single_eval_pos = n_tr

        nan_a = torch.full(
            (n_te, A_train.shape[1]), torch.nan, device=device, dtype=dtype
        )
        nan_y = torch.full(
            (n_te, Y_train.shape[1]), torch.nan, device=device, dtype=dtype
        )

        X_all = torch.cat([X_train, X_test], dim=0).unsqueeze(1)  # (S, 1, Dx)
        Z_all = torch.cat([Z_train, Z_test], dim=0).unsqueeze(1)  # (S, 1, Dz)
        W_all = torch.cat([W_train, W_test], dim=0).unsqueeze(1)  # (S, 1, Dw)
        A_all = torch.cat([A_train, nan_a], dim=0).unsqueeze(1)  # (S, 1, 1)
        Y_all = torch.cat([Y_train, nan_y], dim=0).unsqueeze(1)  # (S, 1, 1)

        return (
            {"main": X_all},
            {"main": Z_all},
            {"main": W_all},
            {"main": A_all},
            {"main": Y_all},
            single_eval_pos,
        )

    @torch.no_grad()
    def estimate_cate(
        self,
        X_train: torch.Tensor,
        Z_train: torch.Tensor,
        W_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        Z_test: torch.Tensor,
        W_test: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **forward_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Zero-shot proximal ITE estimation.

        Args:
            X_train / X_test : baseline covariates   [N, Dx]
            Z_train / Z_test : treatment-inducing proxies [N, Dz]
            W_train / W_test : outcome-inducing proxies   [N, Dw]
            A_train          : binary treatment            [N, 1]
            Y_train          : factual outcome             [N, 1]

        Returns:
            dict with keys:
                "cate_mean"  — predicted ITE mean,     shape [N_te]
                "cate_std"   — predicted ITE std-dev,  shape [N_te]
        """
        x, z, w, a, y, single_eval_pos = self._pack_eval_io(
            X_train,
            Z_train,
            W_train,
            A_train,
            Y_train,
            X_test,
            Z_test,
            W_test,
            device=device,
            dtype=dtype,
        )
        out = self.forward(
            x, z, w, a, y, single_eval_pos=single_eval_pos, **forward_kwargs
        )

        cate_mean = out["cate_mean"][:, single_eval_pos:, :].squeeze()
        cate_std = out["cate_std"][:, single_eval_pos:, :].squeeze()
        return {"cate_mean": cate_mean, "cate_std": cate_std}

    def forward(
        self,
        x: torch.Tensor | dict,
        z: torch.Tensor | dict | None = None,
        w: torch.Tensor | dict | None = None,
        a: torch.Tensor | dict | None = None,
        y: torch.Tensor | dict | None = None,
        single_eval_pos: int | None = None,
        *,
        zw_mask_prob: float = 0.0,
        x_mask_prob: float = 0.0,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Core forward pass.

        Args:
            x : (seq_len, batch, Dx)  — baseline covariates
            z : (seq_len, batch, Dz)  — treatment-inducing proxies
            w : (seq_len, batch, Dw)  — outcome-inducing proxies
            a : (seq_len, batch, 1)   — treatment  (NaN at test positions)
            y : (seq_len, batch, 1)   — outcome    (NaN at test positions)
            single_eval_pos : first test-row index
            zw_mask_prob : probability of replacing Z and W type tokens with
                           TYPE_MASK (CFG masking; active only when training)
            x_mask_prob  : same for X tokens

        Returns:
            dict with "cate_mean" and "cate_std", each shape (batch, seq, 1)
        """
        # ------------------------------------------------------------------
        # 0. Normalise inputs to dicts before any shape inspection
        # ------------------------------------------------------------------
        x = self._wrap_dict(x)
        z = self._wrap_dict(z)
        w = self._wrap_dict(w)

        # A and Y may be absent; create zero-length placeholders so the rest
        # of the code stays uniform.
        seq_len, batch_size, _ = x["main"].shape
        device, dtype = x["main"].device, x["main"].dtype

        if a is None:
            a = {
                "main": torch.full(
                    (seq_len, batch_size, 1), torch.nan, device=device, dtype=dtype
                )
            }
        else:
            a = self._wrap_dict(a)

        if y is None:
            y = {
                "main": torch.full(
                    (seq_len, batch_size, 1), torch.nan, device=device, dtype=dtype
                )
            }
        else:
            y = self._wrap_dict(y)

        if z is None:
            raise ValueError("Z (treatment-inducing proxy) must be provided.")
        if w is None:
            raise ValueError("W (outcome-inducing proxy) must be provided.")

        num_x_features = x["main"].shape[2]
        num_z_features = z["main"].shape[2]
        num_w_features = w["main"].shape[2]

        assert single_eval_pos is not None, "single_eval_pos must be provided."
        single_eval_pos_ = int(single_eval_pos)

        # ------------------------------------------------------------------
        # 1. Hard anti-leakage: zero out A and Y at test positions
        # ------------------------------------------------------------------
        y["main"][single_eval_pos_:] = torch.nan
        a["main"][single_eval_pos_:] = torch.nan

        # ------------------------------------------------------------------
        # 2. Encode each variable group
        # ------------------------------------------------------------------

        # --- X ---
        embedded_x = self._encode_feature_group(
            dict(x), self.x_encoder, seq_len, batch_size, single_eval_pos_
        )  # (b, s, f_x, e)

        # --- Z ---
        embedded_z = self._encode_feature_group(
            dict(z), self.z_encoder, seq_len, batch_size, single_eval_pos_
        )  # (b, s, f_z, e)

        # --- W ---
        embedded_w = self._encode_feature_group(
            dict(w), self.w_encoder, seq_len, batch_size, single_eval_pos_
        )  # (b, s, f_w, e)

        # --- A ---
        embedded_a = self._encode_scalar_group(
            dict(a), self.a_encoder, single_eval_pos_
        ).unsqueeze(
            2
        )  # (b, s, 1, e)

        # --- Y ---
        embedded_y = self._encode_scalar_group(
            dict(y), self.y_encoder, single_eval_pos_
        ).unsqueeze(
            2
        )  # (b, s, 1, e)

        del x, z, w, a, y

        # ------------------------------------------------------------------
        # 3. Add learnable type embeddings
        #
        # Z and W masking is joint: we draw a single per-sample Bernoulli and
        # apply it to both, so the model always sees either both roles labelled
        # or neither — preventing the asymmetric inference problem where only
        # one of the two proxy roles is unknown.
        # ------------------------------------------------------------------
        if zw_mask_prob > 0.0 and self.training:
            # One joint mask per (batch, feature) — shared across Z and W
            joint_zw_mask = torch.rand(batch_size, 1, 1, device=device) < zw_mask_prob
            # Broadcast to each feature token
            z_mask = joint_zw_mask.expand(batch_size, 1, num_z_features)
            w_mask = joint_zw_mask.expand(batch_size, 1, num_w_features)

            z_type_ids = torch.where(
                z_mask,
                torch.full_like(z_mask, TYPE_MASK, dtype=torch.long),
                torch.full_like(z_mask, TYPE_Z, dtype=torch.long),
            ).expand(batch_size, seq_len, num_z_features)

            w_type_ids = torch.where(
                w_mask,
                torch.full_like(w_mask, TYPE_MASK, dtype=torch.long),
                torch.full_like(w_mask, TYPE_W, dtype=torch.long),
            ).expand(batch_size, seq_len, num_w_features)

            embedded_z = embedded_z + self.type_embedding(z_type_ids)
            embedded_w = embedded_w + self.type_embedding(w_type_ids)
        else:
            embedded_z = self._add_type_embedding(embedded_z, TYPE_Z)
            embedded_w = self._add_type_embedding(embedded_w, TYPE_W)

        embedded_x = self._add_type_embedding(embedded_x, TYPE_X, mask_prob=x_mask_prob)

        # A and Y have fixed, never-masked type embeddings
        a_type = self.type_embedding(
            torch.full(
                (batch_size, seq_len, 1), TYPE_A, device=device, dtype=torch.long
            )
        )
        y_type = self.type_embedding(
            torch.full(
                (batch_size, seq_len, 1), TYPE_Y, device=device, dtype=torch.long
            )
        )
        embedded_a = embedded_a + a_type
        embedded_y = embedded_y + y_type

        # ------------------------------------------------------------------
        # 4. Concatenate all tokens along the feature dimension
        #    Layout: [X tokens | Z tokens | W tokens | A token | Y token]
        #    Shape:  (b, s, f_x + f_z + f_w + 1 + 1, e)
        # ------------------------------------------------------------------
        embedded_input = torch.cat(
            [embedded_x, embedded_z, embedded_w, embedded_a, embedded_y], dim=2
        )

        if torch.isnan(embedded_input).any():
            raise ValueError(
                "NaN detected in embedded_input after encoding. "
                "Ensure NaN-handling encoders are used for A and Y."
            )

        # ------------------------------------------------------------------
        # 5. Feature-positional embeddings (optional, for permutation symmetry)
        # ------------------------------------------------------------------
        num_feature_tokens = embedded_input.shape[2]
        embedded_input = self._add_feature_positional_embeddings(
            embedded_input,
            data_dags=data_dags,
            num_features=num_feature_tokens,
            seq_len=seq_len,
        )

        # ------------------------------------------------------------------
        # 6. Transformer
        # ------------------------------------------------------------------
        encoder_out = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos_,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # (b, s, f_total, e)  [encoder caches only train positions]

        if self.transformer_decoder:
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_
            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], dim=1)

        # ------------------------------------------------------------------
        # 7. Aggregation and ITE head
        #
        # We give privileged weight to the A token when forming the CATE
        # representation, since the CATE is a contrast in A.  We compute a
        # weighted sum: the A-token representation is upweighted relative to
        # the mean of all other tokens.
        # ------------------------------------------------------------------
        a_token_idx = (
            num_x_features + num_z_features + num_w_features
        )  # first of the two scalar tokens

        a_repr = encoder_out[:, :, a_token_idx, :]  # (b, s, e)
        other_repr = torch.cat(
            [
                encoder_out[:, :, :a_token_idx, :],
                encoder_out[:, :, a_token_idx + 1 :, :],
            ],
            dim=2,
        ).mean(
            dim=2
        )  # (b, s, e)

        cate_representation = a_repr + other_repr  # (b, s, e)

        # Predict mean and log-variance for a Gaussian NLL training loss
        ite_params = self.ite_head(cate_representation)  # (b, s, 2)
        cate_mean = ite_params[..., 0:1]  # (b, s, 1)
        cate_log_var = ite_params[..., 1:2]  # (b, s, 1)
        cate_std = torch.exp(0.5 * cate_log_var)  # (b, s, 1)

        return {
            "cate_mean": cate_mean,
            "cate_std": cate_std,
            "cate_log_var": cate_log_var,
        }

    # ------------------------------------------------------------------
    # Positional embedding
    # ------------------------------------------------------------------

    def _add_feature_positional_embeddings(
        self,
        embedded_input: torch.Tensor,
        *,
        data_dags: Iterable[nx.DiGraph] | None,
        num_features: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Injects feature-specific identifiers into the latent embeddings to
        preserve column identity across subjects.

        Because Transformers are naturally permutation-invariant, they cannot
        distinguish between different covariates of the same type (e.g., two
        different 'X' variables) without these embeddings[cite: 154]. This method
        breaks that symmetry, allowing the model to track the specific influence
        of individual proxies consistently throughout the context window[cite: 115, 154].

        Args:
            embedded_input (torch.Tensor): The latent representation of the dataset
                with shape (batch, seq, num_features, ninp)[cite: 115].
            data_dags (Iterable[nx.DiGraph] | None): Optional structural causal graphs
                associated with each batch (reserved for future DAG-aware encoding)[cite: 90].
            num_features (int): The total number of feature tokens per subject.
            seq_len (int): The total number of subjects in the in-context sequence.

        Returns:
            torch.Tensor: The input tensor with additive positional noise or
                learned vectors, maintaining the original shape[cite: 154].

        Supported Modes:
            - 'normal_rand_vec' / 'uni_rand_vec': Adds fixed random noise to
              uniquely identify each feature slot via random projections[cite: 154].
            - 'learned': Uses a learnable embedding table to assign a
              distinct vector to each feature index[cite: 154].
            - 'subspace': Projects random noise through a linear layer to
              constrain feature identities to a specific latent subspace.
        """
        with isolate_torch_rng(self.seed, device=embedded_input.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3]),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embedded_input = embedded_input + embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (embedded_input.shape[2], embedded_input.shape[3]),
                        device=embedded_input.device,
                        dtype=embedded_input.dtype,
                    )
                    * 2
                    - 1
                )
                embedded_input = embedded_input + embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[torch.randint(0, w.shape[0], (embedded_input.shape[2],))]
                embedded_input = embedded_input + embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3] // 4),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                embedded_input = embedded_input + embs[None, None]
        return embedded_input
