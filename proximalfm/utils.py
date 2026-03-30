"""
utils.py — standalone re-implementations of four TabPFN components.

Drop-in replacements for:
    from tabpfn.model.encoders import LinearInputEncoderStep, NanHandlingEncoderStep, SequentialEncoder
    from tabpfn.model.layer import PerFeatureEncoderLayer

Only standard PyTorch is required (+ einops, which the main model already uses).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nanmean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.nanmean(x, dim=dim).nan_to_num(nan=0.0)

def _nanstd(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Standard deviation ignoring NaNs along *dim*."""
    mask = ~torch.isnan(x)
    num = mask.sum(dim=dim)

    # Need at least 2 non-NaN elements to compute sample std
    if num.max() < 2:
        return torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype) if dim == 0 else torch.zeros_like(x).sum(dim=dim)

    num_clamp = num.clamp(min=2)
    mean = _nanmean(x, dim=dim)

    # Ensure mean broadcasts correctly for computing variance
    if dim == 0:
        mean_expanded = mean.unsqueeze(0).expand_as(x)
    else:
        # Assuming other dims if needed, though dim=0 is primary use-case
        mean_expanded = mean.unsqueeze(dim).expand_as(x)

    var = torch.where(mask, (x - mean_expanded)**2, torch.zeros_like(x)).sum(dim=dim) / (num_clamp - 1)

    # Return 0 std if there's less than 2 elements
    return torch.where(num < 2, torch.zeros_like(var), torch.sqrt(var))

# ===========================================================================
#  SeqEncStep — abstract base for all encoder steps
# ===========================================================================

class SeqEncStep(nn.Module):
    """
    Base class for a single step inside a SequentialEncoder pipeline.

    Each step receives a *state dict* (str → Tensor), reads from ``in_keys``,
    writes its outputs back into the dict under ``out_keys``, and returns the
    updated dict.

    Subclasses implement either:
      • ``_forward(*tensors, **kwargs)``  — stateless steps, or
      • ``_fit(*tensors, **kwargs)`` + ``_transform(*tensors, **kwargs)``
        — steps that compute statistics from the training portion of the
          sequence and then apply them to the full sequence.

    The ``single_eval_pos`` kwarg (passed through ``**kwargs``) marks the
    first test-row index; steps that need it (e.g. NaN-mean computation)
    read it from kwargs.
    """

    def __init__(
        self,
        in_keys:  tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        super().__init__()
        self.in_keys  = in_keys
        self.out_keys = out_keys

    # -- override one of these two pairs --

    def _forward(self, *tensors: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _fit(self, *tensors: torch.Tensor, **kwargs: Any) -> None:
        raise NotImplementedError

    def _transform(self, *tensors: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    # -- machinery (do not override) --

    def forward(
        self,
        state: dict[str, torch.Tensor],
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        args = [state[k] for k in self.in_keys]

        has_fit = type(self)._fit is not SeqEncStep._fit   # subclass overrode _fit

        if has_fit:
            # Re-fit whenever single_eval_pos > 0 (i.e. we have a real train set
            # in front of us), or when caching is disabled.
            single_eval_pos = kwargs.get("single_eval_pos", None)
            if single_eval_pos or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            out = self._forward(*args, **kwargs)

        assert isinstance(out, tuple), f"{type(self).__name__}._forward/transform must return a tuple"
        assert len(out) == len(self.out_keys), (
            f"{type(self).__name__}: expected {len(self.out_keys)} outputs, got {len(out)}"
        )
        state.update({k: v for k, v in zip(self.out_keys, out)})
        return state


# ===========================================================================
#  SequentialEncoder
# ===========================================================================

class SequentialEncoder(nn.Sequential):
    """
    Chains several SeqEncSteps into one encoder.

    The encoder maintains a running *state dict* (str → Tensor).  Each step
    reads its inputs from the dict and writes its outputs back.  The final
    output is ``state[output_key]``.

    If the caller passes a bare Tensor instead of a dict, it is automatically
    wrapped under the first step's first ``in_key``.

    Args:
        *steps: SeqEncStep instances to execute in order.
        output_key: Key whose value is returned from ``forward``.
    """

    def __init__(self, *steps: SeqEncStep, output_key: str = "output"):
        super().__init__(*steps)
        self.output_key = output_key

    def forward(self, state: dict | torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Accept a bare tensor for convenience.
        if not isinstance(state, dict):
            first_step = self[0]
            state = {first_step.in_keys[0]: state}

        for step in self:
            state = step(state, **kwargs)

        return state[self.output_key]


# ===========================================================================
#  LinearInputEncoderStep
# ===========================================================================

class LinearInputEncoderStep(SeqEncStep):
    """
    Projects concatenated input tensors through a single linear layer.

    The step is stateless (no train-set statistics): it simply concatenates
    all tensors listed in ``in_keys`` along the last dimension and passes
    them through ``nn.Linear(num_features, emsize)``.

    Args:
        num_features:       Total number of input features (after concat).
        emsize:             Output embedding dimension.
        replace_nan_by_zero: If True, NaN values are zeroed before the
                            linear projection (useful when NaN-handling is
                            done upstream).
        bias:               Whether the linear layer uses a bias term.
        in_keys:            State-dict keys to read and concatenate.
        out_keys:           State-dict key under which the result is stored.
    """

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys:  tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *tensors: torch.Tensor, **kwargs: Any) -> None:
        pass  # stateless

    def _transform(self, *tensors: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        x = torch.cat(tensors, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


# ===========================================================================
#  NormaliseAndClipEncoderStep
# ===========================================================================

class NormaliseAndClipEncoderStep(SeqEncStep):
    """
    Normalises each feature using mean/std computed over the training rows only,
    then clips to [-100, 100].
    """

    def __init__(
        self,
        in_keys:  tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        super().__init__(in_keys, out_keys)
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)
        self.register_buffer("feature_stds_", torch.tensor([]), persistent=False)

    def _fit(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> None:
        """Compute mean and std over the training portion [0 : single_eval_pos]."""
        x_train = x[:single_eval_pos]
        self.feature_means_ = _nanmean(x_train, dim=0)
        stds = _nanstd(x_train, dim=0)
        # Handle zero standard deviation
        stds[stds == 0] = 1.0
        # Handle NaN standard deviation (e.g., when n < 2)
        stds[torch.isnan(stds)] = 1.0
        self.feature_stds_ = stds

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        means = self.feature_means_.unsqueeze(0).expand_as(x)
        stds = self.feature_stds_.unsqueeze(0).expand_as(x)
        x = (x - means) / stds
        return (torch.clamp(x, min=-100.0, max=100.0),)

# ===========================================================================
#  NanHandlingEncoderStep
# ===========================================================================

class NanHandlingEncoderStep(SeqEncStep):
    """
    Handles NaN / ±Inf values for variables that may be missing at test time
    (Treatment A and Outcome Y in the proximal CATE model).

    Two-step process
    ----------------
    1. **Fit** (on training rows only): compute per-feature means from the
       observed training data.  Stored in a non-persistent buffer so that it
       is never saved as part of the model checkpoint.

    2. **Transform** (on all rows):
       a. Build a *NaN-indicator* tensor that encodes the original missingness
          pattern (NaN → -2, +Inf → +2, -Inf → +4, observed → 0).
       b. Impute NaN / ±Inf positions with the training-set mean so that the
          downstream linear encoder receives a numerically valid tensor.

    The output is ``(imputed_x, nan_indicator)`` — two tensors written under
    ``out_keys`` (defaults: ``"main"`` and ``"nan_indicators"``).  Both are
    then concatenated by the following LinearInputEncoderStep.

    Args:
        keep_nans: If False, only the imputed tensor is returned (no indicator).
        in_keys:   Must be a single key — the raw scalar tensor.
        out_keys:  Keys for the two outputs (imputed value, indicator).
    """

    # Sentinel values used in the indicator channel.
    NAN_INDICATOR     = -2.0
    INF_INDICATOR     =  2.0
    NEG_INF_INDICATOR =  4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys:   tuple[str, ...] = ("main",),
        out_keys:  tuple[str, ...] = ("main", "nan_indicators"),
    ):
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects exactly one input key."
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        # Non-persistent: re-computed on every forward pass; never checkpointed.
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    def _fit(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> None:
        """Compute mean over the training portion [0 : single_eval_pos]."""
        self.feature_means_ = _nanmean(x[:single_eval_pos], dim=0)

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Build indicator before any mutation.
        if self.keep_nans:
            indicator = (
                torch.isnan(x).to(x.dtype)          * self.NAN_INDICATOR
                + (torch.isinf(x) & (x > 0)).to(x.dtype) * self.INF_INDICATOR
                + (torch.isinf(x) & (x < 0)).to(x.dtype) * self.NEG_INF_INDICATOR
            )
        else:
            indicator = torch.zeros_like(x)

        # 2. Impute NaN / ±Inf with training-set means.
        bad_mask = torch.isnan(x) | torch.isinf(x)
        x = x.clone()
        means = self.feature_means_.unsqueeze(0).expand_as(x)
        x[bad_mask] = means[bad_mask]

        return x, indicator


# ===========================================================================
#  PerFeatureEncoderLayer
# ===========================================================================

class PerFeatureEncoderLayer(nn.Module):
    """
    One transformer layer for the Per-Feature-Net (PFN) paradigm.

    Input / output shape: ``(batch, seq, features, d_model)``

    Each forward pass applies three sub-layers in sequence, each followed by
    Post-LayerNorm:

    1. **Feature attention** — Multi-head self-attention *across the feature
       dimension* (within each subject/row).  This lets the model integrate
       information from different covariates of the same subject.

    2. **Item attention** — Multi-head self-attention *across the sequence
       dimension* (across subjects).  At test positions (``>= single_eval_pos``)
       only the training rows are used as keys/values — test items attend to
       train items but not to each other (no leakage).

    3. **Feed-forward MLP** — Two-layer MLP with GELU or ReLU activation.

    The ``zero_init`` option initialises all output projections to zero so that
    every layer starts as the identity function.  This is a form of depth
    regularisation described in the "ReZero" / "SkipInit" literature and is
    critical for stable training of deep PFN stacks.

    Args:
        d_model:        Embedding dimension.
        nhead:          Number of attention heads (must divide d_model).
        dim_feedforward: Hidden size of the MLP (defaults to 4 × d_model).
        activation:     ``"gelu"`` or ``"relu"``.
        zero_init:      Initialise output projections to zero.
        attention_between_features: If False, skip the feature-attention
            sub-layer (useful when features_per_group == 1).
        precomputed_kv: Ignored — present for API compatibility with the
            original TabPFN layer.
        **kwargs:       Extra kwargs are silently ignored for forward
            compatibility with callers that pass TabPFN-specific options.
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "gelu",
        zero_init: bool = True,
        attention_between_features: bool = True,
        precomputed_kv: Any = None,   # accepted but ignored
        **kwargs: Any,                # absorb extra TabPFN kwargs
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.attention_between_features = attention_between_features

        # ---- feature attention ----
        if attention_between_features:
            self.feat_attn = nn.MultiheadAttention(
                d_model, nhead, batch_first=True, bias=True
            )
            self.norm1 = nn.LayerNorm(d_model)

        # ---- item (sequence) attention ----
        self.item_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, bias=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # ---- MLP ----
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # ---- optional zero-init of all output projections ----
        if zero_init:
            for module in [self.item_attn, self.mlp]:
                # MHA: out_proj; MLP: last Linear
                out_proj = (
                    module.out_proj if hasattr(module, "out_proj")
                    else module[-1]         # last layer of nn.Sequential
                )
                nn.init.zeros_(out_proj.weight)
                if out_proj.bias is not None:
                    nn.init.zeros_(out_proj.bias)

            if attention_between_features:
                nn.init.zeros_(self.feat_attn.out_proj.weight)
                if self.feat_attn.out_proj.bias is not None:
                    nn.init.zeros_(self.feat_attn.out_proj.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,  # API compat, unused
        att_src: torch.Tensor | None = None,          # cross-attn source
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            state:            ``(batch, seq, features, d_model)``
            single_eval_pos:  First test-row index.  Test rows attend only to
                              train rows during item-attention.
            att_src:          If provided (decoder mode), item-attention uses
                              this as keys/values instead of the train portion.
                              Shape: ``(batch, train_seq, features, d_model)``.
        Returns:
            ``(batch, seq, features, d_model)`` — same shape as input.
        """
        assert state.ndim == 4, (
            f"Expected shape (batch, seq, features, d_model), got {tuple(state.shape)}"
        )
        if single_eval_pos is None:
            single_eval_pos = 0

        B, S, F, E = state.shape

        # ----------------------------------------------------------------
        # 1. Feature attention  — operates on the F dimension
        #    Reshape: (B, S, F, E)  →  (B*S, F, E)
        # ----------------------------------------------------------------
        if self.attention_between_features and F > 1:
            x_feat = state.reshape(B * S, F, E)
            attn_out, _ = self.feat_attn(x_feat, x_feat, x_feat, need_weights=False)
            x_feat = x_feat + attn_out          # residual
            x_feat = self.norm1(x_feat)
            state  = x_feat.reshape(B, S, F, E)

        # ----------------------------------------------------------------
        # 2. Item attention  — operates on the S dimension
        #    Reshape: (B, S, F, E)  →  (B*F, S, E)
        # ----------------------------------------------------------------
        x_item = state.permute(0, 2, 1, 3).contiguous().reshape(B * F, S, E)

        if att_src is not None:
            # Decoder mode: attend to a separate key/value source.
            kv_src = att_src.permute(0, 2, 1, 3).reshape(
                B * F, att_src.shape[1], E
            )
            attn_out, _ = self.item_attn(x_item, kv_src, kv_src, need_weights=False)
            x_item = x_item + attn_out
        elif single_eval_pos > 0:
            # Split into train block and test block; handle separately so
            # test rows cannot attend to each other (no label leakage).
            x_train = x_item[:, :single_eval_pos, :]   # (B*F, n_tr, E)
            x_test  = x_item[:, single_eval_pos:, :]   # (B*F, n_te, E)

            # Train rows: full self-attention among training items.
            tr_out, _ = self.item_attn(
                x_train, x_train, x_train, need_weights=False
            )
            x_train = x_train + tr_out

            # Test rows: cross-attend to training keys/values only.
            if x_test.shape[1] > 0:
                te_out, _ = self.item_attn(
                    x_test, x_train, x_train, need_weights=False
                )
                x_test = x_test + te_out
                x_item = torch.cat([x_train, x_test], dim=1)
            else:
                x_item = x_train
        else:
            # No split needed (single_eval_pos == 0): full self-attention.
            attn_out, _ = self.item_attn(x_item, x_item, x_item, need_weights=False)
            x_item = x_item + attn_out

        x_item = self.norm2(x_item)
        state  = x_item.reshape(B, F, S, E).permute(0, 2, 1, 3)   # back to (B, S, F, E)

        # ----------------------------------------------------------------
        # 3. MLP  — applied identically to every (subject, feature) token
        # ----------------------------------------------------------------
        state = self.norm3(state + self.mlp(state))

        return state