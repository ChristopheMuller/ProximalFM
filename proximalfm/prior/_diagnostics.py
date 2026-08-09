"""Mechanism input-composition diagnostics for the MLP-SCM prior.

Answers, for a single generated mechanism (``f_A``, ``f_Y``): *how much of the
signal handed to this mechanism comes from each parent block?* Two estimators,
from wiring to realised weights:

1. **Wiring share** -- each block's column count (the caller derives shares
   from the widths it already knows; nothing to compute here). Parent blocks
   are standardised before concatenation, so column-count share equals
   input-variance share.
2. **First-layer variance share** (:func:`first_layer_variances`) -- the first
   layer of :class:`~proximalfm.prior._mlp_scm.MLPSCM` is linear, so each input
   block's contribution to the pre-activation variance is exact: no fitting,
   no re-forward. Captures what the block-sparse / dropout-masked
   initialisation actually zeroes out, i.e. how the realised function deviates
   from the wiring.

Nothing here consumes RNG or re-runs a mechanism; the generator still wraps
collection in ``torch.random.fork_rng()`` as a guard, so that enabling
diagnostics can never change the generated data.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor

from ._mlp_scm import MLPSCM


def block_slices(widths: Mapping[str, int]) -> Dict[str, slice]:
    """Column slice of each named block in the concatenated mechanism input.

    ``widths`` must be ordered as the blocks were concatenated. Zero-width
    blocks get an empty slice.
    """
    slices: Dict[str, slice] = {}
    start = 0
    for name, width in widths.items():
        slices[name] = slice(start, start + int(width))
        start += int(width)
    return slices


def first_layer_variances(
    mlp: MLPSCM, x: Tensor, slices: Mapping[str, slice]
) -> Dict[str, float]:
    """Per-block variance injected through the mechanism's first linear layer.

    For the first layer ``h = W x + b``, block ``B`` contributes
    ``h_B = W[:, B] @ x[:, B]``; its injected variance is ``Var_rows(h_B)``
    averaged over hidden units. Exact (one matmul per block), and sensitive to
    the sparse initialisation: a block whose columns were zeroed by block-wise
    dropout injects nothing regardless of its width.

    Returns raw variances, not shares -- callers choose the denominator
    (with or without the ``eps`` block) to match the share they are plotting.
    Cross-block covariances are ignored by construction: shares normalised by
    the sum of these variances attribute shared signal proportionally.
    """
    weight = mlp.layers[0].weight
    variances: Dict[str, float] = {}
    for name, sl in slices.items():
        if sl.stop - sl.start == 0:
            variances[name] = 0.0
            continue
        partial = x[:, sl] @ weight[:, sl].T
        variances[name] = float(partial.var(dim=0, unbiased=False).mean())
    return variances


def mechanism_diagnostics(
    mlp: MLPSCM, x: Tensor, widths: Mapping[str, int]
) -> Dict[str, Dict[str, float]]:
    """All input-composition measurements for one mechanism.

    ``widths`` maps block name to column count, ordered as concatenated into
    ``x``. Returns ``{"widths", "first_layer_var"}``.
    """
    slices = block_slices(widths)
    return {
        "widths": {k: int(v) for k, v in widths.items()},
        "first_layer_var": first_layer_variances(mlp, x, slices),
    }
