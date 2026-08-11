"""
data.py

Build the standardized benchmark data dict from the RAW, sparse
"training-validation" format (the ``batch_*.pt`` + ``metadata.json`` layout
written by ``proximalfm.prior._genload.SavePriorDataset`` and by
``synt_eval/build_proximal_datasets.py``).

The benchmark historically consumed its own single-file ``data.pt``; this
module lets it consume the val-format directory directly, so the same files on
disk serve both in-training validation and benchmarking.

This module is deliberately free of any ``proximalfm`` import: the benchmark
must stay runnable in environments where proximalfm is not installed
(``run.py`` only requires it when *generating* data). ``sparse2dense``
is therefore vendored verbatim from ``proximalfm.prior._genload``; a unit test
in ``tests/test_bench_data.py`` keeps the two in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

Tensor = torch.Tensor


def sparse2dense(
    sparse_tensor: Tensor,
    row_lengths: Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Reconstruct a zero-padded dense tensor from its sparse representation.

    Vendored from ``proximalfm.prior._genload.sparse2dense`` (kept in sync by
    a unit test) so that loading val-format files does not require proximalfm.
    """
    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(
        sparse_tensor
    ), "data length must match sum of row_lengths"

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense


def _squeeze_trailing(t: Tensor) -> Tensor:
    """Normalize per-row scalars to [B, T] (priors may yield [B, T, 1])."""
    if t.dim() == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return t


def _has_replicates(K_eff: Optional[Tensor]) -> bool:
    """True when every episode carries more than one Monte-Carlo replicate.

    ``K_eff`` is the *realized* replicate count after ``prior_mc_mem_budget``
    capping, so it is the only trustworthy witness that ``MU0``/``MU1`` are an
    average rather than a single draw. A mixed pool (some episodes at ``K = 1``)
    is refused wholesale: one scalar target per benchmark, no per-episode
    switching between two different estimands.
    """
    if K_eff is None:
        return False
    return bool((K_eff > 1).all())


def build_benchmark_dict(
    *,
    U: Tensor,
    X: Tensor,
    W: Tensor,
    Z: Tensor,
    A: Tensor,
    Y_obs: Tensor,
    Y0: Tensor,
    Y1: Tensor,
    d_X: Tensor,
    d_W: Tensor,
    d_Z: Tensor,
    train_size: int,
    MU0: Optional[Tensor] = None,
    MU1: Optional[Tensor] = None,
    S2: Optional[Tensor] = None,
    K_eff: Optional[Tensor] = None,
) -> dict:
    """Standardize a raw pool of datasets into the benchmark data dict.

    Covariates/proxies are standardized with statistics computed ONLY on the
    first ``train_size`` rows (the context the methods may see — no test
    leakage), and ``Y``/``true_ITEs``/``true_CATEs`` are scaled by the observed
    training outcomes' mean/std. Schema matches what ``run.py`` saves as
    ``data.pt``.

    ``U``/``X``/``W``/``Z`` are dense ``[B, T, d]``; ``A``/``Y_obs``/``Y0``/
    ``Y1`` are ``[B, T]`` (a trailing singleton dim is squeezed off).

    Two ground truths, and the difference is the whole point of the benchmark:

    ``true_ITEs``
        ``Y1 − Y0`` on the query rows — one *realized* draw of the individual
        effect, carrying that row's own ``U`` and noise.
    ``true_CATEs``
        ``MU1 − MU0``, the Monte-Carlo estimate of ``tau*(x) = E[Y1 − Y0 | X=x]``
        that ``SCMPrior`` builds from ``K_eff`` replicates per query row. This is
        the estimand ProximalFM is trained on; scoring against ``true_ITEs``
        instead buries every method under an irreducible noise floor.

    ``true_CATEs`` is ``None`` unless ``MU0``/``MU1`` are given **and** every
    episode has ``K_eff > 1``. At ``K = 1`` the "Monte-Carlo estimate" is a
    single draw — the same object as the ITE, with none of its honesty — so the
    gate is what keeps a ``--mc_cate 1`` set from silently scoring as a CATE set.
    """
    train_size = int(train_size)
    B, T, _ = X.shape
    n_test = T - train_size

    A, Y_obs, Y0, Y1 = (_squeeze_trailing(t) for t in (A, Y_obs, Y0, Y1))

    def safe_standardize(x: Tensor) -> Tensor:
        if x.shape[-1] == 0:
            return x
        ctx = x[:, :train_size]
        mu = ctx.mean(dim=1, keepdim=True)
        std = ctx.std(dim=1, keepdim=True)
        # Prevent division by zero for zero-variance or zero-padded features
        std = torch.where(std < 1e-5, torch.ones_like(std), std)
        return (x - mu) / std

    X_std = safe_standardize(X)
    W_std = safe_standardize(W)
    Z_std = safe_standardize(Z)
    U_std = safe_standardize(U)

    y_train_obs = Y_obs[:, :train_size]
    y_ctx_mean = y_train_obs.mean(dim=1, keepdim=True)
    y_ctx_std = y_train_obs.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
    Y_obs_std = (Y_obs - y_ctx_mean) / y_ctx_std

    true_ITEs_std = (
        ((Y1[:, train_size:] - Y0[:, train_size:]) / y_ctx_std).cpu().numpy()
    )

    true_CATEs_std = None
    cate_s2_std = None
    cate_K_eff = None
    if MU0 is not None and MU1 is not None and _has_replicates(K_eff):
        MU0, MU1 = (_squeeze_trailing(t) for t in (MU0, MU1))
        true_CATEs_std = (
            ((MU1[:, train_size:] - MU0[:, train_size:]) / y_ctx_std).cpu().numpy()
        )
        cate_K_eff = K_eff.cpu()
        if S2 is not None:
            # S2 is a *variance* of the target, so it takes the square of the
            # scale factor that true_CATEs takes.
            S2 = _squeeze_trailing(S2)
            cate_s2_std = (S2[:, train_size:] / y_ctx_std**2).cpu().numpy()

    return {
        "U": U_std.cpu(),
        "X": X_std.cpu(),
        "Z": Z_std.cpu(),
        "W": W_std.cpu(),
        "A": A.cpu(),
        "Y": Y_obs_std.cpu(),
        "d_X": d_X.cpu(),
        "d_Z": d_Z.cpu(),
        "d_W": d_W.cpu(),
        "true_ITEs": true_ITEs_std,
        "true_CATEs": true_CATEs_std,
        "cate_s2": cate_s2_std,
        "cate_K_eff": cate_K_eff,
        "N_TRAIN": train_size,
        "N_TEST": n_test,
        "N_BATCH": B,
    }


def load_benchmark_from_val_dir(val_dir: str | Path) -> dict:
    """Load a val-format directory and return the standardized benchmark dict.

    Reads every ``batch_*.pt`` file in ``val_dir`` (sparse, non-nested layout),
    reconstructs the dense tensors — padding every file to the directory-wide
    max feature width so multi-file dirs concatenate safely — and standardizes
    via :func:`build_benchmark_dict`.

    Requires all datasets to share a single ``seq_len`` and ``train_size``
    (the benchmark dict has one scalar ``N_TRAIN``/``N_TEST``).

    ``MU0``/``MU1``/``S2``/``K_eff`` are picked up when the files carry them —
    every file written by ``proximalfm.prior._genload.write_batch_file`` does,
    but sets predating that writer do not, and those load as ITE-only.
    """
    val_dir = Path(val_dir)
    files = sorted(val_dir.glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files found in {val_dir}")

    payloads = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    for f, p in zip(files, payloads):
        if p["X"].dim() != 1:
            raise ValueError(
                f"{f}: expected sparse (1D) covariate tensors; nested val "
                "files are not supported by the benchmark loader."
            )

    seq_lens = torch.cat([p["seq_lens"] for p in payloads])
    T = int(seq_lens[0].item())
    if not (seq_lens == T).all():
        raise ValueError("All datasets must share a single seq_len.")

    train_sizes = torch.cat([p["train_sizes"] for p in payloads])
    train_size = int(train_sizes[0].item())
    if not (train_sizes == train_size).all():
        raise ValueError(
            "The benchmark format requires a single train_size across all "
            "datasets (got varying train_sizes)."
        )

    def densify(field: str, dims_key: str) -> Tensor:
        max_d = max(int(p[dims_key].max().item()) for p in payloads)
        parts = []
        for p in payloads:
            dims = p[dims_key]
            b = dims.shape[0]
            parts.append(
                sparse2dense(p[field], dims.repeat_interleave(T), max_len=max_d).view(
                    b, T, max_d
                )
            )
        return torch.cat(parts, dim=0)

    def flat(field: str) -> Tensor:
        return torch.cat([_squeeze_trailing(p[field]) for p in payloads], dim=0)

    def flat_opt(field: str) -> Optional[Tensor]:
        if not all(field in p for p in payloads):
            return None
        return flat(field)

    return build_benchmark_dict(
        U=densify("U", "d_U"),
        X=densify("X", "d_X"),
        W=densify("W", "d_W"),
        Z=densify("Z", "d_Z"),
        A=flat("A"),
        Y_obs=flat("Y_obs"),
        Y0=flat("Y0"),
        Y1=flat("Y1"),
        d_X=torch.cat([p["d_X"] for p in payloads]),
        d_W=torch.cat([p["d_W"] for p in payloads]),
        d_Z=torch.cat([p["d_Z"] for p in payloads]),
        train_size=train_size,
        MU0=flat_opt("MU0"),
        MU1=flat_opt("MU1"),
        S2=flat_opt("S2"),
        K_eff=flat_opt("K_eff"),
    )
