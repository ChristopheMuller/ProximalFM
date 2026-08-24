"""
Our implementation of NMMR — Nonparametric Marginal Moment Regression
================================================
Causal CATE estimator based on kernel-weighted U/V-statistic loss.

Reference: Kompa et al. "Deep Learning Methods for Proximal Inference via Maximum Moment Restriction" (NeurIPS 2022)

Model:  h(A, W, X) → Ŷ,  trained so that  E[(Y - h(A,W,X)) k(A,Z,X)] = 0
CATE:   E_W[h(1,W,X) - h(0,W,X)]  marginalized over the empirical dist of W_train

Tuning entry point: NMMRConfig — every knob lives there with sensible defaults.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


#: NMMRConfig fields a tuner may vary *without* retraining. Only the trajectory
#: *length* qualifies: every other knob changes the optimisation itself, so a
#: candidate needs its own run. `fit_trajectory` pauses at each requested epoch
#: count, and the benchmark wrapper validates `tune.sweep` against this same
#: set, so the two cannot drift apart. Mirrors `pmmr.SOLVE_ONLY_PARAMS`.
#:
#: Note what is deliberately *absent*: `w_strategy`, `x_cond_kernel_scale` and
#: `x_cond_top_k` cost nothing to vary — they touch only the W-marginalisation
#: at prediction time — but `score` is a moment loss on the bridge h(A,W,X)
#: against Y and never evaluates that path, so every value would score
#: identically and the "winner" would be whichever came first. Free to
#: re-evaluate is not the same as selectable; both are required.
SWEEP_ONLY_PARAMS = frozenset({"n_epochs"})


#: Rows used to estimate the median-heuristic bandwidth. Deliberately *not*
#: `kernel_precompute_max_n`: the bandwidth must be a function of the data
#: alone, so that turning the precomputation on or off cannot move it. Tying
#: the two together makes `kernel_precompute_max_n=0` resolve sigma from two
#: rows, which is a different model, not a slower one.
_MEDIAN_HEURISTIC_MAX_N = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NMMRConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    hidden_dim: int = 64  # width of each hidden layer
    n_layers: int = 3  # number of hidden layers (depth before output)

    # ── Kernel ────────────────────────────────────────────────────────────────
    # The training moment kernel uses one of two deliberately simple bandwidth
    # rules. This is a method-level choice; fixed and median variants are
    # benchmarked separately rather than searched as one categorical axis.
    # ``fixed`` reproduces the reference NMMR RBF's length_scale=1. ``median``
    # uses the off-diagonal median pairwise distance of the fit's kernel inputs.
    # The current benchmark geometry keeps binary A unchanged and scales Z/X
    # by their context standard deviations.
    kernel_width: Literal["fixed", "median"] = "median"

    # ── Loss ──────────────────────────────────────────────────────────────────
    # "u_stat": U-statistic (zeros diagonal of K, unbiased but noisier)
    # "v_stat": V-statistic (keeps diagonal, biased but lower variance)
    loss: Literal["u_stat", "v_stat"] = "u_stat"

    # ── Optimisation ──────────────────────────────────────────────────────────
    n_epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularisation on network weights

    # ── Inference ─────────────────────────────────────────────────────────────
    # CATE prediction marginalises over W_train in chunks to avoid OOM.
    predict_batch_size: int = 128

    # ── Reproducibility / device ──────────────────────────────────────────────
    seed: int = 42
    # "cpu", "cuda", "mps", or None to auto-select (cuda > mps > cpu).
    #
    # Inside the benchmark this is always set explicitly, from the method's
    # resolved ComputeConfig. It has to be: auto-selection grabbed MPS in *every*
    # process of a spawn pool, and concurrent multi-process Metal access stalls.
    # `benchmarks.compute` owns the choice now; None is kept for direct use of
    # NMMREstimator outside the harness.
    device: Optional[str] = None

    # ── CATE prediction / W marginalisation ──────────────────────────────────
    w_strategy: Literal["independent", "conditional"] = "independent"

    # For conditional W|X prediction: same multiplier semantics, on the median
    # heuristic of the scaled training X. A fixed width degenerates the same way
    # — at large d_X every softmax weight is equal and `conditional` silently
    # becomes `independent`.
    x_cond_kernel_scale: float = 1.0
    x_cond_top_k: int | None = 512

    # Chunk over W/reference samples at prediction time.
    w_predict_batch_size: int = 1024

    # ── Kernel precomputation ────────────────────────────────────────────────
    # The moment kernel K(A, Z, X) depends on the data alone — no model
    # parameters, no grad — so `fit_trajectory` builds the full n x n matrix
    # once and each minibatch reads its own submatrix, instead of recomputing
    # identical entries once per minibatch per epoch. Above this many rows the
    # matrix (4n^2 bytes: 67 MB at 4096) is not worth holding on the device and
    # the per-batch path is used instead. The two are bit-identical, so this is
    # a time/memory trade and never a numerical one.
    kernel_precompute_max_n: int = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _MLP(nn.Module):
    """Fully-connected MLP with ReLU on hidden layers, linear output."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()

        dims = [input_dim] + [hidden_dim] * n_layers + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers:  # hidden layers only
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sq_dists(X: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, shape (n, n), for X of shape (n, d).

    Deliberately the subtraction form rather than the ‖x‖² + ‖y‖² - 2xᵀy gemm
    identity: the latter is faster but differs in the last ulp, and relatively
    most so for near-identical rows — exactly the near-diagonal region that
    dominates K. `tests/test_bench_nmmr_sweep.py` pins bit-identical weights
    across code paths, so the arithmetic here is load-bearing.
    """
    # Vectorised: use broadcasting over the last two dimensions.
    # X.T has shape (d, n); unsqueeze gives (d, n, 1) and (d, 1, n).
    t = X.T  # (d, n)
    diff = t.unsqueeze(2) - t.unsqueeze(1)  # (d, n, n)
    return (diff**2).sum(dim=0)  # (n, n)


def _rbf_from_sq_dists(sq_dist: torch.Tensor, length_scale: float) -> torch.Tensor:
    """K = exp(-d² / (2σ²)) from precomputed squared distances."""
    return torch.exp(-sq_dist / (2.0 * (length_scale + 1e-8) ** 2))


def _rbf_kernel_matrix(X: torch.Tensor, length_scale: float = 1.0) -> torch.Tensor:
    """
    Compute the nxn RBF (Gaussian) kernel matrix for a dataset of n row-vectors.

        K[i,j] = exp( -‖X[i] - X[j]‖² / (2 * length_scale²) )

    Args:
        X:            Tensor of shape (n, d).
                      length_scale: Bandwidth parameter σ, an *absolute* width.
                      The estimator resolves it from its selected width rule.

    Returns:
        K: Tensor of shape (n, n).
    """
    return _rbf_from_sq_dists(_sq_dists(X), length_scale)


def _median_bandwidth(sq_dist: torch.Tensor) -> float:
    """Median-heuristic bandwidth: σ = median_{i≠j} ‖xᵢ - xⱼ‖.

    Taken as the median *distance* (the median commutes with the square root,
    so it is read straight off the squared distances) rather than any of the
    other spellings in circulation, because that is the convention the sibling
    kernel methods already use — `kernels.ColumnWiseGaussianKernel`'s
    `2 * median² * scale` under `exp(-d²/σ)` and `plearner._median_bandwidth`
    both put the median entry of K at exp(-1/2). Here it is used only as one of
    the two explicit NMMR method variants.

    The diagonal is excluded: it contributes n exact zeros out of n², which
    drags the median down by a factor that depends on n, so including it would
    make the bandwidth depend on the context size. Degenerate inputs (all rows
    identical, a single row) leave the median at 0; we fall back to 1.0 rather
    than dividing by zero, which is the only case where an absolute width is
    the right answer.
    """
    n = sq_dist.shape[0]
    if n < 2:
        return 1.0

    iu = torch.triu_indices(n, n, offset=1, device=sq_dist.device)
    med = float(sq_dist[iu[0], iu[1]].median())

    if not (med > 0.0) or not np.isfinite(med):
        return 1.0
    return float(np.sqrt(med))


def _nmmr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    K: torch.Tensor,
    loss: Literal["u_stat", "v_stat"],
) -> torch.Tensor:
    """
    Kernel-weighted moment loss.

        L = (r^T K r) / normaliser,  r = target - pred

    U-statistic: normaliser = n(n-1), diagonal of K zeroed out (unbiased).
    V-statistic: normaliser = n²,     diagonal kept (biased).

    Args:
        pred:   (n, 1) model predictions.
        target: (n, 1) observed outcomes.
        K:      (n, n) kernel matrix; not mutated.
        loss:   which statistic to use.

    Returns:
        Scalar loss tensor.
    """
    r = target - pred  # (n, 1)
    n = r.shape[0]

    if loss == "u_stat":
        if n <= 1:
            return pred.sum() * 0.0
        K = K.clone()
        K.fill_diagonal_(0.0)
        normaliser = n * (n - 1)
    else:
        normaliser = n * n

    return (r.T @ K @ r)[0, 0] / normaliser


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class NMMREstimator:
    """
    Sklearn-style CATE estimator implementing NMMR.

    Usage::

        cfg = NMMRConfig(hidden_dim=128, n_epochs=200, lr=5e-4)
        est = NMMREstimator(cfg)
        est.fit(X_train, A_train, Y_train, W=W_train, Z=Z_train)
        cate = est.predict(X_test)   # shape (n_test,)

    The ``cfg`` dataclass is the single place to tune all hyperparameters.
    """

    def __init__(self, cfg: NMMRConfig = NMMRConfig()) -> None:
        self.cfg = cfg
        self._model: _MLP | None = None
        self._X_train: torch.Tensor | None = None
        self._W_train: torch.Tensor | None = None
        self._dim_A: int = 1
        # Absolute RBF widths, resolved from the data at fit time (the config
        # carries multipliers, not widths). The W|X one is filled in on first
        # use of the `conditional` strategy.
        self._length_scale_: float = 1.0
        self._x_cond_length_scale_: float | None = None
        self._device = self._select_device(self.cfg.device)
        self.diagnostics_: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> "NMMREstimator":
        """
        Train the NMMR model.

        Args:
            X: Covariates,               shape (n, d_X).
            A: Treatment (binary),  shape (n,) or (n, d_A).
            Y: Observed outcome,          shape (n,).
            W: Outcome-inducing proxy,    shape (n, d_W).
            Z: Treatment-inducing proxy,  shape (n, d_Z).

        Returns:
            self (for method chaining).

        This is exactly ``fit_trajectory(..., milestones=(cfg.n_epochs,))`` run
        to exhaustion — one code path, so a single fit and a swept one cannot
        diverge. See :meth:`fit_trajectory` for the split and why it is exact.
        """
        for _ in self.fit_trajectory(
            X, A, Y, W=W, Z=Z, milestones=(self.cfg.n_epochs,)
        ):
            pass
        return self

    def fit_trajectory(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
        milestones: Sequence[int],
    ) -> Iterator[tuple[int, "NMMREstimator"]]:
        """Train once, pausing after each milestone epoch so the caller can score.

        Yields ``(epoch, self)`` once epoch ``m`` has completed, for every ``m``
        in ``milestones``, then resumes training from exactly that state. The
        run lasts ``max(milestones)`` epochs, so a search over ``n_epochs``
        costs one training run rather than one per value — the counterpart of
        :meth:`PMMREstimator.prepare`'s split, except that here what is shared
        is a prefix of the optimisation rather than a set of matrices.

        **The weights yielded at ``m`` are bit-identical to a fresh** ``fit``
        **with** ``n_epochs=m``. Nothing in the loop reads ``cfg.n_epochs``
        except the bound; the seeds are set once here and the MLP is built
        before the first epoch, so both the initialisation and the whole
        ``randperm`` stream are independent of where the run stops. That is what
        makes ``sweep`` a re-spelling rather than an approximation, and
        ``tests/test_bench_nmmr_sweep.py`` pins it.

        Two things a caller must not break, both of which would silently
        invalidate that claim rather than raise:

        - **Do not consume RNG between yields.** :meth:`score` does not (a
          ``no_grad`` forward plus an RBF gram), and neither does the
          ``eval()``/``train()`` round trip it induces, since :class:`_MLP` has
          no dropout or batch norm. Anything that draws — resampling the
          validation split per milestone, say — would shift every subsequent
          minibatch.
        - **Score before advancing.** The yielded object is ``self``, mutated in
          place; the next epoch overwrites it. Holding one for later gives the
          final weights, not the milestone's. Keeping the *score* rather than a
          state dict is what keeps memory flat in the number of milestones.

        Args are as :meth:`fit`, plus ``milestones``: epoch counts at which to
        yield, deduplicated and sorted; must be non-empty and >= 1.
        """
        cfg = self.cfg

        stops = sorted({int(m) for m in milestones})
        if not stops or stops[0] < 1:
            raise ValueError(
                f"fit_trajectory needs a non-empty list of epoch counts >= 1, "
                f"got {list(milestones)!r}."
            )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        X_t, A_t, Y_t, W_t, Z_t = self._to_tensors(X, A, Y, W, Z)

        self._W_train = W_t
        self._X_train = X_t
        self._dim_A = A_t.shape[1]

        self._Z_scale = torch.std(Z_t, dim=0, unbiased=False, keepdim=True).clamp_min(
            1e-6
        )
        self._X_scale = torch.std(X_t, dim=0, unbiased=False, keepdim=True).clamp_min(
            1e-6
        )

        n = X_t.shape[0]

        # The bandwidth is a property of the data, so it is resolved once, here,
        # from the fit's own kernel inputs — never per minibatch, which would
        # make sigma a function of `batch_size`. Nothing below draws from the
        # RNG, so the minibatch stream is unaffected by any of it.
        with torch.no_grad():
            k_input_all = self._kernel_input(A_t, Z_t, X_t)
            sq_dist = (
                _sq_dists(k_input_all) if n <= cfg.kernel_precompute_max_n else None
            )
            self._length_scale_ = self._resolve_bandwidth(
                k_input_all,
                cfg.kernel_width,
                # Only when it covers exactly the rows the resolver would have
                # picked itself, so that sigma cannot depend on the precompute.
                sq_dist=sq_dist if n <= _MEDIAN_HEURISTIC_MAX_N else None,
            )

            # One kernel for the whole fit. Each minibatch's K is a submatrix of
            # this one, and `index_select` twice recovers it *exactly*: every
            # entry is the same per-pair arithmetic on the same inputs, so the
            # yielded weights stay bit-identical to the per-batch path (and to
            # every checkpoint `tests/test_bench_nmmr_sweep.py` pins).
            K_full: torch.Tensor | None = None
            if sq_dist is not None:
                K_full = _rbf_from_sq_dists(sq_dist, self._length_scale_)
                del sq_dist

        self._record_kernel_diagnostics(A_t, Z_t, X_t, K_full=K_full)

        input_dim = self._dim_A + W_t.shape[1] + X_t.shape[1]
        self._model = _MLP(input_dim, cfg.hidden_dim, cfg.n_layers).to(self._device)

        optimizer = optim.Adam(
            self._model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        todo = set(stops)

        for epoch in range(1, stops[-1] + 1):
            # Re-asserted every epoch rather than once before the loop: a
            # milestone score leaves the module in eval(). Inert for this
            # architecture, but the loop must not depend on what the caller did.
            self._model.train()
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]

                bA, bW, bX, bY = A_t[idx], W_t[idx], X_t[idx], Y_t[idx]

                optimizer.zero_grad()

                pred = self._model(torch.cat([bA, bW, bX], dim=1))

                if K_full is not None:
                    K = K_full.index_select(0, idx).index_select(1, idx)
                else:
                    K = _rbf_kernel_matrix(
                        self._kernel_input(bA, Z_t[idx], bX),
                        length_scale=self._length_scale_,
                    )

                loss = _nmmr_loss(pred, bY, K, cfg.loss)
                loss.backward()
                optimizer.step()

            if epoch in todo:
                yield epoch, self

    def score(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        *,
        W: np.ndarray,
        Z: np.ndarray,
        loss: Literal["u_stat", "v_stat"] | None = None,
    ) -> float:
        """Held-out kernel moment loss (lower is better) for model selection.

        The NMMR objective is E[(Y - h(A,W,X)) k(A,Z,X)] = 0, so the natural
        validation metric is the same kernel-weighted moment loss evaluated on
        rows the model was not trained on.

        The scoring rows use the same fixed or median width rule as the model.
        Candidates within one width variant are therefore ranked by one
        candidate-independent functional. The two width variants are separate
        benchmark methods because their tuning losses use different kernels.

        ``loss`` defaults to the model's **own** statistic (``cfg.loss``), so
          each estimator is validated on the objective it was trained for.
          Scoring everything on the V-statistic instead — as this used to —
          adds V's diagonal term, which is exactly ``MSE/n``: a U-statistic
          model would then be ranked partly by an ordinary-regression fit on
          ``(A, W, X)``, the confounded quantity the method exists to avoid.
          ``cfg.loss`` is therefore a fixed property of a method entry, not a
          tunable, and :class:`benchmarks.methods.NMMR_Wrapper` refuses to see
          it inside a ``tune.grid``.
        """
        if loss is None:
            loss = self.cfg.loss
        if self._model is None:
            raise RuntimeError("Call fit() before score().")

        X_t, A_t, Y_t, W_t, Z_t = self._to_tensors(X, A, Y, W, Z)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.cat([A_t, W_t, X_t], dim=1))
            k_input = self._kernel_input(A_t, Z_t, X_t)
            K = _rbf_kernel_matrix(
                k_input,
                length_scale=self._resolve_bandwidth(k_input, self.cfg.kernel_width),
            )
            val = _nmmr_loss(pred, Y_t, K, loss)

        return float(val)

    def predict(
        self,
        X: np.ndarray,
        *,
        W: np.ndarray | None = None,
        strategy: Literal["independent", "conditional"] | None = None,
    ) -> np.ndarray:
        """
        Predict treatment effect h(1,W,X)-h(0,W,X).

        If W is provided, returns the paired proxy-conditional contrast.

        If W is not provided:
            - independent: average over marginal empirical W_train.
            - conditional: approximate E[. | X=x] using kernel weights over X_train.
        """
        if self._model is None or self._W_train is None:
            raise RuntimeError("Call fit() before predict().")

        X_t = self._as_2d_tensor(X, name="X")
        strategy = strategy or self.cfg.w_strategy

        if W is not None:
            W_t = self._as_2d_tensor(W, name="W")
            return self._predict_effect_paired_w(X_t, W_t)

        if strategy == "independent":
            return self._predict_effect_independent_w(X_t)

        if strategy == "conditional":
            return self._predict_effect_conditional_w(X_t)

        raise ValueError(f"Unknown W prediction strategy: {strategy!r}")

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _select_device(device: str | None = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _as_2d_tensor(self, arr: np.ndarray, *, name: str) -> torch.Tensor:
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")

        return torch.tensor(arr, dtype=torch.float32, device=self._device)

    def _to_tensors(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        def t(arr: np.ndarray, shape: tuple | None = None) -> torch.Tensor:
            out = torch.tensor(arr, dtype=torch.float32, device=self._device)
            if shape is not None:
                out = out.view(*shape)
            return out

        n = X.shape[0]
        return (
            t(X).view(n, -1) if X.ndim == 1 else t(X),
            t(A).view(n, -1),
            t(Y).view(n, 1),
            t(W).view(n, -1) if W.ndim == 1 else t(W),
            t(Z).view(n, -1) if Z.ndim == 1 else t(Z),
        )

    def _kernel_input(
        self,
        A_t: torch.Tensor,
        Z_t: torch.Tensor,
        X_t: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenated moment-kernel input ``(A, Z/sd, X/sd)``.

        One definition for three call sites — training, :meth:`score` and the
        diagnostics. Together with the median-heuristic bandwidth, the
        per-column scaling keeps the median rule stable across episodes while
        preserving the benchmark's binary treatment geometry, so the three
        call sites must not drift apart.
        """
        return torch.cat(
            [A_t, Z_t / self._Z_scale, X_t / self._X_scale],
            dim=1,
        )

    def _resolve_bandwidth(
        self,
        k_input: torch.Tensor,
        kernel_width: Literal["fixed", "median"],
        *,
        sq_dist: torch.Tensor | None = None,
    ) -> float:
        """Resolve the selected method-level width into an absolute RBF width.

        ``sq_dist`` is the caller's already-computed distance matrix, if it has
        one *and* it covers the same rows this would have used; otherwise the
        rows are strided down to ``_MEDIAN_HEURISTIC_MAX_N`` first, since the
        median of a strided subsample is a stable approximation and the full
        matrix may not fit. The stride is deterministic rather than random so
        it cannot perturb :meth:`fit_trajectory`'s RNG stream.
        """
        if kernel_width == "fixed":
            return 1.0
        if kernel_width != "median":
            raise ValueError(
                "kernel_width must be either 'fixed' or 'median', "
                f"got {kernel_width!r}."
            )

        if sq_dist is None:
            n = k_input.shape[0]
            if n > _MEDIAN_HEURISTIC_MAX_N:
                step = (n + _MEDIAN_HEURISTIC_MAX_N - 1) // _MEDIAN_HEURISTIC_MAX_N
                k_input = k_input[::step]
            sq_dist = _sq_dists(k_input)

        return _median_bandwidth(sq_dist)

    def _record_kernel_diagnostics(
        self,
        A_t: torch.Tensor,
        Z_t: torch.Tensor,
        X_t: torch.Tensor,
        *,
        K_full: torch.Tensor | None = None,
    ) -> None:
        """Measure how much moment signal the kernel actually carries.

        The diagnostics make both width rules visible. Median mode should have
        an off-diagonal median near ``exp(-1/2)``, while fixed mode can become
        close to the identity as the standardized kernel dimension grows.

        That matters because the whole moment condition lives in K's
        **off-diagonal**. The V-statistic's diagonal term is just ``MSE/n``, so
        a near-diagonal K quietly turns the objective into ordinary regression
        on ``(A, W, X)`` — the confounded fit the method exists to avoid — while
        the U-statistic, which zeroes the diagonal, keeps only a vanishing
        remainder. Neither failure is visible in the loss curve: both look like
        healthy convergence.

        Hence these keys, measured once per fit on the leading
        ``batch_size`` rows (no RNG — a permutation here would shift the
        training minibatch stream and change results for a diagnostic):

        - ``kernel_dim`` — ``1 + d_Z + d_X``, the dimension driving the effect.
        - ``kernel_length_scale`` — the resolved absolute σ.
        - ``kernel_diag_mean`` / ``kernel_diag_std`` — checks the unit diagonal.
        - ``kernel_offdiag_min`` / ``_p25`` / ``_median`` / ``_mean`` /
          ``_p75`` / ``_max`` — the distribution of off-diagonal similarities.
          Small values mean the moment condition is barely being enforced; read
          them before reading an NMMR number.
        """
        m = min(A_t.shape[0], self.cfg.batch_size)
        if m < 2:
            return

        with torch.no_grad():
            k_input = self._kernel_input(A_t[:m], Z_t[:m], X_t[:m])
            if K_full is not None:
                K = K_full[:m, :m]
            else:
                K = _rbf_kernel_matrix(k_input, length_scale=self._length_scale_)
            off = K[~torch.eye(m, dtype=torch.bool, device=self._device)]

            self.diagnostics_ = {
                "kernel_dim": float(k_input.shape[1]),
                "kernel_length_scale": float(self._length_scale_),
                "kernel_diag_mean": float(K.diagonal().mean()),
                "kernel_diag_std": float(K.diagonal().std(unbiased=False)),
                "kernel_offdiag_min": float(off.min()),
                "kernel_offdiag_p25": float(torch.quantile(off, 0.25)),
                "kernel_offdiag_median": float(off.median()),
                "kernel_offdiag_mean": float(off.mean()),
                "kernel_offdiag_p75": float(torch.quantile(off, 0.75)),
                "kernel_offdiag_max": float(off.max()),
            }

    def _predict_effect_paired_w(
        self,
        X_t: torch.Tensor,
        W_t: torch.Tensor,
    ) -> np.ndarray:
        assert self._model is not None

        n = X_t.shape[0]

        if W_t.shape[0] == 1:
            W_t = W_t.expand(n, -1)
        elif W_t.shape[0] != n:
            raise ValueError(
                "When W is provided to predict(), it must either have one row "
                "or the same number of rows as X."
            )

        cate = np.empty(n, dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n, self.cfg.predict_batch_size):
                X_b = X_t[start : start + self.cfg.predict_batch_size]
                W_b = W_t[start : start + self.cfg.predict_batch_size]
                b = X_b.shape[0]

                A1 = torch.ones(b, self._dim_A, device=self._device)
                A0 = torch.zeros(b, self._dim_A, device=self._device)

                y1 = self._model(torch.cat([A1, W_b, X_b], dim=1))
                y0 = self._model(torch.cat([A0, W_b, X_b], dim=1))

                cate[start : start + b] = (y1 - y0).squeeze(1).cpu().numpy()

        return cate

    def _predict_effect_independent_w(self, X_t: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        assert self._W_train is not None

        n_test = X_t.shape[0]
        n_W = self._W_train.shape[0]
        cate = np.empty(n_test, dtype=np.float32)

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n_test, self.cfg.predict_batch_size):
                X_b = X_t[start : start + self.cfg.predict_batch_size]
                b = X_b.shape[0]

                effect_sum = torch.zeros(b, device=self._device)
                count = 0

                for w_start in range(0, n_W, self.cfg.w_predict_batch_size):
                    W_c = self._W_train[
                        w_start : w_start + self.cfg.w_predict_batch_size
                    ]
                    diff = self._effect_grid(X_b, W_c)  # shape (b, m)

                    effect_sum += diff.sum(dim=1)
                    count += W_c.shape[0]

                cate[start : start + b] = (effect_sum / count).cpu().numpy()

        return cate

    def _effect_grid(
        self,
        X_b: torch.Tensor,
        W_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return h(1,W_j,X_i)-h(0,W_j,X_i) for every pair (X_i, W_j).

        Shape:
            X_b:   (b, d_X)
            W_ref: (m, d_W)

        Returns:
            diff:  (b, m)
        """
        assert self._model is not None

        b = X_b.shape[0]
        m = W_ref.shape[0]

        X_exp = X_b.repeat_interleave(m, dim=0)
        W_exp = W_ref.repeat(b, 1)

        A1 = torch.ones(b * m, self._dim_A, device=self._device)
        A0 = torch.zeros(b * m, self._dim_A, device=self._device)

        y1 = self._model(torch.cat([A1, W_exp, X_exp], dim=1)).view(b, m)
        y0 = self._model(torch.cat([A0, W_exp, X_exp], dim=1)).view(b, m)

        return y1 - y0

    def _predict_effect_conditional_w(self, X_t: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        assert self._W_train is not None
        assert self._X_train is not None
        assert self._X_scale is not None

        cfg = self.cfg
        n_test = X_t.shape[0]
        cate = np.empty(n_test, dtype=np.float32)

        X_train_scaled = self._X_train / self._X_scale

        if self._x_cond_length_scale_ is None:
            # Lazily, because most method entries never take this path, and a
            # second n x n distance matrix per fit is not free. Cached on the
            # estimator: it is a function of X_train alone, so it must not vary
            # between prediction batches.
            self._x_cond_length_scale_ = (
                self._resolve_bandwidth(X_train_scaled, "median")
                * cfg.x_cond_kernel_scale
            )

        self._model.eval()
        with torch.no_grad():
            for start in range(0, n_test, cfg.predict_batch_size):
                X_b = X_t[start : start + cfg.predict_batch_size]
                b = X_b.shape[0]

                X_b_scaled = X_b / self._X_scale

                # Squared distances between test X and train X.
                dist2 = torch.cdist(X_b_scaled, X_train_scaled).pow(2)

                log_weights = -dist2 / (2.0 * (self._x_cond_length_scale_ + 1e-8) ** 2)

                if cfg.x_cond_top_k is not None:
                    k = min(cfg.x_cond_top_k, self._X_train.shape[0])
                    top_log_weights, top_idx = torch.topk(log_weights, k=k, dim=1)

                    weights = torch.softmax(top_log_weights, dim=1)  # (b, k)
                    W_local = self._W_train[top_idx]  # (b, k, d_W)
                else:
                    weights = torch.softmax(log_weights, dim=1)  # (b, n_train)
                    W_local = self._W_train.unsqueeze(0).expand(b, -1, -1)

                cate_b = self._effect_rowwise_weighted_w(X_b, W_local, weights)

                cate[start : start + b] = cate_b.cpu().numpy()

        return cate

    def _effect_rowwise_weighted_w(
        self,
        X_b: torch.Tensor,
        W_local: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted conditional effect for each row of X_b.

        Args:
            X_b:      (b, d_X)
            W_local:  (b, k, d_W), local/reference W values for each X row.
            weights:  (b, k), sums to 1 across k.

        Returns:
            cate:     (b,)
        """
        assert self._model is not None

        b, k, d_W = W_local.shape

        X_exp = X_b.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        W_exp = W_local.reshape(b * k, d_W)

        A1 = torch.ones(b * k, self._dim_A, device=self._device)
        A0 = torch.zeros(b * k, self._dim_A, device=self._device)

        y1 = self._model(torch.cat([A1, W_exp, X_exp], dim=1)).view(b, k)
        y0 = self._model(torch.cat([A0, W_exp, X_exp], dim=1)).view(b, k)

        diff = y1 - y0

        return (weights * diff).sum(dim=1)
