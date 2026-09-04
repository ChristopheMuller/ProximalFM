"""Marginalise proxy-conditional CATE predictions back to CATE(X).

The wrapped estimator owns causal-effect estimation.  This module only changes
its prediction target: it evaluates ``tau_hat(x, w, z)`` on aligned context
proxy atoms and integrates those predictions against a local empirical estimate
of ``P(W, Z | X=x)``.  It is therefore useful for ordinary backdoor learners and
any future single-output estimator with the same prediction interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from benchmarks.methods.base import CATEEstimator

__all__ = ["IntegrationSpec", "ProxyMarginalizer"]


def _matrix(name: str, value, *, n: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}.")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has {len(array)} rows; expected {n}.")
    return array


@dataclass(frozen=True)
class IntegrationSpec:
    """One proxy-integration rule applied to a shared neighbour ordering."""

    method: str
    n_points: int
    kernel_scale: float | None = None

    @classmethod
    def from_dict(cls, value, *, where: str) -> "IntegrationSpec":
        if not isinstance(value, dict):
            raise ValueError(f"{where} must be a mapping.")
        unknown = set(value) - {"method", "n_points", "kernel_scale"}
        if unknown:
            raise ValueError(f"{where} has unknown key(s) {sorted(unknown)}.")

        method = value.get("method")
        if method not in {"knn", "kernel"}:
            raise ValueError(f"{where}.method must be 'knn' or 'kernel'.")
        try:
            raw_n_points = value["n_points"]
        except KeyError:
            raise ValueError(f"{where}.n_points must be a positive integer.") from None
        if isinstance(raw_n_points, bool) or not isinstance(
            raw_n_points, (int, np.integer)
        ):
            raise ValueError(f"{where}.n_points must be a positive integer.")
        n_points = int(raw_n_points)
        if n_points < 1:
            raise ValueError(f"{where}.n_points must be a positive integer.")

        if method == "knn":
            if "kernel_scale" in value:
                raise ValueError(
                    f"{where}.kernel_scale is only valid for method='kernel'."
                )
            return cls(method=method, n_points=n_points)

        try:
            kernel_scale = float(value.get("kernel_scale", 1.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}.kernel_scale must be a finite positive number."
            ) from None
        if not np.isfinite(kernel_scale) or kernel_scale <= 0:
            raise ValueError(f"{where}.kernel_scale must be a finite positive number.")
        return cls(
            method=method,
            n_points=n_points,
            kernel_scale=kernel_scale,
        )


class ProxyMarginalizer(CATEEstimator):
    """Wrap a proxy-conditional estimator and integrate out its query proxies."""

    def __init__(
        self,
        estimator: CATEEstimator,
        *,
        integration: dict | None = None,
        outputs: list[dict] | None = None,
        predict_batch_size: int = 4096,
    ) -> None:
        if isinstance(predict_batch_size, bool) or not isinstance(
            predict_batch_size, (int, np.integer)
        ):
            raise ValueError("predict_batch_size must be a positive integer.") from None
        predict_batch_size = int(predict_batch_size)
        if predict_batch_size < 1:
            raise ValueError("predict_batch_size must be a positive integer.")
        if outputs is not None and not isinstance(outputs, list):
            raise ValueError("outputs must be a list of mappings.")
        if integration is not None and outputs:
            raise ValueError("Set either integration or outputs, not both.")
        if integration is None and not outputs:
            raise ValueError("ProxyMarginalizer requires integration or outputs.")

        self.estimator = estimator
        self.predict_batch_size = predict_batch_size
        self._specs: dict[str | None, IntegrationSpec] = {}

        if outputs:
            names: set[str] = set()
            for position, output in enumerate(outputs):
                where = f"outputs[{position}]"
                if not isinstance(output, dict):
                    raise ValueError(f"{where} must be a mapping.")
                unknown = set(output) - {"name", "integration"}
                if unknown:
                    raise ValueError(f"{where} has unknown key(s) {sorted(unknown)}.")
                name = output.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{where}.name must be a non-empty string.")
                if name in names:
                    raise ValueError(f"duplicate output name {name!r}.")
                names.add(name)
                self._specs[name] = IntegrationSpec.from_dict(
                    output.get("integration"), where=f"{where}.integration"
                )
        else:
            self._specs[None] = IntegrationSpec.from_dict(
                integration, where="integration"
            )

        self._X_context: np.ndarray | None = None
        self._W_context: np.ndarray | None = None
        self._Z_context: np.ndarray | None = None
        self._inv_width: np.ndarray | None = None

    def fit(self, X, A, Y, *, W=None, Z=None) -> "ProxyMarginalizer":
        if W is None or Z is None:
            raise ValueError("ProxyMarginalizer requires both W and Z during fit().")

        X = _matrix("X", X)
        n = len(X)
        if n == 0:
            raise ValueError("ProxyMarginalizer requires at least one context row.")
        W = _matrix("W", W, n=n)
        Z = _matrix("Z", Z, n=n)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        if len(A) != n or len(Y) != n:
            raise ValueError(f"A and Y must have {n} rows; got {len(A)} and {len(Y)}.")

        self._X_context = np.array(X, copy=True)
        self._W_context = np.array(W, copy=True)
        self._Z_context = np.array(Z, copy=True)

        sigmas = np.empty(X.shape[1], dtype=np.float64)
        for d in range(X.shape[1]):
            column = np.asarray(X[:, [d]], dtype=np.float64)
            distances = cdist(column, column, metric="euclidean")
            sigma = 2.0 * np.median(distances) ** 2
            sigmas[d] = sigma if np.isfinite(sigma) and sigma > 0 else 1e-3
        self._inv_width = 1.0 / np.sqrt(sigmas)

        self.estimator.fit(X, A, Y, W=W, Z=Z)
        return self

    def _nearest_atoms(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the shared maximum-support indices and scaled distances."""
        assert self._X_context is not None
        assert self._inv_width is not None

        max_points = min(
            max(spec.n_points for spec in self._specs.values()),
            len(self._X_context),
        )
        if X.shape[1] == 0:
            distances = np.zeros((len(X), len(self._X_context)), dtype=np.float64)
        else:
            delta = (
                np.asarray(X, dtype=np.float64)[:, None, :]
                - np.asarray(self._X_context, dtype=np.float64)[None, :, :]
            ) * self._inv_width[None, None, :]
            distances = np.hypot.reduce(delta, axis=2)

        indices = np.argsort(distances, axis=1, kind="stable")[:, :max_points]
        selected = np.take_along_axis(distances, indices, axis=1)
        return indices, selected

    def _predict_atoms(self, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
        assert self._W_context is not None
        assert self._Z_context is not None

        n_query, n_atoms = indices.shape
        flat_indices = indices.reshape(-1)
        X_atoms = np.repeat(X, n_atoms, axis=0)
        W_atoms = self._W_context[flat_indices]
        Z_atoms = self._Z_context[flat_indices]

        result = np.empty(len(X_atoms), dtype=np.float64)
        for start in range(0, len(X_atoms), self.predict_batch_size):
            stop = min(start + self.predict_batch_size, len(X_atoms))
            prediction = np.asarray(
                self.estimator.predict(
                    X_atoms[start:stop],
                    W=W_atoms[start:stop],
                    Z=Z_atoms[start:stop],
                )
            ).reshape(-1)
            if len(prediction) != stop - start:
                raise ValueError(
                    "wrapped estimator returned the wrong number of predictions: "
                    f"expected {stop - start}, got {len(prediction)}."
                )
            result[start:stop] = prediction
        return result.reshape(n_query, n_atoms)

    @staticmethod
    def _kernel_weights(distances: np.ndarray, scale: float) -> np.ndarray:
        minimum = distances[:, :1]
        difference = distances - minimum
        with np.errstate(over="ignore", invalid="ignore"):
            logits = -difference * (distances + minimum) / scale
        logits = np.where(difference == 0, 0.0, logits)

        bad_rows = ~np.isfinite(minimum[:, 0])
        logits = np.nan_to_num(logits, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        if np.any(bad_rows):
            weights[bad_rows] = 1.0 / distances.shape[1]
        return weights

    def predict(self, X, *, W=None, Z=None) -> np.ndarray:
        return next(iter(self.predict_variants(X, W=W, Z=Z).values()))

    def predict_variants(self, X, *, W=None, Z=None) -> dict[str | None, np.ndarray]:
        if (
            self._X_context is None
            or self._W_context is None
            or self._Z_context is None
        ):
            raise ValueError("Call fit() before predict().")

        X = _matrix("X", X)
        if X.shape[1] != self._X_context.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} columns; expected {self._X_context.shape[1]}."
            )
        if len(X) == 0:
            return {name: np.empty(0, dtype=np.float64) for name in self._specs}

        indices, distances = self._nearest_atoms(X)
        predictions = self._predict_atoms(X, indices)

        results: dict[str | None, np.ndarray] = {}
        for name, spec in self._specs.items():
            n_points = min(spec.n_points, predictions.shape[1])
            values = predictions[:, :n_points]
            if spec.method == "knn":
                results[name] = values.mean(axis=1)
            else:
                weights = self._kernel_weights(
                    distances[:, :n_points], float(spec.kernel_scale)
                )
                results[name] = np.sum(values * weights, axis=1)
        return results

    def diagnostics(self) -> dict[str, float]:
        return self.estimator.diagnostics()

    def tuning_time(self) -> float:
        return self.estimator.tuning_time()
