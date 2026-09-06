"""Compose paired empirical ``(X, U)`` rows with a synthetic proximal SCM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

from semisynt_eval.mechanisms import (
    CaseConfig,
    fit_mechanism,
    mechanism_target_capability,
)
from semisynt_eval.source import PreparedCSV

Tensor = torch.Tensor


@dataclass(frozen=True)
class Episode:
    """One fully materialized, non-replicated benchmark episode."""

    U: Tensor
    X: Tensor
    W: Tensor
    Z: Tensor
    A: Tensor
    A_probs: Tensor
    Y_obs: Tensor
    Y0: Tensor
    Y1: Tensor
    CATE: Optional[Tensor]
    d_U: int
    d_X: int
    d_W: int
    d_Z: int
    source_row_indices: Tensor


class CSVProximalDataset:
    """A generic CSV source whose paired feature rows become empirical X/U.

    There is deliberately no replicate or Monte Carlo API here. The bundled
    mechanisms expose an analytic CATE. A future mechanism may instead declare
    ``ite_only`` and return no CATE, in which case the benchmark uses Y1-Y0.
    """

    def __init__(
        self,
        source: PreparedCSV,
        *,
        mechanism: str,
        mechanism_config: CaseConfig,
        row_sampling: str = "without_replacement",
    ) -> None:
        if row_sampling != "without_replacement":
            raise ValueError("row_sampling must be without_replacement")
        self.source = source
        self.name = f"real_csv:{source.path.stem}"
        self.mechanism_name = mechanism
        self.mechanism_config = mechanism_config
        self.row_sampling = row_sampling
        self.target_capability = mechanism_target_capability(
            mechanism, config=self.mechanism_config
        )
        self.options: dict[str, Any] = {
            "source_file": source.path.name,
            "source_sha256": source.digest,
            "source_rows": len(source.calibration_row_indices)
            + len(source.evaluation_row_indices),
            "calibration_rows": len(source.calibration_row_indices),
            "evaluation_rows": len(source.evaluation_row_indices),
            "partition_seed": source.partition_seed,
            "encoded_columns": list(source.columns),
            "x_columns": list(source.x_columns),
            "u_columns": list(source.u_columns),
            "mechanism": mechanism,
            "mechanism_config": asdict(self.mechanism_config),
            "row_sampling": row_sampling,
            "target_capability": self.target_capability,
        }

    @property
    def d_X(self) -> int:
        return len(self.source.x_indices)

    @property
    def d_U(self) -> int:
        return len(self.source.u_indices)

    @property
    def d_W(self) -> int:
        return self.d_U

    @property
    def d_Z(self) -> int:
        return self.d_U

    def _sample_indices(self, n: int, rng: torch.Generator) -> Tensor:
        n_source = int(self.source.evaluation_values.shape[0])
        if n > n_source:
            raise ValueError(
                f"{self.source.path.name} has {n_source} evaluation rows but "
                f"episode needs {n}; reduce n_calibration or the episode size"
            )
        return torch.randperm(n_source, generator=rng)[:n]

    def generate(self, *, seed: int, n_rows: int) -> Episode:
        """Generate one observational dataset and its potential outcomes."""
        if n_rows < 2:
            raise ValueError("n_rows must be >= 2")
        rng = torch.Generator().manual_seed(int(seed))
        rows = self._sample_indices(n_rows, rng)
        x = self.source.X_eval[rows]
        u = self.source.U_eval[rows]
        source_rows = torch.tensor(
            self.source.evaluation_row_indices, dtype=torch.long
        )[rows]
        mechanism = fit_mechanism(
            self.mechanism_name,
            self.source.X_cal,
            self.source.U_cal,
            rng,
            config=self.mechanism_config,
        )
        generated = mechanism.generate(x, u)
        w, z = generated.W, generated.Z
        propensity, a = generated.A_probs, generated.A
        y0, y1, y_obs = generated.Y0, generated.Y1, generated.Y_obs
        cate = generated.CATE
        if mechanism.target_capability == "exact_cate" and cate is None:
            raise RuntimeError(
                f"{self.mechanism_name} declares exact_cate but returned no CATE"
            )
        if mechanism.target_capability == "ite_only" and cate is not None:
            raise RuntimeError(
                f"{self.mechanism_name} declares ite_only but returned a CATE"
            )

        for name, tensor in (
            ("X", x),
            ("U", u),
            ("W", w),
            ("Z", z),
            ("A_probs", propensity),
            ("Y0", y0),
            ("Y1", y1),
        ):
            if tensor.shape[0] != n_rows or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{self.name}: invalid {name} block {tuple(tensor.shape)}"
                )
        if min(self.d_W, self.d_Z) < self.d_U:
            raise ValueError("proxy widths must be at least the latent width")

        return Episode(
            U=u.float(),
            X=x.float(),
            W=w.float(),
            Z=z.float(),
            A=a.float(),
            A_probs=propensity.float(),
            Y_obs=y_obs.float(),
            Y0=y0.float(),
            Y1=y1.float(),
            CATE=None if cate is None else cate.float(),
            d_U=self.d_U,
            d_X=self.d_X,
            d_W=self.d_W,
            d_Z=self.d_Z,
            source_row_indices=source_rows,
        )
