"""Compose paired empirical ``(X, U)`` rows with a synthetic proximal SCM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

from semisynt_eval.mechanisms import (
    CaseConfig,
    MechanismOutput,
    mechanism_target_capability,
)
from semisynt_eval.mechanisms.calibrated_latent_proxy import stable_seed
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
    U_source: Tensor
    V: Tensor
    bridge0: Tensor
    bridge1: Tensor


@dataclass(frozen=True)
class EvaluationRows:
    """One ordered empirical row sample shared across all factorial cases."""

    X: Tensor
    U_source: Tensor
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

    def sample_rows(
        self, *, master_seed: int, replicate: int, n_rows: int
    ) -> EvaluationRows:
        """Select paired evaluation rows from an explicit named stream."""
        if n_rows < 2:
            raise ValueError("n_rows must be >= 2")
        n_source = int(self.source.evaluation_values.shape[0])
        if n_rows > n_source:
            raise ValueError(
                f"{self.source.path.name} has {n_source} evaluation rows but "
                f"episode needs {n_rows}; reduce n_calibration or the episode size"
            )
        if replicate < 1:
            raise ValueError("replicate must be one-based and positive")
        row_seed = stable_seed(master_seed, self.source.path.name, replicate, "rows")
        rng = torch.Generator().manual_seed(row_seed)
        rows = torch.randperm(n_source, generator=rng)[:n_rows]
        x = self.source.X_eval[rows]
        u_source = self.source.U_eval[rows]
        source_rows = torch.tensor(
            self.source.evaluation_row_indices, dtype=torch.long
        )[rows]
        return EvaluationRows(
            X=x,
            U_source=u_source,
            source_row_indices=source_rows,
        )

    def materialize_episode(
        self, rows: EvaluationRows, generated: MechanismOutput
    ) -> Episode:
        """Convert native float64 output to the canonical float32 episode."""
        x, u_source = rows.X, rows.U_source
        n_rows = x.shape[0]
        w, z = generated.W, generated.Z
        propensity, a = generated.A_probs, generated.A
        y0, y1, y_obs = generated.Y0, generated.Y1, generated.Y_obs
        cate = generated.CATE
        if self.target_capability == "exact_cate" and cate is None:
            raise RuntimeError(
                f"{self.mechanism_name} declares exact_cate but returned no CATE"
            )
        if self.target_capability == "ite_only" and cate is not None:
            raise RuntimeError(
                f"{self.mechanism_name} declares ite_only but returned a CATE"
            )
        if any(
            value is None
            for value in (
                generated.U,
                generated.V,
                generated.bridge0,
                generated.bridge1,
            )
        ):
            raise RuntimeError(
                f"{self.mechanism_name} did not return paired latent diagnostics"
            )

        for name, tensor in (
            ("X", x),
            ("U_source", u_source),
            ("U", generated.U),
            ("V", generated.V),
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
            U=generated.U.float(),
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
            source_row_indices=rows.source_row_indices,
            U_source=u_source.float(),
            V=generated.V.float(),
            bridge0=generated.bridge0.float(),
            bridge1=generated.bridge1.float(),
        )
