"""Compose paired empirical ``(X, U)`` rows with a synthetic proximal SCM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch

from semi_synt_eval.mechanisms import MechanismConfig, build_mechanism
from semi_synt_eval.source import PreparedCSV

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
        mechanism: str = "linear",
        mechanism_config: MechanismConfig,
        row_sampling: str = "auto",
    ) -> None:
        if row_sampling not in {"auto", "without_replacement", "bootstrap"}:
            raise ValueError(
                "row_sampling must be auto, without_replacement, or bootstrap"
            )
        self.source = source
        self.name = f"real_csv:{source.path.stem}"
        self.mechanism_name = mechanism
        self.mechanism_config = mechanism_config
        self.overlap_eps = float(self.mechanism_config.treatment.overlap_epsilon)
        self.row_sampling = row_sampling
        probe = build_mechanism(
            mechanism,
            self.d_X,
            self.d_U,
            torch.Generator().manual_seed(0),
            config=self.mechanism_config,
        )
        self.target_capability = probe.target_capability
        self.options: dict[str, Any] = {
            "source_file": source.path.name,
            "source_sha256": source.digest,
            "source_rows": int(source.values.shape[0]),
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
        n_source = int(self.source.values.shape[0])
        replace = self.row_sampling == "bootstrap" or (
            self.row_sampling == "auto" and n > n_source
        )
        if not replace and n > n_source:
            raise ValueError(
                f"{self.source.path.name} has {n_source} rows but episode needs {n}; "
                "use row_sampling=auto or bootstrap"
            )
        if replace:
            return torch.randint(n_source, (n,), generator=rng)
        return torch.randperm(n_source, generator=rng)[:n]

    def generate(self, *, seed: int, seq_len: int) -> Episode:
        """Generate one observational dataset and its potential outcomes."""
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        rng = torch.Generator().manual_seed(int(seed))
        rows = self._sample_indices(seq_len, rng)
        x = self.source.X[rows]
        u = self.source.U[rows]
        mechanism = build_mechanism(
            self.mechanism_name,
            self.d_X,
            self.d_U,
            rng,
            config=self.mechanism_config,
        )

        w, z = mechanism.proxies(x, u)
        propensity = mechanism.treatment_probability(x, u, z).clamp(
            self.overlap_eps, 1.0 - self.overlap_eps
        )
        a = (torch.rand(seq_len, 1, generator=rng) < propensity).float()
        y0, y1 = mechanism.potential_outcomes(x, u, w)
        y_obs = a * y1 + (1.0 - a) * y0
        cate = mechanism.cate(x)
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
            if tensor.shape[0] != seq_len or not torch.isfinite(tensor).all():
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
        )
