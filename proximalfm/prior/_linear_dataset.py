from __future__ import annotations

import math
import os
import sys
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import loguniform
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


class LinearPrior:
    def __init__(
        self,
        batch_size: int = 256,
        min_dim_U: int = 3,
        max_dim_U: int = 10,
        min_dim_W: int = 3,
        max_dim_W: int = 10,
        min_dim_Z: int = 3,
        max_dim_Z: int = 10,
        min_dim_X: int = 5,
        max_dim_X: int = 20,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_dim_U <= max_dim_U, "Invalid U dimension range"
        assert min_dim_W <= max_dim_W, "Invalid W dimension range"
        assert min_dim_Z <= max_dim_Z, "Invalid Z dimension range"
        assert min_dim_X <= max_dim_X, "Invalid X dimension range"

        self.min_dim_U = min_dim_U
        self.max_dim_U = max_dim_U
        self.min_dim_W = min_dim_W
        self.max_dim_W = max_dim_W
        self.min_dim_Z = min_dim_Z
        self.max_dim_Z = max_dim_Z
        self.min_dim_X = min_dim_X
        self.max_dim_X = max_dim_X

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(
        min_train_size: Union[int, float], max_train_size: Union[int, float]
    ) -> None:
        if not isinstance(min_train_size, (int, float)) or not isinstance(
            max_train_size, (int, float)
        ):
            raise TypeError("Training sizes must be int or float")

        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert 0 < min_train_size < max_train_size < 1
        else:
            raise ValueError("Both training sizes must be of the same type.")

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int],
        max_seq_len: int,
        log: bool = False,
        replay_small: bool = False,
    ) -> int:
        if min_seq_len is None:
            return max_seq_len
        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
        return seq_len

    @staticmethod
    def sample_train_size(
        min_train_size: Union[int, float],
        max_train_size: Union[int, float],
        seq_len: int,
    ) -> int:
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            return np.random.randint(min_train_size, max_train_size)
        else:
            train_size = np.random.uniform(min_train_size, max_train_size)
            return int(seq_len * train_size)

    @staticmethod
    def adjust_max_features(
        seq_len: int, max_x: int, max_w: int, max_z: int
    ) -> Tuple[int, int, int]:
        total_features = max_x + max_w + max_z + 1
        if seq_len <= 10240:
            limit = 100
        elif 10240 < seq_len <= 20000:
            limit = 80
        elif 20000 < seq_len <= 30000:
            limit = 60
        elif 30000 < seq_len <= 40000:
            limit = 40
        elif 40000 < seq_len <= 50000:
            limit = 30
        elif 50000 < seq_len <= 60000:
            limit = 20
        elif 60000 < seq_len <= 65000:
            limit = 15
        else:
            limit = 10

        if total_features <= limit:
            return max_x, max_w, max_z

        scale = limit / total_features
        return (
            max(1, int(max_x * scale)),
            max(1, int(max_w * scale)),
            max(1, int(max_z * scale)),
        )

    @staticmethod
    def filter_constant_features_and_pad(
        X: Tensor, d: Tensor, max_d: int
    ) -> Tuple[Tensor, Tensor]:
        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            if xi.ndim == 1:
                xi = xi.unsqueeze(0)
            actual_di = min(int(di), xi.shape[-1])
            xi_active = xi[:, :actual_di]
            num_features = xi.shape[-1]
            unique_mask = [
                len(torch.unique(xi_active[:, j])) > 1 for j in range(actual_di)
            ]
            di_new = sum(unique_mask)
            xi_unique = xi_active[:, unique_mask]
            xi_new = F.pad(
                xi_unique, (0, num_features - di_new), mode="constant", value=0
            )
            return xi_new, torch.tensor(di_new, device=xi.device)

        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        if X_new.shape[-1] < max_d:
            X_new = F.pad(X_new, (0, max_d - X_new.shape[-1]), mode="constant", value=0)
        d_new = torch.tensor([min(int(di), max_d) for di in d_new], device=X_new.device)
        return X_new, d_new


class LinearSCMPrior(LinearPrior):
    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_dim_U: int = 3,
        max_dim_U: int = 5,
        min_dim_W: int = 3,
        max_dim_W: int = 5,
        min_dim_Z: int = 3,
        max_dim_Z: int = 5,
        min_dim_X: int = 5,
        max_dim_X: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.device = device

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, ...]:
        seq_len = params["seq_len"]
        dU, dX, dW, dZ = (
            params["dim_U"],
            params["dim_X"],
            params["dim_W"],
            params["dim_Z"],
        )

        def safe_normalize(tensor: Tensor) -> Tensor:
            return (tensor - tensor.mean(dim=0, keepdim=True)) / (
                tensor.std(dim=0, keepdim=True) + 1e-8
            )

        while True:
            U = torch.randn(seq_len, dU, device=self.device)
            U_ = safe_normalize(U)

            W_weights = torch.rand(dU, dW, device=self.device) - 0.5
            W = U_ @ W_weights + torch.randn(seq_len, dW, device=self.device) * 0.2
            W_ = safe_normalize(W)

            Z_weights = torch.rand(dU, dZ, device=self.device) - 0.5
            Z = U_ @ Z_weights + torch.randn(seq_len, dZ, device=self.device) * 0.2
            Z_ = safe_normalize(Z)

            X_weights = torch.rand(dU, dX, device=self.device) - 0.5
            X = U_ @ X_weights + torch.randn(seq_len, dX, device=self.device) * 0.2
            X_ = safe_normalize(X)

            A_input = torch.cat([U_, X_], dim=-1)
            A_weights = (torch.rand(dU + dX, 1, device=self.device) * 0.8) - 0.4
            A_logits = (A_input @ A_weights).squeeze(-1)

            A_logits = (A_logits - A_logits.mean()) / (A_logits.std() + 1e-8)
            A_logits = A_logits * params.get("propensity_scale", 1.0) + params.get(
                "propensity_shift", 0.0
            )
            A_probs = torch.sigmoid(A_logits).clamp(1e-3, 1.0 - 1e-3)
            A = torch.bernoulli(A_probs).long()

            Y_input = torch.cat([U_, X_], dim=-1)
            beta_0 = torch.rand(dU + dX, 1, device=self.device) - 0.5
            beta_cate = (torch.rand(dU + dX, 1, device=self.device) * 0.6) - 0.3

            y0_raw = (Y_input @ beta_0).squeeze(-1) + torch.randn(
                seq_len, device=self.device
            ) * 0.1
            cate_raw = (Y_input @ beta_cate).squeeze(-1) + torch.randn(
                seq_len, device=self.device
            ) * 0.05
            y1_raw = y0_raw + cate_raw

            gamma_hetero = torch.rand(1, device=self.device)
            alphas = torch.rand(seq_len, device=self.device)
            ate_raw = (y1_raw - y0_raw).mean()

            y1 = (alphas + (1 - alphas) * gamma_hetero) * y1_raw + (
                1 - gamma_hetero
            ) * (1 - alphas) * (y0_raw + ate_raw)
            y0 = ((1 - alphas) + alphas * gamma_hetero) * y0_raw + (
                1 - gamma_hetero
            ) * alphas * (y1_raw - ate_raw)
            y = torch.where(A == 1, y1, y0)

            U_padded = F.pad(U, (0, params["max_dim_U"] - dU), value=0.0).unsqueeze(0)
            X_padded = F.pad(X, (0, params["max_dim_X"] - dX), value=0.0).unsqueeze(0)
            W_padded = F.pad(W, (0, params["max_dim_W"] - dW), value=0.0).unsqueeze(0)
            Z_padded = F.pad(Z, (0, params["max_dim_Z"] - dZ), value=0.0).unsqueeze(0)

            d_U_t = torch.tensor([dU], device=self.device, dtype=torch.long)
            d_X_t = torch.tensor([dX], device=self.device, dtype=torch.long)
            d_W_t = torch.tensor([dW], device=self.device, dtype=torch.long)
            d_Z_t = torch.tensor([dZ], device=self.device, dtype=torch.long)

            U_padded, d_U_t = self.filter_constant_features_and_pad(
                U_padded, d_U_t, params["max_dim_U"]
            )
            X_padded, d_X_t = self.filter_constant_features_and_pad(
                X_padded, d_X_t, params["max_dim_X"]
            )
            W_padded, d_W_t = self.filter_constant_features_and_pad(
                W_padded, d_W_t, params["max_dim_W"]
            )
            Z_padded, d_Z_t = self.filter_constant_features_and_pad(
                Z_padded, d_Z_t, params["max_dim_Z"]
            )

            if (d_X_t > 0).all() and (d_W_t > 0).all() and (d_Z_t > 0).all():
                return (
                    U_padded.squeeze(0),
                    X_padded.squeeze(0),
                    W_padded.squeeze(0),
                    Z_padded.squeeze(0),
                    A,
                    A_probs,
                    y,
                    y0,
                    y1,
                    d_U_t.squeeze(0),
                    d_X_t.squeeze(0),
                    d_W_t.squeeze(0),
                    d_Z_t.squeeze(0),
                )

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, ...]:
        batch_size = batch_size or self.batch_size
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)
        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        param_list = []
        global_seq_len, global_train_size = None, None
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len,
                self.max_seq_len,
                log=self.log_seq_len,
                replay_small=self.replay_small,
            )
            global_train_size = self.sample_train_size(
                self.min_train_size, self.max_train_size, global_seq_len
            )

        for gp_idx in range(num_gps):
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len,
                    self.max_seq_len,
                    log=self.log_seq_len,
                    replay_small=self.replay_small,
                )
                gp_train_size = self.sample_train_size(
                    self.min_train_size, self.max_train_size, gp_seq_len
                )
                gp_max_X, gp_max_W, gp_max_Z = self.adjust_max_features(
                    gp_seq_len, self.max_dim_X, self.max_dim_W, self.max_dim_Z
                )
            else:
                gp_seq_len, gp_train_size = global_seq_len, global_train_size
                gp_max_X, gp_max_W, gp_max_Z = (
                    self.max_dim_X,
                    self.max_dim_W,
                    self.max_dim_Z,
                )

            gp_max_W = max(gp_max_W, self.max_dim_U)
            gp_max_Z = max(gp_max_Z, self.max_dim_U)
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            for subgp_idx in range(num_subgps_in_gp):
                actual_subgp_size = min(
                    size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp
                )
                if actual_subgp_size <= 0:
                    break

                dim_U = np.random.randint(self.min_dim_U, self.max_dim_U + 1)
                min_W_req = max(self.min_dim_W, dim_U)
                dim_W = np.random.randint(min_W_req, max(gp_max_W, min_W_req) + 1)
                min_Z_req = max(self.min_dim_Z, dim_U)
                dim_Z = np.random.randint(min_Z_req, max(gp_max_Z, min_Z_req) + 1)
                dim_X = np.random.randint(
                    self.min_dim_X, max(self.min_dim_X, gp_max_X) + 1
                )

                for _ in range(actual_subgp_size):
                    param_list.append(
                        {
                            "seq_len": gp_seq_len,
                            "train_size": gp_train_size,
                            "max_dim_U": self.max_dim_U,
                            "max_dim_X": gp_max_X,
                            "max_dim_W": gp_max_W,
                            "max_dim_Z": gp_max_Z,
                            "dim_U": dim_U,
                            "dim_W": dim_W,
                            "dim_Z": dim_Z,
                            "dim_X": dim_X,
                        }
                    )

        results = [self.generate_dataset(p) for p in param_list]
        (U_l, X_l, W_l, Z_l, A_l, Ap_l, Y_l, Y0_l, Y1_l, dU_l, dX_l, dW_l, dZ_l) = zip(
            *results
        )

        if self.seq_len_per_gp:
            U, X, W, Z = (
                nested_tensor(list(U_l), device=self.device),
                nested_tensor(list(X_l), device=self.device),
                nested_tensor(list(W_l), device=self.device),
                nested_tensor(list(Z_l), device=self.device),
            )
            A, A_probs = nested_tensor(list(A_l), device=self.device), nested_tensor(
                list(Ap_l), device=self.device
            )
            Y_obs, Y0, Y1 = (
                nested_tensor(list(Y_l), device=self.device),
                nested_tensor(list(Y0_l), device=self.device),
                nested_tensor(list(Y1_l), device=self.device),
            )
        else:
            U, X, W, Z = (
                torch.stack(U_l),
                torch.stack(X_l),
                torch.stack(W_l),
                torch.stack(Z_l),
            )
            A, A_probs = torch.stack(A_l), torch.stack(Ap_l)
            Y_obs, Y0, Y1 = torch.stack(Y_l), torch.stack(Y0_l), torch.stack(Y1_l)

        return (
            U,
            X,
            W,
            Z,
            A,
            A_probs,
            Y_obs,
            Y0,
            Y1,
            torch.stack(dU_l),
            torch.stack(dX_l),
            torch.stack(dW_l),
            torch.stack(dZ_l),
            torch.tensor(
                [p["seq_len"] for p in param_list], device=self.device, dtype=torch.long
            ),
            torch.tensor(
                [p["train_size"] for p in param_list],
                device=self.device,
                dtype=torch.long,
            ),
        )


class LinearPriorDataset(IterableDataset):
    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_dim_U: int = 3,
        max_dim_U: int = 10,
        min_dim_W: int = 3,
        max_dim_W: int = 10,
        min_dim_Z: int = 3,
        max_dim_Z: int = 10,
        min_dim_X: int = 5,
        max_dim_X: int = 20,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        self.prior = LinearSCMPrior(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            batch_size_per_subgp=batch_size_per_subgp,
            min_dim_U=min_dim_U,
            max_dim_U=max_dim_U,
            min_dim_W=min_dim_W,
            max_dim_W=max_dim_W,
            min_dim_Z=min_dim_Z,
            max_dim_Z=max_dim_Z,
            min_dim_X=min_dim_X,
            max_dim_X=max_dim_X,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            seq_len_per_gp=seq_len_per_gp,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
            device=device,
        )
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, ...]:
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> LinearPriorDataset:
        return self

    def __next__(self) -> Tuple[Tensor, ...]:
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        return (
            f"LinearPriorDataset(\n"
            f"  batch_size: {self.batch_size}\n"
            f"  features U: {self.prior.min_dim_U} - {self.prior.max_dim_U}\n"
            f"  features X: {self.prior.min_dim_X} - {self.prior.max_dim_X}\n"
            f"  device: {self.device}\n"
            f")"
        )
