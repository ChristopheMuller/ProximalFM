from __future__ import annotations

import random
import numpy as np

import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        if torch.is_tensor(std):
            self._is_zero = bool((std == 0).all())
        else:
            self._is_zero = std == 0

    def forward(self, X):
        if self._is_zero:
            return X
        return X + torch.randn_like(X).mul_(self.std)


class XSampler:
    """Input sampler for generating features for prior datasets.

    Supports multiple feature distribution types:

    - Normal: Standard normal distribution
    - Multinomial: Categorical features with random number of categories
    - Zipf: Power law distributed features
    - Mixed: Random combination of the above

    Parameters
    ----------
    seq_len : int
        Length of sequence to generate.

    num_features : int
        Number of features to generate.

    pre_stats : bool
        Whether to pre-generate statistics for the input features.

    sampling : str, default='mixed'
        Feature sampling strategy ('normal', 'mixed', 'uniform').

    device : str, default='cpu'
        Device to store tensors on.
    """

    def __init__(
        self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"
    ):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(
            means, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(
            0
        )

    def sample(self, return_numpy=False):
        """Generate features according to the specified sampling strategy.

        Returns
        -------
        X : torch.Tensor
            Generated features of shape ``(seq_len, num_features)``.
        """
        samplers = {
            "normal": self.sample_normal_all,
            "mixed": self.sample_mixed,
            "uniform": self.sample_uniform,
        }
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()

        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            noise = torch.randn(self.seq_len, self.num_features, device=self.device)
            X = (self.means + self.stds * noise).float()
        else:
            X = torch.normal(
                0.0, 1.0, (self.seq_len, self.num_features), device=self.device
            ).float()
        return X

    def sample_uniform(self):
        """Generate uniformly distributed features."""
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        """Generate normally distributed features.

        Parameters
        ----------
        n : int
            Index of the feature to generate
        """

        if self.pre_stats:
            noise = torch.randn(self.seq_len, device=self.device)
            return (self.means[0, n] + self.stds[0, n] * noise).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        """Generate categorical features.

        The column is standardised, and two draws make that standardisation
        degenerate: every sample landing in the same category (``std == 0``,
        reachable whenever ``torch.rand`` gives one category almost all the
        mass), and ``seq_len == 1`` (no unbiased ``std`` at all, hence NaN).
        Both would emit an all-NaN column, which then flows into a mechanism's
        exogenous-noise input -- and ``MLPSCM`` only ``nan_to_num``s its
        *output*, so ``process_features`` sees the NaN first.

        Clamped rather than offset by an epsilon: ``std.clamp(min=1e-8)``
        leaves every non-degenerate draw bit-identical, where ``std + 1e-8``
        would perturb them all. A degenerate draw yields the constant-zero
        column (``x - mean`` is already 0), which is the right answer for a
        noise input carrying no information.
        """
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        std, mean = torch.std_mean(x)
        scale = torch.nan_to_num(std, nan=0.0).clamp(min=1e-8)
        return (x - mean) / scale

    def sample_zipf(self):
        """Generate Zipf-distributed features."""
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        """Generate features with mixed distributions."""
        zipf_p, multi_p, normal_p = (
            random.random() * 0.66,
            random.random() * 0.66,
            random.random() * 0.66,
        )
        X = torch.empty(self.seq_len, self.num_features, device=self.device)
        normal_cols = []
        for n in range(self.num_features):
            if random.random() > normal_p:
                normal_cols.append(n)
            elif random.random() > multi_p:
                X[:, n] = self.sample_multinomial()
            elif random.random() > zipf_p:
                X[:, n] = self.sample_zipf()
            else:
                X[:, n] = torch.rand((self.seq_len,), device=self.device)
        if normal_cols:
            noise = torch.randn(self.seq_len, len(normal_cols), device=self.device)
            if self.pre_stats:
                cols = torch.tensor(normal_cols, device=self.device)
                noise = self.means[0, cols] + self.stds[0, cols] * noise
            X[:, normal_cols] = noise
        return X
