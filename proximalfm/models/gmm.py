# models/gmm.py
import typing
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# CHANGE THIS LINE
from .registry import register_model
from .base import BaseCATEModel, InducedTransformerBlock, TransformerBlock, get_mlp


@register_model("gmm")
class ProximalGMM(BaseCATEModel):
    def __init__(self, config):
        super().__init__()
        self.feature_group_size_x = config.feature_group_size_x
        self.feature_group_size_w = config.feature_group_size_w
        self.feature_group_size_z = config.feature_group_size_z

        self.n_gmm = getattr(config, "n_gmm_components", 1)
        icl_dim = config.embed_dim * config.n_cls_cols

        self.x_embed = nn.Linear(config.feature_group_size_x, config.embed_dim)
        self.w_embed = nn.Linear(config.feature_group_size_w, config.embed_dim)
        self.z_embed = nn.Linear(config.feature_group_size_z, config.embed_dim)

        self.a_embed = nn.Linear(1, config.embed_dim)
        self.a_embed_icl = nn.Linear(1, icl_dim)

        self.y_embed_in = nn.Linear(1, config.embed_dim)
        self.y_embed_icl = nn.Linear(1, icl_dim)

        self.col_blocks = nn.ModuleList(
            [
                InducedTransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.col_nhead,
                    n_inducing=config.n_cls_rows,
                    ssmax=True,
                )
                for _ in range(config.col_num_blocks)
            ]
        )
        self.row_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.row_nhead,
                    use_rope=True,
                )
                for _ in range(config.row_num_blocks)
            ]
        )
        self.icl_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=icl_dim, num_heads=config.icl_nhead, ssmax=True
                )
                for _ in range(config.icl_num_blocks)
            ]
        )

        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, config.n_cls_cols, config.embed_dim)
        )
        self.row_ln = nn.LayerNorm(config.embed_dim)
        self.out_ln = nn.LayerNorm(icl_dim)

        # Output size: 3 * K (logits, mean, log_var for each component)
        self.out_mlp = get_mlp(icl_dim, icl_dim * 2, 3 * self.n_gmm)

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        assert y.shape == a.shape, "y and a must have the same shape"

        n_batch, n_rows, n_cols = x.shape
        n_batch, n_train = y.shape

        # ─── 1. INTERNAL NORMALIZATION (Zero Data Leakage) ───────────────────
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, unbiased=False, keepdim=True) + 1e-8
        y_norm = (y - y_mean) / y_std

        def group_norm(tensor: torch.Tensor) -> torch.Tensor:
            context_slice = tensor[:, :n_train, :]
            tensor_mean = context_slice.mean(dim=1, keepdim=True)
            tensor_std = context_slice.std(dim=1, unbiased=False, keepdim=True) + 1e-8
            return (tensor - tensor_mean) / tensor_std

        x_norm = group_norm(x)
        w_norm = group_norm(w)
        z_norm = group_norm(z)

        # ─── 2. EMBEDDING ────────────────────────────────────────────────────
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        idx_matrix_x = (
            idxs.unsqueeze(1)
            + (2 ** torch.arange(self.feature_group_size_x, device=x.device) - 1)
        ) % n_cols
        x = x_norm[:, :, idx_matrix_x]
        emb_x = self.x_embed(x)

        idxs = torch.arange(w.shape[-1], dtype=torch.long, device=w.device)
        idx_matrix_w = (
            idxs.unsqueeze(1)
            + (2 ** torch.arange(self.feature_group_size_w, device=w.device) - 1)
        ) % w.shape[-1]
        w = w_norm[:, :, idx_matrix_w]
        emb_w = self.w_embed(w)

        idxs = torch.arange(z.shape[-1], dtype=torch.long, device=z.device)
        idx_matrix_z = (
            idxs.unsqueeze(1)
            + (2 ** torch.arange(self.feature_group_size_z, device=z.device) - 1)
        ) % z.shape[-1]
        z = z_norm[:, :, idx_matrix_z]
        emb_z = self.z_embed(z)

        emb_a = self.a_embed(a[:, :, None, None])
        emb_y = self.y_embed_in(y_norm[:, :, None, None])

        emb = torch.concatenate([emb_x, emb_w, emb_z], dim=-2)
        emb[:, :n_train] += emb_y + emb_a

        # ─── 3. TRANSFORMER BLOCKS ───────────────────────────────────────────
        for block in self.col_blocks:
            emb = block.col_attn(emb, kv_max_idx=n_train)

        emb = torch.cat(
            [self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2
        )
        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)
        emb = self.row_blocks[-1].row_attn(emb, q_max_idx=self.row_cls_tokens.size(-2))

        emb = self.row_ln(emb).flatten(-2, -1)

        emb[:, :n_train] += self.y_embed_icl(y_norm[:, :, None]) + self.a_embed_icl(
            a[:, :, None]
        )

        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)
        emb = self.icl_blocks[-1](emb[:, n_train:], emb[:, :n_train])

        out = self.out_mlp(self.out_ln(emb))

        # ─── 4. OUTPUT DENORMALIZATION ───────────────────────────────────────
        y_std_expanded = y_std.unsqueeze(-1) if y_std.dim() == 2 else y_std

        # Split outputs into logits, mean, log_var (each of size K)
        # Shape of out is [batch, n_test, 3 * n_gmm]
        logits = out[..., : self.n_gmm]
        pred_mean_norm = out[..., self.n_gmm : 2 * self.n_gmm]
        pred_log_var_norm = out[..., 2 * self.n_gmm :]

        # ITE is a difference, so we scale by std, but DO NOT add the mean back
        pred_mean_raw = pred_mean_norm * y_std_expanded

        # Variance scales by std^2 -> log_var shifts by 2*log(std)
        pred_log_var_raw = pred_log_var_norm + 2 * torch.log(y_std_expanded)

        # Return all components
        return torch.cat([logits, pred_mean_raw, pred_log_var_raw], dim=-1)

    def loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for GMM using robust log-space math.
        outputs: Tensor of shape [batch, n_test, 3 * K] (logits, mean, log_var)
        target: target values
        """
        if target.dim() == 2:
            target = target.unsqueeze(-1)  # Ensure [batch, n_test, 1]

        logits = outputs[..., : self.n_gmm]
        mu = outputs[..., self.n_gmm : 2 * self.n_gmm]
        log_var = outputs[..., 2 * self.n_gmm :]

        log_pi = F.log_softmax(logits, dim=-1)
        z_diff = target - mu
        z_sq = (z_diff * z_diff) * torch.exp(-log_var)
        log_prob = -0.5 * z_sq - 0.5 * log_var
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)

        return -log_mix.mean()

    def estimate_CATE(
        self,
        x: torch.Tensor,
        a_train: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y_train: torch.Tensor,
        out: typing.Optional[torch.Tensor] = None,
        x_test: typing.Optional[torch.Tensor] = None,
        w_test: typing.Optional[torch.Tensor] = None,
        z_test: typing.Optional[torch.Tensor] = None,
        return_log_variance: bool = False,
    ) -> typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Estimates the CATE.
        If return_log_variance is True, returns (expected_mean, overall_log_variance).
        Otherwise, returns just the expected mean.
        """

        if out is None:
            mode = (
                "explicit_test"
                if (x_test is not None and w_test is not None and z_test is not None)
                else "full_sequence"
            )

            if mode == "explicit_test":
                x_full = torch.cat([x, x_test], dim=1)
                w_full = torch.cat([w, w_test], dim=1)
                z_full = torch.cat([z, z_test], dim=1)
            else:
                x_full, w_full, z_full = x, w, z

            # Forward pass returns [batch, n_test, 3K]
            out = self.forward(x=x_full, a=a_train, w=w_full, z=z_full, y=y_train)

        logits = out[..., : self.n_gmm]
        means = out[..., self.n_gmm : 2 * self.n_gmm]
        log_vars = out[..., 2 * self.n_gmm :]

        pi = F.softmax(logits, dim=-1)
        expected_mean = (pi * means).sum(dim=-1, keepdim=True)

        if return_log_variance:
            variances = torch.exp(log_vars)
            expected_x_squared = (pi * (variances + means * means)).sum(
                dim=-1, keepdim=True
            )

            total_variance = torch.clamp(
                expected_x_squared - expected_mean * expected_mean, min=1e-8
            )
            overall_log_variance = torch.log(total_variance)

            return expected_mean, overall_log_variance

        return expected_mean
