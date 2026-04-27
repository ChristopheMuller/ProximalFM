# models/gmm.py
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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

        self.role_ya_gates = nn.Parameter(
            torch.full((3, config.embed_dim), 2.0)
        )

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

    def _normalize_inputs(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize y and feature groups using only the training/context rows.

        Returns:
            x_norm, w_norm, z_norm, y_norm, y_std
        """
        n_train = y.shape[1]

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

        return x_norm, w_norm, z_norm, y_norm, y_std

    def _circular_group_features(
        self,
        tensor: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """
        Circular feature grouping.

        Input:
            tensor: [batch, n_rows, n_features]

        Output:
            grouped: [batch, n_rows, n_features, group_size]
        """
        n_features = tensor.shape[-1]
        device = tensor.device

        idxs = torch.arange(n_features, dtype=torch.long, device=device)
        offsets = 2 ** torch.arange(group_size, dtype=torch.long, device=device) - 1

        idx_matrix = (idxs.unsqueeze(1) + offsets.unsqueeze(0)) % n_features

        return tensor[:, :, idx_matrix]

    def _embed_inputs(
        self,
        x_norm: torch.Tensor,
        a: torch.Tensor,
        w_norm: torch.Tensor,
        z_norm: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed X, W, Z and inject treatment/outcome context into training rows.

        Returns:
            emb: [batch, n_rows, n_feature_tokens, embed_dim]
        """
        n_train = y_norm.shape[1]

        x_grouped = self._circular_group_features(
            x_norm, self.feature_group_size_x
        )
        w_grouped = self._circular_group_features(
            w_norm, self.feature_group_size_w
        )
        z_grouped = self._circular_group_features(
            z_norm, self.feature_group_size_z
        )

        emb_x = self.x_embed(x_grouped)
        emb_w = self.w_embed(w_grouped)
        emb_z = self.z_embed(z_grouped)

        emb_a = self.a_embed(a[:, :, None, None])
        emb_y = self.y_embed_in(y_norm[:, :, None, None])

        ya_context = emb_y + emb_a

        role_gates = torch.sigmoid(self.role_ya_gates).view(
            3, 1, 1, 1, -1
        )

        emb_x[:, :n_train] = emb_x[:, :n_train] + role_gates[0] * ya_context
        emb_w[:, :n_train] = emb_w[:, :n_train] + role_gates[1] * ya_context
        emb_z[:, :n_train] = emb_z[:, :n_train] + role_gates[2] * ya_context

        emb = torch.cat([emb_x, emb_w, emb_z], dim=-2)

        return emb

    def _run_col_attention(
        self,
        emb: torch.Tensor,
        n_train: int,
    ) -> torch.Tensor:
        """
        Run column-wise attention.

        Input:
            emb: [batch, n_rows, n_feature_tokens, embed_dim]

        Output:
            emb: [batch, n_rows, n_feature_tokens, embed_dim]
        """
        for block in self.col_blocks:
            emb = block.col_attn(emb, kv_max_idx=n_train)

        return emb

    def _run_row_attention(
        self,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add row CLS tokens and run row-wise attention.

        Input:
            emb: [batch, n_rows, n_feature_tokens, embed_dim]

        Output:
            row_repr: [batch, n_rows, n_cls_cols * embed_dim]
        """
        n_batch, n_rows = emb.shape[:2]

        cls_tokens = self.row_cls_tokens.expand(n_batch, n_rows, -1, -1)

        emb = torch.cat([cls_tokens, emb], dim=2)

        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)

        emb = self.row_blocks[-1].row_attn(
            emb,
            q_max_idx=self.row_cls_tokens.size(-2),
        )

        row_repr = self.row_ln(emb).flatten(-2, -1)

        return row_repr

    def _add_icl_context(
        self,
        row_repr: torch.Tensor,
        a: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add y/a embeddings to training row representations before ICL attention.

        Input:
            row_repr: [batch, n_rows, icl_dim]

        Output:
            row_repr: [batch, n_rows, icl_dim]
        """
        n_train = y_norm.shape[1]

        row_repr = row_repr.clone()

        row_repr[:, :n_train] = row_repr[:, :n_train] + self.y_embed_icl(
            y_norm[:, :, None]
        ) + self.a_embed_icl(a[:, :, None])

        return row_repr

    def _run_icl_attention(
        self,
        row_repr: torch.Tensor,
        n_train: int,
    ) -> torch.Tensor:
        """
        Run dataset-level in-context attention.

        Input:
            row_repr: [batch, n_rows, icl_dim]

        Output:
            test_repr: [batch, n_test, icl_dim]
        """
        emb = row_repr

        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)

        emb = self.icl_blocks[-1](
            emb[:, n_train:],
            emb[:, :n_train],
        )

        return emb

    def _decode_gmm(
        self,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode normalized GMM parameters.

        Output:
            out: [batch, n_test, 3 * n_gmm]
        """
        return self.out_mlp(self.out_ln(emb))

    def _denormalize_outputs(
        self,
        out: torch.Tensor,
        y_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalized GMM mean/log-variance back to raw outcome scale.

        Important:
            CATE/ITE is a difference, so the mean is multiplied by y_std
            but y_mean is not added back.
        """
        y_std_expanded = y_std.unsqueeze(-1)

        logits = out[..., : self.n_gmm]
        pred_mean_norm = out[..., self.n_gmm : 2 * self.n_gmm]
        pred_log_var_norm = out[..., 2 * self.n_gmm :]

        pred_mean_raw = pred_mean_norm * y_std_expanded
        pred_log_var_raw = pred_log_var_norm + 2 * torch.log(y_std_expanded)

        return torch.cat(
            [logits, pred_mean_raw, pred_log_var_raw],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        assert y.shape == a.shape, "y and a must have the same shape"

        n_train = y.shape[1]

        x_norm, w_norm, z_norm, y_norm, y_std = self._normalize_inputs(
            x=x,
            w=w,
            z=z,
            y=y,
        )

        emb = self._embed_inputs(
            x_norm=x_norm,
            a=a,
            w_norm=w_norm,
            z_norm=z_norm,
            y_norm=y_norm,
        )

        emb = self._run_col_attention(
            emb=emb,
            n_train=n_train,
        )

        row_repr = self._run_row_attention(emb)

        row_repr = self._add_icl_context(
            row_repr=row_repr,
            a=a,
            y_norm=y_norm,
        )

        test_repr = self._run_icl_attention(
            row_repr=row_repr,
            n_train=n_train,
        )

        out_norm = self._decode_gmm(test_repr)

        out = self._denormalize_outputs(
            out=out_norm,
            y_std=y_std,
        )

        return out

    def loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for GMM using robust log-space math.
        outputs: Tensor of shape [batch, n_test, 3 * K] (logits, mean, log_var)
        target: target values
        """
        if target.dim() == 2:
            target = target.unsqueeze(-1)

        logits = outputs[..., : self.n_gmm]
        mu = outputs[..., self.n_gmm : 2 * self.n_gmm]
        log_var = outputs[..., 2 * self.n_gmm :]

        log_var = log_var.clamp(min=-12.0, max=8.0)
        log_pi = F.log_softmax(logits, dim=-1)

        z_sq = (target - mu).pow(2) * torch.exp(-log_var)

        # Proper Gaussian log density.
        log_prob = -0.5 * (z_sq + log_var + math.log(2.0 * math.pi))

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
        log_vars = log_vars.clamp(min=-12.0, max=8.0)

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
