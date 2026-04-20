import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from proximalfm.Structure import ModelConfig

# Assuming these are available in your namespace
from .registry import register_model
from .base import BaseCATEModel, InducedTransformerBlock, TransformerBlock, get_mlp


@register_model("quantile")
class ProximalQuantile(BaseCATEModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.feature_group_size_x = config.feature_group_size_x
        self.feature_group_size_w = config.feature_group_size_w
        self.feature_group_size_z = config.feature_group_size_z

        self.quantiles = getattr(config, "quantiles", [0.1, 0.5, 0.9])
        self.n_quantiles = len(self.quantiles)

        if 0.5 not in self.quantiles:
            raise ValueError(
                "Quantile list must include 0.5 (median) for expected mean estimation."
            )
        self.median_idx = self.quantiles.index(0.5)

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

        self.out_mlp = get_mlp(icl_dim, icl_dim * 2, self.n_quantiles)

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

        # 1. INTERNAL NORMALIZATION (Zero Data Leakage)
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

        # 2. EMBEDDING
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        idx_matrix_x = (idxs.unsqueeze(1) + (2**torch.arange(self.feature_group_size_x, device=x.device) - 1)) % n_cols
        x = x_norm[:, :, idx_matrix_x]
        emb_x = self.x_embed(x)

        idxs = torch.arange(w.shape[-1], dtype=torch.long, device=w.device)
        idx_matrix_w = (idxs.unsqueeze(1) + (2**torch.arange(self.feature_group_size_w, device=w.device) - 1)) % w.shape[-1]
        w = w_norm[:, :, idx_matrix_w]
        emb_w = self.w_embed(w)

        idxs = torch.arange(z.shape[-1], dtype=torch.long, device=z.device)
        idx_matrix_z = (idxs.unsqueeze(1) + (2**torch.arange(self.feature_group_size_z, device=z.device) - 1)) % z.shape[-1]
        z = z_norm[:, :, idx_matrix_z]
        emb_z = self.z_embed(z)

        emb_a = self.a_embed(a[:, :, None, None])
        emb_y = self.y_embed_in(y_norm[:, :, None, None])

        emb = torch.concatenate([emb_x, emb_w, emb_z], dim=-2)
        emb[:, :n_train] += emb_y + emb_a

        # 3. TRANSFORMER BLOCKS
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

        pred_quantiles_norm = self.out_mlp(self.out_ln(emb))

        # 4. OUTPUT DENORMALIZATION
        y_std_expanded = y_std.unsqueeze(-1) if y_std.dim() == 2 else y_std

        # Scale back the predictions using the context factual standard deviation
        pred_quantiles_raw = pred_quantiles_norm * y_std_expanded

        return pred_quantiles_raw

    def loss(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Pinball loss across all predicted quantiles.
        outputs: Tensor of shape [batch, n_test, n_quantiles]
        target: target values [batch, n_test, 1]
        """
        if target.dim() == 2:
            target = target.unsqueeze(-1)

        diff = target - outputs

        # Create tau tensor matching the device of outputs
        tau = torch.tensor(
            self.quantiles, device=outputs.device, dtype=outputs.dtype
        ).view(1, 1, -1)

        # Pinball loss formula: max(tau * diff, (tau - 1) * diff)
        pinball_loss = torch.max(tau * diff, (tau - 1.0) * diff)

        return pinball_loss.mean()

    def estimate_CATE(
        self,
        x: torch.Tensor,
        a_train: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y_train: torch.Tensor,
        x_test: typing.Optional[torch.Tensor] = None,
        w_test: typing.Optional[torch.Tensor] = None,
        z_test: typing.Optional[torch.Tensor] = None,
        return_log_variance: bool = False,
    ) -> typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Estimates the CATE using the median (0.5 quantile).
        If return_log_variance is True, approximates variance using the interquantile range.
        """
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

        # Forward pass returns [batch, n_test, n_quantiles]
        out = self.forward(x=x_full, a=a_train, w=w_full, z=z_full, y=y_train)

        # The point estimate for CATE is the median
        expected_mean = out[..., self.median_idx : self.median_idx + 1]

        if return_log_variance:
            if 0.1 in self.quantiles and 0.9 in self.quantiles:
                idx_low = self.quantiles.index(0.1)
                idx_high = self.quantiles.index(0.9)

                q_low = out[..., idx_low : idx_low + 1]
                q_high = out[..., idx_high : idx_high + 1]

                sigma = (q_high - q_low) / 2.56
                variance = torch.clamp(sigma * sigma, min=1e-8)
                overall_log_variance = torch.log(variance)
            else:
                overall_log_variance = torch.zeros_like(expected_mean) - 10.0

            return expected_mean, overall_log_variance

        return expected_mean
