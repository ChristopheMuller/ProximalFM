import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .base import BaseCATEModel, InducedTransformerBlock, TransformerBlock, get_mlp, DistributionAwareEmbedder


@register_model("cate_decomposition")
class ProximalDecomposedCATE(BaseCATEModel):
    def __init__(self, config):
        super().__init__()
        self.feature_group_size_x = config.feature_group_size_x
        self.feature_group_size_w = config.feature_group_size_w
        self.feature_group_size_z = config.feature_group_size_z

        self.n_gmm = getattr(config, "n_gmm_components", 1)
        icl_dim = config.embed_dim * config.n_cls_cols

        self.use_distribution_aware_embedding = getattr(
            config, "use_distribution_aware_embedding", False
        )

        dist_embed_num_blocks = getattr(config, "dist_embed_num_blocks", 1)
        dist_embed_n_inducing = getattr(config, "dist_embed_n_inducing", 16)
        dist_embed_nhead = getattr(config, "dist_embed_nhead", config.col_nhead)

        if self.use_distribution_aware_embedding:
            self.x_embed = DistributionAwareEmbedder(
                group_size=config.feature_group_size_x,
                embed_dim=config.embed_dim,
                num_heads=dist_embed_nhead,
                num_blocks=dist_embed_num_blocks,
                n_inducing=dist_embed_n_inducing,
            )
            self.w_embed = DistributionAwareEmbedder(
                group_size=config.feature_group_size_w,
                embed_dim=config.embed_dim,
                num_heads=dist_embed_nhead,
                num_blocks=dist_embed_num_blocks,
                n_inducing=dist_embed_n_inducing,
            )
            self.z_embed = DistributionAwareEmbedder(
                group_size=config.feature_group_size_z,
                embed_dim=config.embed_dim,
                num_heads=dist_embed_nhead,
                num_blocks=dist_embed_num_blocks,
                n_inducing=dist_embed_n_inducing,
            )
        else:
            self.x_embed = nn.Linear(config.feature_group_size_x, config.embed_dim)
            self.w_embed = nn.Linear(config.feature_group_size_w, config.embed_dim)
            self.z_embed = nn.Linear(config.feature_group_size_z, config.embed_dim)

        self.a_embed = nn.Linear(1, config.embed_dim)
        self.a_embed_icl = nn.Linear(1, icl_dim)

        self.y_embed_in = nn.Linear(1, config.embed_dim)
        self.y_embed_icl = nn.Linear(1, icl_dim)

        self.role_ya_gates = nn.Parameter(torch.full((3, config.embed_dim), 2.0))

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

        self.ate_head = nn.Linear(icl_dim, 1)
        self.delta_head = nn.Linear(icl_dim, 1)  # TODO: change the delta to a gmm

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

    def _compute_context_stats(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute normalization statistics from training/context rows only.

        Inputs here should be training/context rows.
        """
        stats = {}

        stats["x_mean"] = x.mean(dim=1, keepdim=True)
        stats["x_std"] = x.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        stats["w_mean"] = w.mean(dim=1, keepdim=True)
        stats["w_std"] = w.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        stats["z_mean"] = z.mean(dim=1, keepdim=True)
        stats["z_std"] = z.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        stats["y_mean"] = y.mean(dim=1, keepdim=True)
        stats["y_std"] = y.std(dim=1, unbiased=False, keepdim=True) + 1e-8

        return stats

    def _normalize_with_stats(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        stats: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize rows using cached context statistics.
        """
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        w_norm = (w - stats["w_mean"]) / stats["w_std"]
        z_norm = (z - stats["z_mean"]) / stats["z_std"]

        return x_norm, w_norm, z_norm

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

    def _embed_feature_families(
        self,
        x_norm: torch.Tensor,
        w_norm: torch.Tensor,
        z_norm: torch.Tensor,
        n_train: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed X, W, Z separately, without injecting y/a context.

        Returns:
            emb_x: [batch, n_rows, n_x_tokens, embed_dim]
            emb_w: [batch, n_rows, n_w_tokens, embed_dim]
            emb_z: [batch, n_rows, n_z_tokens, embed_dim]
        """
        x_grouped = self._circular_group_features(x_norm, self.feature_group_size_x)
        w_grouped = self._circular_group_features(w_norm, self.feature_group_size_w)
        z_grouped = self._circular_group_features(z_norm, self.feature_group_size_z)

        if self.use_distribution_aware_embedding:
            emb_x = self.x_embed(x_grouped, n_train)
            emb_w = self.w_embed(w_grouped, n_train)
            emb_z = self.z_embed(z_grouped, n_train)
        else:
            emb_x = self.x_embed(x_grouped)
            emb_w = self.w_embed(w_grouped)
            emb_z = self.z_embed(z_grouped)

        return emb_x, emb_w, emb_z

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

        emb_x, emb_w, emb_z = self._embed_feature_families(
            x_norm=x_norm,
            w_norm=w_norm,
            z_norm=z_norm,
            n_train=n_train,
        )

        emb_a = self.a_embed(a[:, :, None, None])
        emb_y = self.y_embed_in(y_norm[:, :, None, None])

        ya_context = emb_y + emb_a

        role_gates = torch.sigmoid(self.role_ya_gates).view(3, 1, 1, 1, -1)

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

        row_repr[:, :n_train] = (
            row_repr[:, :n_train]
            + self.y_embed_icl(y_norm[:, :, None])
            + self.a_embed_icl(a[:, :, None])
        )

        return row_repr

    def _run_icl_attention(
        self,
        row_repr: torch.Tensor,
        n_train: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run dataset-level in-context attention.
        
        Returns:
            context_repr: [batch, n_train, icl_dim]
            test_repr: [batch, n_test, icl_dim]
        """
        emb = row_repr

        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)


        context_repr = emb[:, :n_train]

        # Final block processes the query/test set using the context set as KV
        test_repr = self.icl_blocks[-1](
            emb[:, n_train:],
            context_repr,
        )

        return context_repr, test_repr

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        context_repr, test_repr = self._run_icl_attention(
            row_repr=row_repr,
            n_train=n_train,
        )
        
        context_repr = self.out_ln(context_repr)
        test_repr = self.out_ln(test_repr)

        # 1. Predict ATE from Context
        pooled_context = context_repr.mean(dim=1)    # shape: [batch, icl_dim]
        pred_context_ate = self.ate_head(pooled_context)  # shape: [batch, 1]
        
        # 2. Predict Delta for Query
        pred_delta = self.delta_head(test_repr).squeeze(-1)  # shape: [batch, n_test]

        return pred_context_ate, pred_delta

    def loss(
        self, 
        outputs: tuple[torch.Tensor, torch.Tensor], 
        target_query: torch.Tensor,
        target_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Dual loss: MSE for Context ATE + MSE for Query CATE.
        """
        pred_context_ate, pred_delta = outputs

        # Ensure shapes are [batch, n_rows]
        if target_query.dim() == 3:
            target_query = target_query.squeeze(-1)
        if target_context.dim() == 3:
            target_context = target_context.squeeze(-1)

        true_context_ate = target_context.mean(dim=1, keepdim=True)
        loss_ate = F.mse_loss(pred_context_ate, true_context_ate)

        pred_query_cate = pred_context_ate + pred_delta
        loss_cate = F.mse_loss(pred_query_cate, target_query)

        return loss_ate + loss_cate

    def estimate_CATE(
        self,
        x: torch.Tensor,
        a_train: torch.Tensor,
        w: torch.Tensor,
        z: torch.Tensor,
        y_train: torch.Tensor,
        return_log_variance: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate CATE for the query set as Context ATE + Query Delta.
        """
        #TODO: retrieve the output of forward as argument here to avoid double forward pass
        pred_context_ate, pred_delta = self.forward(x=x, a=a_train, w=w, z=z, y=y_train)
        
        # Broadcasting [batch, 1] + [batch, n_test]
        pred_cate = pred_context_ate + pred_delta

        if return_log_variance:
            return pred_cate, None  # GMM is gone for now, so log-variance is not predicted

        return pred_cate