import typing, math, torch, torch.nn as nn


class ProximalNanoTabICLv2(nn.Module):
    def __init__(
        self,
        out_dim: int,
        embed_dim: int = 64,
        col_num_blocks: int = 3,
        row_num_blocks: int = 3,
        icl_num_blocks: int = 12,
        col_nhead: int = 8,
        row_nhead: int = 8,
        icl_nhead: int = 8,
        feature_group_size_x: int = 2,
        feature_group_size_w: int = 4,
        feature_group_size_z: int = 2,
        n_cls_cols: int = 4,
        n_cls_rows: int = 128,
    ):
        # classification: max_classes = out_dim (= 10 typically); regression: max_classes = 0, out_dim = n_quantiles
        super().__init__()
        self.feature_group_size_x = feature_group_size_x
        self.feature_group_size_w = feature_group_size_w
        self.feature_group_size_z = feature_group_size_z
        icl_dim = embed_dim * n_cls_cols

        self.x_embed = nn.Linear(feature_group_size_x, embed_dim)
        self.w_embed = nn.Linear(feature_group_size_w, embed_dim)
        self.z_embed = nn.Linear(feature_group_size_z, embed_dim)

        self.a_embed = nn.Linear(1, embed_dim)
        self.a_embed_icl = nn.Linear(1, icl_dim)

        self.y_embed_in = nn.Linear(1, embed_dim)
        self.y_embed_icl = nn.Linear(1, icl_dim)

        self.col_blocks = nn.ModuleList(
            [
                InducedTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=col_nhead,
                    n_inducing=n_cls_rows,
                    ssmax=True,
                )
                for _ in range(col_num_blocks)
            ]
        )
        self.row_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, num_heads=row_nhead, use_rope=True
                )
                for _ in range(row_num_blocks)
            ]
        )
        self.icl_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=icl_dim, num_heads=icl_nhead, ssmax=True)
                for _ in range(icl_num_blocks)
            ]
        )

        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, n_cls_cols, embed_dim)
        )
        self.row_ln = nn.LayerNorm(embed_dim)
        self.out_ln = nn.LayerNorm(icl_dim)
        self.out_mlp = get_mlp(icl_dim, icl_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor, w: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        assert y.shape == a.shape, "y and a must have the same shape"

        n_batch, n_rows, n_cols = x.shape
        n_batch, n_train = y.shape

        x_mean = x[:, :n_train].mean(dim=1, keepdim=True)
        x_std = x[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        x = (x - x_mean) / x_std

        w_mean = w[:, :n_train].mean(dim=1, keepdim=True)
        w_std = w[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        w = (w - w_mean) / w_std

        z_mean = z[:, :n_train].mean(dim=1, keepdim=True)
        z_std = z[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        z = (z - z_mean) / z_std

        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, unbiased=False, keepdim=True) + 1e-8
        y = (y - y_mean) / y_std

        # ----- Embedding: repeated feature grouping -> x embedding -> add y embedding to train

        x = x / (
            x[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        )
        # Vectorized sliding window grouping
        idxs_x = torch.arange(n_cols, dtype=torch.long, device=x.device)
        offsets_x = (2**torch.arange(self.feature_group_size_x, device=x.device) - 1)
        group_idxs_x = (idxs_x[:, None] + offsets_x) % n_cols
        x = x[:, :, group_idxs_x]
        emb_x = self.x_embed(x)  # emb.shape = (n_batch, n_rows, n_cols, embed_dim)

        w = w / (
            w[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        )
        idxs_w = torch.arange(w.shape[-1], dtype=torch.long, device=w.device)
        offsets_w = (2**torch.arange(self.feature_group_size_w, device=w.device) - 1)
        group_idxs_w = (idxs_w[:, None] + offsets_w) % w.shape[-1]
        w = w[:, :, group_idxs_w]
        emb_w = self.w_embed(w)

        z = z / (
            z[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8
        )
        idxs_z = torch.arange(z.shape[-1], dtype=torch.long, device=z.device)
        offsets_z = (2**torch.arange(self.feature_group_size_z, device=z.device) - 1)
        group_idxs_z = (idxs_z[:, None] + offsets_z) % z.shape[-1]
        z = z[:, :, group_idxs_z]
        emb_z = self.z_embed(z)

        emb_a = self.a_embed(a[:, :, None, None])

        emb_y = self.y_embed_in(y[:, :, None, None])

        emb = torch.concatenate([emb_x, emb_w, emb_z], dim=-2)
        emb[:, :n_train] += emb_y + emb_a

        # ----- TF_col: induced self-attention within each column
        for block in self.col_blocks:
            emb = block.col_attn(
                emb, kv_max_idx=n_train
            )  # all rows only attend to training rows

        # ----- TF_row: concat CLS tokens as extra columns -> row attention -> norm + merge cls tokens
        emb = torch.cat(
            [self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2
        )
        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)
        emb = self.row_blocks[-1].row_attn(
            emb, q_max_idx=self.row_cls_tokens.size(-2)
        )  # need only the cls token values
        emb = self.row_ln(emb).flatten(
            -2, -1
        )  # norm + merge cls tokens into one bigger token

        # ----- TF_icl: add y embedding -> self-attention
        # now emb.shape = (n_batch, n_rows, icl_dim)
        emb[:, :n_train] += self.y_embed_icl(y[:, :, None]) + self.a_embed_icl(a[:, :, None])
        for block in self.icl_blocks[:-1]:
            emb = block(
                emb, kv_max_idx=n_train
            )  # all rows only attend to training rows
        emb = self.icl_blocks[-1](
            emb[:, n_train:], emb[:, :n_train]
        )  # need only test predictions

        return self.out_mlp(self.out_ln(emb))  # output MLP


class ClassEmbedding(nn.Embedding):
    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.weight,
            -1 / math.sqrt(self.num_embeddings),
            1 / math.sqrt(self.num_embeddings),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return super().forward(y.squeeze(-1).long())


def get_mlp(n_in: int, n_hidden: int, n_out: int):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out)
    )


class TableAttnBase(
    nn.Module
):  # base class with functions to apply attention on 2D tables instead of 1D sequences
    def row_attn(
        self, q, kv=None, **kwargs
    ):  # apply attention within each row separately
        n_batch, n_rows, n_cols, embed_dim = q.shape
        q, kv = (
            None if t is None else t.flatten(0, 1) for t in [q, kv]
        )  # merge rows dim into batch dim
        # apply attention -> unmerge rows dim from batch dim; dimension -2 might differ because of q_max_idx in kwargs
        return self(q, kv, **kwargs).reshape(n_batch, n_rows, -1, embed_dim)

    def col_attn(
        self, q, kv=None, **kwargs
    ):  # apply attention within each column separately
        return self.row_attn(
            q.transpose(1, 2), None if kv is None else kv.transpose(1, 2), **kwargs
        ).transpose(1, 2)


class InducedTransformerBlock(TableAttnBase):
    def __init__(
        self, embed_dim: int, num_heads: int, n_inducing: int, ssmax: bool = False
    ):
        super().__init__()
        self.tfm1 = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, ssmax=ssmax
        )
        self.tfm2 = TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads
        )  # fixed nb of ind. vectors -> no ssmax
        self.inducing_vectors = nn.Parameter(
            0.02 * torch.randn(1, n_inducing, embed_dim)
        )

    def forward(
        self,
        q,
        kv=None,
        q_max_idx: typing.Optional[int] = None,
        kv_max_idx: typing.Optional[int] = None,
    ):
        kv = self.tfm1(
            self.inducing_vectors.expand(q.shape[0], -1, -1),
            q if kv is None else kv,
            kv_max_idx=kv_max_idx,
        )
        return self.tfm2(q, kv, q_max_idx=q_max_idx)


class TransformerBlock(nn.MultiheadAttention, TableAttnBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rope: bool = False,
        ssmax: bool = False,
    ):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.rope = (
            Rope(head_dim=embed_dim // num_heads, theta=100_000.0) if use_rope else None
        )
        self.ssmax_layer = (
            QASSMax(num_heads=num_heads, head_dim=embed_dim // num_heads)
            if ssmax
            else None
        )
        self.mlp = get_mlp(embed_dim, embed_dim * 2, embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ln_mlp = nn.LayerNorm(embed_dim)

    def forward(
        self,
        q,
        kv=None,
        q_max_idx: typing.Optional[int] = None,
        kv_max_idx: typing.Optional[int] = None,
    ):
        # q.shape: (batch_size, q_len, embed_dim), kv.shape: (batch_size, kv_len, embed_dim)
        x, q = q, self.ln_attn(q)
        kv = q if kv is None else self.ln_attn(kv)
        if kv_max_idx is not None:
            kv = kv[..., :kv_max_idx, :]
        if q_max_idx is not None:
            x, q = x[..., :q_max_idx, :], q[..., :q_max_idx, :]

        x = x + self.attn(q, kv)
        del q, kv  # save memory during inference
        return x + self.mlp(
            self.ln_mlp(x)
        )  # we use pre-norm here and for the attention as well

    def attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Joint projection of (q, k, v), then transpose heads to (batch_size, num_heads, len, head_dim)
        q, k, v = nn.functional._in_projection_packed(
            q, k, k, self.in_proj_weight, self.in_proj_bias
        )
        q, k, v = (
            t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
            for t in [q, k, v]
        )

        q = (
            q if self.ssmax_layer is None else self.ssmax_layer(q=q, n=k.size(-2))
        )  # SSMax (optional)
        q, k = (
            t if self.rope is None else self.rope(t) for t in [q, k]
        )  # RoPE (optional)

        # attention with heads in batch dim (maybe needed for FlashAttention) -> put the head dim back -> out projection
        attn_output = nn.functional.scaled_dot_product_attention(
            *[t.flatten(0, 1) for t in (q, k, v)]
        ).view(q.shape)
        del q, k, v  # save memory during inference
        return self.out_proj(
            attn_output.transpose(-3, -2).flatten(-2, -1)
        )  # (batch_size, q_len, embed_dim)


class Rope(nn.Module):  # rotary positional encoding
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.half = head_dim // 2
        self.register_buffer(
            "inv_freq",
            theta ** torch.linspace(0.0, -1.0, self.half + 1)[:-1],
            persistent=False,
        )
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.register_buffer("cos", torch.empty(0), persistent=False)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape

        if (
            self.cos.numel() == 0
            or self.cos.device != x.device
            or self.cos.size(0) < seq_len
        ):  # need to extend cache
            pos = torch.arange(
                seq_len, device=x.device, dtype=self.inv_freq.dtype
            )  # (seq_len,)
            angles = pos[:, None] * self.inv_freq[None, :]  # (seq_len, half_head_dim)
            self.sin, self.cos = angles.sin(), angles.cos()  # (seq_len, half_head_dim)

        sin, cos = self.sin[:seq_len], self.cos[:seq_len]
        x1, x2 = (
            x[..., : self.half],
            x[..., self.half :],
        )  # (batch_size, num_heads, seq_len, half_head_dim)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class QASSMax(
    nn.Module
):  # query-aware scalable softmax for better context length scaling
    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.base_mlp = get_mlp(1, n_hidden, num_heads * head_dim)
        self.query_mlp = get_mlp(head_dim, n_hidden, head_dim)
        nn.init.zeros_(self.query_mlp[-1].weight)  # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        logn = q.new_tensor(math.log(max(1, n))).view(1, 1)
        return (
            self.base_mlp(logn).view(1, num_heads, 1, head_dim)
            * (1 + torch.tanh(self.query_mlp(q)))
            * q
        )