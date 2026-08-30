import torch
import pytest
from proximalfm._model.rope import (
    broadcat,
    rotate_half_interleaved,
    rotate_half_contiguous,
    apply_rotary_emb,
    apply_learned_rotations,
    RotaryEmbedding,
)


def test_broadcat():
    t1 = torch.randn(1, 10)
    t2 = torch.randn(5, 1)
    out = broadcat((t1, t2), dim=-1)
    assert out.shape == (5, 20)

    # Check if values are correctly broadcasted
    t1_expanded = t1.expand(5, 10)
    t2_expanded = t2.expand(5, 10)
    expected = torch.cat((t1_expanded, t2_expanded), dim=-1)
    assert torch.allclose(out, expected)


def test_rotate_half_interleaved():
    # Input: [1, 2, 3, 4]
    # rearrange to [..., d, 2] -> [[1, 2], [3, 4]]
    # unbind -> x1=[1, 3], x2=[2, 4]
    # stack(-x2, x1) -> [[-2, 1], [-4, 3]]
    # rearrange back -> [-2, 1, -4, 3]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-2.0, 1.0, -4.0, 3.0])
    out = rotate_half_interleaved(x)
    assert torch.allclose(out, expected)


def test_rotate_half_contiguous():
    # Input: [1, 2, 3, 4]
    # half = 2
    # x1 = [1, 2], x2 = [3, 4]
    # cat([-x2, x1]) -> [-3, -4, 1, 2]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    out = rotate_half_contiguous(x)
    assert torch.allclose(out, expected)


def test_apply_rotary_emb_interleaved():
    # freqs shape (seq_len, dim)
    # t shape (batch, seq_len, dim)
    seq_len = 4
    dim = 4
    t = torch.ones(1, seq_len, dim)
    freqs = torch.zeros(seq_len, dim)  # cos=1, sin=0

    # Formula: x * cos + rotate(x) * sin
    # with cos=1, sin=0, out should be t
    out = apply_rotary_emb(freqs, t, interleaved=True)
    assert torch.allclose(out, t)

    # With cos=0, sin=1
    freqs = torch.full((seq_len, dim), 1.57079632679)  # approx pi/2
    out = apply_rotary_emb(freqs, t, interleaved=True)
    expected = rotate_half_interleaved(t)
    assert torch.allclose(out, expected, atol=1e-5)


def test_apply_rotary_emb_non_interleaved():
    seq_len = 4
    dim = 4  # half_dim = 2
    t = torch.ones(1, seq_len, dim)
    freqs = torch.zeros(seq_len, dim // 2)  # cos=1, sin=0

    out = apply_rotary_emb(freqs, t, interleaved=False)
    assert torch.allclose(out, t)

    # With cos=0, sin=1
    freqs = torch.full((seq_len, dim // 2), 1.57079632679)
    out = apply_rotary_emb(freqs, t, interleaved=False)
    expected = rotate_half_contiguous(t)
    assert torch.allclose(out, expected, atol=1e-5)


def test_apply_rotary_emb_start_index():
    seq_len = 2
    dim = 8
    t = torch.ones(1, seq_len, dim)
    # Rotate only from index 2 to 6 (rot_dim = 4)
    freqs = torch.zeros(seq_len, 4)
    out = apply_rotary_emb(freqs, t, start_index=2, interleaved=True)
    assert torch.allclose(out, t)
    assert out.shape == t.shape


def test_apply_learned_rotations():
    t = torch.ones(1, 4, 4)
    rotations = torch.zeros(1, 4, 2)  # cos=1, sin=0 after repeat
    out = apply_learned_rotations(rotations, t)
    assert torch.allclose(out, t)


def test_rotary_embedding_init():
    dim = 16
    # Test different freqs_for
    rope_lang = RotaryEmbedding(dim=dim, freqs_for="lang")
    assert rope_lang.freqs.shape == (dim // 2,)

    rope_pixel = RotaryEmbedding(dim=dim, freqs_for="pixel")
    assert rope_pixel.freqs.shape == (dim // 2,)

    rope_const = RotaryEmbedding(dim=dim, freqs_for="constant", num_freqs=8)
    assert rope_const.freqs.shape == (8,)


def test_rotary_embedding_forward_caching():
    dim = 16
    rope = RotaryEmbedding(dim=dim, cache_if_possible=True)
    t = torch.arange(10).float()

    # First call - should compute and cache
    freqs1 = rope(t, seq_len=10)
    assert rope.cached_freqs is not None
    assert torch.allclose(freqs1, rope.cached_freqs)

    # Second call - should return cached
    freqs2 = rope(t, seq_len=10)
    assert freqs1 is freqs2 or torch.allclose(freqs1, freqs2)


def test_rotary_embedding_get_seq_pos():
    dim = 16
    rope = RotaryEmbedding(dim=dim, interpolate_factor=2.0)
    seq_len = 10
    pos = rope.get_seq_pos(seq_len, device="cpu", dtype=torch.float)
    expected = torch.arange(seq_len).float() / 2.0
    assert torch.allclose(pos, expected)


def test_rotary_embedding_get_scale_xpos():
    dim = 16
    rope = RotaryEmbedding(dim=dim, use_xpos=True)
    t = torch.arange(10).float()
    scale = rope.get_scale(t, seq_len=10)
    assert scale.shape == (10, dim)
    assert rope.cached_scales is not None


def test_rotary_embedding_rotate_queries_or_keys():
    dim = 16
    rope = RotaryEmbedding(dim=dim)
    q = torch.randn(1, 10, dim)

    # Test interleaved
    out_interleaved = rope.rotate_queries_or_keys(q)
    assert out_interleaved.shape == q.shape

    # Test non-interleaved
    rope_ni = RotaryEmbedding(dim=dim, interleaved=False)
    out_ni = rope_ni.rotate_queries_or_keys(q)
    assert out_ni.shape == q.shape


def test_rotary_embedding_rotate_queries_and_keys_xpos():
    dim = 16
    rope = RotaryEmbedding(dim=dim, use_xpos=True)
    q = torch.randn(1, 10, dim)
    k = torch.randn(1, 10, dim)

    rq, rk = rope.rotate_queries_and_keys(q, k)
    assert rq.shape == q.shape
    assert rk.shape == k.shape


def test_rotary_embedding_rotate_queries_with_cached_keys():
    dim = 16
    rope = RotaryEmbedding(dim=dim, use_xpos=True)
    q = torch.randn(1, 5, dim)
    k = torch.randn(1, 10, dim)

    rq, rk = rope.rotate_queries_with_cached_keys(q, k)
    assert rq.shape == q.shape
    assert rk.shape == k.shape


def test_rotary_embedding_get_axial_freqs():
    dim = 16
    rope = RotaryEmbedding(dim=dim)
    # 2D axial freqs
    axial_freqs = rope.get_axial_freqs(10, 10)
    # Since each dim gets a full forward pass and then cat, and forward returns (L, dim)
    # result should be (10, 10, 2*dim)
    assert axial_freqs.shape == (10, 10, 2 * dim)
