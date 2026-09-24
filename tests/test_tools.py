"""Tests for `src.utils.tools` (RoPE helpers and grouped-query repeat_kv)."""

import torch

from src.models.gemma import GemmaRotaryEmbedding
from src.utils.tools import apply_rotary_pos_emb, repeat_kv, rotate_half


def test_rotate_half_swaps_and_negates_halves():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    result = rotate_half(x)

    assert torch.equal(result, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_pos_emb_preserves_vector_norm():
    batch, heads, seq_len, head_dim = 2, 3, 5, 8
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)

    rotary_emb = GemmaRotaryEmbedding(dim=head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    cos, sin = rotary_emb(q, position_ids)

    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

    # Rotation is norm-preserving per position.
    assert torch.allclose(q.norm(dim=-1), q_embed.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_embed.norm(dim=-1), atol=1e-5)


def test_repeat_kv_identity_when_n_rep_is_one():
    hidden_states = torch.randn(1, 2, 5, 4)

    result = repeat_kv(hidden_states, n_rep=1)

    assert result is hidden_states


def test_repeat_kv_repeats_each_head_contiguously():
    hidden_states = torch.randn(1, 2, 5, 4)

    result = repeat_kv(hidden_states, n_rep=3)

    assert result.shape == (1, 6, 5, 4)
    for rep in range(3):
        assert torch.equal(result[:, rep], hidden_states[:, 0])
    for rep in range(3):
        assert torch.equal(result[:, 3 + rep], hidden_states[:, 1])
