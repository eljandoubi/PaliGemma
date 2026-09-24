"""Tests for the Gemma decoder-only language model."""

import torch

from src.models.gemma import GemmaAttention, GemmaForCausalLM, GemmaRMSNorm
from src.utils.kv_cache import KVCache


def test_rms_norm_matches_manual_formula_with_zero_weight():
    norm = GemmaRMSNorm(dim=4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = norm(x)

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(output, expected, atol=1e-5)


def _no_mask(batch, seq_len):
    return torch.zeros(batch, 1, seq_len, seq_len)


def test_attention_output_shape_and_softmax_sums_to_one(tiny_gemma_config):
    attn = GemmaAttention(tiny_gemma_config, layer_idx=0)
    attn.eval()
    batch, seq_len = 2, 5
    hidden_states = torch.randn(batch, seq_len, tiny_gemma_config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    with torch.no_grad():
        attn_output, attn_weights = attn(
            hidden_states=hidden_states,
            attention_mask=_no_mask(batch, seq_len),
            position_ids=position_ids,
        )

    assert attn_output.shape == (batch, seq_len, tiny_gemma_config.hidden_size)
    assert attn_weights.shape == (
        batch,
        tiny_gemma_config.num_attention_heads,
        seq_len,
        seq_len,
    )
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causal_lm_forward_shape(tiny_gemma_config):
    model = GemmaForCausalLM(tiny_gemma_config)
    model.eval()
    batch, seq_len = 1, 6
    inputs_embeds = torch.randn(batch, seq_len, tiny_gemma_config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            attention_mask=_no_mask(batch, seq_len),
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

    assert outputs["logits"].shape == (batch, seq_len, tiny_gemma_config.vocab_size)
    assert "kv_cache" not in outputs


def test_causal_lm_forward_with_kv_cache_decodes_one_token_at_a_time(tiny_gemma_config):
    model = GemmaForCausalLM(tiny_gemma_config)
    model.eval()
    batch, prefill_len = 1, 4
    kv_cache = KVCache()

    prefill_embeds = torch.randn(batch, prefill_len, tiny_gemma_config.hidden_size)
    with torch.no_grad():
        prefill_out = model(
            attention_mask=_no_mask(batch, prefill_len),
            position_ids=torch.arange(prefill_len).unsqueeze(0),
            inputs_embeds=prefill_embeds,
            kv_cache=kv_cache,
        )
    kv_cache = prefill_out["kv_cache"]
    assert kv_cache.num_items() == prefill_len

    next_embed = torch.randn(batch, 1, tiny_gemma_config.hidden_size)
    with torch.no_grad():
        decode_out = model(
            attention_mask=torch.zeros(batch, 1, 1, prefill_len + 1),
            position_ids=torch.tensor([[prefill_len]]),
            inputs_embeds=next_embed,
            kv_cache=kv_cache,
        )

    assert decode_out["logits"].shape == (batch, 1, tiny_gemma_config.vocab_size)
    assert decode_out["kv_cache"].num_items() == prefill_len + 1
