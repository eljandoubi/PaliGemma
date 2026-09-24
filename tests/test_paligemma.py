"""Tests for `PaliGemmaConfig` and `PaliGemmaForConditionalGeneration`."""

import pytest
import torch

from src.models.paligemma import PaliGemmaForConditionalGeneration
from src.utils.kv_cache import KVCache


def test_config_post_init_derives_dependent_fields(tiny_paligemma_config):
    config = tiny_paligemma_config

    assert config.vocab_size == config.text_config.vocab_size == 32
    assert config.vision_config.projection_dim == config.projection_dim == 8
    vision_config = config.vision_config
    expected_num_image_tokens = (vision_config.image_size // vision_config.patch_size) ** 2
    assert config.text_config.num_image_tokens == expected_num_image_tokens


def test_tie_weights_shares_embedding_and_lm_head(tiny_paligemma_config):
    model = PaliGemmaForConditionalGeneration(tiny_paligemma_config)

    model.tie_weights()

    lm_head = model.language_model.lm_head
    embed_tokens = model.language_model.model.embed_tokens
    assert lm_head.weight is embed_tokens.weight


def _num_image_tokens(config) -> int:
    return (config.vision_config.image_size // config.vision_config.patch_size) ** 2


def test_forward_rejects_padded_attention_mask(tiny_paligemma_config):
    model = PaliGemmaForConditionalGeneration(tiny_paligemma_config)
    model.eval()
    num_image_tokens = _num_image_tokens(tiny_paligemma_config)
    seq_len = num_image_tokens + 2
    input_ids = torch.full((1, seq_len), fill_value=5, dtype=torch.long)
    input_ids[:, :num_image_tokens] = tiny_paligemma_config.image_token_index
    attention_mask = torch.ones(1, seq_len)
    attention_mask[0, -1] = 0  # simulate padding, which this model does not support
    pixel_values = torch.randn(
        1,
        3,
        tiny_paligemma_config.vision_config.image_size,
        tiny_paligemma_config.vision_config.image_size,
    )

    with pytest.raises(AssertionError), torch.no_grad():
        model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)


def test_forward_prefill_then_kv_cache_decode_step(tiny_paligemma_config):
    model = PaliGemmaForConditionalGeneration(tiny_paligemma_config)
    model.tie_weights()
    model.eval()
    num_image_tokens = _num_image_tokens(tiny_paligemma_config)
    image_size = tiny_paligemma_config.vision_config.image_size

    input_ids = torch.full((1, num_image_tokens + 3), fill_value=6, dtype=torch.long)
    input_ids[:, :num_image_tokens] = tiny_paligemma_config.image_token_index
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
    assert prefill_out["logits"].shape == (1, input_ids.shape[1], tiny_paligemma_config.vocab_size)

    kv_cache = KVCache()
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
    kv_cache = prefill_out["kv_cache"]
    assert kv_cache.num_items() == input_ids.shape[1]

    next_token = torch.tensor([[7]])
    next_attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=-1)
    with torch.no_grad():
        decode_out = model(
            input_ids=next_token,
            pixel_values=pixel_values,
            attention_mask=next_attention_mask,
            kv_cache=kv_cache,
        )

    assert decode_out["logits"].shape == (1, 1, tiny_paligemma_config.vocab_size)
    assert decode_out["kv_cache"].num_items() == input_ids.shape[1] + 1
