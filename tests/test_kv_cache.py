"""Tests for `src.utils.kv_cache.KVCache`."""

import torch

from src.utils.kv_cache import KVCache


def test_empty_cache_has_zero_items():
    cache = KVCache()
    assert cache.num_items() == 0


def test_update_on_empty_cache_stores_input_as_is():
    cache = KVCache()
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    returned_key, returned_value = cache.update(key, value, layer_idx=0)

    assert torch.equal(returned_key, key)
    assert torch.equal(returned_value, value)
    assert cache.num_items() == 5


def test_update_concatenates_along_sequence_dimension():
    cache = KVCache()
    key1 = torch.randn(1, 2, 5, 4)
    value1 = torch.randn(1, 2, 5, 4)
    cache.update(key1, value1, layer_idx=0)

    key2 = torch.randn(1, 2, 1, 4)
    value2 = torch.randn(1, 2, 1, 4)
    returned_key, returned_value = cache.update(key2, value2, layer_idx=0)

    assert returned_key.shape == (1, 2, 6, 4)
    assert returned_value.shape == (1, 2, 6, 4)
    assert torch.equal(returned_key[..., :5, :], key1)
    assert torch.equal(returned_key[..., 5:, :], key2)
    assert cache.num_items() == 6


def test_layers_are_tracked_independently():
    cache = KVCache()
    key0 = torch.randn(1, 2, 5, 4)
    value0 = torch.randn(1, 2, 5, 4)
    cache.update(key0, value0, layer_idx=0)

    key1 = torch.randn(1, 2, 5, 4)
    value1 = torch.randn(1, 2, 5, 4)
    cache.update(key1, value1, layer_idx=1)

    # `num_items` reports the sequence length of layer 0, unaffected by layer 1.
    assert cache.num_items() == 5
    assert len(cache.key_cache) == 2
    assert torch.equal(cache.key_cache[1], key1)
