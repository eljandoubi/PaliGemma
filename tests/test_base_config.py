"""Tests for `src.configs.base_config.BaseConfig`."""

import pytest

from src.models.gemma import GemmaConfig


def test_from_dict_filters_unknown_keys():
    data = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "some_unknown_field": "should be dropped",
    }

    config = GemmaConfig.from_dict(data)

    assert config.vocab_size == 32
    assert config.hidden_size == 8
    assert not hasattr(config, "some_unknown_field")


def test_from_dict_keeps_defaults_for_missing_optional_fields():
    data = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }

    config = GemmaConfig.from_dict(data)

    assert config.head_dim == 256
    assert config.pad_token_id is None


def test_from_dict_raises_when_required_field_missing():
    with pytest.raises(TypeError):
        GemmaConfig.from_dict({"vocab_size": 32})
