"""Shared fixtures for the test suite: tiny configs and a fake tokenizer.

All model configs here use tiny dimensions so tests run fast on CPU while
still exercising the real forward-pass code paths (attention, RoPE,
grouped-query repeat_kv, KV-cache, vision/text fusion, ...).
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from src.models.gemma import GemmaConfig
from src.models.paligemma import PaliGemmaConfig
from src.models.siglip import SiglipVisionConfig


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Make randomly-initialized modules reproducible across test runs."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_siglip_config() -> SiglipVisionConfig:
    return SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
    )


@pytest.fixture
def tiny_gemma_config() -> GemmaConfig:
    return GemmaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_paligemma_config() -> PaliGemmaConfig:
    vision_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_channels": 3,
        "image_size": 32,
        "patch_size": 16,
    }
    text_config = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    return PaliGemmaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=1,
        projection_dim=8,
        hidden_size=8,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_fast_tokenizer() -> PreTrainedTokenizerFast:
    """A small, fully offline `PreTrainedTokenizerFast` for processor tests."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<bos>": 2,
        "<eos>": 3,
        "this": 4,
        "tower": 5,
        "is": 6,
        "hello": 7,
        "world": 8,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<bos>",
        eos_token="<eos>",
        clean_up_tokenization_spaces=False,
    )
