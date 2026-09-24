"""Tests for image/text preprocessing in `PaliGemmaProcessor`."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.processors.paligemma_processor import (
    PaliGemmaProcessor,
    add_image_tokens_to_prompt,
    normalize,
    process_images,
    rescale,
)


def test_rescale_scales_values():
    image = np.array([[0.0, 128.0, 255.0]], dtype=np.float32)

    result = rescale(image, scale=1.0 / 255)

    assert np.allclose(result, image / 255)
    assert result.dtype == np.float32


def test_normalize_applies_mean_and_std():
    image = np.ones((2, 2, 3), dtype=np.float32)

    result = normalize(image, mean=0.5, std=0.5)

    assert np.allclose(result, 1.0)


def test_process_images_returns_channel_first_normalized_arrays():
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))

    results = process_images(
        images=[image],
        size=4,
        resample=Image.Resampling.BILINEAR,
        rescale_factor=1.0 / 255,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    assert len(results) == 1
    assert results[0].shape == (3, 4, 4)
    # Pure red, rescaled to [0, 1] then normalized with mean/std 0.5 -> [-1, 1].
    assert np.allclose(results[0][0], 1.0, atol=1e-5)
    assert np.allclose(results[0][1], -1.0, atol=1e-5)


def test_add_image_tokens_to_prompt_formats_prefix():
    result = add_image_tokens_to_prompt(
        prefix_prompt="hello ", bos_token="<bos>", image_seq_len=3, image_token="<image>"
    )

    assert result == "<image><image><image><bos>hello \n"


def test_processor_call_builds_expected_input_ids(tiny_fast_tokenizer):
    processor = PaliGemmaProcessor(tiny_fast_tokenizer, num_image_tokens=4, image_size=8)
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))

    result = processor(text=["this tower is"], images=[image])

    assert result["pixel_values"].shape == (1, 3, 8, 8)
    input_ids = result["input_ids"][0].tolist()
    assert input_ids[:4] == [processor.image_token_id] * 4
    assert input_ids[4] == tiny_fast_tokenizer.convert_tokens_to_ids("<bos>")
    assert input_ids[5:8] == [
        tiny_fast_tokenizer.convert_tokens_to_ids(tok) for tok in ("this", "tower", "is")
    ]
    assert torch.equal(result["attention_mask"], torch.ones_like(result["attention_mask"]))


def test_processor_call_requires_matching_text_and_image_counts(tiny_fast_tokenizer):
    processor = PaliGemmaProcessor(tiny_fast_tokenizer, num_image_tokens=4, image_size=8)
    image = Image.new("RGB", (8, 8))

    with pytest.raises(AssertionError):
        processor(text=["a", "b"], images=[image])
