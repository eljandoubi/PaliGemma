"""Tests for the standalone helper functions in `inference.py`."""

from pathlib import Path

import torch
from inference import _sample_top_p, get_model_inputs, move_inputs_to_device

from src.processors.paligemma_processor import PaliGemmaProcessor

SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "samples" / "EiffelTower.jpg"


def test_sample_top_p_restricts_sampling_to_the_nucleus():
    # Only the first token's cumulative probability stays below p=0.5,
    # so every other candidate is zeroed out and sampling is deterministic.
    probs = torch.tensor([[0.99, 0.005, 0.005]])

    next_token = _sample_top_p(probs, p=0.5)

    assert next_token.shape == (1, 1)
    assert next_token.item() == 0


def test_move_inputs_to_device_moves_every_tensor():
    inputs = {"a": torch.zeros(2), "b": torch.ones(3)}

    moved = move_inputs_to_device(inputs, "cpu")

    assert set(moved) == {"a", "b"}
    assert all(tensor.device.type == "cpu" for tensor in moved.values())


def test_get_model_inputs_loads_image_and_tokenizes_prompt(tiny_fast_tokenizer):
    processor = PaliGemmaProcessor(tiny_fast_tokenizer, num_image_tokens=4, image_size=8)

    result = get_model_inputs(processor, "this tower is", str(SAMPLE_IMAGE), device="cpu")

    assert result["pixel_values"].shape[1:] == (3, 8, 8)
    assert result["input_ids"].device.type == "cpu"
    assert result["input_ids"].shape[0] == 1
