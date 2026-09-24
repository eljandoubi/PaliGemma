"""Tests for the SigLIP vision tower."""

import torch

from src.models.siglip import SiglipVisionEmbeddings, SiglipVisionModel


def test_embeddings_output_shape(tiny_siglip_config):
    embeddings = SiglipVisionEmbeddings(tiny_siglip_config)
    num_patches = (tiny_siglip_config.image_size // tiny_siglip_config.patch_size) ** 2
    pixel_values = torch.randn(2, 3, tiny_siglip_config.image_size, tiny_siglip_config.image_size)

    output = embeddings(pixel_values)

    assert output.shape == (2, num_patches, tiny_siglip_config.hidden_size)


def test_vision_model_forward_shape_and_finiteness(tiny_siglip_config):
    model = SiglipVisionModel(tiny_siglip_config)
    model.eval()
    num_patches = (tiny_siglip_config.image_size // tiny_siglip_config.patch_size) ** 2
    pixel_values = torch.randn(2, 3, tiny_siglip_config.image_size, tiny_siglip_config.image_size)

    with torch.no_grad():
        output = model(pixel_values)

    assert output.shape == (2, num_patches, tiny_siglip_config.hidden_size)
    assert torch.isfinite(output).all()
