"""Torch module of SigLIP the image encoder"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import  dataclass, InitVar
import torch
from torch import nn


@dataclass
class SiglipVisionConfig:
    """Config class for SigLIP module"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = None
    kwargs: InitVar[Optional[Dict[str, Any]]] = None

    def __post_init__(self, kwargs: Optional[Dict[str, Any]]) -> None:
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

class SiglipVisionEmbeddings(nn.Module):
    """SigLIP pytorch module"""
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        )
        self.num_positions = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_embeddings=self.num_positions,
                                               embedding_dim=config.hidden_size)
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(self.num_positions).expand((1,-1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Forward method"""
        # pixel_values: [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image,
        # with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches]
        # -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch.
        # Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings

class SiglipEncoder(nn.Module):
    """SigLIP pytorch module"""
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self,inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Forward method"""
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        return inputs_embeds

class SiglipVisionTransformer(nn.Module):
    """SigLIP Transformer module"""
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                           eps=config.layer_norm_eps)

    def forward(self,pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward method"""
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):
    """SigLIP pytorch module"""
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self,pixel_values: torch.Tensor) -> Tuple:
        """Forward method"""
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values)
