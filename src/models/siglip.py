"""Torch module of SigLIP the image encoder"""

from typing import Optional
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from src.configs.base_config import BaseConfig


@dataclass
class SiglipVisionConfig(BaseConfig):
    """Config dataclass for SigLIP module"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None


class SiglipVisionEmbeddings(nn.Module):
    """SigLIP Embedding module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
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
            tensor=torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # pixel_values: [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image,
        # with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width
        # // patch_size
        patch_embeds: torch.FloatTensor = self.patch_embedding(pixel_values)
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


class SiglipAttention(nn.Module):
    """SigLIP Attention module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # Equivalent to 1 / sqrt(self.head_dim)
        self.scale: float = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,
                hidden_states: torch.FloatTensor
                ) -> torch.FloatTensor:
        """Forward method"""
        batch_size, seq_len, _ = hidden_states.size()
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        query_states: torch.FloatTensor = self.q_proj(hidden_states)
        key_states: torch.FloatTensor = self.k_proj(hidden_states)
        value_states: torch.FloatTensor = self.v_proj(hidden_states)
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len,
                                         self.num_heads, self.head_dim
                                         ).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len,
                                     self.num_heads, self.head_dim
                                     ).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len,
                                         self.num_heads, self.head_dim
                                         ).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k).
        # [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = self.scale * \
            torch.matmul(query_states, key_states.transpose(2, 3))
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"""Attention weights should be of size {
                    (
                        batch_size,
                        self.num_heads,
                        seq_len,
                        seq_len)}, but is"""
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads,
        # Num_Patches, Num_Patches]
        attn_weights = F.softmax(attn_weights, dim=-1,
                                 dtype=torch.float32
                                 ).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = F.dropout(attn_weights, p=self.dropout,
                                 training=self.training)
        # Multiply the attention weights by the value states. attn_output:
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"""`attn_output` should be of size {
                    (
                        batch_size,
                        self.num_heads,
                        seq_len,
                        self.head_dim)}, but is"""
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len,
                                       self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)
        return attn_output


class SiglipMLP(nn.Module):
    """MLP module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size,
                             config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size,
                             config.hidden_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # [Batch_Size, Num_Patches, Embed_Dim]
        # -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # Same Shape
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size]
        # -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """SigLIP encoder layer"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.hidden_size,
                                        eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    """SigLIP encoder module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Forward method"""
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """SigLIP Transformer module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                           eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """SigLIP pytorch module"""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values)
