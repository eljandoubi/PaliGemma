"""PaliGemma pytorch module."""

import math
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import torch
from torch import nn
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
from modeling_gemma import GemmaConfig, GemmaForCausalLM
from kv_cache import KVCache


@dataclass
class PaliGemmaConfig:
    """Config class for PaliGemma module"""

    vision_config: Dict[str, Any] = field(default_factory=dict)
    text_config: Dict[str, Any] = field(default_factory=dict)
    ignore_index: int = -100
    image_token_index: int = 256000
    vocab_size: int = 257152
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: Optional[int] = None
    is_encoder_decoder: bool = False

    def __post_init__(self):
        self.vision_config = SiglipVisionConfig(**self.vision_config)
        self.text_config = GemmaConfig(
            **self.text_config,
            pad_token_id=self.pad_token_id)

        # Update the vocab size from text_config
        self.vocab_size = self.text_config.vocab_size

        # Calculate the number of image tokens and update the text config
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim


class PaliGemmaMultiModelProjector(nn.Module):
    """PaliGemma Multimodel projector module"""

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size,
                                config.vision_config.projection_dim, bias=True)

    def forward(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    """PaliGemma pytorch module for conditional generation"""

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.vision_tower = SiglipVisionModel(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModelProjector(
            config=config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.image_token_index = config.image_token_index
        self.language_model = GemmaForCausalLM(config=config.text_config)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def tie_weights(self) -> None:
        "Tie weights of the language model"
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
            self,
            image_features: torch.FloatTensor,
            inputs_embeds: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor]:
        """Merge input ids with image features"""
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / math.sqrt(self.hidden_size)
        # Combine the embeddings of the image tokens,
        # the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim,
                                      dtype=dtype, device=device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (
            input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id
        # We need to expand the masks to the embedding dimension
        # otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(
            -1).expand(-1, -1, embed_dim)
        # Add the text embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where
        # because the sequence length of scaled_image_features is not equal
        # to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(
            pad_mask_expanded,
            torch.zeros_like(final_embedding),
            final_embedding)
        #### CREATE THE ATTENTION MASK ####
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single
            # token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything,
            # since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = attention_mask.cumsum(
                -1).masked_fill_((attention_mask == 0), 1)

        return final_embedding, causal_mask, position_ids

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Dict[str, Union[torch.FloatTensor, KVCache]]:
        """Forward method"""

        assert torch.all(attention_mask == 1), "The input can not be padded "
        # 1. Extract the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(
            pixel_values.to(inputs_embeds.dtype))

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)
        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
