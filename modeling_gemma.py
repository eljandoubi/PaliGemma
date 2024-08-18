"""pytorch module"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, InitVar
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GemmaConfig:
    """Config dataclass for Gemma module"""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropou: float = 0.0
    pad_token_id: Optional[int] = None
    kwargs: InitVar[Optional[Dict[str, Any]]] = None

    def __post_init__(self, kwargs: Optional[Dict[str, Any]]):
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)


class KVCache:
    """Key Value Cache class"""

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        """Count how many token stored"""
        if len(self.key_cache) == 0:
            return 0

        # The shape of the key_cache is
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        return self.key_cache[0].shape[-2]


class GemmaAttention(nn.Module):
    """Gemma Attention"""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Tuple[torch.FloatTensor]:
        """Forward method"""
        return hidden_states, attention_mask, position_ids, kv_cache


class GemmaMLP(nn.Module):
    """Gemma MLP"""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size,
                                   config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size,
                                 config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Forward method"""
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] ->
        # [Batch_Size, Seq_Len, Intermediate_Size]
        # y = F.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] ->
        # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] ->
        # [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(F.gelu(self.gate_proj(
            x), approximate="tanh") * self.up_proj(x))


class GemmaRMSNorm(nn.Module):
    """RMS Norm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        "Normalize x"
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method"""
        return x


class GemmaDecoderLayer(nn.Module):
    """Gemma decoder layer"""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> torch.FloatTensor:
        """Forward method"""
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    """Gemma model"""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx=i)
                                     for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size,
                                 eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Get input embeddings"""
        return self.embed_tokens

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> torch.FloatTensor:
        """Forward method"""
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds * math.sqrt(self.hidden_size)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states,
                                          attention_mask,
                                          position_ids,
                                          kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    """Gemma Causal LM module"""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def tie_weights(self) -> None:
        "Tie weights of the model Embedding and LM head."
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> Dict[str, Union[torch.FloatTensor, KVCache]]:
        """Forward method"""
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(hidden_states).float()

        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
