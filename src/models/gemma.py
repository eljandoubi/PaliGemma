"""pytorch module"""

import math
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from src.configs.base_config import BaseConfig
from src.utils.kv_cache import KVCache
from src.utils.tools import apply_rotary_pos_emb


@dataclass
class GemmaConfig(BaseConfig):
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
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = None


class GemmaRotaryEmbedding(nn.Module):
    """Gemma Rotary Position Embedding"""

    def __init__(self, dim: int,
                 base: float = 10000.,
                 ):
        super().__init__()

        self.dim = dim  # it is set to the head_dim
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim)
        # where i = 0, 1, 2, ..., dim // 2
        inv_freq = base ** (-torch.arange(0, dim, 2) / dim)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self,
                x: torch.FloatTensor,
                position_ids: torch.LongTensor
                ) -> Tuple[torch.FloatTensor]:
        """Forward method"""
        # x: [bs, num_attention_heads, seq_len, head_size]
        # self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len]
            # --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaAttention(nn.Module):
    """Gemma Attention"""

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        self.q_proj = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)

        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)

        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            dim=self.head_dim,
            base=self.rope_theta,
        )

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
                ) -> torch.FloatTensor:
        """Forward method"""

        # [Batch_Size, Seq_Len, Hidden_Size]
        bsz, q_len, _ = hidden_states.size()
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len,
                                         self.num_heads,
                                         self.head_dim
                                         ).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads,
                                     self.head_dim
                                     ).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads,
                                         self.head_dim
                                         ).transpose(1, 2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim],
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx)

        attn_output = F.scaled_dot_product_attention(query_states,
                                                    key_states,
                                                    value_states, 
                                                    attn_mask=attention_mask,
                                                    dropout_p=self.attention_dropout,
                                                    enable_gqa = True
                                                    )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz,
                                                    self.num_heads,
                                                    q_len,
                                                    self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size,
        # Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q,
        # Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q,
        # Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q *
        # Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output


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
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


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
        hidden_states = self.self_attn(
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
