"""Helper functions"""

from typing import Tuple
import torch


def rotate_half(x: torch.FloatTensor) -> torch.FloatTensor:
    """Build the [-x2, x1, -x4, x3, ...] tensor
    for the sin part of the positional encoding."""
    # Takes the first half of the last dimension
    x1 = x[..., : x.shape[-1] // 2]
    # Takes the second half of the last dimension
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.FloatTensor,
                         k: torch.FloatTensor,
                         cos: torch.FloatTensor,
                         sin: torch.FloatTensor,
                         unsqueeze_dim: int = 1,
                         ) -> Tuple[torch.FloatTensor]:
    """Apply rotary position embedding."""
    # Add the head dimension
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat tensor n_rep  time"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim)
