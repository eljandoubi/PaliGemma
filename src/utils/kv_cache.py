"""KVCache Class"""

from typing import List, Tuple
import torch


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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update Cache"""
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's
            # create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len,
            # Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
