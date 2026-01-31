"""
Compatibility utilities for handling breaking changes in dependencies.

This module provides shims for functions that have been removed or relocated
in newer versions of libraries like transformers.
"""

from typing import List, Set, Tuple

import torch


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    This function was removed from transformers 5.0 as part of the removal of
    head pruning functionality. We provide it here for compatibility with
    vendored model code that still requires it.

    Based on the original implementation from huggingface/transformers.

    Args:
        heads: List of the indices of heads to prune.
        n_heads: The number of heads in the model.
        head_size: The size of each head.
        already_pruned_heads: A set of already pruned heads.

    Returns:
        A tuple with the indices of heads to prune (accounting for already pruned)
        and a LongTensor of indices to keep in the layer.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


# Try to import from transformers, fall back to our implementation
try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    # transformers 5.0+ removed this function, use our implementation
    pass

# prune_linear_layer is still available in transformers 5.0
from transformers.pytorch_utils import prune_linear_layer

__all__ = ["find_pruneable_heads_and_indices", "prune_linear_layer"]
