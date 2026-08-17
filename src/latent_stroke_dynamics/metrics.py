"""Distance helpers shared by the embedding-sensitivity experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return cosine distance along the final feature dimension."""

    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} versus {right.shape}")
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return (1.0 - (left * right).sum(dim=-1)).clamp(min=0.0, max=2.0)


def alternating_pair_distances(features: torch.Tensor) -> torch.Tensor:
    """Measure rows (0,1), (2,3), and so on as ordered comparison pairs."""

    if features.shape[0] % 2:
        raise ValueError("The feature batch must contain an even number of rows.")
    paired = features.reshape(features.shape[0] // 2, 2, *features.shape[1:])
    return cosine_distance(paired[:, 0], paired[:, 1])
