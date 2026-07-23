"""Small shared helpers."""

from pathlib import Path

import torch

# Token layout shared by every module that indexes into the 261 tokens DINOv2
# returns per image: [cls] + [4 register tokens] + [256 spatial patch tokens]
# (16x16 grid). Centralized here so downstream analysis modules agree on it.
CLS_IDX = 0
REG_SLICE = slice(1, 5)
SPATIAL_SLICE = slice(5, 261)


def cached(path, compute_fn):
    """Load a torch object from *path* if it exists, else compute, save, and return it.

    Keeps src/ modules focused on pure computation — caching is the caller's
    concern (see rabbit_hull.ipynb), not something each module reimplements.
    """
    path = Path(path)
    if path.exists():
        print(f"Loading cached {path.name}…")
        result = torch.load(path, weights_only=True)
    else:
        result = compute_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"Cached to {path}")
    return result
