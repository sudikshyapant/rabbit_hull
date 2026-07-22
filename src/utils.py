"""Small shared helpers."""

from pathlib import Path

import torch


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
