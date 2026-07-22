"""Centralised configuration for the scaled-down "Into the Rabbit Hull" reimplementation.

Every value that was scaled down from the paper (arXiv:2510.08638) is marked
"# paper: ..." so the gap is visible in one place. See ../README.md for the
full differences-from-paper table.
"""

import os
from pathlib import Path

import torch


def _setup_dirs() -> tuple[Path, Path]:
    """Mount Google Drive (Colab only) and return (cache_dir, results_dir)."""
    in_colab = os.path.exists("/content")
    if in_colab:
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive", force_remount=False)
            base = Path("/content/drive/MyDrive/rabbit_hull")
            print(f"Google Drive mounted — using {base}")
        except Exception:
            base = Path("/content/rabbit_hull")
            print("Drive unavailable — using local Colab storage.")
    else:
        base = Path(__file__).parent.parent

    cache_dir = base / "cache"
    results_dir = base / "result"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, results_dir


_cache_dir, _results_dir = _setup_dirs()

CONFIG = {
    # Model
    "model_name": "facebook/dinov2-with-registers-base",
    "d_model": 768,
    "n_tokens": 261,  # 256 patch + 1 cls + 4 register tokens
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Data
    # sae_train: random sample from the val split, over all classes except the
    #   200 analysis classes below (paper: 1.4M train images).
    # analysis: ~75 images/class over the 200 curated ImageNet-200 classes
    #   (imagenet200.py) — 50/class from val, the rest streamed from train.
    #   Chosen over fewer classes with more images/class: see README.md.
    "n_sae_train": 18_000,
    "n_analysis": 15_000,
    "random_state": 42,
    # Stable SAE
    "n_atoms": 1_000,           # paper: 32,000
    "n_centroids": 4_000,       # paper: 128,000 (same 4x ratio to n_atoms)
    "kmeans_max_tokens": 2_000_000,  # subsample cap for k-means fitting (RAM)
    "sparsity_k": 8,            # paper: 8 (kept as-is)
    "sae_epochs": 50,           # paper: 50
    "sae_lr": 1e-3,             # not specified in paper; chosen default
    "sae_batch_size": 4_096,    # tokens per SGD step; not specified in paper
    # I/O
    "cache_dir": _cache_dir,
    "results_dir": _results_dir,
}
