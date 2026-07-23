"""Scaled-down RISE causal verification for "Elsewhere" concepts (PAPER_NOTES.md Section 3 /
Appendix C.2).

Paper: 8000 pixel-masked forward passes per image, across all classes. Here,
scaled to `config['rise_n_masks']` masks x `config['rise_images_per_class']`
images x `config['rise_n_classes']` top classes (~9,000 total forward passes
at defaults) — feasible on a single Colab GPU session. See README.md's
differences-from-paper table.
"""

import numpy as np
import torch
from PIL import Image

import sae as sae_module
from utils import SPATIAL_SLICE


def random_masks(n_masks: int, grid: int, prob: float, size: tuple, rng: np.random.RandomState) -> np.ndarray:
    """Low-res random binary masks, bilinearly upsampled to *size* = (W, H) — standard RISE construction."""
    w, h = size
    small = (rng.random((n_masks, grid, grid)) < prob).astype(np.uint8) * 255
    masks = np.zeros((n_masks, h, w), dtype=np.float32)
    for i in range(n_masks):
        up = Image.fromarray(small[i]).resize((w, h), Image.BILINEAR)
        masks[i] = np.array(up, dtype=np.float32) / 255.0
    return masks


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Zero out (black out) pixels where *mask* is 0."""
    arr = np.array(image.convert("RGB")).astype(np.float32)
    masked = (arr * mask[..., None]).astype(np.uint8)
    return Image.fromarray(masked)


def _concept_activation(images: list, model, processor, sae, config: dict, concept_idx: int) -> np.ndarray:
    """Mean spatial-token activation of *concept_idx*, per image, for a batch of (possibly masked) images."""
    device = config["device"]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state
        z = sae_module.encode_dense(sae, tokens)
    return z[:, SPATIAL_SLICE, concept_idx].mean(dim=1).cpu().numpy()


def rise_importance(image: Image.Image, model, processor, sae, config: dict, concept_idx: int) -> np.ndarray:
    """RISE per-pixel importance heatmap for *concept_idx* on *image* (Petsiuk et al. 2018)."""
    rng = np.random.RandomState(config["random_state"])
    n_masks = config["rise_n_masks"]
    masks = random_masks(n_masks, config["rise_mask_grid"], config["rise_mask_prob"], image.size, rng)

    activations = np.zeros(n_masks, dtype=np.float32)
    batch_size = config["batch_size"]
    for i in range(0, n_masks, batch_size):
        batch_masks = masks[i : i + batch_size]
        batch_images = [apply_mask(image, m) for m in batch_masks]
        activations[i : i + batch_size] = _concept_activation(
            batch_images, model, processor, sae, config, concept_idx
        )

    # Gamma_RISE(i) ~ E[activation | mask includes i] (PAPER_NOTES.md Section 3)
    heatmap = (masks * activations[:, None, None]).mean(axis=0)
    heatmap /= masks.mean(axis=0) + 1e-8
    return heatmap


def verify_elsewhere_concept(
    image: Image.Image,
    object_region: np.ndarray,
    model,
    processor,
    sae,
    config: dict,
    concept_idx: int,
) -> dict:
    """Compare concept activation with the object masked out vs. a random same-size region masked out.

    `object_region` (grid, grid) bool: since plain ImageNet classification
    images have no ground-truth object mask, the caller approximates "the
    object" as the patches the trained classification probe finds most
    predictive of the true class — a self-consistent proxy, since the paper
    doesn't specify an exact object-removal method beyond RISE-style masking.
    """
    rng = np.random.RandomState(config["random_state"])
    w, h = image.size

    def upsample(region: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray((region * 255).astype(np.uint8)).resize((w, h), Image.NEAREST),
            dtype=np.float32,
        ) / 255.0

    n_object_patches = int(object_region.sum())
    non_object_patches = np.flatnonzero(~object_region)
    n_random = min(n_object_patches, len(non_object_patches))
    random_region = np.zeros_like(object_region)
    random_region.flat[rng.choice(non_object_patches, size=n_random, replace=False)] = True

    keep_all = np.ones((h, w), dtype=np.float32)
    keep_object_masked = upsample(~object_region)  # 0 over the object, 1 elsewhere
    keep_random_masked = upsample(~random_region)

    images = [
        apply_mask(image, keep_all),
        apply_mask(image, keep_object_masked),
        apply_mask(image, keep_random_masked),
    ]
    acts = _concept_activation(images, model, processor, sae, config, concept_idx)
    return {
        "with_object": float(acts[0]),
        "object_masked": float(acts[1]),
        "random_region_masked": float(acts[2]),
    }
