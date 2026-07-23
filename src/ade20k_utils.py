"""ADE20K (scene_parse_150) loading and per-patch label downsampling.

Used only by the segmentation half of `classification_segmentation.ipynb`.
Unlike imagenet-1k, this dataset is public — no HF gating/auth needed.
"""

import numpy as np
from PIL import Image
from scipy import stats


def load_ade20k(config: dict):
    """Load the ADE20K (scene_parse_150) validation split.

    Note: `scene_parse_150`'s HF loading script may need `trust_remote_code=True`
    on recent `datasets` versions; if this specific dataset id has moved on the
    Hub by the time you run this, swap in the current mirror's id here — nothing
    downstream depends on the exact id, only on `image`/`annotation` fields.
    """
    from datasets import load_dataset

    print("Loading ADE20K (scene_parse_150) validation split…")
    ds = load_dataset("scene_parse_150", split="validation", trust_remote_code=True)
    print(f"Loaded {len(ds)} images.")
    return ds


def make_segmentation_split(dataset, config: dict) -> np.ndarray:
    """Random sample of dataset indices for segmentation analysis (config['n_segmentation'])."""
    rng = np.random.RandomState(config["random_state"])
    n = min(config["n_segmentation"], len(dataset))
    return rng.choice(len(dataset), n, replace=False)


def patchify_labels(segmentation_map: Image.Image, grid_size: int = 16, image_size: int = 224) -> np.ndarray:
    """Majority-vote downsample of a pixel-level segmentation map to a grid_size x grid_size patch grid.

    Resizes with nearest-neighbor (integer class labels must not be
    interpolated) to `image_size`, matching the resolution DINOv2's image
    processor uses upstream — so a patch here lines up 1:1 with a spatial
    token from `model_utils.extract_activations` — then majority-votes each
    (image_size / grid_size)-pixel block.
    """
    seg = np.array(segmentation_map.resize((image_size, image_size), Image.NEAREST))
    patch_size = image_size // grid_size
    patches = (
        seg.reshape(grid_size, patch_size, grid_size, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(grid_size, grid_size, -1)
    )
    return stats.mode(patches, axis=-1, keepdims=False).mode


def border_mask(patch_labels: np.ndarray) -> np.ndarray:
    """True where a patch's label differs from any 4-connected neighbor (boundary token flag)."""
    border = np.zeros_like(patch_labels, dtype=bool)
    border[:-1, :] |= patch_labels[:-1, :] != patch_labels[1:, :]
    border[1:, :] |= patch_labels[:-1, :] != patch_labels[1:, :]
    border[:, :-1] |= patch_labels[:, :-1] != patch_labels[:, 1:]
    border[:, 1:] |= patch_labels[:, :-1] != patch_labels[:, 1:]
    return border
