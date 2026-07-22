"""ImageNet-1k validation split loading and train/analysis splitting.

The paper trains its SAE on 1.4M ImageNet train images and analyzes on the
val set separately. We use only the 50,000-image val split for everything,
partitioned into a chunk used to fit the SAE (n_sae_train) and a held-out
chunk reserved for downstream-task/geometry analysis in later phases
(n_analysis). See README.md for why.
"""

import os

import numpy as np


def _hf_login() -> None:
    """Authenticate with HuggingFace (needed: imagenet-1k is a gated dataset).

    Tries, in order: Colab secret, HF_TOKEN env var, existing local
    `huggingface-cli login` credentials (no-op if none of these apply).
    """
    from huggingface_hub import login  # type: ignore

    try:
        from google.colab import userdata  # type: ignore
        login(token=userdata.get("HF_TOKEN"), add_to_git_credential=False)
        print("Authenticated with HuggingFace via Colab secret.")
        return
    except Exception:
        pass

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
        print("Authenticated with HuggingFace via HF_TOKEN env var.")


def load_imagenet_val(config: dict):
    """Load the ImageNet-1k validation split (50,000 images) from HuggingFace."""
    _hf_login()
    from datasets import load_dataset

    print("Loading imagenet-1k validation split (this may take a while)…")
    ds = load_dataset("imagenet-1k", split="validation")
    print(f"Loaded {len(ds)} images.")
    return ds


def make_split(n_total: int, config: dict) -> dict[str, np.ndarray]:
    """Split dataset indices into sae_train / analysis subsets.

    Returns
    -------
    {"sae_train": ndarray, "analysis": ndarray}  (disjoint index arrays)
    """
    n_sae_train = min(config["n_sae_train"], n_total)
    n_analysis = min(config["n_analysis"], n_total - n_sae_train)

    rng = np.random.RandomState(config["random_state"])
    idx = rng.permutation(n_total)

    return {
        "sae_train": idx[:n_sae_train],
        "analysis": idx[n_sae_train : n_sae_train + n_analysis],
    }
