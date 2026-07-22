"""ImageNet-1k loading: a broad sae_train split, plus a curated analysis split.

sae_train is a random sample drawn from the *validation* split, across
whichever classes aren't reserved for analysis (kept broad on purpose — the
SAE dictionary should stay general-purpose, not biased toward 200 classes).

analysis targets ~config['n_analysis'] images spread evenly over the 200
curated ImageNet-200 classes (imagenet200.py) — val only has 50 images/class,
so we fill any remainder per class by streaming (not downloading) matching
images from the *train* split. See README.md for why 200 classes over more
images/class was chosen.
"""

import os

import numpy as np

from imagenet200 import IMAGENET200_WNIDS


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


def _label_ids_for_wnids(dataset, wnids: list[str]) -> list[int]:
    """Map WordNet IDs to *dataset*'s integer label ids (via its ClassLabel feature)."""
    names = dataset.features["label"].names
    return [names.index(w) for w in wnids]


def make_sae_train_split(val_dataset, config: dict) -> np.ndarray:
    """Random sample of val indices for SAE training, excluding the 200 analysis classes.

    Excluding those classes keeps the analysis split's images completely
    unseen by the SAE (the SAE is unsupervised, so this isn't about label
    leakage — it's just a clean separation between "what the dictionary was
    built from" and "what we probe it with" later).
    """
    analysis_ids = set(_label_ids_for_wnids(val_dataset, IMAGENET200_WNIDS))
    labels = np.array(val_dataset["label"])
    eligible = np.where(~np.isin(labels, list(analysis_ids)))[0]

    rng = np.random.RandomState(config["random_state"])
    n = min(config["n_sae_train"], len(eligible))
    return rng.choice(eligible, n, replace=False)


def load_analysis_images(val_dataset, config: dict):
    """~config['n_analysis'] images spread over IMAGENET200_WNIDS.

    Takes every available val image per class first (val is fixed at 50/class,
    taken in dataset order), then streams the train split — filtered to only
    these 200 classes, stopping as soon as every class is topped up — to fill
    the rest of each class's target_per_class quota. Streaming means we only
    download the images we actually keep, not the full train set.

    Returns
    -------
    images : list of PIL images (RGB)
    labels : list of int label ids (same indexing as `images`)
    """
    label_ids = _label_ids_for_wnids(val_dataset, IMAGENET200_WNIDS)
    target_per_class = config["n_analysis"] // len(label_ids)

    val_labels = np.array(val_dataset["label"])
    by_class: dict[int, list] = {}
    for lid in label_ids:
        val_idx = np.where(val_labels == lid)[0][:target_per_class]
        by_class[lid] = [("val", int(i)) for i in val_idx]

    remaining = {lid: target_per_class - len(v) for lid, v in by_class.items()
                 if len(v) < target_per_class}
    if remaining:
        n_needed = sum(remaining.values())
        print(f"Streaming train split to top up {n_needed} images across "
              f"{len(remaining)} classes (val only has 50/class)…")
        from datasets import load_dataset

        train_stream = load_dataset("imagenet-1k", split="train", streaming=True)
        scanned = 0
        for example in train_stream:
            scanned += 1
            lid = example["label"]
            if lid in remaining:
                by_class[lid].append(("train", example["image"]))
                remaining[lid] -= 1
                if remaining[lid] == 0:
                    del remaining[lid]
                if not remaining:
                    break
            if scanned % 100_000 == 0:
                print(f"  scanned {scanned} train images, {len(remaining)} classes left…")

    images, labels = [], []
    for lid, items in by_class.items():
        for source, payload in items:
            img = val_dataset[payload]["image"] if source == "val" else payload
            images.append(img.convert("RGB"))
            labels.append(lid)

    print(f"Analysis set: {len(images)} images across {len(by_class)} classes "
          f"(target was {target_per_class}/class).")
    return images, labels
