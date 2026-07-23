"""ImageNet-1k loading: a broad sae_train split, plus a curated analysis split.

sae_train is a random sample drawn from the *validation* split, across
whichever classes aren't reserved for analysis (kept broad on purpose — the
SAE dictionary should stay general-purpose, not biased toward 200 classes).

analysis targets ~config['n_analysis'] images spread evenly over the 200
curated ImageNet-200 classes (imagenet200.py) — val only has 50 images/class,
so we fill any remainder per class by streaming (not downloading) matching
images from the *train* split. See README.md for why 200 classes over more
images/class was chosen.

IMPORTANT: we load via the generic "parquet" builder pointed at explicit
hf:// file globs, NOT via load_dataset("ILSVRC/imagenet-1k", ...). Going
through that repo's own loader — even with a `data_files` argument meant to
scope it to one split — was observed (twice) to resolve and download every
split's parquet shards anyway (~150GB, ~1-2 hours). The generic "parquet"
loader has no such repo-specific logic: it can only ever touch files that
match the glob we hand it. Run `smoke_test()` before a full load to confirm
this on a single ~480MB file before committing to the real (~6-7GB) pull.
"""

import os

import numpy as np

from imagenet200 import IMAGENET200_LABEL_IDS, IMAGENET200_WNIDS

_WNID_TO_LABEL_ID = dict(zip(IMAGENET200_WNIDS, IMAGENET200_LABEL_IDS))

REPO = "ILSVRC/imagenet-1k"
VAL_GLOB = f"hf://datasets/{REPO}/data/validation-*.parquet"
TRAIN_GLOB = f"hf://datasets/{REPO}/data/train-*.parquet"


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


def _check_standard_ordering(dataset) -> None:
    """Verify *dataset*'s label ints follow the standard ILSVRC2012 ordering.

    `features["label"].names` holds human-readable descriptions
    ("tench, Tinca tinca"), not WNIDs — there's no WNID string to match
    against it. Instead, imagenet200.py hardcodes each WNID's index in the
    standard ordering (WNIDs sorted alphabetically), verified against the
    canonical Keras/TF imagenet_class_index.json. This check confirms that
    assumption holds for *this* dataset — index 1 should always be goldfish
    — so a future dataset revision with different ordering fails loudly
    here rather than silently mislabeling every image.
    """
    names = getattr(dataset.features["label"], "names", None)
    if names is None or len(names) != 1000 or "goldfish" not in names[1].lower():
        got = None if names is None else names[1]
        raise RuntimeError(
            "Dataset's label ordering doesn't match the standard ILSVRC2012 "
            f"ordering this code assumes (expected index 1 to be 'goldfish', "
            f"got {got!r}). The WNID->label-id mapping in imagenet200.py "
            "won't be correct here — stop and let me know before proceeding."
        )


def smoke_test() -> None:
    """Fast (~1 file, ~480MB) check that the glob-scoped loading actually works.

    Run this FIRST, before load_imagenet_val(). Confirms, without a multi-GB
    commitment: (1) only the requested file is fetched — no train shards —
    and (2) the dataset's label ordering matches what imagenet200.py assumes
    (see _check_standard_ordering).
    """
    _hf_login()
    from datasets import load_dataset

    one_file = f"hf://datasets/{REPO}/data/validation-00000-of-00014.parquet"
    print(f"Smoke test: loading a single file ({one_file})…")
    ds = load_dataset("parquet", data_files={"validation": one_file}, split="validation")
    print(f"Loaded {len(ds)} rows from one file (expect ~3,500, i.e. 50,000/14).")

    _check_standard_ordering(ds)
    print("Label ordering OK (matches standard ILSVRC2012 ordering).")
    print("Smoke test passed — safe to run load_imagenet_val().")


def load_imagenet_val(config: dict):
    """Load the ImageNet-1k validation split (50,000 images) from HuggingFace.

    Only ever touches files matching VAL_GLOB (see module docstring for why
    we don't go through the repo's own loading script).
    """
    _hf_login()
    from datasets import load_dataset

    print("Loading imagenet-1k validation split (this may take a while)…")
    ds = load_dataset("parquet", data_files={"validation": VAL_GLOB}, split="validation")
    print(f"Loaded {len(ds)} images.")
    return ds


def _label_ids_for_wnids(dataset, wnids: list[str]) -> list[int]:
    """Map WordNet IDs to *dataset*'s integer label ids.

    Uses the fixed, verified imagenet200.py mapping (see its module
    docstring) rather than dataset.features["label"].names, which holds
    descriptions, not WNIDs. _check_standard_ordering guards the assumption.
    """
    _check_standard_ordering(dataset)
    return [_WNID_TO_LABEL_ID[w] for w in wnids]


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
    fetch the images we actually keep, not the full train set.

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

        train_stream = load_dataset(
            "parquet", data_files={"train": TRAIN_GLOB}, split="train", streaming=True
        )
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
