"""Linear probes + concept-importance machinery shared by classification and segmentation.

Implements Section 6's exact derivation: for a linear probe `Y = A W^T` and the
SAE approximation `A ~= Z D`, `Y = Z (D W^T) = Z W'`. `W'` is the alignment
between dictionary concepts and task outputs; the concept-importance vector is
`phi = E(Z) W'`, decomposable per-concept-per-class via `E(Z) * W'` (elementwise).

All functions here take plain numpy arrays (not torch tensors) — everything
downstream is either an sklearn model or basic linear algebra, and this keeps
this module framework-agnostic; callers densify/convert torch tensors before
calling in.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_val_split(X, y, config: dict, test_size: float = 0.2):
    """Stratified train/val split for probe training (separate from the SAE's own splits)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=config["random_state"])


def train_linear_probe(X_train, y_train, X_val, y_val, config: dict):
    """Multiclass logistic-regression linear probe. Returns (probe, val_accuracy)."""
    probe = LogisticRegression(C=config["probe_C"], max_iter=1000)
    probe.fit(X_train, y_train)
    val_acc = probe.score(X_val, y_val)
    print(f"Linear probe val accuracy: {val_acc:.4f}")
    return probe, val_acc


def concept_importance(Z: np.ndarray, D: np.ndarray, W: np.ndarray):
    """Section 6: W' = D W^T, contrib = E(Z) * W' (per-concept-per-class), phi = sum over concepts.

    Parameters
    ----------
    Z : (n, c) — SAE codes for whatever population the probe was trained/evaluated over
        (e.g. cls-token codes per image for classification, per-token codes for segmentation).
    D : (c, d) — SAE dictionary.
    W : (o, d) — linear probe weight matrix (`probe.coef_` from `train_linear_probe`).

    Returns
    -------
    phi : (o,) — aggregate concept-importance vector, one score per class/output.
    contrib : (c, o) — per-concept-per-class contribution (phi = contrib.sum(axis=0)).
    """
    W_prime = D @ W.T  # (c, o)
    E_Z = Z.mean(axis=0)  # (c,)
    contrib = E_Z[:, None] * W_prime  # (c, o)
    phi = contrib.sum(axis=0)
    return phi, contrib


def top_concepts_for_class(contrib: np.ndarray, class_idx: int, k: int = 10) -> np.ndarray:
    """Indices of the k concepts contributing most positively to *class_idx*."""
    return np.argsort(-contrib[:, class_idx])[:k]


def random_control_indices(c: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.choice(c, size=k, replace=False)


def task_subspace_stats(D: np.ndarray, idx: np.ndarray, random_idx: np.ndarray) -> dict:
    """Fig 3 middle/right: intra-group cosine similarity + singular-value spectrum,
    task concept subset vs. a random control subset of the same size.
    """

    def _stats(sub_idx):
        d_sub = D[sub_idx]
        d_norm = d_sub / np.linalg.norm(d_sub, axis=1, keepdims=True)
        cos = d_norm @ d_norm.T
        iu = np.triu_indices(len(sub_idx), k=1)
        mean_cos = float(cos[iu].mean())
        singular_values = np.linalg.svd(d_sub, compute_uv=False)
        return mean_cos, singular_values

    task_cos, task_sv = _stats(idx)
    random_cos, random_sv = _stats(random_idx)
    return {
        "task_cosine_sim": task_cos,
        "random_cosine_sim": random_cos,
        "task_singular_values": task_sv,
        "random_singular_values": random_sv,
    }
