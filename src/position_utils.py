"""Position-embedding analysis (Section 6 / Appendix I of PAPER_NOTES.md).

Operates on `model_utils.extract_all_layer_activations` output — see that
function's docstring for why it's deliberately not cached to disk.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from utils import SPATIAL_SLICE


def direct_average_basis(layer_acts: np.ndarray) -> np.ndarray:
    """Method 1 (§12): p_i = mean embedding per spatial position, across images.

    Parameters
    ----------
    layer_acts : (n_images, n_tokens, d_model) — activations from a single layer.

    Returns
    -------
    (n_tokens, d_model)
    """
    return layer_acts.mean(axis=0)


def train_position_classifier(layer_acts: np.ndarray, config: dict):
    """Method 2 (§12): linear classifier predicting spatial position index from embedding.

    Uses `SGDClassifier` (linear, log loss) rather than `LogisticRegression`:
    at `config["n_position"]` images x 256 spatial positions x 13 layers this
    is up to ~256,000 samples x 768 dims x 256 classes *per layer*, where
    SGD's mini-batch fitting is far more tractable than exact solvers.

    Returns
    -------
    P : (256, d_model) — the classifier's weight matrix (rows = positional basis).
    accuracy : float — held-out position-decoding accuracy.
    """
    spatial = layer_acts[:, SPATIAL_SLICE, :]  # (n_images, 256, d_model)
    n_images, n_positions, d_model = spatial.shape
    X = spatial.reshape(-1, d_model)
    y = np.tile(np.arange(n_positions), n_images)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config["random_state"]
    )
    clf = SGDClassifier(loss="log_loss", random_state=config["random_state"], max_iter=20, tol=1e-3)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_val, y_val)
    return clf.coef_, float(accuracy)


def stable_rank(M: np.ndarray) -> float:
    """||M||_F^2 / ||M||_2^2 (standard definition; not given explicitly in the paper)."""
    frob_sq = np.linalg.norm(M, "fro") ** 2
    spectral_sq = np.linalg.norm(M, 2) ** 2
    return float(frob_sq / spectral_sq)


def per_image_pca_components(tokens: np.ndarray, variance_threshold: float = 0.95) -> int:
    """# PCA components needed to reach *variance_threshold* cumulative explained variance.

    Parameters
    ----------
    tokens : (n_tokens, d_model) — a single image's tokens.
    """
    pca = PCA()
    pca.fit(tokens)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def remove_positional_subspace(tokens: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project *tokens* (n, d) orthogonal to P's (k, d) row space (Appendix I.2, Fig 27)."""
    Q, _ = np.linalg.qr(P.T)  # (d, k) orthonormal basis for row(P)
    return tokens - tokens @ Q @ Q.T
