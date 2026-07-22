"""K-means centroids approximating conv(A), the convex hull of activations.

The stable SAE constrains dictionary atoms to lie in conv(A). Following the
paper, we approximate conv(A) with a set of centroids (paper: 128,000 over
1.4M images; here: 4,000 over a subsample of our sae_train split — see
README.md).
"""

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def fit_centroids(activations: torch.Tensor, config: dict) -> torch.Tensor:
    """Fit MiniBatchKMeans centroids over a subsample of token activations.

    Parameters
    ----------
    activations : (n_images, n_tokens, d_model) tensor, e.g. the sae_train cache

    Returns
    -------
    Tensor of shape (n_centroids, d_model), dtype float32.
    """
    tokens = activations.reshape(-1, config["d_model"]).float().numpy()

    rng = np.random.RandomState(config["random_state"])
    max_tokens = config["kmeans_max_tokens"]
    if len(tokens) > max_tokens:
        idx = rng.choice(len(tokens), max_tokens, replace=False)
        tokens = tokens[idx]

    print(f"Fitting {config['n_centroids']} centroids on {len(tokens)} tokens…")
    kmeans = MiniBatchKMeans(
        n_clusters=config["n_centroids"],
        random_state=config["random_state"],
        n_init="auto",
    )
    kmeans.fit(tokens)
    print("Centroid fitting done.")
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
