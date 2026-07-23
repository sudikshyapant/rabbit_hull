"""MRH empirical evidence (Section 7.2 / Fig 17 of PAPER_NOTES.md): k-NN geodesics,
Archetypal Analysis vs. SAE, and block structure.
"""

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def build_knn_graph(tokens: np.ndarray, k: int, metric: str = "cosine"):
    """k-NN graph over *tokens* (n, d), symmetrized by union of directed edges
    ("standard symmetrization" — the paper doesn't specify further, Section 14)."""
    graph = kneighbors_graph(tokens, k, mode="distance", metric=metric)
    return graph.maximum(graph.T)


def graph_geodesic_path(graph, src: int, dst: int) -> list[int]:
    """Shortest path node sequence from *src* to *dst* on *graph* (Fig 17 left)."""
    _, predecessors = shortest_path(graph, directed=False, indices=src, return_predecessors=True)
    path = [dst]
    while path[-1] != src:
        prev = predecessors[path[-1]]
        if prev == -9999:
            raise ValueError(f"No path between {src} and {dst} in the k-NN graph.")
        path.append(prev)
    path.reverse()
    return path


def path_points(coords: np.ndarray, steps: int) -> np.ndarray:
    """Evenly-spaced (by arc length) points along the piecewise-linear path through *coords* (m, d).

    With a 2-point *coords* this is plain straight-line interpolation; with a
    graph-geodesic node sequence's coordinates it traces the piecewise-linear
    path — same primitive serves both curves in Fig 17 left.
    """
    if len(coords) == 1:
        return np.repeat(coords, steps, axis=0)
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum[-1]
    targets = np.linspace(0, total, steps)
    points = np.zeros((steps, coords.shape[1]))
    for i, t in enumerate(targets):
        seg_idx = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(coords) - 2))
        seg_len = seg_lengths[seg_idx]
        local_t = (t - cum[seg_idx]) / seg_len if seg_len > 0 else 0.0
        points[i] = coords[seg_idx] + local_t * (coords[seg_idx + 1] - coords[seg_idx])
    return points


def distance_to_manifold(points: np.ndarray, reference_tokens: np.ndarray) -> np.ndarray:
    """Nearest-neighbor distance from each of *points* to *reference_tokens* (Fig 17 left)."""
    nn = NearestNeighbors(n_neighbors=1).fit(reference_tokens)
    dist, _ = nn.kneighbors(points)
    return dist[:, 0]


def simplex_project(V: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al. 2008), vectorized over rows."""
    v = np.atleast_2d(V)
    n = v.shape[1]
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1) - 1
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    rho = cond.sum(axis=1)
    theta = css[np.arange(len(v)), rho - 1] / rho
    result = np.maximum(v - theta[:, None], 0)
    return result if V.ndim > 1 else result[0]


def archetypal_analysis(X: np.ndarray, n_archetypes: int, n_iters: int, rng: np.random.RandomState, lr: float = 0.05):
    """Archetypal Analysis (Cutler & Breiman 1994), PAPER_NOTES.md §14 point 2's exact objective.

    Row-samples convention (X: (n, d), n data points in d dims — e.g. one
    image's 261 tokens), equivalent to the paper's column convention up to
    transposition:
      - B: (n, k) column-simplex-constrained (each column sums to 1, nonneg) —
        archetypes = B^T @ X, (k, d): each archetype is a convex combination of the data.
      - A: (n, k) row-simplex-constrained (each row sums to 1, nonneg) —
        X_hat = A @ archetypes: each data point is a convex combination of archetypes.

    Plain projected-gradient descent on ||X - A(B^T X)||_F^2 (quadratic in
    each of A, B given the other fixed).

    Returns
    -------
    archetypes : (n_archetypes, d)
    B, A : (n, n_archetypes) each
    mse : final reconstruction MSE
    """
    n, d = X.shape
    k = n_archetypes
    A = simplex_project(rng.random((n, k)))
    B = simplex_project(rng.random((k, n))).T  # (n, k), column-simplex

    for _ in range(n_iters):
        Y = B.T @ X  # (k, d) archetypes
        R = A @ Y - X  # (n, d)
        dA = 2 * R @ Y.T
        dB = 2 * X @ (R.T @ A)
        A = simplex_project(A - lr * dA)
        B = simplex_project((B - lr * dB).T).T

    archetypes = B.T @ X
    mse = float(np.mean((A @ archetypes - X) ** 2))
    return archetypes, B, A, mse


def block_structure(archetype_coding: np.ndarray) -> np.ndarray:
    """Hierarchical-clustering reordering of *archetype_coding* rows (n, k) that
    reveals block-diagonal structure when the Gram matrix is reindexed by it (Fig 17 right).
    """
    norm = np.linalg.norm(archetype_coding, axis=1, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    cos = (archetype_coding / norm) @ (archetype_coding / norm).T
    distance = np.clip(1 - cos, 0, None)  # guard against floating-point cos slightly > 1
    condensed = distance[np.triu_indices(len(distance), k=1)]
    linkage_matrix = linkage(condensed, method="average")
    return leaves_list(linkage_matrix)
