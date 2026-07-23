"""Token-type/footprint concepts (Section 8) and dictionary geometry + baselines (Sections 9-11).

`footprint` and `gram_matrix` operate directly on the sparse (indices, values)
representation from `sae.encode_sparse` rather than a densified `(n, t,
n_atoms)` tensor — that tensor would be several GB at analysis-set scale, and
neither computation actually needs it materialized all at once.
"""

import numpy as np
import torch

import sae as sae_module
from utils import CLS_IDX, REG_SLICE, SPATIAL_SLICE

# ---------------------------------------------------------------------------
# Section 8: token-type-specific / footprint concepts
# ---------------------------------------------------------------------------


def footprint(indices: torch.Tensor, values: torch.Tensor, n_atoms: int) -> np.ndarray:
    """omega_i = mean over images of Z[:, :, i] (Section 8, exact formula).

    Returns
    -------
    (n_atoms, n_tokens) — footprint per concept (omega.T from the formula above).
    """
    n_images, n_tokens, k = indices.shape
    omega = torch.zeros(n_tokens, n_atoms, dtype=torch.float32)
    for t in range(n_tokens):
        flat_idx = indices[:, t, :].reshape(-1).long()
        flat_val = values[:, t, :].reshape(-1).float()
        omega[t].scatter_add_(0, flat_idx, flat_val)
    omega /= n_images
    return omega.T.numpy()


def footprint_entropy(footprint_matrix: np.ndarray) -> np.ndarray:
    """Shannon entropy of each concept's footprint distribution over token positions.

    Footprint rows are normalized to sum to 1 first (paper doesn't give an
    explicit normalization beyond "entropy of its token-wise activation
    distribution" — see `PAPER_NOTES.md` §8). Concepts with all-zero
    footprint get entropy 0.
    """
    mass = footprint_matrix.sum(axis=1, keepdims=True)
    probs = np.divide(footprint_matrix, mass, out=np.zeros_like(footprint_matrix), where=mass > 0)
    terms = np.where(probs > 0, probs * np.log(probs), 0.0)
    return -terms.sum(axis=1)


def classify_token_types(footprint_matrix: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Label each concept cls-only/reg-only/spatial-only/mixed by footprint mass fraction.

    At `n_atoms=1000` (vs. the paper's 32,000), expect much smaller absolute
    counts than the paper's "1 cls-only, hundreds reg-only" — a scaled
    expectation, not a bug.
    """
    mass = footprint_matrix.sum(axis=1, keepdims=True)
    mass = np.where(mass > 0, mass, 1.0)
    cls_frac = footprint_matrix[:, CLS_IDX] / mass[:, 0]
    reg_frac = footprint_matrix[:, REG_SLICE].sum(axis=1) / mass[:, 0]
    spatial_frac = footprint_matrix[:, SPATIAL_SLICE].sum(axis=1) / mass[:, 0]

    labels = np.full(footprint_matrix.shape[0], "mixed", dtype=object)
    labels[cls_frac >= threshold] = "cls-only"
    labels[reg_frac >= threshold] = "reg-only"
    labels[spatial_frac >= threshold] = "spatial-only"
    return labels


# ---------------------------------------------------------------------------
# Sections 9/5: co-activation Gram matrix + baselines
# ---------------------------------------------------------------------------


def gram_matrix(indices: torch.Tensor, values: torch.Tensor, n_atoms: int, chunk_size: int = 500) -> np.ndarray:
    """G = Z^T Z (Sections 5/9), accumulated in image chunks to avoid densifying
    the full (n, t, n_atoms) code tensor at once."""
    n_images = indices.shape[0]
    G = torch.zeros(n_atoms, n_atoms, dtype=torch.float64)
    for i in range(0, n_images, chunk_size):
        z_chunk = sae_module.densify(indices[i : i + chunk_size], values[i : i + chunk_size], n_atoms)
        flat = z_chunk.reshape(-1, n_atoms).double()
        G += flat.T @ flat
    return G.numpy()


def random_gram_baseline(G: np.ndarray, rho: float, rng: np.random.RandomState) -> np.ndarray:
    """Appendix E formulas (1)-(2): random symmetric matrix matching G's sparsity/mass."""
    n = G.shape[0]
    U = rng.random((n, n))
    V = rng.random((n, n))
    R = U * (V < rho)
    R_sym = (R + R.T) / 2
    R_tilde = R_sym * (np.linalg.norm(G) / (np.linalg.norm(R_sym) + 1e-12))
    np.fill_diagonal(R_tilde, np.diag(G))
    return R_tilde


def shuffled_gram_baseline(G: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Appendix E formula (3): random permutation of G's off-diagonal entries."""
    n = G.shape[0]
    iu = np.triu_indices(n, k=1)
    permuted = rng.permutation(G[iu])
    S = np.zeros_like(G)
    S[iu] = permuted
    S = S + S.T
    np.fill_diagonal(S, np.diag(G))
    return S


def spectrum(M: np.ndarray) -> np.ndarray:
    """Eigenvalues of symmetric M, descending."""
    return np.linalg.eigvalsh(M)[::-1]


# ---------------------------------------------------------------------------
# Sections 10-11: dictionary geometry + baselines
# ---------------------------------------------------------------------------


def hoyer_score(vectors: np.ndarray) -> np.ndarray:
    """Hoyer sparsity per row: (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)."""
    n = vectors.shape[1]
    l1 = np.abs(vectors).sum(axis=1)
    l2 = np.linalg.norm(vectors, axis=1)
    return (np.sqrt(n) - l1 / (l2 + 1e-12)) / (np.sqrt(n) - 1)


def antipodal_pairs(D: np.ndarray, threshold: float = -0.9) -> list[tuple[int, int, float]]:
    """Pairs (i, j, cos_theta) with cos_theta <= threshold, sorted most-antipodal first."""
    d_norm = D / np.linalg.norm(D, axis=1, keepdims=True)
    cos = d_norm @ d_norm.T
    iu = np.triu_indices(D.shape[0], k=1)
    cos_vals = cos[iu]
    mask = cos_vals <= threshold
    pairs = list(zip(iu[0][mask].tolist(), iu[1][mask].tolist(), cos_vals[mask].tolist()))
    pairs.sort(key=lambda p: p[2])
    return pairs


def random_sphere_baseline(c: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    """Gaussian random matrix, rows L2-normalized to unit norm (Appendix F)."""
    H = rng.normal(size=(c, d))
    return H / np.linalg.norm(H, axis=1, keepdims=True)


def coherence_minimized_baseline(
    c: int, d: int, steps: int, lr: float, rng: np.random.RandomState, device: str = "cpu"
) -> np.ndarray:
    """Gradient-descent frame-potential minimization on the unit sphere.

    An approximate stand-in for the paper's exact TAAP-algorithm Grassmannian
    frame (Appendix F): the paper needed ~6 GPU-hours for this at c=32,000;
    at our c=1,000 scale a plain autograd loop converges in seconds-to-minutes,
    so no TAAP/CUDA-specific implementation is used here — documented
    substitution, not an attempt at an equally optimal frame.
    """
    torch.manual_seed(int(rng.randint(0, 2**31 - 1)))
    H = torch.randn(c, d, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([H], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        h_norm = H / H.norm(dim=1, keepdim=True)
        gram = h_norm @ h_norm.T
        off_diag = gram - torch.diag(torch.diag(gram))
        loss = (off_diag**2).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        h_norm = H / H.norm(dim=1, keepdim=True)
    return h_norm.detach().cpu().numpy()
