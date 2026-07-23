"""Stable sparse autoencoder: single-layer encoder + BatchTopK, D = S @ C decoder.

Implements the paper's objective (Section 2):

    min_{Z,D} ||A - ZD||_F^2   s.t.  Z >= 0, ||Z_i||_0 <= k, D_i in conv(A)

The convex-hull constraint on D is enforced by parametrizing D = S @ C, where
C is a fixed set of k-means centroids approximating conv(A) (kmeans_utils.py)
and S is row-stochastic (each atom is a convex combination of centroids, so
it lies inside conv(A) by construction). Sparsity is enforced via BatchTopK
(Bussmann et al. 2024): keep the top (batch_size * k) activations across the
whole batch, zero the rest — this is what "codes" means throughout.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_topk(preact: torch.Tensor, k: int) -> torch.Tensor:
    """BatchTopK sparsity: nonnegative codes, top (batch_size*k) kept overall."""
    preact = F.relu(preact)
    n_keep = preact.shape[0] * k
    flat = preact.flatten()
    if n_keep >= flat.numel():
        return preact
    threshold = torch.topk(flat, n_keep, sorted=False).values.min()
    return preact * (preact >= threshold)


class StableSAE(nn.Module):
    """Sparse autoencoder with a convex-hull-constrained dictionary D = S @ C."""

    def __init__(self, centroids: torch.Tensor, n_atoms: int, sparsity_k: int, d_model: int):
        super().__init__()
        self.sparsity_k = sparsity_k
        self.register_buffer("centroids", centroids)  # (n_centroids, d_model), fixed
        self.encoder = nn.Linear(d_model, n_atoms)
        self.s_logits = nn.Parameter(torch.randn(n_atoms, centroids.shape[0]) * 0.01)

    def dictionary(self) -> torch.Tensor:
        """D = softmax(s_logits, dim=centroids) @ C  — each atom is a convex mix of C."""
        s = torch.softmax(self.s_logits, dim=1)
        return s @ self.centroids  # (n_atoms, d_model)

    def forward(self, x: torch.Tensor):
        z = batch_topk(self.encoder(x), self.sparsity_k)
        recon = z @ self.dictionary()
        return z, recon


def r2_score(sae: StableSAE, tokens: torch.Tensor, device: str) -> float:
    """Reconstruction R^2 of *sae* on *tokens* (fraction of variance explained)."""
    with torch.no_grad():
        _, recon = sae(tokens.to(device))
        recon = recon.cpu()
    ss_res = ((tokens - recon) ** 2).sum()
    ss_tot = ((tokens - tokens.mean(dim=0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def train_sae(sae: StableSAE, activations: torch.Tensor, config: dict) -> list[dict]:
    """Train *sae* with Adam for config['sae_epochs'] epochs. Returns loss/R2 history."""
    device = config["device"]
    sae.to(device)

    tokens = activations.reshape(-1, config["d_model"]).float()
    n_tokens = len(tokens)
    batch_size = config["sae_batch_size"]

    rng = torch.Generator().manual_seed(config["random_state"])
    eval_idx = torch.randperm(n_tokens, generator=rng)[:50_000]
    eval_tokens = tokens[eval_idx]

    optimizer = torch.optim.Adam(sae.parameters(), lr=config["sae_lr"])
    history = []

    for epoch in range(config["sae_epochs"]):
        perm = torch.randperm(n_tokens)
        total_loss = 0.0
        for i in range(0, n_tokens, batch_size):
            batch = tokens[perm[i : i + batch_size]].to(device)
            optimizer.zero_grad()
            _, recon = sae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        mean_loss = total_loss / n_tokens
        r2 = r2_score(sae, eval_tokens, device)
        history.append({"epoch": epoch, "loss": mean_loss, "r2": r2})
        print(f"Epoch {epoch + 1}/{config['sae_epochs']}  loss={mean_loss:.4f}  R2={r2:.4f}")

    return history


def save_checkpoint(sae: StableSAE, config: dict) -> Path:
    path = Path(config["cache_dir"]) / "checkpoints" / "sae.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), path)
    print(f"SAE checkpoint saved to {path}")
    return path


def load_checkpoint(centroids: torch.Tensor, config: dict) -> StableSAE:
    path = Path(config["cache_dir"]) / "checkpoints" / "sae.pt"
    sae = StableSAE(centroids, config["n_atoms"], config["sparsity_k"], config["d_model"])
    sae.load_state_dict(torch.load(path, weights_only=True))
    return sae


def encode_sparse(
    sae: StableSAE, activations: torch.Tensor, config: dict, batch_size: int = 4_096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode *activations* through *sae*, keeping only the top-k nonzero codes per token.

    A dense (n, t, n_atoms) code tensor would be ~7.8GB at 15,000 analysis
    images / n_atoms=1000 (well past this repo's Drive budget), so this keeps
    just the per-token indices/values that BatchTopK leaves nonzero — see
    `densify` to reconstruct a dense tensor transiently when needed.

    Parameters
    ----------
    activations : (n_images, n_tokens, d_model) tensor

    Returns
    -------
    indices : (n_images, n_tokens, k) int32 — atom ids of the nonzero codes
    values  : (n_images, n_tokens, k) float16 — corresponding code values
    """
    device = config["device"]
    sae.to(device)
    sae.eval()

    n_images, n_tokens, d_model = activations.shape
    k = sae.sparsity_k
    tokens = activations.reshape(-1, d_model).float()

    all_indices, all_values = [], []
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i : i + batch_size].to(device)
            preact = torch.relu(sae.encoder(batch))
            # Per-token top-k (not BatchTopK): the paper's sparsity constraint
            # |supp(z)| <= k is defined per token (Definition 1); BatchTopK is
            # only a training-time relaxation for stable gradients.
            values, indices = torch.topk(preact, k, dim=1)
            all_indices.append(indices.to(torch.int32).cpu())
            all_values.append(values.half().cpu())

    indices = torch.cat(all_indices, dim=0).reshape(n_images, n_tokens, k)
    values = torch.cat(all_values, dim=0).reshape(n_images, n_tokens, k)
    return indices, values


def densify(indices: torch.Tensor, values: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Scatter sparse (indices, values) from `encode_sparse` back to a dense (n, t, n_atoms) tensor."""
    shape = indices.shape[:-1] + (n_atoms,)
    dense = torch.zeros(shape, dtype=values.dtype)
    dense.scatter_(-1, indices.long(), values)
    return dense


def encode_dense(sae: StableSAE, tokens: torch.Tensor) -> torch.Tensor:
    """Per-token top-k encode for a small in-memory batch already on the model's device.

    Unlike `encode_sparse` (which is for caching a whole dataset to disk),
    this returns a dense tensor directly, staying on-device — used where a
    handful of images are encoded on the fly (RISE masking, per-image
    Archetypal Analysis comparisons) and a CPU round-trip would just be
    wasted overhead.
    """
    preact = torch.relu(sae.encoder(tokens))
    values, indices = torch.topk(preact, sae.sparsity_k, dim=-1)
    dense = torch.zeros_like(preact)
    dense.scatter_(-1, indices, values)
    return dense
