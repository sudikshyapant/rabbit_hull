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
