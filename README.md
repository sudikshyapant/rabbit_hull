# Into the Rabbit Hull — scaled-down reimplementation

A scaled-down reimplementation of ["Into the Rabbit Hull"](https://arxiv.org/abs/2510.08638)
(DINOv2 concept discovery via a stable sparse autoencoder + the Minkowski Representation
Hypothesis), sized to run on Colab Pro. Full paper details are in `PAPER_NOTES.md`.

**Status**: Phase 1 only — data → DINOv2 activations → k-means centroids → trained SAE
(`rabbit_hull.ipynb`). Downstream-task and geometry/MRH analysis are follow-up phases (see
the notebook's last cell).

## Run in Colab

1. Upload this folder (or clone the repo) into your Colab environment.
2. Add an `HF_TOKEN` secret (Runtime → Secrets) — the `imagenet-1k` dataset is gated;
   accept its terms on huggingface.co first.
3. Run `rabbit_hull.ipynb` top to bottom. It mounts Google Drive on first run and caches
   activations/centroids/checkpoints under `MyDrive/rabbit_hull/cache/` so they survive
   session restarts.

## Differences from the paper

| Aspect | Paper | This implementation | Why |
|---|---|---|---|
| Dictionary size | 32,000 atoms | 1,000 atoms | fit Colab Pro compute/storage |
| K-means centroids (conv(A) approx.) | 128,000 | 4,000 (same 4x ratio to dict size) | scaled proportionally |
| Sparsity `k` | 8 | 8 | kept as-is — still very sparse at 1,000 atoms |
| Training images | 1.4M ImageNet train | 45,000 (ImageNet-1k *val* split, subset) | avoids the gated ~150GB train download |
| Analysis images | full val split | 5,000 (held out from the same val split) | same reasoning |
| Epochs | 50 | 50 | cheap at this scale |
| SAE learning rate / batch size | not stated in paper | lr=1e-3, batch=4096 tokens | paper doesn't specify these |
| Downstream tasks | classification, segmentation, depth | classification + segmentation (depth deferred) | scoped down for this pass |

Everything else (the `D = S @ C` convex-hull-constrained decoder, BatchTopK sparsity,
model choice, objective) follows the paper as described in `PAPER_NOTES.md`.

## Layout

```
src/
  config.py         all hyperparameters + Colab/Drive path setup
  utils.py           generic cache-or-compute helper
  data_utils.py        ImageNet-1k loading & splitting
  model_utils.py         DINOv2 activation extraction (pure compute; caching is in the notebook)
  kmeans_utils.py           centroid fitting for conv(A) approximation (same)
  sae.py                      the stable SAE model + training loop
rabbit_hull.ipynb    driver notebook
PAPER_NOTES.md       full paper extraction (reference)
```
