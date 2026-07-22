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

**Drive usage**: ~13.3GB (18,000 sae_train + 15,000 analysis = 33,000 images × 261 tokens ×
768 dims, fp16, ~0.4GB per 1,000 images), leaving a small margin under a 14GB quota. Lower
`n_sae_train`/`n_analysis` in `config.py` if you need more headroom — later phases (e.g.
ADE20K for segmentation) will add modestly on top of this. The raw ImageNet-1k download
itself (~6-7GB) is *not* on Drive — HuggingFace's dataset cache defaults to local/ephemeral
Colab storage, so it re-downloads each fresh session (doesn't count against your Drive
quota).

## Differences from the paper

| Aspect | Paper | This implementation | Why |
|---|---|---|---|
| Dictionary size | 32,000 atoms | 1,000 atoms | fit Colab Pro compute/storage |
| K-means centroids (conv(A) approx.) | 128,000 | 4,000 (same 4x ratio to dict size) | scaled proportionally |
| Sparsity `k` | 8 | 8 | kept as-is — still very sparse at 1,000 atoms |
| SAE training images | 1.4M ImageNet train | 18,000 (ImageNet-1k *val* split, random, excl. analysis classes) | avoids the gated ~150GB train download, fits a 14GB Drive quota |
| Analysis images | full val split, all 1,000 classes | 15,000, over a curated 200-class subset (`src/imagenet200.py`, the ImageNet-R/A class list) at ~75 images/class (50 from val + streamed top-up from train) | fits the same Drive budget; 200 classes at 75/class was chosen over fewer classes with more images/class — see the design discussion below |
| Epochs | 50 | 50 | cheap at this scale |
| SAE learning rate / batch size | not stated in paper | lr=1e-3, batch=4096 tokens | paper doesn't specify these |
| Downstream tasks | classification, segmentation, depth | classification + segmentation (depth deferred) | scoped down for this pass |

**Why 200 classes at ~75 images/class, not fewer classes with more images each?**
Per-class depth has strongly diminishing returns well past ~16-75 images/class for linear
probing on strong frozen features, while class *diversity* is what several of the paper's
own claims depend on — e.g. it explicitly checks that "Elsewhere" concepts recur "across
diverse object categories" to argue the effect isn't an artifact of specific classes
(`PAPER_NOTES.md` §7 / Appendix C.2). A narrow 50-class subset would undercut exactly that
kind of check, and this particular 200-class list is already animal/dog-breed-heavy, so
narrowing it further would skew it more.

Everything else (the `D = S @ C` convex-hull-constrained decoder, BatchTopK sparsity,
model choice, objective) follows the paper as described in `PAPER_NOTES.md`.

## Layout

```
src/
  config.py         all hyperparameters + Colab/Drive path setup
  utils.py           generic cache-or-compute helper
  imagenet200.py      the 200-class WordNet ID list used for the analysis split
  data_utils.py         ImageNet-1k loading & splitting (sae_train + analysis)
  model_utils.py           DINOv2 activation extraction (pure compute; caching is in the notebook)
  kmeans_utils.py             centroid fitting for conv(A) approximation (same)
  sae.py                        the stable SAE model + training loop
rabbit_hull.ipynb    driver notebook
PAPER_NOTES.md       full paper extraction (reference)
```
