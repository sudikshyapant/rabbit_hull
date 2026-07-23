# Into the Rabbit Hull — scaled-down reimplementation

A scaled-down reimplementation of ["Into the Rabbit Hull"](https://arxiv.org/abs/2510.08638)
(DINOv2 concept discovery via a stable sparse autoencoder + the Minkowski Representation
Hypothesis), sized to run on Colab Pro. Full paper details are in `PAPER_NOTES.md`.

**Status**: all planned phases implemented, across 4 notebooks (run in order, each after
`rabbit_hull.ipynb`):

1. `rabbit_hull.ipynb` — data → DINOv2 activations → k-means centroids → trained SAE.
2. `classification_segmentation.ipynb` — concept importance, classification & segmentation
   (ADE20K) task recruitment, Elsewhere concepts, border concepts.
3. `dictionary_geometry.ipynb` — token-type/footprint concepts, co-activation spectrum,
   dictionary geometry & baselines.
4. `mrh_analysis.ipynb` — position-embedding analysis, MRH empirical evidence (k-NN
   geodesics, Archetypal Analysis, block structure).

Depth estimation (one of the paper's three downstream tasks) and the DinoVision
visualization tool stay out of scope — see the differences table below.

## Run in Colab

1. Upload this folder (or clone the repo) into your Colab environment.
2. Add an `HF_TOKEN` secret (Runtime → Secrets) — the `imagenet-1k` dataset is gated;
   accept its terms on huggingface.co first.
3. Run `rabbit_hull.ipynb` top to bottom. It mounts Google Drive on first run and caches
   activations/centroids/checkpoints under `MyDrive/rabbit_hull/cache/` so they survive
   session restarts.
4. Then run `classification_segmentation.ipynb`, `dictionary_geometry.ipynb`, and
   `mrh_analysis.ipynb`, in that order (each reuses the previous notebooks' cached
   activations/latents where possible — see each notebook's Setup section).

**Drive usage**: ~13.3GB for Phase 1 (18,000 sae_train + 15,000 analysis = 33,000 images ×
261 tokens × 768 dims, fp16, ~0.4GB per 1,000 images), leaving a small margin under a 14GB
quota. Lower `n_sae_train`/`n_analysis` in `config.py` if you need more headroom. On top of
this, `classification_segmentation.ipynb` adds sparse SAE latents (~200MB, not dense — see
`sae.encode_sparse`) plus ADE20K activations (~0.4GB per 1,000 images at
`n_segmentation=2,000`); `dictionary_geometry.ipynb` and `mrh_analysis.ipynb` add no new
Drive usage (`mrh_analysis.ipynb`'s per-layer position activations are deliberately *not*
persisted — see `model_utils.extract_all_layer_activations`'s docstring). The raw
ImageNet-1k download itself (~6-7GB) is *not* on Drive — HuggingFace's dataset cache
defaults to local/ephemeral Colab storage, so it re-downloads each fresh session (doesn't
count against your Drive quota).

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
| Elsewhere-concept causal verification (RISE) | 8000 masked forward passes/image, all classes | 300 masks × 3 images × top-10 classes (~9,000 passes total) | 8000/image × thousands of images is infeasible on a single Colab session |
| Elsewhere-concept object-removal method | not fully specified (RISE-style masking) | proxy: patches scoring highest against the classification probe's class-weight direction | plain ImageNet classification images have no ground-truth object mask |
| Segmentation dataset | ADE20K, full | `scene_parse_150` (public ADE20K mirror), 2,000 validation images | fits Colab session time; public, no HF gating unlike imagenet-1k |
| Grassmannian frame baseline | exact TAAP algorithm, ~6 GPU-hours at c=32,000 | gradient-descent frame-potential minimization, seconds-to-minutes at c=1,000 | TAAP's cost was driven by the paper's much larger dictionary; a plain autograd loop suffices at this scale |
| Position-embedding analysis images | 1,000,000 | 1,000 (not persisted to Drive — recomputed per session) | all-layer activations at this scale are ~10GB; cheaper to recompute than store |
| Position classifier | not specified | `SGDClassifier` (linear, log loss) rather than exact `LogisticRegression` | 256-class, up to 256,000-sample, 13-layer sweep needs a scalable solver |
| k-NN graph `k` (MRH geodesics) | "standard" (unspecified) | 15 | paper doesn't give a value; documented choice |
| Archetypal Analysis scope | dataset-wide comparisons implied | per-image (261 tokens), 20 images, archetype counts 3-30 | matches the paper's own "~10 archetypes per image" framing; keeps AA's alternating optimization cheap |

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
  utils.py           generic cache-or-compute helper + shared token-layout constants
  imagenet200.py      the 200-class WordNet ID list used for the analysis split
  data_utils.py         ImageNet-1k loading & splitting (sae_train + analysis)
  model_utils.py           DINOv2 activation extraction (last-layer + all-layer; pure compute)
  kmeans_utils.py             centroid fitting for conv(A) approximation (same)
  sae.py                        the stable SAE model + training loop + sparse/dense encoding
  probing.py                     linear probes + concept-importance machinery (§6)
  ade20k_utils.py                  ADE20K (scene_parse_150) loading + patch-label downsampling
  rise.py                           scaled-down RISE causal verification for Elsewhere concepts
  geometry.py                        footprint/token-type concepts + dictionary geometry (§8-11)
  position_utils.py                   position-embedding analysis (§12)
  mrh_utils.py                          k-NN geodesics, Archetypal Analysis, block structure (§14)
rabbit_hull.ipynb                Phase 1: activations → centroids → trained SAE
classification_segmentation.ipynb  Phase 2: concept importance, classification/segmentation, Elsewhere/border concepts (§6-7)
dictionary_geometry.ipynb            Phase 3: token-type concepts, co-activation, dictionary geometry (§8-11)
mrh_analysis.ipynb                     Phase 4: position embeddings, MRH empirical evidence (§12, §14)
PAPER_NOTES.md       full paper extraction (reference)
```
