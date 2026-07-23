"""DINOv2-B (+ registers) loading and token-activation extraction.

Extracts all 261 last-layer tokens per image (256 patch + 1 cls + 4 register),
matching the paper's `f: X -> R^(t x d)` with d=768, t=261.
"""

import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


def load_model(config: dict):
    """Load DINOv2-B-with-registers and its image processor."""
    print(f"Loading model: {config['model_name']}")
    processor = AutoImageProcessor.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"]).to(config["device"])
    model.eval()
    print("Model loaded.")
    return model, processor


def extract_activations(images: list, model, processor, config: dict) -> torch.Tensor:
    """Run DINOv2 forward passes and return last-hidden-state tokens.

    Parameters
    ----------
    images : list of PIL images

    Returns
    -------
    Tensor of shape (len(images), n_tokens, d_model), dtype float16, on CPU.

    Note: the full result is held in CPU RAM before the caller writes it to
    disk (~0.4GB per 1,000 images at fp16 — e.g. ~12GB for 30,000 images).
    Reduce n_sae_train/n_analysis in config.py if this doesn't fit your Colab
    runtime's RAM or Drive quota.
    """
    device = config["device"]
    batch_size = config["batch_size"]
    all_tokens = []

    for i in tqdm(range(0, len(images), batch_size), desc="Extracting activations"):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        tokens = out.last_hidden_state  # (B, n_tokens, d_model)
        all_tokens.append(tokens.half().cpu())

    return torch.cat(all_tokens, dim=0)


def extract_all_layer_activations(images: list, model, processor, config: dict) -> torch.Tensor:
    """Run DINOv2 forward passes and return tokens from every hidden-state layer.

    Used only by the position-embedding analysis (`position_utils.py`), which
    needs per-layer accuracy/rank curves. Deliberately NOT wrapped in
    `utils.cached()` / not written to Drive: even at `config["n_position"]`
    (default 1,000) images this is ~10GB dense fp16 across the model's 13
    hidden-state layers — cheaper to recompute (a few minutes on a Colab GPU)
    than to store.

    Returns
    -------
    Tensor of shape (len(images), n_layers, n_tokens, d_model), dtype float16, on CPU.
    """
    device = config["device"]
    batch_size = config["batch_size"]
    all_tokens = []

    for i in tqdm(range(0, len(images), batch_size), desc="Extracting per-layer activations"):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        layers = torch.stack(out.hidden_states, dim=1)  # (B, n_layers, n_tokens, d_model)
        all_tokens.append(layers.half().cpu())

    return torch.cat(all_tokens, dim=0)
