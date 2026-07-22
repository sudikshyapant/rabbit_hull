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
    disk (~18GB for 45,000 images at fp16). Reduce n_sae_train in config.py
    if this doesn't fit your Colab runtime's RAM.
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
