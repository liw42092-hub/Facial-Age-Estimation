from __future__ import annotations

from pathlib import Path

import torch

from models.age_model import AgeEstimator


def load_age_model(checkpoint_path: str | Path, device: torch.device, backbone: str | None = None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint (missing model_state_dict): {checkpoint_path}")

    method = checkpoint.get("method", "dldl")
    saved_backbone = checkpoint.get("backbone", backbone or "resnet50")
    if backbone is not None and backbone != saved_backbone:
        raise ValueError(
            f"Checkpoint uses backbone '{saved_backbone}', but '{backbone}' was requested."
        )
    num_outputs = checkpoint.get("num_outputs", 1 if method == "regression" else 101)
    model = AgeEstimator(saved_backbone, num_outputs=num_outputs, pretrained=False, method=method)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
