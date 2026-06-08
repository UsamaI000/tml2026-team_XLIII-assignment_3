"""Model factory for Assignment 3 robustness.

IMPORTANT: This file intentionally keeps the default torchvision ResNet stem.
The server appears to instantiate torchvision.models.resnet18/34/50(weights=None)
and only replaces the final fc layer. Therefore we must not replace conv1 or
maxpool, otherwise the submitted state_dict will not load on the server.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

NUM_CLASSES = 9
ALLOWED_MODELS = {"resnet18", "resnet34", "resnet50"}


def get_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Return a submission-compatible torchvision ResNet.

    The architecture exactly follows the task template:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 9)

    Args:
        model_name: One of "resnet18", "resnet34", "resnet50".
        num_classes: Number of output classes. Assignment requires 9.

    Returns:
        torch.nn.Module producing logits of shape (N, num_classes).
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        model = resnet18(weights=None)
    elif model_name == "resnet34":
        model = resnet34(weights=None)
    elif model_name == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unknown model_name={model_name!r}. Use one of {sorted(ALLOWED_MODELS)}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_state_dict_for_submission(model_name: str, checkpoint_path: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load a saved state_dict into the exact submission-compatible model."""
    model = get_model(model_name)
    state = torch.load(checkpoint_path, map_location=device)

    # Some accidental checkpoints may wrap the state dict. Be permissive locally,
    # but training scripts in this repo save the raw state_dict only.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def sanity_check_model(model: nn.Module, device: str | torch.device = "cpu") -> None:
    """Check assignment-required input/output shape."""
    model.eval().to(device)
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32, device=device)
        y = model(x)
    if tuple(y.shape) != (1, NUM_CLASSES):
        raise RuntimeError(f"Invalid output shape: expected (1, {NUM_CLASSES}), got {tuple(y.shape)}")
