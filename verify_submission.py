from __future__ import annotations

import argparse
import torch

from models import get_model, NUM_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint is server-compatible.")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    model = get_model(args.model_name)
    state = torch.load(args.checkpoint, map_location="cpu")

    if not isinstance(state, dict):
        raise TypeError("Checkpoint is not a state_dict-like dictionary. Save only model.state_dict().")

    # Reject common wrapped checkpoint format for submission safety.
    if "state_dict" in state or "model_state_dict" in state:
        raise ValueError(
            "Checkpoint appears wrapped. The server expects only the raw model.state_dict(). "
            "Use the best_*_state_dict.pt saved by these scripts."
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        out = model(x)

    assert tuple(out.shape) == (1, NUM_CLASSES), f"Expected output (1, {NUM_CLASSES}), got {tuple(out.shape)}"

    print("OK: checkpoint is a raw .pt state_dict")
    print(f"OK: state_dict loads into torchvision {args.model_name}(weights=None) with fc=9")
    print("OK: input shape (1, 3, 32, 32) -> output shape", tuple(out.shape))
    print("Use this same model_name in submission.py:", args.model_name)


if __name__ == "__main__":
    main()
