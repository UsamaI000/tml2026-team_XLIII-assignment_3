from __future__ import annotations

import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    args = parser.parse_args()

    data = np.load(args.npz_path)
    print("Keys:", data.files)
    images = data["images"]
    labels = data["labels"]
    print("Images:", images.shape, images.dtype, images.min(), images.max())
    print("Labels:", labels.shape, labels.dtype, labels.min(), labels.max())
    x = torch.from_numpy(images).float() / 255.0
    y = torch.from_numpy(labels).long()
    print("After template preprocessing:", x.shape, x.dtype, float(x.min()), float(x.max()))
    print("Label range:", y.min().item(), "to", y.max().item())


if __name__ == "__main__":
    main()
