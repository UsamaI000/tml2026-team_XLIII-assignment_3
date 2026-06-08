"""Adversarial attacks in raw [0, 1] image space.

No normalization is used anywhere. PGD clamps adversarial examples to [0, 1]
and the L-infinity epsilon ball around the original image.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def clamp_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, eps: float) -> torch.Tensor:
    upper = torch.clamp(x_orig + eps, 0.0, 1.0)
    lower = torch.clamp(x_orig - eps, 0.0, 1.0)
    return torch.max(torch.min(x_adv, upper), lower)


def pgd_linf_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted L-infinity PGD attack for inputs in [0, 1]."""
    was_training = model.training
    model.eval()

    x_orig = x.detach()
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = clamp_linf(x_adv, x_orig, eps)
    else:
        x_adv = x_orig.clone()

    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = clamp_linf(x_adv, x_orig, eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if was_training:
        model.train()
    return x_adv.detach()


def fgsm_linf_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
) -> torch.Tensor:
    """Untargeted FGSM attack for inputs in [0, 1]."""
    was_training = model.training
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    with torch.no_grad():
        x_adv = x_adv + eps * grad.sign()
        x_adv = clamp_linf(x_adv, x.detach(), eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if was_training:
        model.train()
    return x_adv.detach()
