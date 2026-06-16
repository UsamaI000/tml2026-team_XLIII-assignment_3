"""PGD adversarial training — improved version.

Key changes vs original:
  1. MultiStepLR (milestones 100/150, gamma 0.1) instead of CosineAnnealingLR
     — matches the original Madry AT recipe, avoids LR being too low early.
  2. Epsilon curriculum: linearly ramp train_eps from eps_start to train_eps
     over --eps_warmup_epochs epochs. Stabilises early training.
  3. AWP (Adversarial Weight Perturbation): perturbs model weights during
     inner maximisation for stronger regularisation (+2-3% robust acc).
     Controlled by --awp_gamma (0 = disabled).
  4. Label smoothing (--label_smoothing, default 0.1).
  5. CutMix augmentation during training (--cutmix_prob, default 0.0).
  6. Defaults updated: lr 0.1, train_eps 8/255, train_alpha 2/255,
     train_steps 10, adv_weight 0.6, epochs 200.

Recommended launch command:
  python train_pgd.py \
    --npz_path ./data/train.npz \
    --output_dir runs/pgd_resnet50_eps8_multistep_awp \
    --model_name resnet50 \
    --epochs 200 \
    --lr 0.1 \
    --train_eps 0.03137 \
    --train_alpha 0.00784 \
    --train_steps 10 \
    --adv_weight 0.6 \
    --awp_gamma 0.01 \
    --cutmix_prob 0.5 \
    --patience 30 \
    --min_epochs 60 \
    --clean_acc_floor 0.55 \
    --seed 20
"""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from attacks import pgd_linf_attack
from data import make_loaders
from models import get_model, sanity_check_model
from utils import (
    AverageMeter,
    EarlyStoppingConfig,
    RobustEarlyStopping,
    accuracy_from_logits,
    append_history,
    get_device,
    save_json,
    save_state_dict,
    set_seed,
)


# ---------------------------------------------------------------------------
# AWP helper
# ---------------------------------------------------------------------------

class AWP:
    """Adversarial Weight Perturbation (Wu et al., NeurIPS 2020).

    Perturbs model weights in the direction that maximises the adversarial
    loss, then runs the backward pass at the perturbed weights. This acts as
    an implicit regulariser that smooths the loss landscape and improves
    robust generalisation by ~2-3%.

    Usage:
        awp = AWP(model, optimizer, gamma=0.01)
        # inside training loop, after computing x_adv:
        awp.perturb(x_adv, y, criterion)
        loss = criterion(model(x_adv), y)
        loss.backward()
        optimizer.step()
        awp.restore()
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, gamma: float = 0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self._backup: dict[str, torch.Tensor] = {}
        self._backup_eps: dict[str, torch.Tensor] = {}

    def perturb(self, x_adv: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> None:
        """Compute weight gradient and add scaled perturbation.

        Note: must NOT be decorated with @torch.no_grad() — we need gradients
        w.r.t. model parameters. x_adv is used as a fixed input (detached).
        """
        if self.gamma == 0.0:
            return
        self.model.train()
        # x_adv arrives detached from pgd_linf_attack; use it as a fixed input.
        # We only need grads w.r.t. model weights, not the input.
        x_fixed = x_adv.detach()
        with torch.enable_grad():
            loss = criterion(self.model(x_fixed), y)
            grad = torch.autograd.grad(
                loss,
                [p for p in self.model.parameters() if p.requires_grad],
                create_graph=False,
            )

        self._backup.clear()
        self._backup_eps.clear()
        for (name, param), g in zip(
            [(n, p) for n, p in self.model.named_parameters() if p.requires_grad], grad
        ):
            self._backup[name] = param.data.clone()
            if g is not None:
                norm = g.norm()
                if norm > 1e-8:
                    perturbation = self.gamma * g / norm
                    self._backup_eps[name] = perturbation
                    param.data.add_(perturbation)

    @torch.no_grad()
    def restore(self) -> None:
        """Restore original weights after the backward pass."""
        if self.gamma == 0.0:
            return
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()
        self._backup_eps.clear()


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def cutmix_batch(x: torch.Tensor, y: torch.Tensor, prob: float, num_classes: int):
    """Apply CutMix to a batch. Returns mixed x and (y_a, y_b, lam) for loss."""
    if torch.rand(()).item() > prob:
        return x, y, None, None, 1.0

    lam = torch.distributions.Beta(1.0, 1.0).sample().item()
    B, C, H, W = x.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    perm = torch.randperm(B, device=x.device)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    y_b = y[perm]
    return x_mix, y, y_b, perm, lam


def cutmix_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Epsilon curriculum
# ---------------------------------------------------------------------------

def get_train_eps(args, epoch: int) -> float:
    """Linearly ramp epsilon from args.eps_start to args.train_eps."""
    if args.eps_warmup_epochs <= 0 or epoch >= args.eps_warmup_epochs:
        return args.train_eps
    eps_start = getattr(args, "eps_start", 2 / 255)
    frac = epoch / args.eps_warmup_epochs
    return eps_start + frac * (args.train_eps - eps_start)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(args, model):
    if args.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                         weight_decay=args.weight_decay, nesterov=True)
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("optimizer must be sgd or adamw")


def build_scheduler(args, optimizer):
    if args.scheduler == "multistep":
        milestones = [int(m) for m in args.lr_milestones.split(",")]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    raise ValueError("scheduler must be multistep or cosine")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch_pgd(model, loader, optimizer, criterion, device, args, epoch, scaler=None, awp=None):
    model.train()
    loss_meter = AverageMeter()
    clean_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()

    train_eps = get_train_eps(args, epoch)
    # alpha scales with eps so steps remain meaningful
    train_alpha = min(args.train_alpha, train_eps / 4)
    num_classes = 9

    use_awp = awp is not None and args.awp_gamma > 0
    use_cutmix = args.cutmix_prob > 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Optional CutMix on clean images before generating adversarial examples.
        # We apply CutMix to the adversarial inputs only (not the PGD source),
        # so the adversarial examples still target the true labels.
        x_adv = pgd_linf_attack(
            model, x, y,
            eps=train_eps,
            alpha=train_alpha,
            steps=args.train_steps,
            random_start=True,
        )

        # CutMix on adversarial batch
        if use_cutmix:
            x_adv_mix, ya, yb, _, lam = cutmix_batch(x_adv, y, args.cutmix_prob, num_classes)
        else:
            x_adv_mix, ya, yb, lam = x_adv, y, None, 1.0

        # AWP: perturb weights before the backward pass
        if use_awp:
            awp.perturb(x_adv, y, criterion)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits_clean = model(x)
                logits_adv = model(x_adv_mix)
                loss_clean = criterion(logits_clean, y)
                if yb is not None:
                    loss_adv = cutmix_loss(criterion, logits_adv, ya, yb, lam)
                else:
                    loss_adv = criterion(logits_adv, ya)
                loss = (1.0 - args.adv_weight) * loss_clean + args.adv_weight * loss_adv
            scaler.scale(loss).backward()
            if use_awp:
                awp.restore()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_clean = model(x)
            logits_adv = model(x_adv_mix)
            loss_clean = criterion(logits_clean, y)
            if yb is not None:
                loss_adv = cutmix_loss(criterion, logits_adv, ya, yb, lam)
            else:
                loss_adv = criterion(logits_adv, ya)
            loss = (1.0 - args.adv_weight) * loss_clean + args.adv_weight * loss_adv
            loss.backward()
            if use_awp:
                awp.restore()
            optimizer.step()

        bs = y.size(0)
        loss_meter.update(loss.item(), bs)
        clean_acc_meter.update(accuracy_from_logits(logits_clean.detach(), y), bs)
        adv_acc_meter.update(accuracy_from_logits(logits_adv.detach(), y), bs)

    return loss_meter.avg, clean_acc_meter.avg, adv_acc_meter.avg


@torch.no_grad()
def evaluate_clean(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_from_logits(logits, y), bs)
    return loss_meter.avg, acc_meter.avg


def evaluate_pgd(model, loader, criterion, device, args):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for batch_idx, (x, y) in enumerate(loader):
        if args.eval_max_batches is not None and batch_idx >= args.eval_max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = pgd_linf_attack(
            model, x, y,
            eps=args.eval_eps,
            alpha=args.eval_alpha,
            steps=args.eval_steps,
            random_start=True,
        )
        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)
        bs = y.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_from_logits(logits, y), bs)
    return loss_meter.avg, acc_meter.avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PGD adversarial training — improved.")

    # Data / output
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")

    # Model
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"])

    # Optimiser
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial LR. 0.1 is standard for SGD+MultiStepLR AT.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--amp", action="store_true")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="multistep",
                        choices=["multistep", "cosine"],
                        help="multistep: Madry recipe. cosine: original behaviour.")
    parser.add_argument("--lr_milestones", type=str, default="100,150",
                        help="Comma-separated epoch milestones for MultiStepLR.")
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    # PGD training attack
    parser.add_argument("--train_eps", type=float, default=8 / 255,
                        help="Training epsilon. Default 8/255 = 0.03137.")
    parser.add_argument("--train_alpha", type=float, default=2 / 255,
                        help="Training step size. Default 2/255.")
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--adv_weight", type=float, default=0.6,
                        help="Weight of adversarial loss. 1-adv_weight applied to clean loss.")

    # Epsilon curriculum
    parser.add_argument("--eps_warmup_epochs", type=int, default=15,
                        help="Linearly ramp train_eps from eps_start to train_eps over this many epochs. 0 = disabled.")
    parser.add_argument("--eps_start", type=float, default=2 / 255,
                        help="Starting epsilon for curriculum.")

    # AWP
    parser.add_argument("--awp_gamma", type=float, default=0.01,
                        help="AWP perturbation magnitude. 0 = disabled.")

    # CutMix
    parser.add_argument("--cutmix_prob", type=float, default=0.5,
                        help="Probability of applying CutMix to adversarial batch. 0 = disabled.")

    # Label smoothing
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # PGD evaluation attack
    parser.add_argument("--eval_eps", type=float, default=8 / 255)
    parser.add_argument("--eval_alpha", type=float, default=2 / 255)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_max_batches", type=int, default=None)

    # Early stopping
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_epochs", type=int, default=60)
    parser.add_argument("--clean_acc_floor", type=float, default=0.55)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    train_loader, val_loader, _ = make_loaders(
        args.npz_path,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augment_train=not args.no_augment,
    )

    model = get_model(args.model_name).to(device)
    sanity_check_model(model, device)

    # Label smoothing helps calibration and robust accuracy
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # Use plain CE for evaluation so metrics are comparable across runs
    criterion_eval = nn.CrossEntropyLoss()

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    awp = AWP(model, optimizer, gamma=args.awp_gamma) if args.awp_gamma > 0 else None

    stopper = RobustEarlyStopping(
        EarlyStoppingConfig(
            patience=args.patience,
            min_epochs=args.min_epochs,
            clean_acc_floor=args.clean_acc_floor,
        )
    )

    best_path = os.path.join(args.output_dir, f"best_{args.model_name}_pgd_state_dict.pt")
    last_path = os.path.join(args.output_dir, f"last_{args.model_name}_pgd_state_dict.pt")
    history_path = os.path.join(args.output_dir, "history.csv")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_clean_acc, train_adv_acc = train_one_epoch_pgd(
            model, train_loader, optimizer, criterion, device, args, epoch, scaler, awp
        )
        val_clean_loss, val_clean_acc = evaluate_clean(model, val_loader, criterion_eval, device)
        val_pgd_loss, val_pgd_acc = evaluate_pgd(model, val_loader, criterion_eval, device, args)
        scheduler.step()

        score = 0.5 * val_clean_acc + 0.5 * val_pgd_acc
        improved, should_stop = stopper.step(epoch, score=score, val_clean_acc=val_clean_acc,
                                             train_acc=train_clean_acc)
        if improved:
            save_state_dict(model, best_path)
        save_state_dict(model, last_path)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_eps": get_train_eps(args, epoch),
            "train_loss": train_loss,
            "train_clean_acc": train_clean_acc,
            "train_adv_acc": train_adv_acc,
            "val_clean_loss": val_clean_loss,
            "val_clean_acc": val_clean_acc,
            "val_pgd_loss": val_pgd_loss,
            "val_pgd_acc": val_pgd_acc,
            "score": score,
            "best_epoch": stopper.best_epoch,
            "best_ema": stopper.best_value,
            "seconds": time.time() - start,
        }
        append_history(history_path, row)
        print(row, flush=True)

        if should_stop:
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}")
            break

    save_json(
        {
            "best_epoch": stopper.best_epoch,
            "best_value_ema": stopper.best_value,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
            "selection_metric": "0.5 * val_clean_acc + 0.5 * val_pgd_acc, with clean_acc_floor gate",
        },
        os.path.join(args.output_dir, "summary.json"),
    )


if __name__ == "__main__":
    main()