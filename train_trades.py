from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

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


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    The EMA checkpoint is still a normal torchvision ResNet state_dict.
    It can be submitted exactly like the normal model checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = OrderedDict()
        self.backup = OrderedDict()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            self.shadow[name].mul_(self.decay).add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = OrderedDict()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param.copy_(self.backup[name])

        self.backup = OrderedDict()

    @torch.no_grad()
    def state_dict(self, model: nn.Module):
        current_state = model.state_dict()
        ema_state = OrderedDict()

        param_names = set(self.shadow.keys())

        for key, value in current_state.items():
            if key in param_names:
                ema_state[key] = self.shadow[key].detach().cpu().clone()
            else:
                ema_state[key] = value.detach().cpu().clone()

        return ema_state


def save_ema_state_dict(ema: ModelEMA, model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ema.state_dict(model), path)


def build_optimizer(args, model):
    if args.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    if args.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    raise ValueError("optimizer must be sgd or adamw")


def trades_linf_attack(
    model: nn.Module,
    x: torch.Tensor,
    steps: int,
    eps: float,
    alpha: float,
    random_start: bool = True,
):
    """
    Generate adversarial examples for TRADES.

    Unlike normal PGD adversarial training, this attack maximizes the KL divergence
    between model predictions on clean images and adversarial images.

    Images are assumed to be in raw [0, 1] space.
    """

    model.eval()

    x_clean = x.detach()

    with torch.no_grad():
        clean_logits = model(x_clean)
        clean_probs = F.softmax(clean_logits, dim=1)

    if random_start:
        x_adv = x_clean + torch.empty_like(x_clean).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        adv_logits = model(x_adv)

        loss_kl = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            clean_probs,
            reduction="batchmean",
        )

        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())

        delta = torch.clamp(x_adv - x_clean, min=-eps, max=eps)
        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    return x_adv


def pgd_linf_attack_ce(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    eps: float,
    alpha: float,
    random_start: bool = True,
):
    """
    Standard CE-based PGD attack used only for validation.

    Images are assumed to be in raw [0, 1] space.
    """

    model.eval()

    x_clean = x.detach()

    if random_start:
        x_adv = x_clean + torch.empty_like(x_clean).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())

        delta = torch.clamp(x_adv - x_clean, min=-eps, max=eps)
        x_adv = torch.clamp(x_clean + delta, 0.0, 1.0).detach()

    return x_adv


def train_one_epoch_trades(
    model,
    loader,
    optimizer,
    criterion,
    device,
    args,
    scaler=None,
    ema=None,
):
    model.train()

    loss_meter = AverageMeter()
    clean_loss_meter = AverageMeter()
    robust_loss_meter = AverageMeter()
    clean_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = trades_linf_attack(
            model=model,
            x=x,
            steps=args.train_steps,
            eps=args.train_eps,
            alpha=args.train_alpha,
            random_start=True,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                clean_logits = model(x)
                adv_logits = model(x_adv)

                clean_loss = criterion(clean_logits, y)

                robust_loss = F.kl_div(
                    F.log_softmax(adv_logits, dim=1),
                    F.softmax(clean_logits.detach(), dim=1),
                    reduction="batchmean",
                )

                loss = clean_loss + args.beta * robust_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            clean_logits = model(x)
            adv_logits = model(x_adv)

            clean_loss = criterion(clean_logits, y)

            robust_loss = F.kl_div(
                F.log_softmax(adv_logits, dim=1),
                F.softmax(clean_logits.detach(), dim=1),
                reduction="batchmean",
            )

            loss = clean_loss + args.beta * robust_loss
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = y.size(0)

        loss_meter.update(loss.item(), bs)
        clean_loss_meter.update(clean_loss.item(), bs)
        robust_loss_meter.update(robust_loss.item(), bs)
        clean_acc_meter.update(accuracy_from_logits(clean_logits.detach(), y), bs)
        adv_acc_meter.update(accuracy_from_logits(adv_logits.detach(), y), bs)

    return (
        loss_meter.avg,
        clean_loss_meter.avg,
        robust_loss_meter.avg,
        clean_acc_meter.avg,
        adv_acc_meter.avg,
    )


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

        x_adv = pgd_linf_attack_ce(
            model=model,
            x=x,
            y=y,
            steps=args.eval_steps,
            eps=args.eval_eps,
            alpha=args.eval_alpha,
            random_start=True,
        )

        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)

        bs = y.size(0)

        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_from_logits(logits, y), bs)

    return loss_meter.avg, acc_meter.avg


def evaluate_clean_and_pgd(model, val_loader, criterion, device, args):
    val_clean_loss, val_clean_acc = evaluate_clean(model, val_loader, criterion, device)
    val_pgd_loss, val_pgd_acc = evaluate_pgd(model, val_loader, criterion, device, args)
    score = 0.5 * val_clean_acc + 0.5 * val_pgd_acc

    return val_clean_loss, val_clean_acc, val_pgd_loss, val_pgd_acc, score


def main():
    parser = argparse.ArgumentParser(
        description="TRADES adversarial training with task-template preprocessing."
    )

    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
    )

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--train_eps", type=float, default=6 / 255)
    parser.add_argument("--train_alpha", type=float, default=1 / 255)
    parser.add_argument("--train_steps", type=int, default=7)

    parser.add_argument("--beta", type=float, default=4.0)

    parser.add_argument("--eval_eps", type=float, default=8 / 255)
    parser.add_argument("--eval_alpha", type=float, default=2 / 255)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_max_batches", type=int, default=None)

    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min_epochs", type=int, default=40)
    parser.add_argument("--clean_acc_floor", type=float, default=0.55)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler() if args.amp and device.type == "cuda" else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    stopper = RobustEarlyStopping(
        EarlyStoppingConfig(
            patience=args.patience,
            min_epochs=args.min_epochs,
            clean_acc_floor=args.clean_acc_floor,
        )
    )

    ema_stopper = None
    if args.use_ema:
        ema_stopper = RobustEarlyStopping(
            EarlyStoppingConfig(
                patience=args.patience,
                min_epochs=args.min_epochs,
                clean_acc_floor=args.clean_acc_floor,
            )
        )

    best_path = os.path.join(args.output_dir, f"best_{args.model_name}_trades_state_dict.pt")
    last_path = os.path.join(args.output_dir, f"last_{args.model_name}_trades_state_dict.pt")

    best_ema_path = os.path.join(args.output_dir, f"best_{args.model_name}_trades_ema_state_dict.pt")
    last_ema_path = os.path.join(args.output_dir, f"last_{args.model_name}_trades_ema_state_dict.pt")

    history_path = os.path.join(args.output_dir, "history.csv")

    print("=" * 100)
    print("TRADES ADVERSARIAL TRAINING")
    print("=" * 100)
    print(f"model_name       : {args.model_name}")
    print(f"device           : {device}")
    print(f"epochs           : {args.epochs}")
    print(f"batch_size       : {args.batch_size}")
    print(f"lr               : {args.lr}")
    print(f"weight_decay     : {args.weight_decay}")
    print(f"optimizer        : {args.optimizer}")
    print(f"train_eps        : {args.train_eps}")
    print(f"train_alpha      : {args.train_alpha}")
    print(f"train_steps      : {args.train_steps}")
    print(f"beta             : {args.beta}")
    print(f"eval_eps         : {args.eval_eps}")
    print(f"eval_alpha       : {args.eval_alpha}")
    print(f"eval_steps       : {args.eval_steps}")
    print(f"use_ema          : {args.use_ema}")
    print(f"ema_decay        : {args.ema_decay if args.use_ema else None}")
    print("=" * 100)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        (
            train_loss,
            train_clean_loss,
            train_robust_loss,
            train_clean_acc,
            train_adv_acc,
        ) = train_one_epoch_trades(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            args=args,
            scaler=scaler,
            ema=ema,
        )

        val_clean_loss, val_clean_acc, val_pgd_loss, val_pgd_acc, score = evaluate_clean_and_pgd(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        ema_val_clean_loss = None
        ema_val_clean_acc = None
        ema_val_pgd_loss = None
        ema_val_pgd_acc = None
        ema_score = None
        ema_improved = False

        if args.use_ema:
            ema.apply_shadow(model)

            (
                ema_val_clean_loss,
                ema_val_clean_acc,
                ema_val_pgd_loss,
                ema_val_pgd_acc,
                ema_score,
            ) = evaluate_clean_and_pgd(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
            )

            ema.restore(model)

        scheduler.step()

        improved, should_stop_normal = stopper.step(
            epoch,
            score=score,
            val_clean_acc=val_clean_acc,
            train_acc=train_clean_acc,
        )

        if improved:
            save_state_dict(model, best_path)

        save_state_dict(model, last_path)

        should_stop_ema = False

        if args.use_ema:
            ema_improved, should_stop_ema = ema_stopper.step(
                epoch,
                score=ema_score,
                val_clean_acc=ema_val_clean_acc,
                train_acc=train_clean_acc,
            )

            if ema_improved:
                save_ema_state_dict(ema, model, best_ema_path)

            save_ema_state_dict(ema, model, last_ema_path)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_clean_loss": train_clean_loss,
            "train_robust_loss": train_robust_loss,
            "train_clean_acc": train_clean_acc,
            "train_adv_acc_trades": train_adv_acc,
            "val_clean_loss": val_clean_loss,
            "val_clean_acc": val_clean_acc,
            "val_pgd_loss": val_pgd_loss,
            "val_pgd_acc": val_pgd_acc,
            "score": score,
            "best_epoch": stopper.best_epoch,
            "best_ema": stopper.best_value,
            "ema_val_clean_loss": ema_val_clean_loss,
            "ema_val_clean_acc": ema_val_clean_acc,
            "ema_val_pgd_loss": ema_val_pgd_loss,
            "ema_val_pgd_acc": ema_val_pgd_acc,
            "ema_score": ema_score,
            "ema_best_epoch": ema_stopper.best_epoch if ema_stopper is not None else None,
            "ema_best_value": ema_stopper.best_value if ema_stopper is not None else None,
            "seconds": time.time() - start,
        }

        append_history(history_path, row)

        if args.use_ema:
            print(
                {
                    "epoch": epoch,
                    "lr": row["lr"],
                    "train_loss": train_loss,
                    "train_clean_loss": train_clean_loss,
                    "train_robust_loss": train_robust_loss,
                    "train_clean_acc": train_clean_acc,
                    "train_adv_acc_trades": train_adv_acc,
                    "val_clean_acc": val_clean_acc,
                    "val_pgd_acc": val_pgd_acc,
                    "score": score,
                    "ema_val_clean_acc": ema_val_clean_acc,
                    "ema_val_pgd_acc": ema_val_pgd_acc,
                    "ema_score": ema_score,
                    "best_epoch": stopper.best_epoch,
                    "ema_best_epoch": ema_stopper.best_epoch,
                    "seconds": row["seconds"],
                },
                flush=True,
            )
        else:
            print(
                {
                    "epoch": epoch,
                    "lr": row["lr"],
                    "train_loss": train_loss,
                    "train_clean_loss": train_clean_loss,
                    "train_robust_loss": train_robust_loss,
                    "train_clean_acc": train_clean_acc,
                    "train_adv_acc_trades": train_adv_acc,
                    "val_clean_acc": val_clean_acc,
                    "val_pgd_acc": val_pgd_acc,
                    "score": score,
                    "best_epoch": stopper.best_epoch,
                    "seconds": row["seconds"],
                },
                flush=True,
            )

        if args.use_ema:
            if should_stop_normal and should_stop_ema:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best normal epoch: {stopper.best_epoch}. "
                    f"Best EMA epoch: {ema_stopper.best_epoch}."
                )
                break
        else:
            if should_stop_normal:
                print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}")
                break

    summary = {
        "best_epoch": stopper.best_epoch,
        "best_value_ema": stopper.best_value,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "selection_metric": "0.5 * val_clean_acc + 0.5 * val_pgd_acc, with clean_acc_floor gate",
        "method": "TRADES",
        "beta": args.beta,
        "use_ema": args.use_ema,
    }

    if args.use_ema:
        summary.update(
            {
                "ema_best_epoch": ema_stopper.best_epoch,
                "ema_best_value": ema_stopper.best_value,
                "ema_best_checkpoint": best_ema_path,
                "ema_last_checkpoint": last_ema_path,
                "ema_decay": args.ema_decay,
            }
        )

    save_json(summary, os.path.join(args.output_dir, "summary.json"))

    print("=" * 100)
    print("Training finished.")
    print(f"Best normal checkpoint: {best_path}")
    print(f"Last normal checkpoint: {last_path}")

    if args.use_ema:
        print(f"Best EMA checkpoint: {best_ema_path}")
        print(f"Last EMA checkpoint: {last_ema_path}")
        print("For EMA submission, use:")
        print(f"MODEL_NAME = '{args.model_name}'")
        print(f"MODEL_PATH = '{best_ema_path}'")

    print("=" * 100)


if __name__ == "__main__":
    main()