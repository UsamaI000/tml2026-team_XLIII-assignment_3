from __future__ import annotations

import argparse
import os
import time

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


def build_optimizer(args, model):
    if args.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("optimizer must be sgd or adamw")


def train_one_epoch_pgd(model, loader, optimizer, criterion, device, args, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    clean_acc_meter = AverageMeter()
    adv_acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Generate PGD examples in raw [0, 1] space.
        x_adv = pgd_linf_attack(
            model,
            x,
            y,
            eps=args.train_eps,
            alpha=args.train_alpha,
            steps=args.train_steps,
            random_start=True,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits_clean = model(x)
                logits_adv = model(x_adv)
                loss_clean = criterion(logits_clean, y)
                loss_adv = criterion(logits_adv, y)
                loss = (1.0 - args.adv_weight) * loss_clean + args.adv_weight * loss_adv
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss_clean = criterion(logits_clean, y)
            loss_adv = criterion(logits_adv, y)
            loss = (1.0 - args.adv_weight) * loss_clean + args.adv_weight * loss_adv
            loss.backward()
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
            model,
            x,
            y,
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


def main():
    parser = argparse.ArgumentParser(description="PGD adversarial training with task-template preprocessing.")
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--train_eps", type=float, default=4 / 255)
    parser.add_argument("--train_alpha", type=float, default=1 / 255)
    parser.add_argument("--train_steps", type=int, default=5)
    parser.add_argument("--adv_weight", type=float, default=0.4)

    parser.add_argument("--eval_eps", type=float, default=8 / 255)
    parser.add_argument("--eval_alpha", type=float, default=2 / 255)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--eval_max_batches", type=int, default=None)

    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument("--clean_acc_floor", type=float, default=0.60)
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
            model, train_loader, optimizer, criterion, device, args, scaler
        )
        val_clean_loss, val_clean_acc = evaluate_clean(model, val_loader, criterion, device)
        val_pgd_loss, val_pgd_acc = evaluate_pgd(model, val_loader, criterion, device, args)
        scheduler.step()

        score = 0.5 * val_clean_acc + 0.5 * val_pgd_acc
        improved, should_stop = stopper.step(epoch, score=score, val_clean_acc=val_clean_acc, train_acc=train_clean_acc)
        if improved:
            save_state_dict(model, best_path)
        save_state_dict(model, last_path)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
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
