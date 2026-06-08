from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_from_logits(logits.detach(), y), bs)

    return loss_meter.avg, acc_meter.avg


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


def build_optimizer(args, model):
    if args.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    if args.optimizer.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError("optimizer must be sgd or adamw")


def main():
    parser = argparse.ArgumentParser(description="Clean sanity baseline, submission-compatible preprocessing.")
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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=20)
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

    best_path = os.path.join(args.output_dir, f"best_{args.model_name}_clean_state_dict.pt")
    last_path = os.path.join(args.output_dir, f"last_{args.model_name}_clean_state_dict.pt")
    history_path = os.path.join(args.output_dir, "history.csv")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate_clean(model, val_loader, criterion, device)
        scheduler.step()

        improved, should_stop = stopper.step(epoch, score=val_acc, val_clean_acc=val_acc, train_acc=train_acc)
        if improved:
            save_state_dict(model, best_path)
        save_state_dict(model, last_path)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_clean_loss": val_loss,
            "val_clean_acc": val_acc,
            "score": val_acc,
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
        },
        os.path.join(args.output_dir, "summary.json"),
    )


if __name__ == "__main__":
    main()
