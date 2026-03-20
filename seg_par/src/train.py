"""
train.py — Training loop for fibre instance segmentation
=========================================================

Usage:
    python train.py --data_dir ./fibre_dataset --epochs 50

Key features:
  - Mask R-CNN with ResNet-50-FPN-V2 backbone (COCO pretrained)
  - Cosine annealing LR schedule with linear warmup
  - Automatic mixed precision (AMP) for faster training
  - Best-checkpoint saving based on validation mask AP
  - TensorBoard logging
  - Graceful resume from checkpoint

Requirements:
    pip install torch torchvision albumentations pycocotools tensorboard
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FibreDataset, collate_fn, get_train_transforms, get_val_transforms
from model import build_model, count_parameters
from evaluate import evaluate_coco


# ─── Helpers ──────────────────────────────────────────────────────────────────

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer  and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  ✓ resumed from epoch {ckpt['epoch']}  (best AP={ckpt.get('best_ap', 0):.4f})")
    return ckpt.get("epoch", 0), ckpt.get("best_ap", 0.0)


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, scaler, epoch, writer):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += losses.item()
        global_step = epoch * len(loader) + step

        if step % 10 == 0:
            writer.add_scalar("train/loss_total",      losses.item(),                       global_step)
            writer.add_scalar("train/loss_classifier", loss_dict["loss_classifier"].item(), global_step)
            writer.add_scalar("train/loss_box_reg",    loss_dict["loss_box_reg"].item(),    global_step)
            writer.add_scalar("train/loss_mask",       loss_dict["loss_mask"].item(),       global_step)
            writer.add_scalar("train/loss_objectness", loss_dict["loss_objectness"].item(), global_step)
            writer.add_scalar("train/loss_rpn_box",    loss_dict["loss_rpn_box_reg"].item(),global_step)

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}  step {step:4d}/{len(loader)}  "
                  f"loss={losses.item():.4f}  ({elapsed:.0f}s elapsed)")

    return total_loss / len(loader)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="./fibre_dataset")
    p.add_argument("--out_dir",        default="./runs/fibre_maskrcnn")
    p.add_argument("--image_size",     type=int,   default=512)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--warmup_epochs",  type=int,   default=3)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--backbone",       default="maskrcnn_resnet50_fpn_v2",
                   choices=["maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2"])
    p.add_argument("--trainable_layers", type=int, default=3)
    p.add_argument("--resume",         default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no_amp",         action="store_true", help="Disable mixed precision")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    # ── Datasets & loaders ──
    train_ds = FibreDataset(args.data_dir, split="train",
                            transforms=get_train_transforms(args.image_size))
    val_ds   = FibreDataset(args.data_dir, split="val",
                            transforms=get_val_transforms(args.image_size))
    print(f"Train: {len(train_ds)} samples   Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    # ── Model ──
    model = build_model(args.backbone, pretrained=True,
                        trainable_backbone_layers=args.trainable_layers)
    model.to(device)
    total, trainable = count_parameters(model)
    print(f"Parameters: {total:,} total  {trainable:,} trainable")

    # ── Optimiser & scheduler ──
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if (not args.no_amp and device.type == "cuda") else None

    # ── Resume ──
    start_epoch, best_ap = 0, 0.0
    if args.resume:
        start_epoch, best_ap = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1

    # ── Save config ──
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{args.epochs-1}   LR={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, writer)
        scheduler.step()

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ── Validation every 2 epochs (COCO AP metrics) ──
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print("  Running validation...")
            metrics = evaluate_coco(model, val_loader, device)
            ap_mask = metrics.get("AP_mask", 0.0)

            writer.add_scalar("val/AP_mask",    metrics.get("AP_mask",    0), epoch)
            writer.add_scalar("val/AP50_mask",  metrics.get("AP50_mask",  0), epoch)
            writer.add_scalar("val/AP75_mask",  metrics.get("AP75_mask",  0), epoch)
            writer.add_scalar("val/AP_box",     metrics.get("AP_box",     0), epoch)

            print(f"  AP_mask={ap_mask:.4f}  AP50={metrics.get('AP50_mask',0):.4f}  "
                  f"AP75={metrics.get('AP75_mask',0):.4f}  AP_box={metrics.get('AP_box',0):.4f}")

            # ── Save best checkpoint ──
            if ap_mask > best_ap:
                best_ap = ap_mask
                save_checkpoint({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_ap":   best_ap,
                    "metrics":   metrics,
                }, os.path.join(args.out_dir, "best.pth"))

        # ── Save latest checkpoint every 5 epochs ──
        if epoch % 5 == 0:
            save_checkpoint({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_ap":   best_ap,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pth"))

    print(f"\n✓ Training complete.  Best AP_mask = {best_ap:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
