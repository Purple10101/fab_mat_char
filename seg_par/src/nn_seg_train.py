"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260220

nn_seg_train.py
training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score

from seg_par.src.nn_seg_data import PSegDataset, build_deeplab_instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def instance_losses(out, batch, w_center=1.0, w_off=1.0):
    """
    out: [B,5,H,W] where
      out[:,0:2] = fg logits
      out[:,2:3] = center logits
      out[:,3:5] = offsets
    batch: dict with fg [B,H,W], center [B,1,H,W], offsets [B,2,H,W]
    """
    fg_t = batch["fg"]                      # [B,H,W] long
    center_t = batch["center"]              # [B,1,H,W] float
    offsets_t = batch["offsets"]            # [B,2,H,W] float

    fg_logits = out[:, 0:2]
    center_logits = out[:, 2:3]
    offsets_pred = out[:, 3:5]

    # foreground semantic loss
    loss_fg = F.cross_entropy(fg_logits, fg_t)

    # center heatmap loss
    loss_center = F.binary_cross_entropy_with_logits(center_logits, center_t)

    # offsets loss only where fg == 1
    fg_mask = (fg_t == 1).unsqueeze(1).float()  # [B,1,H,W]
    # avoid dividing by 0
    denom = fg_mask.sum().clamp_min(1.0)
    loss_off = (F.smooth_l1_loss(offsets_pred * fg_mask, offsets_t * fg_mask, reduction="sum") / denom)

    return loss_fg + w_center * loss_center + w_off * loss_off, {
        "loss_fg": loss_fg.item(),
        "loss_center": loss_center.item(),
        "loss_off": loss_off.item()
    }

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        out = model(imgs)["out"]  # [B,5,H,W]
        loss, parts = instance_losses(out, targets, w_center=1.0, w_off=0.1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item() * imgs.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        out = model(imgs)["out"]
        loss, _ = instance_losses(out, targets, w_center=1.0, w_off=0.1)

        total += loss.item() * imgs.size(0)

    return total / len(loader.dataset)


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    logits: [B,2,H,W]
    targets: [B,H,W] {0,1}
    """
    probs = torch.softmax(logits, dim=1)[:, 1]      # foreground prob [B,H,W]
    targets_f = targets.float()

    intersection = (probs * targets_f).sum(dim=(1,2))
    union = probs.sum(dim=(1,2)) + targets_f.sum(dim=(1,2))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def main():
    dataset = PSegDataset()

    train_loader = DataLoader(
        dataset.train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset.val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = build_deeplab_instance(num_out=5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, 31):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        va_loss = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        # Save best by lowest validation loss
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "best_deeplab_instances.pt")
            print("  saved best model")


if __name__ == "__main__":
    main()