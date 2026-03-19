"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260220

nn_seg_train.py
training script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from seg_par.src.nn_seg_data import PSegDataset, build_deeplab_instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    # Standard BCE is hates how many background pixels we have for a thin particle,
    # the Gaussian peak probably covers <1% of the image, so the model learns to
    # ignore peaks and just predict zero everywhere.
    # Focal loss down-weights easy well-classified pixels (confident background)
    # and concentrates gradient on the hard ones (the heatmap peaks).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - p_t) ** gamma
    loss = alpha * focal_weight * bce
    return loss.mean()

def instance_losses(out, batch, w_center=1.0, w_off=0.1):
    """
    out: [B,4,H,W] where
      out[:,0:1] = fg logits (binary)
      out[:,1:2] = center logits (binary heatmap)
      out[:,2:4] = offsets (dx, dy)
    batch: dict with fg [B,H,W], center [B,1,H,W], offsets [B,2,H,W]
    """
    fg_t = batch["fg"]                      # [B,H,W] long {0,1}
    center_t = batch["center"]              # [B,1,H,W] float 0..1
    offsets_t = batch["offsets"]            # [B,2,H,W] float

    fg_logits = out[:, 0:1]                 # [B,1,H,W]
    center_logits = out[:, 1:2]             # [B,1,H,W]
    offsets_pred = out[:, 2:4]              # [B,2,H,W]

    # foreground loss (binary)
    loss_fg = F.binary_cross_entropy_with_logits(
        fg_logits, fg_t.unsqueeze(1).float()
    )

    loss_center = focal_loss(center_logits, center_t)

    # offsets loss only where fg == 1
    fg_mask = (fg_t == 1).unsqueeze(1).float()  # [B,1,H,W]
    # weight each fg pixel by its distance from its instance centroid,
    # so far-flung ends of long particles aren't drowned out by easy central pixels
    offset_magnitude = (offsets_t ** 2).sum(dim=1, keepdim=True).sqrt()  # [B,1,H,W]
    dist_weight = 1.0 + offset_magnitude / (offset_magnitude.max().clamp_min(1.0))
    dist_weight = dist_weight * fg_mask

    denom = dist_weight.sum().clamp_min(1.0)
    loss_off = (
            F.smooth_l1_loss(offsets_pred * dist_weight, offsets_t * dist_weight, reduction="sum")
            / denom
    )

    total = loss_fg + w_center * loss_center + w_off * loss_off
    return total, {
        "loss_fg": loss_fg.item(),
        "loss_center": loss_center.item(),
        "loss_off": loss_off.item(),
    }

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        out = model(imgs)["out"]  # [B,4,H,W]
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

    model = build_deeplab_instance(num_out=4).to(device)

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