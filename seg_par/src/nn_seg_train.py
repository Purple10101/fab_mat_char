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
from sklearn.metrics import f1_score

from part_seg.src.nn_seg_data import PSegDataset, build_deeplab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targs = []

    ce = nn.CrossEntropyLoss()

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        out = model(imgs)["out"]  # [B,2,H,W]
        loss = ce(out, masks) + 0.5 * dice_loss_from_logits(out, masks)

        total_loss += loss.item() * imgs.size(0)

        pred = out.argmax(dim=1)  # [B,H,W]
        all_preds.append(pred.flatten().cpu())
        all_targs.append(masks.flatten().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targs = torch.cat(all_targs).numpy()
    f1 = f1_score(all_targs, all_preds, average="binary", zero_division=0)

    return total_loss / len(loader.dataset), f1


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        out = model(imgs)["out"]
        loss = ce(out, masks) + 0.5 * dice_loss_from_logits(out, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


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

    # Display Example:
    # subject = "f28b702df5"
    # plt.imshow(dataset.total_dataset[subject][1])
    # plt.axis("off")
    # plt.show()

    train_loader = DataLoader(
        dataset.train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,   # set 0 first (Windows), increase later if stable
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset.val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = build_deeplab(num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_f1 = -1.0
    for epoch in range(1, 31):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_f1 = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), "best_deeplab_particles.pt")
            print("  saved best model")


if __name__ == "__main__":
    main()