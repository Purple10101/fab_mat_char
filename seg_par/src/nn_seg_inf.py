"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260220

nn_seg_inf.py
inference script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import ndimage

from part_seg.src.nn_seg_data import PSegDataset, build_deeplab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def visualise_val_predictions(
    model,
    val_dataset,
    device,
    n=6,
    thresh=0.5,
    seed=0,
    show_instances=True,
):
    """
    Visualise model performance on the validation dataset.

    Shows for each sample:
      - RGB image
      - Predicted binary mask (foreground prob > thresh)
      - Ground-truth binary mask
      - (Optional) Predicted instance labels via connected components

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model returning dict with key "out" -> logits [B,2,H,W].
    val_dataset : torch.utils.data.Dataset
        Returns (image, mask) where image is [3,H,W] float, mask is [H,W] long (0/1).
    device : torch.device
    n : int
        Number of validation samples to display.
    thresh : float
        Threshold on foreground probability.
    seed : int
        Random seed for reproducible sampling.
    show_instances : bool
        If True, add an "Instances" panel showing connected-component labels.
    """
    model.eval()

    rng = random.Random(seed)
    idxs = list(range(len(val_dataset)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    cols = 4 if show_instances else 3
    rows = len(idxs)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # make [rows, cols]

    for r, idx in enumerate(idxs):
        img, gt = val_dataset[idx]  # img: [3,H,W], gt: [H,W]
        x = img.unsqueeze(0).to(device)  # [1,3,H,W]

        out = model(x)["out"]  # [1,2,H,W] logits
        prob_fg = torch.softmax(out, dim=1)[:, 1]  # [1,H,W]
        pred = (prob_fg[0] > thresh).to(torch.uint8).cpu().numpy()  # [H,W] 0/1

        # GT to numpy
        gt_np = gt.cpu().numpy().astype(np.uint8)  # [H,W] 0/1

        # Image to numpy for plotting
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H,W,3]

        # --- Panel 1: RGB image ---
        ax = axes[r, 0]
        ax.imshow(img_np)
        ax.set_title(f"Image (idx={idx})")
        ax.axis("off")

        # --- Panel 2: Pred mask ---
        ax = axes[r, 1]
        ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Pred mask (t={thresh:.2f})")
        ax.axis("off")

        # --- Panel 3: GT mask ---
        ax = axes[r, 2]
        ax.imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        ax.set_title("GT mask")
        ax.axis("off")

        # --- Optional Panel 4: Pred instances via connected components ---
        if show_instances:
            labels, num = ndimage.label(pred)  # 0..num
            ax = axes[r, 3]
            ax.imshow(labels, cmap="viridis")
            ax.set_title(f"Pred instances (n={num})")
            ax.axis("off")

    plt.tight_layout()
    plt.show()

model = build_deeplab(num_classes=2).to(device)
model.load_state_dict(torch.load("best_deeplab_particles.pt", map_location=device))

dataset = PSegDataset()
visualise_val_predictions(
    model=model,
    val_dataset=dataset.val_ds,
    device=device,
    n=6,
    thresh=0.5,
    show_instances=True
)