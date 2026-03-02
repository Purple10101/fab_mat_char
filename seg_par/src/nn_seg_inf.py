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

from seg_par.src.nn_seg_data import PSegDataset, build_deeplab_instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_center_peaks(center_prob, fg_mask=None, thresh=0.3, min_dist=5):
    """
    center_prob: [H,W] float (0..1)
    fg_mask: [H,W] bool/0-1 optional, restrict peaks to fg
    Returns list of (y,x) peak coords.
    """
    c = center_prob.copy()
    if fg_mask is not None:
        c = c * fg_mask.astype(c.dtype)

    # simple non-maximum suppression via max filter
    max_f = ndimage.maximum_filter(c, size=min_dist)
    peaks = (c == max_f) & (c >= thresh)

    ys, xs = np.where(peaks)
    # sort peaks by confidence descending
    conf = c[ys, xs]
    order = np.argsort(-conf)
    return list(zip(ys[order], xs[order]))

def decode_instances_from_centers_offsets(fg_mask, centers_yx, offsets, max_assign_dist=10.0):
    """
    fg_mask: [H,W] uint8/bool
    centers_yx: list[(y,x)]
    offsets: [2,H,W] float where offsets[0]=dx, offsets[1]=dy (pixel -> center)
    Returns:
      labels: [H,W] int32, 0=bg, 1..N instances
    """
    h, w = fg_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if len(centers_yx) == 0:
        return labels

    centers = np.array([(y, x) for (y, x) in centers_yx], dtype=np.float32)  # [N,2]

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0:
        return labels

    dx = offsets[0, ys, xs]
    dy = offsets[1, ys, xs]
    vote_y = ys.astype(np.float32) + dy
    vote_x = xs.astype(np.float32) + dx

    # distance from each pixel's vote to each detected center
    # dists: [P,N]
    dists = (vote_y[:, None] - centers[None, :, 0]) ** 2 + (vote_x[:, None] - centers[None, :, 1]) ** 2
    nn = np.argmin(dists, axis=1)
    nn_dist = np.sqrt(dists[np.arange(dists.shape[0]), nn])

    # optional rejection if vote is too far from any center
    keep = nn_dist <= max_assign_dist
    ys_k = ys[keep]
    xs_k = xs[keep]
    nn_k = nn[keep]

    labels[ys_k, xs_k] = (nn_k + 1).astype(np.int32)  # 1..N
    return labels


@torch.no_grad()
def visualise_val_predictions(
    model,
    val_dataset,
    device,
    n=6,
    fg_thresh=0.5,
    center_thresh=0.3,
    min_peak_dist=7,
    max_assign_dist=12.0,
    seed=0,
    show_centers=True,
    show_gt_instances=True,
):
    """
    Visualise instance-segmentation performance on the validation dataset.

    Expects the model to output logits in model(x)["out"] with shape [B,5,H,W]:
      - out[:,0:2] : foreground logits (bg/fg)
      - out[:,2:3] : center heatmap logits
      - out[:,3:5] : offsets (dx, dy) from pixel -> instance center

    Expects val_dataset[idx] to return:
      (image, targets_dict) where targets_dict contains:
        - "fg":      [H,W] long (0/1)
        - "center":  [1,H,W] float (0..1)
        - "offsets": [2,H,W] float
      Optionally, if present:
        - "id_map":  [H,W] long instance IDs (0=bg, 1..K)

    Parameters
    ----------
    model : torch.nn.Module
    val_dataset : torch.utils.data.Dataset
    device : torch.device
    n : int
    fg_thresh : float
        Threshold applied to predicted fg probability.
    center_thresh : float
        Threshold for accepting center peaks.
    min_peak_dist : int
        Window size for non-maximum suppression when finding peaks.
    max_assign_dist : float
        Maximum distance (in pixels) between a pixel's voted center and a detected
        center peak to accept assignment.
    seed : int
    show_centers : bool
        If True, overlay detected center peaks on the center heatmap.
    show_gt_instances : bool
        If True and "id_map" exists, plot GT instances.
    """

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _find_center_peaks(center_prob: np.ndarray, fg_mask: np.ndarray | None,
                           thresh: float, min_dist: int):
        c = center_prob
        if fg_mask is not None:
            c = c * fg_mask.astype(c.dtype)

        # Non-maximum suppression via max filter
        max_f = ndimage.maximum_filter(c, size=min_dist)
        peaks = (c == max_f) & (c >= thresh)

        ys, xs = np.where(peaks)
        if len(xs) == 0:
            return []

        conf = c[ys, xs]
        order = np.argsort(-conf)  # high -> low
        return list(zip(ys[order], xs[order]))

    def _decode_instances(fg_mask: np.ndarray, centers_yx: list[tuple[int, int]],
                          offsets: np.ndarray, max_dist: float):
        """
        fg_mask: [H,W] uint8/bool
        centers_yx: list[(y,x)]
        offsets: [2,H,W] float where offsets[0]=dx, offsets[1]=dy
        returns labels [H,W] int32 (0=bg, 1..N)
        """
        h, w = fg_mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        if len(centers_yx) == 0:
            return labels

        centers = np.array([(y, x) for (y, x) in centers_yx], dtype=np.float32)  # [N,2]

        ys, xs = np.where(fg_mask > 0)
        if len(xs) == 0:
            return labels

        dx = offsets[0, ys, xs]
        dy = offsets[1, ys, xs]

        vote_y = ys.astype(np.float32) + dy
        vote_x = xs.astype(np.float32) + dx

        # squared distances [P,N]
        d2 = (vote_y[:, None] - centers[None, :, 0]) ** 2 + (vote_x[:, None] - centers[None, :, 1]) ** 2
        nn = np.argmin(d2, axis=1)
        nn_dist = np.sqrt(d2[np.arange(d2.shape[0]), nn])

        keep = nn_dist <= max_dist
        labels[ys[keep], xs[keep]] = (nn[keep] + 1).astype(np.int32)
        return labels

    model.eval()

    rng = random.Random(seed)
    idxs = list(range(len(val_dataset)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    cols = 5
    rows = len(idxs)

    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, idx in enumerate(idxs):
        img, t = val_dataset[idx]  # img: [3,H,W], t: dict

        x = img.unsqueeze(0).to(device)  # [1,3,H,W]
        out_t = model(x)["out"]          # [1,5,H,W]
        out = out_t[0].detach().cpu().numpy()  # [5,H,W]

        # --- predicted fg ---
        fg_logits = out[0:2]  # [2,H,W]
        fg_prob = torch.softmax(torch.from_numpy(fg_logits), dim=0)[1].numpy()  # [H,W]
        pred_fg = (fg_prob > fg_thresh).astype(np.uint8)

        # --- predicted centers ---
        center_logits = out[2]               # [H,W]
        center_prob = _sigmoid(center_logits)

        # --- predicted offsets ---
        offsets = out[3:5]  # [2,H,W] dx, dy

        centers = _find_center_peaks(center_prob, fg_mask=pred_fg, thresh=center_thresh, min_dist=min_peak_dist)
        pred_instances = _decode_instances(pred_fg, centers, offsets, max_dist=max_assign_dist)
        n_inst = int(pred_instances.max())

        # --- GT ---
        gt_fg = t["fg"].detach().cpu().numpy().astype(np.uint8)
        gt_instances = None
        if show_gt_instances and ("id_map" in t):
            gt_instances = t["id_map"].detach().cpu().numpy().astype(np.int32)

        # --- image to numpy ---
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # Panel 1: image
        ax = axes[r, 0]
        ax.imshow(img_np)
        ax.set_title(f"Image (idx={idx})")
        ax.axis("off")

        # Panel 2: predicted fg mask
        ax = axes[r, 1]
        ax.imshow(pred_fg, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Pred FG (t={fg_thresh:.2f})")
        ax.axis("off")

        # Panel 3: center heatmap (+ peaks)
        ax = axes[r, 2]
        ax.imshow(center_prob, cmap="gray", vmin=0, vmax=1)
        if show_centers and len(centers) > 0:
            cy, cx = zip(*centers)
            ax.scatter(cx, cy, s=18)
        ax.set_title(f"Center prob (peaks={len(centers)})")
        ax.axis("off")

        # Panel 4: predicted instances
        ax = axes[r, 3]
        ax.imshow(pred_instances, cmap="viridis")
        ax.set_title(f"Pred instances (n={n_inst})")
        ax.axis("off")

        # Panel 5 (optional): GT instances
        if cols == 5:
            ax = axes[r, 4]
            ax.imshow(gt_instances, cmap="viridis")
            ax.set_title("GT instances")
            ax.axis("off")

    plt.tight_layout()
    plt.show()








model = build_deeplab_instance().to(device)
model.load_state_dict(torch.load("best_deeplab_instances.pt", map_location=device))

dataset = PSegDataset()
visualise_val_predictions(
    model=model,
    val_dataset=dataset.val_ds,
    device=device
)