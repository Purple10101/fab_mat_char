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
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as nd_label

from seg_par.src.nn_seg_data import PSegDataset, build_deeplab_instance
from seg_par.src.prt_inst import (Particle, particles_from_instance_map, show_particle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_skeleton_segments(center_prob: np.ndarray, fg_mask: np.ndarray,
                                thresh: float, min_segment_length: int = 5):
    """
    Threshold the predicted skeleton heatmap and label connected components.
    Each connected component represents one predicted particle.

    We mask by fg_mask first to suppress spurious skeleton activations
    outside the foreground region.

    min_segment_length maps naturally to minimum physical fibre length —
    fragments shorter than this are discarded as noise.

    Returns:
      seg_labels: [H,W] int32, 0=background, 1..N = skeleton segments
      n_segments: int
    """
    # suppress skeleton predictions outside foreground
    skel_mask = (center_prob >= thresh) & (fg_mask > 0)

    # label connected components — each is a candidate particle skeleton
    seg_labels, n_segments = nd_label(skel_mask)

    # discard fragments shorter than min_segment_length
    for seg_id in range(1, n_segments + 1):
        if (seg_labels == seg_id).sum() < min_segment_length:
            seg_labels[seg_labels == seg_id] = 0

    # re-label compactly after filtering
    seg_labels, n_segments = nd_label(seg_labels > 0)

    return seg_labels.astype(np.int32), n_segments


def _decode_instances_from_skeleton(fg_mask: np.ndarray, seg_labels: np.ndarray,
                                     offsets: np.ndarray, max_dist: float = 12.0):
    """
    Assign each foreground pixel to a skeleton segment via offset voting.

    Each fg pixel casts a vote by adding its predicted offset vector to its
    position — the vote lands near the skeleton. We then find the nearest
    skeleton pixel to that vote and inherit its segment label.

    This replaces the centroid-based decode — rather than voting towards a
    single point, pixels vote towards the nearest point on a skeleton line,
    so long thin particles are handled naturally.

    Returns:
      labels: [H,W] int32, 0=background, 1..N = instance ids
    """
    h, w = fg_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)

    if seg_labels.max() == 0:
        return labels

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0:
        return labels

    dx = offsets[0, ys, xs]
    dy = offsets[1, ys, xs]

    # where each pixel thinks the skeleton is
    vote_x = np.clip(xs.astype(np.float32) + dx, 0, w - 1).astype(np.int32)
    vote_y = np.clip(ys.astype(np.float32) + dy, 0, h - 1).astype(np.int32)

    # look up which skeleton segment the vote lands in
    voted_labels = seg_labels[vote_y, vote_x]

    # compute distance from vote to nearest skeleton pixel —
    # votes that land too far from any skeleton are likely noise and discarded
    skel_mask = (seg_labels > 0).astype(np.uint8)
    _, nearest = distance_transform_edt(1 - skel_mask, return_indices=True)

    nearest_skel_y = nearest[0, vote_y, vote_x]
    nearest_skel_x = nearest[1, vote_y, vote_x]
    dist = np.sqrt((vote_y - nearest_skel_y) ** 2 + (vote_x - nearest_skel_x) ** 2)

    keep = (voted_labels > 0) & (dist <= max_dist)
    labels[ys[keep], xs[keep]] = voted_labels[keep]

    return labels


@torch.no_grad()
def visualise_val_predictions(
    model,
    val_dataset,
    device,
    n=6,
    fg_thresh=0.5,
    center_thresh=0.3,
    min_segment_length=5,
    max_assign_dist=12.0,
    seed=0,
    show_centers=True,
):
    """
    Visualise instance-segmentation performance on the validation dataset.
    """

    model.eval()

    rng = random.Random(seed)
    idxs = list(range(len(val_dataset)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    cols = 5
    rows = len(idxs)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, idx in enumerate(idxs):
        img, t = val_dataset[idx]  # img: [3,H,W], t: dict

        x = img.unsqueeze(0).to(device)
        out_t = model(x)["out"]  # [1,4,H,W]
        out = out_t[0].detach().cpu().numpy()

        # --- predicted fg ---
        fg_logit = out[0]
        fg_prob = 1.0 / (1.0 + np.exp(-fg_logit))
        pred_fg = (fg_prob > fg_thresh).astype(np.uint8)

        # --- predicted skeleton heatmap ---
        center_logit = out[1]
        center_prob = 1.0 / (1.0 + np.exp(-center_logit))

        # --- predicted offsets ---
        offsets = out[2:4]  # [2,H,W]

        # extract skeleton segments then assign fg pixels via offset voting
        seg_labels, n_segs = _extract_skeleton_segments(
            center_prob, pred_fg, thresh=center_thresh,
            min_segment_length=min_segment_length
        )
        pred_instances = _decode_instances_from_skeleton(
            pred_fg, seg_labels, offsets, max_dist=max_assign_dist
        )
        n_inst = int(pred_instances.max())

        # --- GT ---
        gt_fg = t["fg"].detach().cpu().numpy().astype(np.uint8)

        # --- image to numpy ---
        img_np = img.permute(1, 2, 0).cpu().numpy()

        c = 0

        # Panel 1: image
        ax = axes[r, c]; c += 1
        ax.imshow(img_np)
        ax.set_title(f"Image (idx={idx})")
        ax.axis("off")

        # Panel 2: predicted fg mask
        ax = axes[r, c]; c += 1
        ax.imshow(pred_fg, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Pred FG (t={fg_thresh:.2f})")
        ax.axis("off")

        # Panel 3: GT fg mask
        ax = axes[r, c]; c += 1
        ax.imshow(gt_fg, cmap="gray", vmin=0, vmax=1)
        ax.set_title("GT FG")
        ax.axis("off")

        # Panel 4: predicted skeleton heatmap + segment overlay
        ax = axes[r, c]; c += 1
        ax.imshow(center_prob, cmap="gray", vmin=0, vmax=1)
        if show_centers and n_segs > 0:
            # overlay skeleton segment centroids
            for seg_id in range(1, n_segs + 1):
                ys, xs = np.where(seg_labels == seg_id)
                ax.scatter(xs.mean(), ys.mean(), s=18)
        ax.set_title(f"Skeleton prob (segs={n_segs})")
        ax.axis("off")

        # Panel 5: predicted instances
        ax = axes[r, c]; c += 1
        ax.imshow(pred_instances, cmap="viridis")
        ax.set_title(f"Pred instances (n={n_inst})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_particles_from_val(
        model,
        val_dataset,
        device,
        n=6,
        fg_thresh=0.5,
        center_thresh=0.3,
        min_segment_length=5,
        max_assign_dist=12.0,
        seed=0,
        min_area_px=5,
):
    model.eval()

    rng = random.Random(seed)
    idxs = list(range(len(val_dataset)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    results = []

    for idx in idxs:
        img, t = val_dataset[idx]  # img: [3,H,W]

        x = img.unsqueeze(0).to(device)
        out_t = model(x)["out"]  # [1,4,H,W]
        out = out_t[0].detach().cpu().numpy()

        # predicted fg
        fg_logit = out[0]
        fg_prob = 1.0 / (1.0 + np.exp(-fg_logit))
        pred_fg = (fg_prob > fg_thresh).astype(np.uint8)

        # predicted skeleton heatmap
        center_logit = out[1]
        center_prob = 1.0 / (1.0 + np.exp(-center_logit))

        # predicted offsets
        offsets = out[2:4]

        # extract skeleton segments then assign fg pixels via offset voting
        seg_labels, n_segs = _extract_skeleton_segments(
            center_prob, pred_fg, thresh=center_thresh,
            min_segment_length=min_segment_length
        )
        pred_instances = _decode_instances_from_skeleton(
            pred_fg, seg_labels, offsets, max_dist=max_assign_dist
        )

        # image to numpy for crop
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H,W,3], float 0..1

        particles = particles_from_instance_map(img_np, pred_instances, min_area_px=min_area_px)

        results.append({
            "idx": idx,
            "img_np": img_np,
            "pred_fg": pred_fg,
            "center_prob": center_prob,
            "seg_labels": seg_labels,
            "n_segs": n_segs,
            "pred_instances": pred_instances,
            "particles": particles,
        })

    return results


def main():
    notable_keys = ["34f4fb273d",
                    "59424b060a",
                    "62a54f335d",
                    ]

    model = build_deeplab_instance(num_out=4).to(device)
    model.load_state_dict(torch.load("best_deeplab_instances.pt", map_location=device))

    dataset = PSegDataset()
    val_dataset = dataset.get_subset_torch_dataset(notable_keys)
    results = get_particles_from_val(
        model=model,
        val_dataset=dataset.val_ds,
        device=device
    )
    for particle in results[1]["particles"]:
        show_particle(particle, results[1]["img_np"])
        print()


if __name__ == '__main__':
    main()