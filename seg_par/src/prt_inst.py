"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260303

prt_inst.py
particle objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class Particle:
    id: int
    # bounding box (x0, y0, x1, y1)
    bbox: tuple[int, int, int, int]
    # cropped views
    crop_img: np.ndarray  # [h,w,3] float
    crop_mask: np.ndarray  # [h,w] bool
    # geometry
    area_px: int
    centroid_xy: tuple[float, float]  # full-image (cx, cy)

def show_particle(
    p: Particle,
    full_img: np.ndarray | None = None,
    mode="full",              # "crop" or "full"
    overlay=True,
    draw_bbox=True,
    draw_centroid=True,
    draw_outline=True,
):
    """
    p: Particle object
    full_img: [H,W,3] numpy image (required if mode="full")
    mode:
        "crop" -> show cropped particle only
        "full" -> show full image with particle highlighted
    """

    plt.figure()

    if mode == "crop":
        plt.imshow(p.crop_img)

        if overlay:
            plt.imshow(p.crop_mask.astype(float), alpha=0.35, cmap="gray")

        plt.title(f"Particle {p.id} | area={p.area_px}px")
        plt.axis("off")
        plt.show()
        return

    if full_img is None:
        raise ValueError("full_img must be provided when mode='full'")

    plt.imshow(full_img)

    x0, y0, x1, y1 = p.bbox

    if overlay:
        overlay_mask = np.zeros(full_img.shape[:2], dtype=float)
        overlay_mask[y0:y1+1, x0:x1+1] = p.crop_mask.astype(float)
        plt.imshow(overlay_mask, alpha=0.35, cmap="Reds")

    if draw_bbox:
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

    if draw_centroid:
        cx, cy = p.centroid_xy
        plt.scatter(cx, cy, c="cyan", s=40)

    if draw_outline:
        from scipy import ndimage
        mask_full = np.zeros(full_img.shape[:2], dtype=bool)
        mask_full[y0:y1+1, x0:x1+1] = p.crop_mask
        edge = mask_full ^ ndimage.binary_erosion(mask_full)
        ys, xs = np.where(edge)
        plt.scatter(xs, ys, s=1, c="lime")

    plt.title(f"Particle {p.id} | area={p.area_px}px")
    plt.axis("off")
    plt.show()

def show_particles_full(
    particles: list[Particle],
    full_img: np.ndarray,
    overlay=True,
    draw_bbox=False,
    draw_centroid=False,
    draw_outline=True,
    label_ids=False,
    alpha=0.25,
    min_area_px=0,
):
    """
    Show the full image with ALL particles highlighted.

    particles: list of Particle (with bbox, crop_mask, centroid_xy, area_px)
    full_img: [H,W,3] numpy image
    """
    H, W = full_img.shape[:2]

    plt.figure(figsize=(10, 8))
    plt.imshow(full_img)

    # Build one combined mask overlay (fast)
    if overlay:
        overlay_mask = np.zeros((H, W), dtype=float)
        for p in particles:
            if p.area_px < min_area_px:
                continue
            x0, y0, x1, y1 = p.bbox
            overlay_mask[y0:y1+1, x0:x1+1] = np.maximum(
                overlay_mask[y0:y1+1, x0:x1+1],
                p.crop_mask.astype(float)
            )
        plt.imshow(overlay_mask, alpha=alpha, cmap="Reds")

    # Optional outlines / bbox / centroid / labels
    for p in particles:
        if p.area_px < min_area_px:
            continue

        x0, y0, x1, y1 = p.bbox

        if draw_outline:
            mask_full = np.zeros((H, W), dtype=bool)
            mask_full[y0:y1+1, x0:x1+1] = p.crop_mask
            edge = mask_full ^ ndimage.binary_erosion(mask_full)
            ys, xs = np.where(edge)
            plt.scatter(xs, ys, s=1, c="lime")

        if draw_bbox:
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
            plt.gca().add_patch(rect)

        if draw_centroid:
            cx, cy = p.centroid_xy
            plt.scatter(cx, cy, c="cyan", s=25)

        if label_ids:
            cx, cy = p.centroid_xy
            plt.text(cx, cy, str(p.id), color="white", fontsize=10,
                     ha="center", va="center")

    plt.title(f"All particles (n={sum(p.area_px >= min_area_px for p in particles)})")
    plt.axis("off")
    plt.show()

def particles_from_instance_map(img_np: np.ndarray, inst: np.ndarray, min_area_px: int = 5) -> list[Particle]:
    parts: list[Particle] = []
    ids = np.unique(inst)
    ids = ids[ids != 0]

    for pid in ids:
        mask = (inst == pid)
        area_px = int(mask.sum())
        if area_px < min_area_px:
            continue

        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        crop_img = img_np[y0:y1+1, x0:x1+1].copy()
        crop_mask = mask[y0:y1+1, x0:x1+1].copy()

        cx, cy = float(xs.mean()), float(ys.mean())
        equiv_d_px = float(2.0 * np.sqrt(area_px / np.pi))

        parts.append(Particle(
            id=int(pid),
            bbox=(x0, y0, x1, y1),
            crop_img=crop_img,
            crop_mask=crop_mask,
            area_px=area_px,
            centroid_xy=(cx, cy),
        ))

    return parts

def main():
    print()

if __name__ == '__main__':
    main()