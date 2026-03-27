"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

infer.py
fibre instance segmentation inference + GT comparison

Output per sample (saved to CONFIG["out_dir"]):
    <n>_overlay.png     — image with coloured predicted instance masks
    <n>_pred.json       — predictions (scores, boxes, RLE masks)
    <n>_comparison.png  — 4-panel GT vs prediction figure
    <n>_fibres.png      — grid of every detected fibre, isolated on black
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import torchvision.transforms.functional as TF
from pycocotools import mask as coco_mask_utils

from msg.src.ss4.seg.model import build_model


# ════════════════════════════════════════════════════════════
#  CONFIG — edit this block, then run `python infer.py`
# ════════════════════════════════════════════════════════════

CONFIG = {
    # Path to the saved checkpoint from train.py
    "checkpoint":   "runs/fibre_maskrcnn/best.pth",

    # Root of the fibre dataset (must contain manifest.json)
    "data_dir":     "./ss4/fibre_dataset",

    # Which split to sample from: "train", "val", "test", or "all"
    "split":        "val",

    # How many random samples to run (0 = entire split)
    "n_samples":    5,

    # Random seed for sample selection (None = different each run)
    "seed":         42,

    # Where to write output files
    "out_dir":      "./predictions",

    # Model input size (must match training)
    "image_size":   512,

    # Detections below this confidence are discarded
    "score_thresh": 0.4,

    # Backbone variant — must match the checkpoint
    "backbone":     "maskrcnn_resnet50_fpn_v2",

    # Mask overlay transparency (0 = invisible, 1 = opaque)
    "mask_alpha":   0.45,

    # Max columns in the per-fibre grid figure
    "fibre_grid_cols": 6,
}

# ════════════════════════════════════════════════════════════


# ─── Colour palette ───────────────────────────────────────────────────────────

PALETTE = [
    (230, 80,  60),  (60,  130, 220), (80,  190, 100), (220, 170, 50),
    (160, 80,  200), (50,  200, 190), (230, 110, 160), (120, 120, 120),
    (255, 150, 50),  (50,  220, 255), (180, 230, 80),  (200, 80,  130),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Fibre dataclass ──────────────────────────────────────────────────────────

@dataclass
class Fibre:
    """
    One detected fibre instance. All properties are computed from the
    predicted binary mask and are immediately available after construction.

    Usage examples:
        fibres = [...]                              # list built during inference

        # Iterate
        for f in fibres:
            print(f.instance_id, f.score, f.area, f.centroid)

        # Filter
        confident = [f for f in fibres if f.score > 0.8]
        large     = [f for f in fibres if f.area  > 500]

        # Sort
        by_length = sorted(fibres, key=lambda f: f.length, reverse=True)

        # Extract raw pixels from the original image
        img_arr      = np.array(orig_image)
        fibre_pixels = img_arr[f.mask]             # shape: (N_pixels, 3)
    """

    instance_id : int
    score       : float          # model confidence in [0, 1]
    mask        : np.ndarray     # H×W bool — True where this fibre was detected
    box_xyxy    : tuple          # (x0, y0, x1, y1) bounding box in image pixels

    # Derived — computed automatically in __post_init__
    area        : int   = field(init=False)  # number of fibre pixels
    centroid    : tuple = field(init=False)  # (x, y) centre of mass
    bbox_wh     : tuple = field(init=False)  # (width, height) of bounding box
    length      : float = field(init=False)  # long-axis length via PCA (px)
    orientation : float = field(init=False)  # long-axis angle in degrees

    def __post_init__(self):
        ys, xs   = np.where(self.mask)
        self.area = int(self.mask.sum())

        if self.area == 0:
            self.centroid = self.bbox_wh = (0.0, 0.0)
            self.length = self.orientation = 0.0
            return

        self.centroid = (float(xs.mean()), float(ys.mean()))

        x0, y0, x1, y1 = self.box_xyxy
        self.bbox_wh    = (float(x1 - x0), float(y1 - y0))

        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        coords -= coords.mean(axis=0)
        if len(coords) >= 2:
            _, _, vt       = np.linalg.svd(coords, full_matrices=False)
            principal      = vt[0]
            proj           = coords @ principal
            self.length    = float(proj.max() - proj.min())
            self.orientation = float(
                np.degrees(np.arctan2(principal[1], principal[0])) % 180
            )
        else:
            self.length = self.orientation = 0.0

    def __repr__(self):
        return (
            f"Fibre(id={self.instance_id}, score={self.score:.2f}, "
            f"area={self.area}px, length={self.length:.1f}px, "
            f"orientation={self.orientation:.1f}deg, "
            f"centroid=({self.centroid[0]:.1f}, {self.centroid[1]:.1f}))"
        )


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_path, image_size):
    img         = Image.open(image_path).convert("RGB")
    orig_size   = img.size
    img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    tensor      = TF.to_tensor(img_resized)
    tensor      = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, img, orig_size


# ─── Overlay helpers ──────────────────────────────────────────────────────────

def build_instance_overlay(base_image, fibres_or_masks, scores=None,
                            boxes=None, alpha=0.45, draw_boxes=True):
    """Accepts either a list[Fibre] or a list of raw binary mask arrays."""
    W, H    = base_image.size
    overlay = np.array(base_image).astype(np.float32)

    if fibres_or_masks and isinstance(fibres_or_masks[0], Fibre):
        masks_list  = [f.mask for f in fibres_or_masks]
        scores_list = [f.score for f in fibres_or_masks]
        boxes_list  = [f.box_xyxy for f in fibres_or_masks]
    else:
        masks_list  = fibres_or_masks
        scores_list = scores or []
        boxes_list  = boxes  or []

    for i, mask_bin in enumerate(masks_list):
        color        = PALETTE[i % len(PALETTE)]
        mask_pil     = Image.fromarray(mask_bin.astype(np.uint8) * 255)
        mask_resized = np.array(mask_pil.resize((W, H), Image.NEAREST)) > 127
        for c in range(3):
            overlay[:, :, c][mask_resized] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][mask_resized]
            )

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    if draw_boxes and boxes_list and scores_list:
        draw = ImageDraw.Draw(result)
        for i, (box, score) in enumerate(zip(boxes_list, scores_list)):
            color           = PALETTE[i % len(PALETTE)]
            x0, y0, x1, y1 = box
            sx, sy          = W / 512, H / 512
            draw.rectangle([x0*sx, y0*sy, x1*sx, y1*sy], outline=color, width=2)
            draw.text((x0*sx + 3, y0*sy + 2), f"{score:.2f}", fill=color)

    return result, len(masks_list)


def decode_gt_masks(mask_rgb_path, fibre_meta):
    mask_rgb = np.array(Image.open(mask_rgb_path).convert("RGB"), dtype=np.uint8)
    masks = []
    for fibre in fibre_meta:
        rgb    = np.array(fibre["mask_rgb"], dtype=np.uint8)
        binary = np.all(mask_rgb == rgb, axis=-1)
        if binary.sum() > 0:
            masks.append(binary)
    return masks


# ─── Per-fibre grid ───────────────────────────────────────────────────────────

def plot_fibres(orig_image, fibres, out_path, max_cols=6):
    """
    Display each detected fibre individually in a grid.

    Each cell shows the fibre cropped tightly to its mask, with
    the background zeroed to black and a coloured outline tracing
    the fibre edge. Caption shows id, score, area, length, orientation.

    Plug-and-play: just call plot_fibres(orig_image, fibres, path).
    """
    if not fibres:
        print("  plot_fibres: no fibres to display")
        return

    img_arr  = np.array(orig_image)        # H×W×3 uint8
    H, W     = img_arr.shape[:2]
    n        = len(fibres)
    n_cols   = min(n, max_cols)
    n_rows   = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.6, n_rows * 2.8))
    fig.patch.set_facecolor("#111111")

    # Flatten axes to a simple list regardless of grid shape
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).flat)

    for ax in axes_flat:
        ax.set_facecolor("#111111")
        ax.axis("off")

    for idx, fibre in enumerate(fibres):
        ax    = axes_flat[idx]
        color = PALETTE[idx % len(PALETTE)]
        mask  = fibre.mask                 # H×W bool

        # Tight bounding crop with a small pad
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad    = 8
        r0 = max(r0 - pad, 0);  r1 = min(r1 + pad, H - 1)
        c0 = max(c0 - pad, 0);  c1 = min(c1 + pad, W - 1)

        crop_img  = img_arr[r0:r1+1, c0:c1+1].copy()
        crop_mask = mask[r0:r1+1, c0:c1+1]

        # Zero out background
        crop_img[~crop_mask] = 0

        # Coloured outline — 1-px boundary ring (mask XOR eroded mask)
        boundary = crop_mask & ~binary_erosion(crop_mask, iterations=1)
        for c in range(3):
            crop_img[:, :, c][boundary] = color[c]

        ax.imshow(crop_img)
        ax.set_title(
            f"#{fibre.instance_id}  score={fibre.score:.2f}",
            color="white", fontsize=8, fontweight="bold", pad=3,
        )
        ax.set_xlabel(
            f"area={fibre.area}px  "
            f"len={fibre.length:.0f}px  "
            f"{fibre.orientation:.0f}°",
            color="#aaaaaa", fontsize=7,
        )

    # Hide unused grid cells
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(
        f"{os.path.basename(out_path).replace('_fibres.png', '')} — extracted fibres",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  fibres grid  → {os.path.basename(out_path)}")


# ─── JSON serialisation ───────────────────────────────────────────────────────

def fibres_to_json(fibres):
    out = []
    for f in fibres:
        binary        = f.mask.astype(np.uint8)
        rle           = coco_mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")
        out.append({
            "instance_id":     f.instance_id,
            "score":           round(f.score, 4),
            "box_xyxy":        [round(v, 2) for v in f.box_xyxy],
            "area":            f.area,
            "centroid":        [round(v, 2) for v in f.centroid],
            "length_px":       round(f.length, 2),
            "orientation_deg": round(f.orientation, 2),
            "mask_rle":        rle,
        })
    return out


# ─── Comparison figure ────────────────────────────────────────────────────────

def make_comparison_figure(orig_image, gt_masks, fibres, out_path,
                            sample_info=None, alpha=0.45):
    W, H = orig_image.size

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")

    title_kw = dict(color="white",   fontsize=11, fontweight="bold", pad=8)
    label_kw = dict(color="#aaaaaa", fontsize=9)

    axes[0].imshow(orig_image)
    axes[0].set_title("Original image", **title_kw)
    if sample_info:
        axes[0].set_xlabel(sample_info, **label_kw)

    if gt_masks:
        gt_overlay, n_gt = build_instance_overlay(
            orig_image, gt_masks, draw_boxes=False, alpha=alpha)
    else:
        gt_overlay, n_gt = orig_image, 0
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f"Ground truth  ({n_gt} fibres)", **title_kw)

    if fibres:
        pred_overlay, n_pred = build_instance_overlay(
            orig_image, fibres, draw_boxes=True, alpha=alpha)
    else:
        pred_overlay, n_pred = orig_image, 0
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Predicted  ({n_pred} fibres)", **title_kw)

    def union(mask_list):
        acc = np.zeros((H, W), dtype=np.float32)
        for m in mask_list:
            r = np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            ) / 255.0
            acc = np.maximum(acc, r)
        return acc > 0.5

    gt_union   = union(gt_masks)
    pred_union = union([f.mask for f in fibres])
    tp = gt_union  &  pred_union
    fn = gt_union  & ~pred_union
    fp = ~gt_union &  pred_union

    diff = np.zeros((H, W, 3), dtype=np.float32)
    diff[tp] = [0.20, 0.85, 0.40]
    diff[fn] = [0.90, 0.25, 0.25]
    diff[fp] = [0.30, 0.50, 1.00]

    grey = np.array(orig_image.convert("L").resize((W, H)),
                    dtype=np.float32) / 255.0
    bg   = np.stack([grey * 0.35] * 3, axis=-1)
    hit  = (tp | fn | fp).astype(np.float32)[..., None]
    axes[3].imshow(np.clip(diff * hit + bg * (1 - hit), 0, 1))
    axes[3].set_title("Coverage diff", **title_kw)

    n_tp = int(tp.sum())
    prec = n_tp / max(int(pred_union.sum()), 1)
    rec  = n_tp / max(int(gt_union.sum()),   1)
    axes[3].set_xlabel(f"pixel  precision={prec:.2f}  recall={rec:.2f}", **label_kw)
    axes[3].legend(
        handles=[
            mpatches.Patch(color=(0.20, 0.85, 0.40), label="True positive"),
            mpatches.Patch(color=(0.90, 0.25, 0.25), label="Missed  (FN)"),
            mpatches.Patch(color=(0.30, 0.50, 1.00), label="False alarm (FP)"),
        ],
        loc="lower right", fontsize=8, framealpha=0.55,
        facecolor="#111111", labelcolor="white",
    )

    stem = os.path.basename(out_path).replace("_comparison.png", "")
    plt.suptitle(stem, color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  comparison   → {os.path.basename(out_path)}"
          f"  (prec={prec:.2f}  rec={rec:.2f})")
    return prec, rec


# ─── GT lookup ────────────────────────────────────────────────────────────────

def build_gt_lookup(data_dir, split):
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {data_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    lookup = {}
    for s in manifest["samples"]:
        if split != "all" and s["split"] != split:
            continue
        stem = os.path.splitext(os.path.basename(s["image"]))[0]
        lookup[stem] = {
            "image_path": os.path.join(data_dir, s["image"]),
            "mask_path":  os.path.join(data_dir, s["mask"]),
            "fibres":     s["fibres"],
            "split":      s["split"],
        }
    return lookup



# ─── Public API (callable from ss4.py) ───────────────────────────────────────

def preprocess_array(image_array, image_size):
    """
    Preprocess a raw image array (cv2 BGR or RGB numpy uint8) into a
    normalised tensor ready for the model.

    Returns:
        tensor    : Float[3, image_size, image_size]  normalised input
        pil_image : PIL.Image (RGB)                   original for visualisation
    """
    # cv2 loads BGR — convert to RGB
    rgb = image_array[:, :, ::-1].copy() if image_array.ndim == 3 else image_array
    pil_image   = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    img_resized = pil_image.resize((image_size, image_size), Image.BILINEAR)
    tensor      = TF.to_tensor(img_resized)
    tensor      = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, pil_image


def run_inference(
    model,
    image,
    device,
    image_size    = 512,
    score_thresh  = 0.4,
    debug         = False,
    debug_out_dir = "./ss4/seg/predictions",
    debug_stem    = "inference",
):
    """
    Run instance segmentation on a single image and return all detected fibres.

    Parameters
    ----------
    model         : torch.nn.Module  loaded, eval-mode Mask R-CNN
    image         : np.ndarray (H x W x 3, BGR or RGB uint8)  OR  PIL.Image
                    Accepts whatever cv2.imread() or PIL.Image.open() gives you.
    device        : torch.device
    image_size    : int    must match the size the model was trained with
    score_thresh  : float  detections below this confidence are dropped
    debug         : bool   when True, saves overlay + fibre grid + JSON to disk
                           and prints a per-fibre summary to stdout
    debug_out_dir : str    folder for debug outputs (created if absent)
    debug_stem    : str    filename prefix for debug outputs

    Returns
    -------
    list[Fibre]
        One Fibre per detected instance, sorted by descending score.
        Each Fibre exposes:
            .instance_id  .score   .mask (H x W bool)  .box_xyxy
            .area         .centroid  .length  .orientation  .bbox_wh

    Example (from ss4.py)
    ---------------------
        from msg.src.ss4.infer import run_inference

        # basic call
        fibres = run_inference(self.model, image, self.device)

        # with debug visuals saved to disk
        fibres = run_inference(self.model, image, self.device,
                               debug=True, debug_stem="frame_0042")

        for f in fibres:
            print(f.instance_id, f.length, f.orientation)
    """
    # Normalise input to numpy RGB array + PIL image
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        arr       = np.array(pil_image)
        tensor, _ = preprocess_array(arr, image_size)
    else:
        tensor, pil_image = preprocess_array(image, image_size)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model([tensor.to(device)])[0]

    preds = {k: v.cpu().numpy() for k, v in output.items()}

    # Build Fibre objects
    fibres = []
    for i in range(len(preds["scores"])):
        if preds["scores"][i] < score_thresh:
            continue
        fibres.append(Fibre(
            instance_id = i,
            score       = float(preds["scores"][i]),
            mask        = (preds["masks"][i, 0] > 0.5),
            box_xyxy    = tuple(float(v) for v in preds["boxes"][i]),
        ))

    fibres.sort(key=lambda f: f.score, reverse=True)

    # Debug output
    if debug:
        os.makedirs(debug_out_dir, exist_ok=True)
        print(f"[run_inference]  {len(fibres)} fibre(s) detected  "
              f"(thresh={score_thresh})")
        for f in fibres:
            print(f"  {f}")

        overlay, _ = build_instance_overlay(
            pil_image, fibres, alpha=0.45, draw_boxes=True)
        overlay.save(os.path.join(debug_out_dir, f"{debug_stem}_overlay.png"))

        plot_fibres(
            pil_image, fibres,
            out_path=os.path.join(debug_out_dir, f"{debug_stem}_fibres.png"),
        )

        with open(os.path.join(debug_out_dir, f"{debug_stem}_pred.json"), "w") as fh:
            json.dump(fibres_to_json(fibres), fh, indent=2)

        print(f"  debug outputs  → {debug_out_dir}/{debug_stem}_*")

    return fibres

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Split  : {cfg['split']}   n_samples={cfg['n_samples'] or 'all'}")

    model = build_model(cfg["backbone"], pretrained=False)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded : {cfg['checkpoint']}")

    gt_lookup   = build_gt_lookup(cfg["data_dir"], cfg["split"])
    image_paths = [v["image_path"] for v in gt_lookup.values()]

    n = cfg["n_samples"]
    if n and n < len(image_paths):
        rng = random.Random(cfg["seed"])
        image_paths = rng.sample(image_paths, n)

    print(f"Running inference on {len(image_paths)} sample(s)...\n")
    os.makedirs(cfg["out_dir"], exist_ok=True)

    with torch.no_grad():
        for image_path in image_paths:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            tensor, orig_image, _ = preprocess(image_path, cfg["image_size"])

            output = model([tensor.to(device)])[0]
            preds  = {k: v.cpu().numpy() for k, v in output.items()}

            # ── Build Fibre objects ──────────────────────────────────────────
            fibres = []
            for i in range(len(preds["scores"])):
                if preds["scores"][i] < cfg["score_thresh"]:
                    continue
                fibres.append(Fibre(
                    instance_id = i,
                    score       = float(preds["scores"][i]),
                    mask        = (preds["masks"][i, 0] > 0.5),
                    box_xyxy    = tuple(float(v) for v in preds["boxes"][i]),
                ))

            print(f"[{stem}]  {len(fibres)} fibres detected")
            for f in fibres:
                print(f"  {f}")

            # ── Overlay PNG ──────────────────────────────────────────────────
            overlay, _ = build_instance_overlay(
                orig_image, fibres,
                alpha=cfg["mask_alpha"], draw_boxes=True,
            )
            overlay.save(os.path.join(cfg["out_dir"], f"{stem}_overlay.png"))

            # ── Per-fibre grid ───────────────────────────────────────────────
            plot_fibres(
                orig_image, fibres,
                out_path=os.path.join(cfg["out_dir"], f"{stem}_fibres.png"),
                max_cols=cfg["fibre_grid_cols"],
            )

            # ── JSON ─────────────────────────────────────────────────────────
            with open(os.path.join(cfg["out_dir"], f"{stem}_pred.json"), "w") as f_out:
                json.dump(fibres_to_json(fibres), f_out, indent=2)

            # ── Comparison figure ────────────────────────────────────────────
            gt_info  = gt_lookup[stem]
            gt_masks = decode_gt_masks(gt_info["mask_path"], gt_info["fibres"])
            make_comparison_figure(
                orig_image, gt_masks, fibres,
                out_path=os.path.join(cfg["out_dir"], f"{stem}_comparison.png"),
                sample_info=f"split={gt_info['split']}  GT fibres={len(gt_masks)}",
                alpha=cfg["mask_alpha"],
            )

            # ────────────────────────────────────────────────────────────────
            # ACCESS INDIVIDUAL FIBRES HERE
            # `fibres` is a list[Fibre] — one entry per detected fibre.
            # Each Fibre has: .instance_id, .score, .mask, .box_xyxy,
            #                 .area, .centroid, .length, .orientation, .bbox_wh
            #
            # Examples:
            #   biggest      = max(fibres, key=lambda f: f.area)
            #   img_arr      = np.array(orig_image)
            #   fibre_pixels = img_arr[fibres[0].mask]  # raw RGB pixels
            # ────────────────────────────────────────────────────────────────

            print()

    print(f"✓  Done — outputs in '{cfg['out_dir']}'")


if __name__ == "__main__":
    main()