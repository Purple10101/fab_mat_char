"""
infer.py — Fibre instance segmentation inference + GT comparison
================================================================
No command line arguments needed. Edit the CONFIG block below,
then just run:

    python infer.py

Output per sample (saved to CONFIG["out_dir"]):
    <name>_overlay.png     — image with coloured predicted instance masks
    <name>_pred.json       — predictions (scores, boxes, RLE masks)
    <name>_comparison.png  — 4-panel GT vs prediction figure

Requirements:
    pip install torch torchvision matplotlib pycocotools pillow numpy
"""

import json
import os
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from pycocotools import mask as coco_mask_utils

from model import build_model


# ════════════════════════════════════════════════════════════
#  CONFIG — edit this block, then run `python infer.py`
# ════════════════════════════════════════════════════════════

CONFIG = {
    # Path to the saved checkpoint from train.py
    "checkpoint":   "runs/fibre_maskrcnn/best.pth",

    # Root of the fibre dataset (must contain manifest.json)
    "data_dir":     "./fibre_dataset",

    # Which split to sample from: "train", "val", "test", or "all"
    "split":        "val",

    # How many random samples to compare (0 = run on every sample in the split)
    "n_samples":    5,

    # Random seed for sample selection (set to None for a different pick each run)
    "seed":         42,

    # Where to write output files
    "out_dir":      "./predictions",

    # Model input size (must match what you trained with)
    "image_size":   512,

    # Detections below this confidence are discarded
    "score_thresh": 0.4,

    # Backbone variant — must match the checkpoint
    "backbone":     "maskrcnn_resnet50_fpn_v2",

    # Mask overlay transparency (0 = invisible, 1 = opaque)
    "mask_alpha":   0.45,
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


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_path, image_size):
    img         = Image.open(image_path).convert("RGB")
    orig_size   = img.size                                       # (W, H)
    img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    tensor      = TF.to_tensor(img_resized)
    tensor      = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor, img, orig_size


# ─── Overlay helpers ──────────────────────────────────────────────────────────

def build_instance_overlay(base_image, masks_binary, scores=None,
                            boxes=None, alpha=0.45, draw_boxes=True):
    W, H    = base_image.size
    overlay = np.array(base_image).astype(np.float32)

    for i, mask_bin in enumerate(masks_binary):
        color       = PALETTE[i % len(PALETTE)]
        mask_pil    = Image.fromarray((mask_bin * 255).astype(np.uint8))
        mask_resized = np.array(mask_pil.resize((W, H), Image.NEAREST)) > 127
        for c in range(3):
            overlay[:, :, c][mask_resized] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][mask_resized]
            )

    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    if draw_boxes and boxes is not None and scores is not None:
        draw = ImageDraw.Draw(result)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            color      = PALETTE[i % len(PALETTE)]
            x0, y0, x1, y1 = box
            sx, sy     = W / 512, H / 512
            draw.rectangle([x0*sx, y0*sy, x1*sx, y1*sy], outline=color, width=2)
            draw.text((x0*sx + 3, y0*sy + 2), f"{score:.2f}", fill=color)

    return result, len(masks_binary)


def decode_gt_masks(mask_rgb_path, fibre_meta):
    mask_rgb = np.array(Image.open(mask_rgb_path).convert("RGB"), dtype=np.uint8)
    masks = []
    for fibre in fibre_meta:
        rgb    = np.array(fibre["mask_rgb"], dtype=np.uint8)
        binary = np.all(mask_rgb == rgb, axis=-1).astype(np.uint8)
        if binary.sum() > 0:
            masks.append(binary)
    return masks


def predictions_to_json(pred_masks, pred_scores, pred_boxes, pred_labels):
    out = []
    for i in range(len(pred_scores)):
        binary        = (pred_masks[i, 0] > 0.5).astype(np.uint8)
        rle           = coco_mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")
        out.append({
            "instance_id": i,
            "score":       float(pred_scores[i]),
            "label":       int(pred_labels[i]),
            "box_xyxy":    [float(v) for v in pred_boxes[i]],
            "mask_rle":    rle,
        })
    return out


# ─── Comparison figure ────────────────────────────────────────────────────────

def make_comparison_figure(orig_image, gt_masks, pred_masks, pred_scores,
                            pred_boxes, out_path, sample_info=None, alpha=0.45):
    W, H = orig_image.size

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")

    title_kw = dict(color="white",   fontsize=11, fontweight="bold", pad=8)
    label_kw = dict(color="#aaaaaa", fontsize=9)

    # Panel 0 — original
    axes[0].imshow(orig_image)
    axes[0].set_title("Original image", **title_kw)
    if sample_info:
        axes[0].set_xlabel(sample_info, **label_kw)

    # Panel 1 — ground truth
    if gt_masks:
        gt_overlay, n_gt = build_instance_overlay(
            orig_image, gt_masks, draw_boxes=False, alpha=alpha)
    else:
        gt_overlay, n_gt = orig_image, 0
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f"Ground truth  ({n_gt} fibres)", **title_kw)

    # Panel 2 — predictions
    pred_binary = [(m[0] > 0.5).astype(np.uint8) for m in pred_masks]
    if pred_binary:
        pred_overlay, n_pred = build_instance_overlay(
            orig_image, pred_binary,
            scores=pred_scores, boxes=pred_boxes,
            draw_boxes=True, alpha=alpha)
    else:
        pred_overlay, n_pred = orig_image, 0
    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Predicted  ({n_pred} fibres)", **title_kw)

    # Panel 3 — pixel diff map
    def union(mask_list):
        acc = np.zeros((H, W), dtype=np.float32)
        for m in mask_list:
            r = np.array(
                Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
            ) / 255.0
            acc = np.maximum(acc, r)
        return acc > 0.5

    gt_union   = union(gt_masks)
    pred_union = union(pred_binary)
    tp = gt_union  &  pred_union
    fn = gt_union  & ~pred_union
    fp = ~gt_union &  pred_union

    diff = np.zeros((H, W, 3), dtype=np.float32)
    diff[tp] = [0.20, 0.85, 0.40]
    diff[fn] = [0.90, 0.25, 0.25]
    diff[fp] = [0.30, 0.50, 1.00]

    grey    = np.array(orig_image.convert("L").resize((W, H)),
                       dtype=np.float32) / 255.0
    bg      = np.stack([grey * 0.35] * 3, axis=-1)
    hit     = (tp | fn | fp).astype(np.float32)[..., None]
    axes[3].imshow(np.clip(diff * hit + bg * (1 - hit), 0, 1))
    axes[3].set_title("Coverage diff", **title_kw)

    n_tp  = int(tp.sum())
    prec  = n_tp / max(int(pred_union.sum()), 1)
    rec   = n_tp / max(int(gt_union.sum()),   1)
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
    print(f"  comparison → {os.path.basename(out_path)}"
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Split  : {cfg['split']}   n_samples={cfg['n_samples'] or 'all'}")

    # Load model
    model = build_model(cfg["backbone"], pretrained=False)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded : {cfg['checkpoint']}")

    # Build GT lookup and select samples
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
            stem   = os.path.splitext(os.path.basename(image_path))[0]
            tensor, orig_image, _ = preprocess(image_path, cfg["image_size"])

            output = model([tensor.to(device)])[0]
            preds  = {k: v.cpu().numpy() for k, v in output.items()}

            # Filter by score threshold
            keep        = preds["scores"] >= cfg["score_thresh"]
            pred_masks  = preds["masks"][keep]
            pred_scores = preds["scores"][keep]
            pred_boxes  = preds["boxes"][keep]
            pred_labels = preds["labels"][keep]

            # Overlay PNG
            pred_binary = [(m[0] > 0.5).astype(np.uint8) for m in pred_masks]
            overlay, n_det = build_instance_overlay(
                orig_image, pred_binary,
                scores=pred_scores, boxes=pred_boxes,
                alpha=cfg["mask_alpha"], draw_boxes=True,
            )
            overlay.save(os.path.join(cfg["out_dir"], f"{stem}_overlay.png"))

            # Prediction JSON
            with open(os.path.join(cfg["out_dir"], f"{stem}_pred.json"), "w") as f:
                json.dump(
                    predictions_to_json(pred_masks, pred_scores,
                                        pred_boxes, pred_labels),
                    f, indent=2,
                )

            print(f"[{stem}]  detected={n_det}")

            # Comparison figure
            gt_info  = gt_lookup[stem]
            gt_masks = decode_gt_masks(gt_info["mask_path"], gt_info["fibres"])
            make_comparison_figure(
                orig_image, gt_masks,
                pred_masks, pred_scores, pred_boxes,
                out_path=os.path.join(cfg["out_dir"], f"{stem}_comparison.png"),
                sample_info=f"split={gt_info['split']}  GT fibres={len(gt_masks)}",
                alpha=cfg["mask_alpha"],
            )

    print(f"\n✓  Done — outputs in '{cfg['out_dir']}'")


if __name__ == "__main__":
    main()
