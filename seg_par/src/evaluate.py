"""
evaluate.py — COCO-style evaluation for fibre instance segmentation
====================================================================
Returns AP (averaged over IoU thresholds) for both masks and boxes,
matching the standard COCO evaluation protocol used by detectron2, MMDet, etc.
"""

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils


# ─── Prediction collector ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device, score_thresh=0.3):
    """
    Run inference over a DataLoader and collect predictions + ground truths
    in COCO-annotation format for pycocotools evaluation.
    """
    model.eval()

    gt_annotations   = []
    pred_annotations = []
    images_info      = []
    ann_id = 1

    for batch_idx, (images, targets) in enumerate(loader):
        images_gpu = [img.to(device) for img in images]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images_gpu)

        for i, (output, target) in enumerate(zip(outputs, targets)):
            image_id = int(target["image_id"].item())
            H, W     = images[i].shape[-2], images[i].shape[-1]

            images_info.append({"id": image_id, "height": H, "width": W})

            # ── Ground truth ──
            gt_masks  = target["masks"].cpu().numpy()   # N×H×W uint8
            gt_labels = target["labels"].cpu().numpy()  # N

            for j in range(len(gt_masks)):
                rle = coco_mask_utils.encode(
                    np.asfortranarray(gt_masks[j])
                )
                rle["counts"] = rle["counts"].decode("utf-8")
                bbox = coco_mask_utils.toBbox(rle).tolist()
                area = float(coco_mask_utils.area(rle))
                gt_annotations.append({
                    "id":           ann_id,
                    "image_id":     image_id,
                    "category_id":  int(gt_labels[j]),
                    "segmentation": rle,
                    "bbox":         bbox,
                    "area":         area,
                    "iscrowd":      0,
                })
                ann_id += 1

            # ── Predictions ──
            pred_masks  = output["masks"].cpu().numpy()   # N×1×H×W float
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_boxes  = output["boxes"].cpu().numpy()

            for j in range(len(pred_scores)):
                if pred_scores[j] < score_thresh:
                    continue
                # threshold mask at 0.5
                binary_mask = (pred_masks[j, 0] > 0.5).astype(np.uint8)
                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                pred_annotations.append({
                    "image_id":     image_id,
                    "category_id":  int(pred_labels[j]),
                    "segmentation": rle,
                    "bbox":         pred_boxes[j].tolist(),
                    "score":        float(pred_scores[j]),
                })

    return images_info, gt_annotations, pred_annotations


# ─── COCO eval runner ─────────────────────────────────────────────────────────

def evaluate_coco(model, loader, device, score_thresh=0.3):
    """
    Run full COCO evaluation and return a dict of AP metrics.

    Returns:
        {
          "AP_mask":   float,   # mask AP  @ IoU 0.50:0.95
          "AP50_mask": float,   # mask AP  @ IoU 0.50
          "AP75_mask": float,   # mask AP  @ IoU 0.75
          "AP_box":    float,   # box  AP  @ IoU 0.50:0.95
          "AP50_box":  float,
          "AP75_box":  float,
        }
    """
    images_info, gt_anns, pred_anns = collect_predictions(
        model, loader, device, score_thresh
    )

    if len(gt_anns) == 0:
        print("  Warning: no ground-truth annotations found in this split.")
        return {}

    # ── Build COCO GT object in-memory ──
    coco_gt = COCO()
    coco_gt.dataset = {
        "images":     images_info,
        "annotations": gt_anns,
        "categories":  [{"id": 1, "name": "fibre"}],
    }
    coco_gt.createIndex()

    if len(pred_anns) == 0:
        print("  Warning: no predictions above score threshold.")
        return {"AP_mask": 0.0, "AP50_mask": 0.0, "AP75_mask": 0.0,
                "AP_box":  0.0, "AP50_box":  0.0, "AP75_box":  0.0}

    coco_dt = coco_gt.loadRes(pred_anns)

    metrics = {}
    for iou_type, prefix in [("segm", "mask"), ("bbox", "box")]:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats   # array of 12 standard COCO metrics
        metrics[f"AP_{prefix}"]   = float(stats[0])
        metrics[f"AP50_{prefix}"] = float(stats[1])
        metrics[f"AP75_{prefix}"] = float(stats[2])

    return metrics
