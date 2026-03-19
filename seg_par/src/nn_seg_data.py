"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260220

nn_seg_data.py
data objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import medial_axis

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGES_PATH = Path(__file__).resolve().parents[1] / "data" / "images"
SEGMAPS_PATH = Path(__file__).resolve().parents[1] / "data" / "segmaps"
CACHE_PATH   = Path(__file__).resolve().parent / "cache"


@dataclass
class InstanceTargets:
    fg: torch.Tensor       # [H,W] long 0/1
    center: torch.Tensor   # [1,H,W] float
    offsets: torch.Tensor  # [2,H,W] float


def seg_rgb_to_instance_ids(seg_rgb: np.ndarray) -> np.ndarray:
    """
    seg_rgb: [H,W,3] uint8 colored instance map.
    Returns:
      inst: [H,W] int32, 0=background, 1..N = instances
    """
    if seg_rgb.ndim != 3 or seg_rgb.shape[2] != 3:
        raise ValueError(f"Expected seg_rgb [H,W,3], got shape={seg_rgb.shape}")

    h, w, _ = seg_rgb.shape
    flat = seg_rgb.reshape(-1, 3)

    # unique colors -> integer ids
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    inst = inv.reshape(h, w).astype(np.int32)

    # background assumed to be most frequent color
    bg_id = np.bincount(inv).argmax()
    inst = inst.copy()
    inst[inst == bg_id] = 0

    # re-label remaining ids to 1..N compactly
    mask = inst != 0
    inst_ids = np.unique(inst[mask])
    remap = {old: i + 1 for i, old in enumerate(inst_ids)}
    for old, new in remap.items():
        inst[inst == old] = new

    return inst


def instance_to_fg_boundary(inst: np.ndarray):
    """
    inst: [H,W] int32 with 0 background.
    Returns:
      fg: [H,W] uint8 {0,1}
      boundary: [H,W] uint8 {0,1} (1 on boundaries between instances)
    """
    fg = (inst > 0).astype(np.uint8)

    boundary = np.zeros_like(fg, dtype=np.uint8)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(inst, shift=(dy, dx), axis=(0, 1))
        boundary |= ((inst != shifted) & (inst > 0) & (shifted > 0)).astype(np.uint8)

    # particle-vs-background edges
    er = ndi.binary_erosion(fg, iterations=1)
    boundary |= (fg & (~er)).astype(np.uint8)

    return fg, boundary


def _instance_skeleton(mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    """
    Compute a dilated medial axis skeleton for a single binary instance mask.

    We use medial_axis over skeletonize because it produces cleaner, more
    connected results on thin (2px) structures — skeletonize can fragment.

    The skeleton is dilated slightly to give the model a more forgiving target;
    on a 2px wide particle the skeleton is essentially the particle itself so
    dilation is modest.

    mask:      [H,W] bool or uint8, single instance
    dilate_px: dilation radius in pixels
    returns:   [H,W] uint8 {0,1}
    """
    skel = medial_axis(mask).astype(np.uint8)

    if dilate_px > 0:
        struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
        skel = binary_dilation(skel, structure=struct).astype(np.uint8)

    return skel


def idmap_to_targets(id_map: np.ndarray, dilate_px: int = 2):
    """
    id_map: [H,W] int64, 0=background, 1..K instances
    returns:
      fg      [H,W] int64
      center  [1,H,W] float32  — dilated skeleton heatmap {0,1}
      offsets [2,H,W] float32  — (dx, dy) towards nearest skeleton pixel

    Skeleton-based targets replace the previous Gaussian centroid approach.
    For long thin particles (fibres), the medial axis is a far more natural
    representative than a centroid — no foreground pixel is ever far from the
    skeleton, keeping offset magnitudes small and the regression task tractable.

    Performance note: distance_transform_edt is called once on the full
    skeleton image rather than per-instance, which is significantly faster
    for dense images with many particles.
    """
    if id_map.ndim != 2:
        raise ValueError(f"Expected id_map [H,W], got shape={id_map.shape}")

    h, w = id_map.shape
    fg = (id_map > 0).astype(np.int64)

    center = np.zeros((h, w), dtype=np.float32)
    offsets = np.zeros((2, h, w), dtype=np.float32)

    ids = np.unique(id_map)
    ids = ids[ids != 0]

    # build the full skeleton image across all instances first,
    # then call distance_transform_edt once — much faster than per-instance
    full_skel = np.zeros((h, w), dtype=np.uint8)

    for inst_id in ids:
        mask = (id_map == inst_id)
        if mask.sum() < 5:
            continue

        skel = _instance_skeleton(mask, dilate_px=dilate_px)
        full_skel = np.maximum(full_skel, skel)

    # accumulate skeleton heatmap
    center = full_skel.astype(np.float32)

    # single distance transform over the full skeleton —
    # for each pixel, find coordinates of nearest skeleton pixel
    _, nearest = distance_transform_edt(1 - full_skel, return_indices=True)
    # nearest: [2,H,W] — nearest[0]=row, nearest[1]=col of closest skel px

    # compute offsets only for foreground pixels
    ys, xs = np.where(fg > 0)
    nearest_y = nearest[0, ys, xs]
    nearest_x = nearest[1, ys, xs]

    offsets[0, ys, xs] = nearest_x - xs   # dx
    offsets[1, ys, xs] = nearest_y - ys   # dy

    return fg, center[None, ...], offsets


class TorchSegDataset(Dataset):
    def __init__(self, data_dict, size=(256, 256), dilate_px=2):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.size = size
        self.dilate_px = dilate_px

        CACHE_PATH.mkdir(exist_ok=True)

    def _cache_path(self, stem: str) -> Path:
        # cache key encodes all parameters that affect the target computation
        # so changing size or dilate_px automatically busts the cache
        return CACHE_PATH / f"{stem}_{self.dilate_px}_{self.size[0]}x{self.size[1]}.npz"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        stem = self.keys[idx]
        image, segmap = self.data[stem]

        # resize
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        segmap = TF.resize(segmap, self.size, interpolation=InterpolationMode.NEAREST)

        # image -> tensor
        image_t = TF.to_tensor(image)  # [3,H,W], float32 0..1

        cache_file = self._cache_path(stem)

        if cache_file.exists():
            # targets already computed — just load from disk
            cached = np.load(cache_file)
            fg_np      = cached["fg"]
            center_np  = cached["center"]
            offsets_np = cached["offsets"]
        else:
            # first time seeing this sample — compute and cache to disk
            # so subsequent epochs just do a fast np.load
            seg_np = np.array(segmap, dtype=np.uint8)
            id_map = seg_rgb_to_instance_ids(seg_np).astype(np.int64)
            fg_np, center_np, offsets_np = idmap_to_targets(id_map, dilate_px=self.dilate_px)
            np.savez_compressed(cache_file, fg=fg_np, center=center_np, offsets=offsets_np)

        fg      = torch.from_numpy(fg_np).long()
        center  = torch.from_numpy(center_np).float()
        offsets = torch.from_numpy(offsets_np).float()

        return image_t, {"fg": fg, "center": center, "offsets": offsets}


class PSegDataset:
    def __init__(self):
        self.total_dataset = self._fetch_dataset()
        self.train, self.val = self._train_val_split()

        self.train_ds = TorchSegDataset(self.train)
        self.val_ds = TorchSegDataset(self.val)

    def get_subset_torch_dataset(self, keys=None, n=None, split=None):
        """
        Returns a TorchSegDataset containing a subset of the dataset.

        Args:
            keys (list[str] | None): specific keys to fetch
            n (int | None): randomly sample n examples
            split (str | None): 'train' or 'val' to pull from split

        Returns:
            TorchSegDataset
        """
        if split == "train":
            source = self.train
        elif split == "val":
            source = self.val
        else:
            source = self.total_dataset

        all_keys = list(source.keys())

        if keys is not None:
            selected_keys = keys
        else:
            if n is not None:
                selected_keys = random.sample(all_keys, min(n, len(all_keys)))
            else:
                selected_keys = all_keys

        subset_dict = {k: source[k] for k in selected_keys}

        return TorchSegDataset(subset_dict)

    def _fetch_dataset(self):
        ret_val = {}
        for img_path in IMAGES_PATH.iterdir():
            if not img_path.is_file():
                continue

            stem = img_path.stem
            seg_path = SEGMAPS_PATH / f"{stem}.png"
            if not seg_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")

            # IMPORTANT: ensure segmap is RGB so color->instance works reliably
            segmap = Image.open(seg_path).convert("RGB")

            ret_val[stem] = [image, segmap]

        return ret_val

    def _train_val_split(self):
        split_idx = int(len(self.total_dataset) * 0.8)

        keys = list(self.total_dataset.keys())
        random.shuffle(keys)

        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]

        train = {k: self.total_dataset[k] for k in train_keys}
        val = {k: self.total_dataset[k] for k in val_keys}

        return train, val


def build_deeplab_instance(num_out: int = 4):
    """
    Outputs 4 channels:
      0: fg logits
      1: center/skeleton logits
      2: offset dx (regression)
      3: offset dy (regression)
    """
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_out, kernel_size=1)
    return model


def main():
    # -------------------------------------------------------------------------
    # Keys to visualise — swap notable_keys for any dict of {category: [keys]}
    # or replace with e.g. {"All": list(dataset.total_dataset.keys())}
    # -------------------------------------------------------------------------
    notable_keys = {
        "Extra Interesting": [
            "34f4fb273d",
            "59424b060a",
            "62a54f335d",
        ],
        "Long and thin": [
            "08549eb98f",
            "0e41d62d5d",
            "2387be5eaf",
            "2897b777fe",
            "2bc87a8698",
            "2bf4aa0195",
            "2d6f268052",
            "3f97a9e821",
            "5f42a8d4a9",
            "707120d0f5",
        ],
        "Overlapping": [
            "208b16bbb7",
            "40afb05b44",
        ],
        "Other": [
            "2807b90ea9",
        ],
    }

    # -------------------------------------------------------------------------
    # Plot settings — change skeleton_alpha to control overlay opacity (0..1)
    # dilate_px must match what your dataset was built with
    # -------------------------------------------------------------------------
    skeleton_alpha = 0.6
    dilate_px = 2

    dataset = PSegDataset()

    for category, keys in notable_keys.items():
        n = len(keys)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        # two panels per sample: instance id map | skeleton overlay
        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
        fig.suptitle(category, fontsize=14, fontweight="bold")
        axes = np.array(axes).reshape(rows * 2, cols)

        for i, key in enumerate(keys):
            row = (i // cols) * 2
            col = i % cols

            img, seg = dataset.total_dataset[key]
            seg_np = np.array(seg, dtype=np.uint8)
            id_map = seg_rgb_to_instance_ids(seg_np).astype(np.int64)

            # compute full skeleton across all instances in this image
            ids = np.unique(id_map)
            ids = ids[ids != 0]
            full_skel = np.zeros(id_map.shape, dtype=np.uint8)
            for inst_id in ids:
                mask = (id_map == inst_id)
                if mask.sum() < 5:
                    continue
                full_skel = np.maximum(full_skel, _instance_skeleton(mask, dilate_px=dilate_px))

            # panel 1: instance id map alone
            axes[row, col].imshow(id_map, cmap="viridis")
            axes[row, col].set_title(f"{key}\n(n={len(ids)})", fontsize=8)
            axes[row, col].axis("off")

            # panel 2: instance id map with skeleton overlaid
            axes[row + 1, col].imshow(id_map, cmap="viridis")
            axes[row + 1, col].imshow(
                np.ma.masked_where(full_skel == 0, full_skel),  # only show skeleton pixels
                cmap="autumn",
                alpha=skeleton_alpha,
            )
            axes[row + 1, col].set_title(f"{key} + skeleton", fontsize=8)
            axes[row + 1, col].axis("off")

        # hide unused subplot pairs
        for j in range(len(keys), rows * cols):
            row = (j // cols) * 2
            col = j % cols
            axes[row, col].axis("off")
            axes[row + 1, col].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()