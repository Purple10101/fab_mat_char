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

import numpy as np
from scipy import ndimage as ndi  # <-- FIX: needed for binary_erosion

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGES_PATH = Path(__file__).resolve().parents[1] / "data" / "images"
SEGMAPS_PATH = Path(__file__).resolve().parents[1] / "data" / "segmaps"


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

def _gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Paint a gaussian peak centered at (cx, cy)."""
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return g

def idmap_to_targets(id_map: np.ndarray, sigma: float = 3.0):
    """
    id_map: [H,W] int64, 0=background, 1..K instances
    returns:
      fg      [H,W] int64
      center  [1,H,W] float32  (0..1)
      offsets [2,H,W] float32  (dx, dy) towards instance centroid
    """
    if id_map.ndim != 2:
        raise ValueError(f"Expected id_map [H,W], got shape={id_map.shape}")

    h, w = id_map.shape
    fg = (id_map > 0).astype(np.int64)

    center = np.zeros((h, w), dtype=np.float32)
    offsets = np.zeros((2, h, w), dtype=np.float32)

    ids = np.unique(id_map)
    ids = ids[ids != 0]

    for inst_id in ids:
        ys, xs = np.where(id_map == inst_id)
        if len(xs) < 5:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())

        # center heatmap peak (gaussian)
        g = _gaussian_2d(h, w, cx, cy, sigma).astype(np.float32)
        center = np.maximum(center, g)  # max over instances

        # offsets: for pixels in this instance, vector points to center
        offsets[0, ys, xs] = cx - xs
        offsets[1, ys, xs] = cy - ys

    return fg, center[None, ...], offsets


class TorchSegDataset(Dataset):
    def __init__(self, data_dict, size=(256, 256), sigma=3.0):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.size = size
        self.sigma = sigma

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

        # segmap (RGB) -> instance id map
        seg_np = np.array(segmap, dtype=np.uint8)        # [H,W,3]
        id_map = seg_rgb_to_instance_ids(seg_np).astype(np.int64)  # [H,W]

        fg_np, center_np, offsets_np = idmap_to_targets(id_map, sigma=self.sigma)

        fg = torch.from_numpy(fg_np).long()                 # [H,W]
        center = torch.from_numpy(center_np).float()        # [1,H,W]
        offsets = torch.from_numpy(offsets_np).float()      # [2,H,W]

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
        import random

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
      1: center logits
      2: offset dx (regression)
      3: offset dy (regression)
    """
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_out, kernel_size=1)
    return model


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = PSegDataset()
    keys = list(dataset.total_dataset.keys())

    batch_size = 12
    rows, cols = 4, 4  # 4x4 grid (we'll leave last 4 blank if needed)

    for batch_start in range(0, len(keys), batch_size):
        batch_keys = keys[batch_start: batch_start + batch_size]

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten()

        for i, key in enumerate(batch_keys):
            img, seg = dataset.total_dataset[key]

            seg_np = np.array(seg, dtype=np.uint8)
            id_map = seg_rgb_to_instance_ids(seg_np)

            axes[i].imshow(id_map)
            axes[i].set_title(f"{key}\n(n={len(np.unique(id_map)) - 1})", fontsize=8)
            axes[i].axis("off")

        # Hide unused subplots (since 4x4=16 but we show 12)
        for j in range(len(batch_keys), rows * cols):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """
    Notable examples in dataset:
    
    Extra Interesting:
    "34f4fb273d"
    "59424b060a"
    "62a54f335d"
    
    
    Long and thin:
    "08549eb98f"
    "0e41d62d5d"
    "2387be5eaf"
    "2897b777fe"
    "2bc87a8698"
    "2bf4aa0195"
    "2d6f268052"
    "3f97a9e821"
    "5f42a8d4a9"
    "707120d0f5"
    
    Overlapping:
    "208b16bbb7"
    "40afb05b44"
    
    Other:
    "2807b90ea9"
    
    """
    main()