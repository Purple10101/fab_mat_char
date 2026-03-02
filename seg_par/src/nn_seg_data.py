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
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

IMAGES_PATH = Path(__file__).resolve().parents[1] / "data" / "images"
SEGMAPS_PATH = Path(__file__).resolve().parents[1] / "data" / "segmaps"


from dataclasses import dataclass

@dataclass
class InstanceTargets:
    fg: torch.Tensor       # [H,W] long 0/1
    center: torch.Tensor   # [1,H,W] float
    offsets: torch.Tensor  # [2,H,W] float


def _gaussian_2d(h, w, cx, cy, sigma):
    # small helper to paint a gaussian peak
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return g


def idmap_to_targets(id_map: np.ndarray, sigma: float = 3.0):
    """
    id_map: [H,W] int64, 0=background, 1..K instances
    returns fg [H,W], center [1,H,W], offsets [2,H,W]
    """
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
    def __init__(self, data_dict, size=(256,256), sigma=3.0):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.size = size
        self.sigma = sigma

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        stem = self.keys[idx]
        image, segmap = self.data[stem]

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        segmap = TF.resize(segmap, self.size, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)  # [3,H,W], float32 0..1

        # IMPORTANT: keep IDs, do NOT binarise
        id_map = np.array(segmap, dtype=np.int64)  # [H,W]

        fg_np, center_np, offsets_np = idmap_to_targets(id_map, sigma=self.sigma)

        fg = torch.from_numpy(fg_np).long()                 # [H,W]
        center = torch.from_numpy(center_np).float()        # [1,H,W]
        offsets = torch.from_numpy(offsets_np).float()      # [2,H,W]

        return image, {"fg": fg, "center": center, "offsets": offsets}

class PSegDataset:
    def __init__(self):
        self.total_dataset = self._fetch_dataset()
        self.train, self.val = self._train_val_split()

        self.train_ds = TorchSegDataset(self.train)
        self.val_ds   = TorchSegDataset(self.val)

    def _fetch_dataset(self):

        ret_val = {}
        for img_path in IMAGES_PATH.iterdir():
            if not img_path.is_file():
                continue

            stem = img_path.stem
            seg_path = SEGMAPS_PATH / f"{stem}.png"

            image = Image.open(img_path).convert("RGB")
            segmap = Image.open(seg_path)

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

def build_deeplab_instance(num_out=5):
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_out, kernel_size=1)
    return model
