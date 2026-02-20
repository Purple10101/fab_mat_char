"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260220

nn_seg_data.py
data objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

IMAGES_PATH = Path(__file__).resolve().parents[1] / "data" / "images"
SEGMAPS_PATH = Path(__file__).resolve().parents[1] / "data" / "segmaps"

class TorchSegDataset(Dataset):
    def __init__(self, data_dict, size=(256,256)):
        self.keys = list(data_dict.keys())
        self.data = data_dict
        self.size = size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        stem = self.keys[idx]
        image, segmap = self.data[stem]

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        segmap = TF.resize(segmap, self.size, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)  # [3,H,W], float32 0..1

        seg_np = np.array(segmap, dtype=np.int64)  # [H,W], instance IDs
        # Convert instance IDs -> binary foreground mask
        seg_np = (seg_np > 0).astype(np.int64)

        mask = torch.from_numpy(seg_np).long()     # [H,W], values 0 or 1

        return image, mask

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
            segmap = Image.open(seg_path).convert("L")

            ret_val[stem] = [image, segmap]

        return ret_val

    def _train_val_split(self):
        split_idx = int(len(self.total_dataset) * 0.8)

        keys = list(self.total_dataset.keys())

        train_keys = keys[:split_idx]
        val_keys = keys[split_idx:]

        train = {k: self.total_dataset[k] for k in train_keys}
        val = {k: self.total_dataset[k] for k in val_keys}

        return train, val

def build_deeplab(num_classes=2):
    model = deeplabv3_resnet50(weights="DEFAULT")  # pretrained backbone
    # Replace classifier head to output num_classes
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model
