"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260320

auto_fibre.py
automated fibre dataset creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Configuration defaults

DEFAULT_CONFIG = {
    "image_size": (512, 512),        # (width, height) in pixels
    "n_images": 500,                 # number of image/mask pairs to generate
    "fibres_per_image": (5, 20),     # (min, max) fibres per image
    "fibre_width": (8, 16),          # (min, max) fibre thickness in pixels
    "fibre_length": (80, 350),       # (min, max) fibre length in pixels
    "curved_fraction": 0.1,          # fraction of fibres that are curved
    "curvature": (0.05, 0.1),        # (min, max) curvature amplitude
    "bg_color": (240, 235, 225),     # background colour (RGB)
    "bg_noise_std": 8,               # Gaussian noise std on background
    "fibre_noise_std": 6,            # Gaussian noise std on fibres
    "fibre_colors": [                # pool of colours for the RGB image
        (180, 80,  60),
        (60,  120, 200),
        (80,  170, 90),
        (200, 150, 50),
        (140, 70,  180),
        (50,  180, 170),
        (220, 100, 140),
        (100, 100, 100),
    ],
    "seed": 42,
    "split_ratio": (0.7, 0.15, 0.15),
}


def generate_instance_palette(n, rng):
    """
    Generate n visually distinct RGB colours for instance masks.
    Spreads hues evenly around the colour wheel so adjacent fibres
    are easy to distinguish. All colours are bright enough to never
    be confused with the black (0,0,0) background.
    """
    colors = []
    for i in range(n):
        hue = (i / max(n, 1) + rng.uniform(-0.03, 0.03)) % 1.0
        h = hue * 6
        sector = int(h)
        f = h - sector
        r, g, b = [
            (1.0, 0.0,   1.0-f),
            (1.0-f, 1.0, 0.0  ),
            (0.0,   1.0, f    ),
            (0.0,   1.0-f, 1.0),
            (f,     0.0, 1.0  ),
            (1.0,   f,   0.0  ),
        ][sector % 6]
        scale = rng.uniform(0.55, 1.0)
        rgb = (
            int(100 + r * 155 * scale),
            int(100 + g * 155 * scale),
            int(100 + b * 155 * scale),
        )
        colors.append(rgb)
    return colors

def make_straight_fibre(cx, cy, angle, length):
    half = length / 2
    return [
        (cx - math.cos(angle) * half, cy - math.sin(angle) * half),
        (cx + math.cos(angle) * half, cy + math.sin(angle) * half),
    ]

def make_curved_fibre(cx, cy, angle, length, amplitude, n_points=40):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ts = np.linspace(-length / 2, length / 2, n_points)
    freq = 2 * math.pi / length
    offsets = amplitude * length * np.sin(freq * ts)
    xs = cx + ts * cos_a - offsets * sin_a
    ys = cy + ts * sin_a + offsets * cos_a
    return list(zip(xs.tolist(), ys.tolist()))

def draw_fibre(draw, points, width, color):
    flat = [c for pt in points for c in pt]
    draw.line(flat, fill=color, width=width, joint="curve")
    r = max(width // 2, 1)
    for x, y in points[::max(1, len(points) // 8)]:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

def generate_sample(cfg, rng):
    W, H = cfg["image_size"]

    # RGB image w a noisy background
    bg = np.full((H, W, 3), cfg["bg_color"], dtype=np.float32)
    bg += rng.normal(0, cfg["bg_noise_std"], bg.shape)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    image = Image.fromarray(bg, "RGB")
    img_draw = ImageDraw.Draw(image)

    # Instance mask
    mask = Image.new("RGB", (W, H), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)

    n_fibres = rng.randint(*cfg["fibres_per_image"])
    instance_colors = generate_instance_palette(n_fibres, rng)

    fibre_meta = []

    for fi in range(n_fibres):
        cx     = rng.uniform(0, W)
        cy     = rng.uniform(0, H)
        angle  = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(*cfg["fibre_length"])
        width  = int(rng.uniform(*cfg["fibre_width"]))
        curved = rng.random() < cfg["curved_fraction"]

        img_color  = cfg["fibre_colors"][rng.randint(0, len(cfg["fibre_colors"]))]
        inst_color = instance_colors[fi]

        if curved:
            amp    = rng.uniform(*cfg["curvature"])
            points = make_curved_fibre(cx, cy, angle, length, amp)
        else:
            points = make_straight_fibre(cx, cy, angle, length)
            amp    = 0.0

        draw_fibre(img_draw, points, width, img_color)
        draw_fibre(mask_draw, points, width, inst_color)

        fibre_meta.append({
            "instance_id": fi,
            "mask_rgb": list(inst_color),
            "curved": curved,
            "amplitude": round(float(amp), 4),
            "length": round(float(length), 2),
            "width": width,
        })

    # mild blur n noise on RGB image
    image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_arr = np.array(image, dtype=np.float32)
    img_arr += rng.normal(0, cfg["fibre_noise_std"], img_arr.shape)
    image = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8), "RGB")

    return image, mask, fibre_meta

def build_dataset(cfg, out_dir):
    rng = np.random.RandomState(cfg["seed"]) if cfg["seed"] is not None else np.random.RandomState()

    n = cfg["n_images"]
    tr, va, te = cfg["split_ratio"]
    n_train = int(n * tr)
    n_val   = int(n * va)
    n_test  = n - n_train - n_val
    split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    manifest = []
    counters  = {"train": 0, "val": 0, "test": 0}

    for i, split in enumerate(split_labels):
        idx = counters[split]
        counters[split] += 1

        img_rel  = os.path.join(split, "images", f"fibre_{idx:04d}.png")
        mask_rel = os.path.join(split, "masks",  f"fibre_{idx:04d}_mask.png")
        img_path  = os.path.join(out_dir, img_rel)
        mask_path = os.path.join(out_dir, mask_rel)

        os.makedirs(os.path.dirname(img_path),  exist_ok=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        image, mask, fibre_meta = generate_sample(cfg, rng)
        image.save(img_path)
        mask.save(mask_path)

        manifest.append({
            "image": img_rel,
            "mask": mask_rel,
            "split": split,
            "n_fibres": len(fibre_meta),
            "fibres": fibre_meta,
        })

        print(f"[{i+1:>4}/{n}] {split:5s}  fibres={len(fibre_meta):2d}  {img_rel}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"config": cfg, "samples": manifest}, f, indent=2)

    print(f"\n✓  Dataset saved to: {out_dir}")
    print(f"   train={n_train}  val={n_val}  test={n_test}")
    print(f"   Masks: RGB PNG — black bg, unique colour per fibre instance")
    print(f"   Manifest: {manifest_path}  (includes instance_id → mask_rgb lookup)")
    print()
    print("To extract per-instance binary masks in PyTorch/numpy:")
    print("   import numpy as np; from PIL import Image")
    print("   mask = np.array(Image.open('fibre_0000_mask.png'))  # H x W x 3")
    print("   rgb  = [180, 95, 210]   # from manifest fibres[i]['mask_rgb']")
    print("   inst = np.all(mask == rgb, axis=-1)                 # H x W bool")

def parse_args():
    p = argparse.ArgumentParser(description="Synthetic fibre instance segmentation dataset")
    p.add_argument("--out_dir",         default="./fibre_dataset")
    p.add_argument("--n_images",        type=int,   default=DEFAULT_CONFIG["n_images"])
    p.add_argument("--img_size",        type=int,   default=512)
    p.add_argument("--min_fibres",      type=int,   default=5)
    p.add_argument("--max_fibres",      type=int,   default=20)
    p.add_argument("--curved_fraction", type=float, default=0.6)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["n_images"]         = args.n_images
    cfg["image_size"]       = (args.img_size, args.img_size)
    cfg["fibres_per_image"] = (args.min_fibres, args.max_fibres)
    cfg["curved_fraction"]  = args.curved_fraction
    cfg["seed"]             = args.seed
    build_dataset(cfg, args.out_dir)