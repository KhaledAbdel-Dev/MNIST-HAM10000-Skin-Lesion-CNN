# make_augmentation_examples.py
import argparse, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

def load_image(data_dir: Path, image_id: str) -> Image.Image:
    # HAM10000 stores images split across two folders; try jpg then png
    for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = data_dir / sub / f"{image_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Could not find {image_id} in {data_dir}")

def random_resized_crop(img: Image.Image, out_size=224, scale=(0.7, 1.0), ratio=(0.8, 1.25)):
    # simple deterministic-ish crop for demo: pick center crop at a random scale in range mid-point
    w, h = img.size
    target_area = w*h*((scale[0]+scale[1])/2.0)
    aspect = (ratio[0]+ratio[1])/2.0
    crop_w = int(round((target_area*aspect)**0.5))
    crop_h = int(round((target_area/aspect)**0.5))
    crop_w = min(crop_w, w); crop_h = min(crop_h, h)
    left = max(0, (w - crop_w)//2)
    top  = max(0, (h - crop_h)//2)
    img_cropped = img.crop((left, top, left+crop_w, top+crop_h))
    return img_cropped.resize((out_size, out_size), Image.BICUBIC)

def save_im(im: Image.Image, path: Path, title: str):
    plt.figure(figsize=(3.8,3.8))
    plt.imshow(im)
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../archive", help="Path to HAM10000 root (contains metadata + image folders)")
    ap.add_argument("--image_id", default="ISIC_0027419", help="e.g., ISIC_0027419")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(data_dir, args.image_id)

    # Original (resize to square canvas for consistent display)
    original_disp = img.copy()

    # Flipped (horizontal and small chance vertical)
    flipped = ImageOps.mirror(img)

    # Random Resized Crop to 224
    cropped = random_resized_crop(img, out_size=224)

    save_im(original_disp, out_dir / "aug_original.png", "Original")
    save_im(flipped, out_dir / "aug_flipped.png", "Flipped")
    save_im(cropped, out_dir / "aug_cropped.png", "Cropped (224×224)")

    print("Saved:",
          out_dir / "aug_original.png",
          out_dir / "aug_flipped.png",
          out_dir / "aug_cropped.png", sep="\n")

if __name__ == "__main__":
    main()
