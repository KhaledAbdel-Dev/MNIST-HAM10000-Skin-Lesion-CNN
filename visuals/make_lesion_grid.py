#!/usr/bin/env python3
# make_lesion_grid.py
# Create a grid of sample HAM10000 lesion images: 1..N per class
# Saves figures/lesion_grid.png (and optional PDF)

import argparse, os, random
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def find_image(data_dir: Path, image_id: str):
    # HAM10000 stores images across two dirs; try common extensions
    for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = data_dir / sub / f"{image_id}{ext}"
            if p.exists():
                return p
    return None

def center_crop_to_square(im: Image.Image):
    w, h = im.size
    if w == h:
        return im
    if w > h:
        left = (w - h) // 2
        return im.crop((left, 0, left + h, h))
    else:
        top = (h - w) // 2
        return im.crop((0, top, w, top + w))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to HAM10000 root (contains metadata and image folders)")
    ap.add_argument("--metadata", default="HAM10000_metadata.csv", help="Metadata CSV filename (default: HAM10000_metadata.csv)")
    ap.add_argument("--n_per_class", type=int, default=1, help="Number of examples per class (default: 1)")
    ap.add_argument("--size", type=int, default=256, help="Tile size in pixels (default: 256)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--save_pdf", action="store_true", help="Also save a PDF alongside the PNG")
    args = ap.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    meta_path = data_dir / args.metadata
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find metadata CSV at {meta_path}")

    df = pd.read_csv(meta_path)
    df = df[df["dx"].isin(CLASSES)].copy()

    # sample rows per class
    samples = []
    for c in CLASSES:
        sub = df[df["dx"] == c]
        if len(sub) == 0:
            raise RuntimeError(f"No rows found for class {c}")
        take = min(args.n_per_class, len(sub))
        sampled = sub.sample(n=take, random_state=args.seed)
        for _, row in sampled.iterrows():
            samples.append((c, row["image_id"]))

    # create grid: rows = classes, cols = n_per_class
    rows = len(CLASSES)
    cols = args.n_per_class
    figsize = (cols * 2.6, rows * 2.6)  # scale for readability

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # fill grid
    idx = 0
    for r, c_name in enumerate(CLASSES):
        for col in range(cols):
            ax = axes[r][col]
            ax.axis("off")
            # If not enough samples to fill, pad with blank
            if idx >= len(samples) or samples[idx][0] != c_name:
                ax.set_facecolor("#f5f5f5")
                continue
            _, image_id = samples[idx]
            idx += 1

            p = find_image(data_dir, image_id)
            if p is None:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, f"Missing:\n{image_id}", ha="center", va="center", fontsize=9)
                continue

            im = Image.open(p).convert("RGB")
            im = center_crop_to_square(im).resize((args.size, args.size), Image.BICUBIC)

            ax.imshow(im)
            # title on leftmost tile only, to reduce clutter
            if col == 0:
                ax.set_title(c_name, fontsize=12, loc="left", pad=6)

    fig.tight_layout()
    outdir = Path("figures"); outdir.mkdir(parents=True, exist_ok=True)

    # filename includes N (images per class) for clarity
    fname = outdir / f"lesion_grid_{args.n_per_class}x_per_class.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    if args.save_pdf:
        plt.savefig(outdir / f"lesion_grid_{args.n_per_class}x_per_class.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {fname}")
    if args.save_pdf:
        print(f"Saved {outdir / f'lesion_grid_{args.n_per_class}x_per_class.pdf'}")

if __name__ == "__main__":
    main()
