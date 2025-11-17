
import argparse, json, csv, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torch.nn.functional as F
import timm

CLASS_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6
}
IDX2CLASS = {v: k for k, v in CLASS_MAP.items()}

def load_metadata(csv_path: str):
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows = [r for r in rows if r["dx"] in CLASS_MAP]
    return rows

def train_val_split(rows, val_frac=0.2, seed=42):
    # lesion-level split (use lesion_id since patient_id not in CSV)
    rng = random.Random(seed)
    lesion_to_rows = {}
    for r in rows:
        lid = r.get("lesion_id", r.get("image_id", "unknown"))
        lesion_to_rows.setdefault(lid, []).append(r)
    lesions = list(lesion_to_rows.keys())
    rng.shuffle(lesions)
    cut = int(len(lesions) * (1 - val_frac))
    train_lesions = set(lesions[:cut])
    train_rows, val_rows = [], []
    for lid, items in lesion_to_rows.items():
        if lid in train_lesions:
            train_rows.extend(items)
        else:
            val_rows.extend(items)
    return train_rows, val_rows

class HamDataset(Dataset):
    def __init__(self, rows, img_dirs, transform=None):
        self.rows = rows
        self.img_dirs = img_dirs
        self.transform = transform
    def __len__(self): return len(self.rows)
    def _resolve_path(self, image_id):
        for d in self.img_dirs:
            p = Path(d) / f"{image_id}.jpg"
            if p.exists(): return str(p)
        for d in self.img_dirs:
            p = Path(d) / f"{image_id}.png"
            if p.exists(): return str(p)
        raise FileNotFoundError(image_id)
    def __getitem__(self, idx):
        r = self.rows[idx]
        image_id = r["image_id"]
        y = CLASS_MAP[r["dx"]]
        path = self._resolve_path(image_id)
        img = read_image(path).float() / 255.0
        if img.shape[0] == 1: img = img.repeat(3,1,1)
        if img.shape[0] == 4: img = img[:3]
        if self.transform: img = self.transform(img)
        return img, y, image_id

def build_transforms(img_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.7,1.0), ratio=(0.8,1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

@torch.no_grad()
def evaluate(model, loader, device, num_classes, tta=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        if tta:
            logits1 = model(x)
            x2 = torch.flip(x, dims=[3])
            logits2 = model(x2)
            prob = F.softmax(logits1, dim=1) * 0.5 + F.softmax(logits2, dim=1) * 0.5
        else:
            logits = model(x)
            prob = F.softmax(logits, dim=1)
        y_pred.append(prob.argmax(dim=1).cpu().numpy())
        y_prob.append(prob.cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True, help="outputs/<run_name> containing checkpoints/best.pt and config.json")
    ap.add_argument("--backbone", type=str, default=None, help="override backbone; otherwise read from config.json")
    ap.add_argument("--img_size", type=int, default=None, help="override img_size; otherwise read from config.json")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.load(open(run_dir / "config.json"))
    backbone = args.backbone or cfg.get("backbone","efficientnet_b0")
    img_size = args.img_size or int(cfg.get("img_size",224))
    seed = int(cfg.get("seed",42))
    val_frac = float(cfg.get("val_frac",0.2))

    data_dir = Path(args.data_dir)
    rows = load_metadata(str(data_dir / "HAM10000_metadata.csv"))
    img_dirs = [str(data_dir / "HAM10000_images_part_1"), str(data_dir / "HAM10000_images_part_2")]
    _, val_rows = train_val_split(rows, val_frac=val_frac, seed=seed)

    val_tf = build_transforms(img_size, train=False)
    val_ds = HamDataset(val_rows, img_dirs, transform=val_tf)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(CLASS_MAP)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt)

    y_true, y_pred, y_prob = evaluate(model, val_loader, device, num_classes, tta=args.tta)

    # Save arrays to run_dir
    np.save(run_dir / "best_val_y_true.npy", y_true)
    np.save(run_dir / "best_val_y_pred.npy", y_pred)
    np.save(run_dir / "best_val_y_prob.npy", y_prob)
    print("Exported arrays to", run_dir)

if __name__ == "__main__":
    main()
