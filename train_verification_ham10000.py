
import argparse, os, random, time, json, math, shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF

import timm

# Optional metrics
try:
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -----------------------------
# Utility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

def macro_f1(y_true, y_pred, num_classes):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    return f1_score(y_true_np, y_pred_np, average="macro")

# -----------------------------
# Dataset
# -----------------------------
import csv

# Binary classification: nv (Melanocytic nevi) vs. all other types
CLASS_MAP = {
    "akiec": 0,  # Not nv
    "bcc":   0,  # Not nv
    "bkl":   0,  # Not nv
    "df":    0,  # Not nv
    "mel":   0,  # Not nv
    "nv":    1,  # Melanocytic nevi (TARGET CLASS)
    "vasc":  0,  # Not nv
}
IDX2CLASS = {0: "not_nv", 1: "nv"}  # Binary class names

def load_metadata(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Expected headers include: image_id, dx (label), dx_type, age, sex, localization, patient_id
            rows.append(r)
    return rows

def train_val_split(rows: List[Dict], val_frac=0.2, seed=42) -> Tuple[List[Dict], List[Dict]]:
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
    def __init__(self, rows: List[Dict], img_dirs: List[str], transform=None):
        self.rows = rows
        self.img_dirs = img_dirs
        self.transform = transform
    def __len__(self):
        return len(self.rows)
    def _resolve_path(self, image_id: str) -> str:
        # Print debug info for first image
        if not hasattr(self, '_printed_debug'):
            print(f"Trying to resolve first image: {image_id}")
            self._printed_debug = True
            
        # Try different image extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            for d in self.img_dirs:
                p = Path(d) / f"{image_id}{ext}"
                if p.exists():
                    if not hasattr(self, '_found_first'):
                        print(f"Found first image at: {p}")
                        self._found_first = True
                    return str(p)
                
        # If we get here, no image was found
        if not hasattr(self, '_reported_missing'):
            print(f"Failed to find image: {image_id}")
            print(f"Searched in directories: {self.img_dirs}")
            self._reported_missing = True
            
        raise FileNotFoundError(f"Image {image_id} not found in {self.img_dirs}")
    def __getitem__(self, idx):
        r = self.rows[idx]
        image_id = r["image_id"]
        label = CLASS_MAP[r["dx"]]
        path = self._resolve_path(image_id)
        img = read_image(path).float() / 255.0  # [C,H,W], 0..1
        # Ensure 3 channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), image_id

# -----------------------------
# Augmentations
# -----------------------------
class RandomApplyJitter(nn.Module):
    def __init__(self, p=0.8, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        super().__init__()
        self.p = p
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                             saturation=saturation, hue=hue)
    def forward(self, x):
        if random.random() < self.p:
            return self.jitter(x)
        return x

def build_transforms(img_size=224, is_train=True):
    if is_train:
        # NO ORIENTATION-BASED AUGMENTATION: Removed RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
        # Simple resize is kept (non-augmenting) to maintain fixed input size for the model
        # This version keeps ONLY color/intensity augmentations for verification purposes
        return transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),  # Simple resize (no cropping/flipping)
            # Keep only color jitter - NO spatial augmentations
            RandomApplyJitter(p=0.8, brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            # Normalize w/ ImageNet stats (since we use pretrained backbones)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        # NO ORIENTATION-BASED AUGMENTATION: Removed CenterCrop
        # Simple resize is kept (non-augmenting) to maintain fixed input size for the model
        return transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((img_size, img_size)),  # Simple resize (no cropping/flipping)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

# -----------------------------
# Mixup/CutMix
# -----------------------------
def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_mixup_cutmix(x, y, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0):
    lam = 1.0
    if mixup_alpha > 0 and cutmix_alpha > 0:
        if np.random.rand() < 0.5:
            # mixup
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(x.size(0), device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, one_hot(y_a, num_classes), one_hot(y_b, num_classes), lam, "mixup"
        else:
            # cutmix
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            index = torch.randperm(x.size(0), device=x.device)
            y_a, y_b = y, y[index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
            return x, one_hot(y_a, num_classes), one_hot(y_b, num_classes), lam, "cutmix"
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, one_hot(y_a, num_classes), one_hot(y_b, num_classes), lam, "mixup"
    elif cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        index = torch.randperm(x.size(0), device=x.device)
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x, one_hot(y_a, num_classes), one_hot(y_b, num_classes), lam, "cutmix"
    else:
        return x, one_hot(y, num_classes), None, 1.0, "none"

# -----------------------------
# Losses
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, targets_onehot):
        # targets_onehot: [B, C]
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        ce = -targets_onehot * log_prob  # [B, C]
        focal = (1 - prob) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        if self.reduction == 'mean':
            return focal.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# -----------------------------
# EMA
# -----------------------------
class ModelEma:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy_model(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema.to(device=self.device)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for k in ema_params.keys():
            ema_params[k].mul_(self.decay).add_(model_params[k] * (1. - self.decay))

def deepcopy_model(model):
    import copy
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

# -----------------------------
# Build Model
# -----------------------------
def build_backbone(name: str, num_classes: int, pretrained=True):
    name = name.lower()
    if name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    elif name == "densenet121":
        model = timm.create_model("densenet121", pretrained=pretrained, num_classes=num_classes)
    elif name == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model

# -----------------------------
# Training/Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, criterion, mixup_alpha, cutmix_alpha, grad_clip, ema: Optional[ModelEma] = None):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    y_true_all, y_pred_all = [], []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup/CutMix
        imgs, y_a, y_b, lam, aug_kind = apply_mixup_cutmix(imgs, labels, num_classes, mixup_alpha, cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=scaler is not None):
            logits = model(imgs)
            if y_b is None:
                loss = criterion(logits, y_a)
                preds = logits.argmax(dim=1)
                y_true = labels
            else:
                # mixup/cutmix loss
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                preds = logits.argmax(dim=1)
                y_true = labels  # for running accuracy

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        correct += (preds == y_true).sum().item()
        y_true_all.append(y_true.detach().cpu())
        y_pred_all.append(preds.detach().cpu())

    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    acc = correct / max(total, 1)
    f1 = macro_f1(y_true_all, y_pred_all, num_classes)
    return total_loss / max(total, 1), acc, f1

@torch.no_grad()
def evaluate(model, loader, device, num_classes, tta=False):
    model.eval()
    total, correct = 0, 0
    y_true_all, y_pred_all, y_prob_all = [], [], []

    def forward_once(x):
        logits = model(x)
        return logits, F.softmax(logits, dim=1)

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # TTA DISABLED: TTA uses horizontal flipping which is orientation-based
        # Always use single forward pass for this verification
        logits, prob = forward_once(imgs)
        preds = logits.argmax(dim=1)

        total += imgs.size(0)
        correct += (preds == labels).sum().item()
        y_true_all.append(labels.cpu())
        y_pred_all.append(preds.cpu())
        y_prob_all.append(prob.cpu())

    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()
    y_prob_all = torch.cat(y_prob_all).numpy()

    acc = correct / max(total, 1)
    f1 = f1_score(y_true_all, y_pred_all, average="macro") if SKLEARN_OK else float("nan")
    metrics = {"acc": acc, "macro_f1": f1}
    if SKLEARN_OK:
        try:
            auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr")
            metrics["macro_auc"] = float(auc)
        except Exception:
            pass
    return metrics, y_true_all, y_pred_all, y_prob_all

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "densenet121", "convnext_tiny"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_dir)
    meta_csv = data_dir / "HAM10000_metadata.csv"
    # Print some debug information
    print(f"Looking for metadata file at: {meta_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata file not found at {meta_csv}")
        
    img_dirs = [str(data_dir / "HAM10000_images_part_1"),
                str(data_dir / "HAM10000_images_part_2")]
    
    # Print debug information about image directories
    for img_dir in img_dirs:
        print(f"Looking for images in: {img_dir}")
        if not Path(img_dir).exists():
            print(f"Warning: Image directory not found at {img_dir}")
    
    # Print number of rows found in metadata
    rows = load_metadata(str(meta_csv))
    print(f"Found {len(rows)} total rows in metadata")
    rows = [r for r in rows if r["dx"] in CLASS_MAP]
    print(f"Found {len(rows)} valid rows with known classes")

    rows = load_metadata(str(meta_csv))
    rows = [r for r in rows if r["dx"] in CLASS_MAP]  # filter to valid lesion types

    train_rows, val_rows = train_val_split(rows, val_frac=args.val_frac, seed=args.seed)

    train_tf = build_transforms(args.img_size, is_train=True)
    val_tf = build_transforms(args.img_size, is_train=False)

    train_ds = HamDataset(train_rows, img_dirs, transform=train_tf)
    val_ds = HamDataset(val_rows, img_dirs, transform=val_tf)

    # Class weights (optional)
    labels_train = [CLASS_MAP[r["dx"]] for r in train_rows]
    class_counts = np.bincount(labels_train, minlength=2)  # Binary classification: 2 classes
    print(f"Binary class distribution - not_nv: {class_counts[0]}, nv: {class_counts[1]}")
    class_weights = (class_counts.sum() / np.maximum(class_counts, 1))
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    num_classes = 2  # Binary classification: nv vs. not-nv
    model = build_backbone(args.backbone, num_classes=num_classes, pretrained=True).to(device)

    # Loss
    if args.focal:
        alpha_vec = class_weights_t if args.use_class_weights else None
        criterion = FocalLoss(gamma=args.gamma, alpha=alpha_vec, reduction='mean')
        def loss_fn(logits, targets_onehot):
            return criterion(logits, targets_onehot)
    else:
        if args.use_class_weights:
            ce = nn.CrossEntropyLoss(weight=class_weights_t)
        else:
            ce = nn.CrossEntropyLoss()
        def loss_fn(logits, targets_onehot):
            # Convert onehot to hard labels for CE
            labels = targets_onehot.argmax(dim=1)
            return ce(logits, labels)

    # Optimizer & scheduler (cosine with warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs
    def lr_schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    scaler = torch.amp.GradScaler(device) if device == "cuda" else None

    from copy import deepcopy
    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema else None

    # Outputs - Add suffixes to indicate: no orientation augmentation + binary nv classification
    run_name = args.run_name or f"{args.backbone}_img{args.img_size}_seed{args.seed}_no_orientation_binary_nv"
    out_dir = Path("outputs") / run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "gradcam").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_f1 = -1.0
    epochs_no_improve = 0
    patience = 7

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scaler, device, num_classes,
            loss_fn, args.mixup_alpha, args.cutmix_alpha, args.grad_clip, ema
        )
        scheduler.step()

        # Evaluate (use EMA if enabled)
        eval_model = ema.ema if ema is not None else model
        metrics, y_true, y_pred, y_prob = evaluate(eval_model, val_loader, device, num_classes, tta=args.tta)

        improved = metrics.get("macro_f1", 0.0) > best_f1
        if improved:
            best_f1 = metrics["macro_f1"]
            torch.save(eval_model.state_dict(), out_dir / "checkpoints" / "best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        log = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_acc": metrics.get("acc", None),
            "val_macro_f1": metrics.get("macro_f1", None),
            "val_macro_auc": metrics.get("macro_auc", None),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": round(time.time() - t0, 2),
        }
        print(json.dumps(log))

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best Macro F1: {best_f1:.4f}")

    # Optional: Temperature scaling for calibration
    if args.calibrate:
        try:
            eval_model = ema.ema if ema is not None else model
            eval_model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt", map_location=device))
            temperature = calibrate_temperature(eval_model, val_loader, device)
            torch.save({"state_dict": eval_model.state_dict(), "temperature": temperature},
                       out_dir / "checkpoints" / "best_calibrated.pt")
            print(f"Calibrated temperature: {temperature:.4f}")
        except Exception as e:
            print(f"Calibration failed: {e}")

# -----------------------------
# Temperature scaling
# -----------------------------
@torch.no_grad()
def calibrate_temperature(model, loader, device):
    # learn a single scalar T minimizing NLL on val set
    T = torch.ones(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _eval():
        loss_sum, n = 0.0, 0
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x) / T
            loss = F.cross_entropy(logits, y, reduction='sum')
            loss_sum += loss
            n += x.size(0)
        return loss_sum / n

    def closure():
        optimizer.zero_grad()
        loss = _eval()
        loss.backward()
        return loss

    for _ in range(10):
        optimizer.step(closure)
    return T.detach().item()

if __name__ == "__main__":
    main()
