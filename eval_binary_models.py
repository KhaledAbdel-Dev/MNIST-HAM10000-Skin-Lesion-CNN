"""
Evaluate trained binary classification models and generate predictions/metrics.
Works with both original and no-orientation variants.
"""

import argparse, json, csv, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import torch.nn.functional as F
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Binary classification mapping
CLASS_MAP = {
    "akiec": 0,  # Not nv
    "bcc":   0,  # Not nv
    "bkl":   0,  # Not nv
    "df":    0,  # Not nv
    "mel":   0,  # Not nv
    "nv":    1,  # Melanocytic nevi (TARGET CLASS)
    "vasc":  0,  # Not nv
}
IDX2CLASS = {0: "not_nv", 1: "nv"}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_metadata(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows = [r for r in rows if r["dx"] in CLASS_MAP]
    return rows

def train_val_split(rows, val_frac=0.2, seed=42):
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
    
    def __len__(self):
        return len(self.rows)
    
    def _resolve_path(self, image_id: str) -> str:
        for ext in ['.jpg', '.jpeg', '.png']:
            for d in self.img_dirs:
                p = Path(d) / f"{image_id}{ext}"
                if p.exists():
                    return str(p)
        raise FileNotFoundError(f"Image {image_id} not found")
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        image_id = r["image_id"]
        label = CLASS_MAP[r["dx"]]
        path = self._resolve_path(image_id)
        img = read_image(path).float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), image_id

def build_transforms(img_size=224):
    """Simple validation transforms."""
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

def build_model(backbone: str, num_classes: int, pretrained=False):
    if backbone == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    elif backbone == "densenet121":
        model = timm.create_model("densenet121", pretrained=pretrained, num_classes=num_classes)
    elif backbone == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        y_true_all.append(labels.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())
        y_prob_all.append(probs.cpu().numpy())
    
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)
    
    return y_true, y_pred, y_prob

def save_results(y_true, y_pred, y_prob, output_dir):
    """Save predictions and generate visualizations."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    np.save(output_dir / "best_val_y_true.npy", y_true)
    np.save(output_dir / "best_val_y_pred.npy", y_pred)
    np.save(output_dir / "best_val_y_prob.npy", y_prob)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob[:, 1])
    
    # Save summary
    summary = {
        "acc": float(acc),
        "macro_f1": float(f1),
        "macro_auc": float(auc)
    }
    with open(figures_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=['not_nv', 'nv'],
                                   output_dict=True)
    with open(figures_dir / "classification_report.txt", 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=['not_nv', 'nv']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['not_nv', 'nv'],
               yticklabels=['not_nv', 'nv'])
    plt.title('Confusion Matrix - Binary Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save confusion matrix as CSV
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=['not_nv', 'nv'], columns=['not_nv', 'nv'])
    cm_df.to_csv(figures_dir / "confusion_matrix.csv")
    
    # Per-class F1 scores
    per_class_f1 = {
        'not_nv': report['not_nv']['f1-score'],
        'nv': report['nv']['f1-score']
    }
    
    plt.figure(figsize=(8, 5))
    plt.bar(per_class_f1.keys(), per_class_f1.values(), color=['#FF6B6B', '#4ECDC4'])
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(per_class_f1.items()):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_f1.png", dpi=300)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                       help="Path to the model output directory")
    parser.add_argument("--data_dir", type=str, default="archive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_dir = Path(args.run_dir)
    
    # Load config
    with open(run_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    print(f"Evaluating model: {run_dir.name}")
    print(f"Backbone: {config['backbone']}")
    print(f"Image size: {config['img_size']}")
    
    # Load data
    data_dir = Path(args.data_dir)
    meta_csv = data_dir / "HAM10000_metadata.csv"
    img_dirs = [
        str(data_dir / "HAM10000_images_part_1"),
        str(data_dir / "HAM10000_images_part_2")
    ]
    
    rows = load_metadata(str(meta_csv))
    _, val_rows = train_val_split(rows, val_frac=args.val_frac, seed=args.seed)
    
    print(f"Validation samples: {len(val_rows)}")
    
    # Build dataset
    transform = build_transforms(config['img_size'])
    val_ds = HamDataset(val_rows, img_dirs, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=False)
    
    # Load model
    model = build_model(config['backbone'], num_classes=2, pretrained=False)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    print("Running evaluation...")
    y_true, y_pred, y_prob = evaluate(model, val_loader, device)
    
    print("Saving results and generating visualizations...")
    save_results(y_true, y_pred, y_prob, run_dir)
    
    print("✓ Evaluation complete!")

if __name__ == "__main__":
    main()
