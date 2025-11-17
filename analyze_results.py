
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# No seaborn; single plot per chart; no custom colors.

HAM_CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]

def load_arrays(run_dir: Path, prefix="best_val"):
    y_true = np.load(run_dir / f"{prefix}_y_true.npy")
    y_pred = np.load(run_dir / f"{prefix}_y_pred.npy")
    y_prob = np.load(run_dir / f"{prefix}_y_prob.npy")
    return y_true, y_pred, y_prob

def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6.5,5.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect='auto')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    # also save CSV
    import csv
    with open(str(out_path).replace(".png",".csv"), "w", newline="") as f:
        w = csv.writer(f)
        for row in cm:
            w.writerow(list(row))

def save_classification_report(y_true, y_pred, class_names, out_txt, out_csv):
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    # text
    with open(out_txt, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    # csv
    import csv
    keys = list(rep.keys())
    fields = ["label","precision","recall","f1-score","support"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for k in class_names + ["accuracy","macro avg","weighted avg"]:
            row = rep[k] if k in rep else {"precision":"","recall":"","f1-score":"","support":""}
            if k == "accuracy":
                w.writerow([k,"","",rep["accuracy"],sum(rep[c]["support"] for c in class_names)])
            else:
                w.writerow([k, row.get("precision",""), row.get("recall",""), row.get("f1-score",""), row.get("support","")])

def save_per_class_f1_bar(y_true, y_pred, class_names, out_path):
    from sklearn.metrics import f1_score
    f1s = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(class_names)), f1s)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    # also save CSV
    import csv
    with open(str(out_path).replace(".png",".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","f1"])
        for c, v in zip(class_names, f1s):
            w.writerow([c, float(v)])

def save_roc_curves(y_true, y_prob, class_names, out_path):
    from sklearn.metrics import roc_curve, auc
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(np.int32)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=cname)
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def save_reliability_diagram(y_true, y_prob, out_path, n_bins=15):
    # top-class confidence calibration
    conf = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correct = (preds == y_true).astype(np.int32)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(conf, bins) - 1
    bin_acc = []
    bin_conf = []
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(conf[mask].mean())
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(bin_conf, bin_acc, marker="o")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram (Top-Class)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def save_confidence_histograms(y_true, y_prob, out_path):
    conf = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    correct = preds == y_true
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.hist(conf[correct], bins=20, alpha=0.7, label="Correct")
    ax.hist(conf[~correct], bins=20, alpha=0.7, label="Incorrect")
    ax.set_xlabel("Top-Class Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to outputs/<run_name> directory containing *_y_*.npy")
    ap.add_argument("--class_names", type=str, default=None, help="Comma-separated class names in order; default=HAM order")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = HAM_CLASSES if args.class_names is None else [s.strip() for s in args.class_names.split(",")]

    y_true, y_pred, y_prob = load_arrays(run_dir)
    # Save figures/tables
    save_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
    save_classification_report(y_true, y_pred, class_names, out_dir / "classification_report.txt", out_dir / "classification_report.csv")
    save_per_class_f1_bar(y_true, y_pred, class_names, out_dir / "per_class_f1.png")
    save_roc_curves(y_true, y_prob, class_names, out_dir / "roc_curves.png")
    save_reliability_diagram(y_true, y_prob, out_dir / "reliability.png")
    save_confidence_histograms(y_true, y_prob, out_dir / "confidence_hist.png")

    # Summary text
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        macro_auc = float("nan")
    summary = {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc)
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved figures to:", out_dir)

if __name__ == "__main__":
    main()
