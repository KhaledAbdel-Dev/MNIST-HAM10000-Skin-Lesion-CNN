"""
Compare performance across different model variants:
- Original (with orientation augmentation) vs No-orientation
- Multi-class (7-way) vs Binary (nv vs not-nv)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, roc_auc_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(output_dir):
    """Load results from an output directory."""
    output_dir = Path(output_dir)
    
    results = {
        'name': output_dir.name,
        'exists': output_dir.exists()
    }
    
    if not output_dir.exists():
        return results
    
    # Load config
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load summary if it exists
    summary_path = output_dir / "figures" / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load predictions
    y_true_path = output_dir / "best_val_y_true.npy"
    y_pred_path = output_dir / "best_val_y_pred.npy"
    y_prob_path = output_dir / "best_val_y_prob.npy"
    
    if y_true_path.exists() and y_pred_path.exists():
        results['y_true'] = np.load(y_true_path)
        results['y_pred'] = np.load(y_pred_path)
        if y_prob_path.exists():
            results['y_prob'] = np.load(y_prob_path)
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(results['y_true'], results['y_pred'])
        results['macro_f1'] = f1_score(results['y_true'], results['y_pred'], average='macro')
        
        # For binary classification
        if len(np.unique(results['y_true'])) == 2:
            results['binary'] = True
            results['precision'] = f1_score(results['y_true'], results['y_pred'], average='binary', pos_label=1)
            try:
                results['auc'] = roc_auc_score(results['y_true'], results['y_prob'][:, 1])
            except:
                pass
        else:
            results['binary'] = False
            try:
                results['auc'] = roc_auc_score(results['y_true'], results['y_prob'], multi_class='ovr')
            except:
                pass
    
    return results


def create_comparison_plots(results_dict, output_dir="outputs/comparisons"):
    """Create comprehensive comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out results that don't exist
    results_dict = {k: v for k, v in results_dict.items() if v.get('exists', False) and 'y_true' in v}
    
    if not results_dict:
        print("No valid results found to compare!")
        return
    
    print(f"Comparing {len(results_dict)} models:")
    for name in results_dict.keys():
        print(f"  - {name}")
    
    # 1. Metrics Comparison Bar Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    names = list(results_dict.keys())
    accuracies = [results_dict[name].get('accuracy', 0) for name in names]
    f1_scores = [results_dict[name].get('macro_f1', 0) for name in names]
    aucs = [results_dict[name].get('auc', 0) for name in names]
    
    # Accuracy comparison
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), accuracies, color=sns.color_palette("husl", len(names)))
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, ha='center', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # F1 Score comparison
    ax = axes[0, 1]
    bars = ax.bar(range(len(names)), f1_scores, color=sns.color_palette("husl", len(names)))
    ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, ha='center', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # AUC comparison
    ax = axes[1, 0]
    bars = ax.bar(range(len(names)), aucs, color=sns.color_palette("husl", len(names)))
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('AUC Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, ha='center', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, aucs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table data
    table_data = []
    for name in names:
        row = [
            name.replace('efficientnet_b0_img224_seed42_', ''),
            f"{results_dict[name].get('accuracy', 0):.4f}",
            f"{results_dict[name].get('macro_f1', 0):.4f}",
            f"{results_dict[name].get('auc', 0):.4f}",
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, 
                     colLabels=['Model Variant', 'Accuracy', 'Macro F1', 'AUC'],
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")
    plt.close()
    
    # 2. Confusion Matrix Comparison (for multi-class only)
    multiclass_results = {k: v for k, v in results_dict.items() if not v.get('binary', False)}
    if len(multiclass_results) > 0:
        n_models = len(multiclass_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        for idx, (name, result) in enumerate(multiclass_results.items()):
            cm = confusion_matrix(result['y_true'], result['y_pred'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cbar_kws={'label': 'Normalized Count'})
            axes[idx].set_title(f'Confusion Matrix\n{name.replace("_", " ")}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'confusion_matrices_comparison.png'}")
        plt.close()
    
    # 3. Binary Classification Comparison (if applicable)
    binary_results = {k: v for k, v in results_dict.items() if v.get('binary', False)}
    if len(binary_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        names = list(binary_results.keys())
        
        # Confusion matrices for binary
        for idx, (name, result) in enumerate(binary_results.items()):
            cm = confusion_matrix(result['y_true'], result['y_pred'])
            
            ax = axes[idx] if len(binary_results) > 1 else axes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['not_nv', 'nv'], yticklabels=['not_nv', 'nv'],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'Binary Classification\n{name.replace("_", " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Hide extra subplot if only one binary model
        if len(binary_results) == 1:
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "binary_confusion_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'binary_confusion_comparison.png'}")
        plt.close()
    
    # 4. Detailed Metrics Table (CSV)
    metrics_data = []
    for name, result in results_dict.items():
        row = {
            'Model': name,
            'Type': 'Binary' if result.get('binary', False) else 'Multi-class',
            'Accuracy': result.get('accuracy', 0),
            'Macro F1': result.get('macro_f1', 0),
            'AUC': result.get('auc', 0),
        }
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "metrics_comparison.csv", index=False)
    print(f"Saved: {output_dir / 'metrics_comparison.csv'}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    """Main comparison function."""
    
    # Define the models to compare - BINARY ONLY
    models = {
        'binary_with_augmentation': 'outputs/efficientnet_b0_img224_seed42_binary_nv',
        'binary_no_orientation': 'outputs/efficientnet_b0_img224_seed42_no_orientation_binary_nv',
    }
    
    print("Loading results from model directories...")
    results = {}
    for label, path in models.items():
        print(f"\nLoading: {label}")
        print(f"  Path: {path}")
        result = load_results(path)
        if result['exists']:
            if 'y_true' in result:
                print(f"  ✓ Results loaded successfully")
                print(f"    - Accuracy: {result.get('accuracy', 0):.4f}")
                print(f"    - Macro F1: {result.get('macro_f1', 0):.4f}")
                print(f"    - AUC: {result.get('auc', 0):.4f}")
                results[label] = result
            else:
                print(f"  ✗ No predictions found (model may not be trained yet)")
        else:
            print(f"  ✗ Directory does not exist (model not trained yet)")
    
    if not results:
        print("\n⚠ No trained models found to compare!")
        print("Please train at least one model first.")
        return
    
    print(f"\n\nCreating comparison visualizations for {len(results)} models...")
    create_comparison_plots(results)
    
    print("\n✓ Comparison complete! Check the 'outputs/comparisons' folder.")


if __name__ == "__main__":
    main()
