import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

sizes = [3000, 800]
labels = ['Train (80%)', 'Validation (20%)']
colors = ['#10b981','#94a3b8']

plt.figure(figsize=(4,4))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
plt.title("Patient-Level Dataset Split", fontsize=14)
plt.savefig("figures/train_val_split.png", dpi=300)
print("Saved figures/train_val_split.png")
