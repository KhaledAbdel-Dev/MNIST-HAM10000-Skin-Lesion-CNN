import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

# Load metadata
df = pd.read_csv("../archive/HAM10000_metadata.csv")
order = df["dx"].value_counts().index
plt.figure(figsize=(8,5))
sns.barplot(x=df["dx"].value_counts().index,
            y=df["dx"].value_counts().values,
            palette=["#3b82f6" if c=="nv" else "#94a3b8" for c in order])
plt.title("HAM10000 Class Distribution", fontsize=16)
plt.xlabel("Lesion Type")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=300)
print("Saved figures/class_distribution.png")
