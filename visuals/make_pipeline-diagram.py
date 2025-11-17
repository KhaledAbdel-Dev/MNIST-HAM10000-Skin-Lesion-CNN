# make_pipeline_diagram.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.figure(figsize=(10,2))
ax = plt.gca()
ax.axis("off")

def box(x, text):
    ax.add_patch(FancyBboxPatch((x,0.25), 1.7, 0.8, boxstyle="round,pad=0.15"))
    ax.text(x+0.85, 0.65, text, ha="center", va="center", fontsize=11)

labels = ["Raw", "Resize / Crop", "Flip", "Normalize", "Model"]
x = 0.3
for i, lab in enumerate(labels):
    box(x, lab)
    if i < len(labels)-1:
        ax.annotate("", xy=(x+1.9,0.65), xytext=(x+1.9-0.7,0.65), arrowprops=dict(arrowstyle="->", lw=1.5))
    x += 2.2

plt.xlim(0, 12)
plt.ylim(0, 1.3)
plt.tight_layout()
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/pipeline_diagram.png", dpi=300, bbox_inches="tight")
print("Saved figures/pipeline_diagram.png")
