import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,2))
ax.axis("off")

def box(x, text, color):
    ax.add_patch(FancyBboxPatch((x,0.2), 1.2, 0.6, boxstyle="round,pad=0.1", fc=color))
    ax.text(x+0.6, 0.5, text, ha="center", va="center", color="white", fontsize=12)

box(0, "Input Image", "#3b82f6")
ax.arrow(1.3,0.5,0.3,0, head_width=0.1, color="gray")
box(1.8, "CNN (EfficientNet-B0)", "#10b981")
ax.arrow(3.1,0.5,0.3,0, head_width=0.1, color="gray")
box(3.6, "Predicted Class", "#f59e0b")

plt.xlim(-0.5,5)
plt.savefig("figures/flow_diagram.png", dpi=300)
print("Saved figures/flow_diagram.png")
