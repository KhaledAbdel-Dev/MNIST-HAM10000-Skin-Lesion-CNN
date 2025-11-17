import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7,1.5))
ax.axis("off")

ax.add_patch(FancyBboxPatch((0,0.2),1.5,0.6,boxstyle="round,pad=0.1",fc="#3b82f6"))
ax.text(0.75,0.5,"Baseline",ha="center",va="center",color="white")
ax.arrow(1.6,0.5,2.5,0,head_width=0.1,length_includes_head=True,color="gray")
ax.add_patch(FancyBboxPatch((4.2,0.2),2.5,0.6,boxstyle="round,pad=0.1",fc="#10b981"))
ax.text(5.45,0.5,"Improved Pipeline\n(Focal Loss + Calib + Aug)",ha="center",va="center",color="white")
plt.xlim(-0.5,7)
plt.savefig("figures/pipeline_arrow.png",dpi=300)
print("Saved figures/pipeline_arrow.png")
