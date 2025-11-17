import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10,1.8))
ax.axis("off")

stages = ["Input (224×224)", "Conv Stem", "Blocks 1–7 (MBConv)", "Global Avg Pool", "Dense (7)"]
colors = ["#3b82f6","#60a5fa","#10b981","#fbbf24","#f97316"]
x = 0
for i, (s,c) in enumerate(zip(stages, colors)):
    ax.add_patch(FancyBboxPatch((x,0.2), 1.8, 0.6, boxstyle="round,pad=0.1", fc=c))
    ax.text(x+0.9,0.5,s,ha="center",va="center",fontsize=10,color="white")
    if i < len(stages)-1:
        ax.arrow(x+1.8,0.5,0.3,0,head_width=0.1,color="gray")
    x += 2.3
plt.xlim(-0.3,11)
plt.savefig("figures/efficientnet_blockdiagram.png",dpi=300)
print("Saved figures/efficientnet_blockdiagram.png")


plt.figure(figsize=(6,1))
steps = ["Edges","Patterns","Shape","Diagnosis"]
for i,txt in enumerate(steps):
    plt.text(i*2,0,txt,fontsize=12,ha='center')
    if i < len(steps)-1:
        plt.arrow(i*2+0.5,0,1.0,0,length_includes_head=True,head_width=0.1)
plt.axis('off')
plt.savefig("figures/learning_progression.png",dpi=300)
print("Saved figures/learning_progression.png")
