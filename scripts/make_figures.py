"""Generate the presentation figures (Sapienza-colored) into outputs/figs/.
Run before scripts/generate_ppt.py."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
MAROON, TEAL, GREY = "#822433", "#15707E", "#6E6E70"
DARK, LIGHT = "#262629", "#EAEEEF"
TEAL2, TEAL3 = "#3E9AA8", "#7BB8C2"

models = ["custom_cnn", "yolo_pico", "yolov8n", "yolov8s"]
params = [28813, 184300, 2692548, 9843604]
acc = [0.9996, 0.2073, 1.0, 1.0]
cols = [MAROON, TEAL3, TEAL2, TEAL]
lat = [0.514, 5.491, 4.346, 6.067]
disk = [0.125, 0.567, 5.594, 19.918]
SAMPLE = ROOT / "data/raw/chess-positions/test"

# Fig 1 — accuracy vs params
fig, ax = plt.subplots(figsize=(8, 4.4))
for m, p, a, c in zip(models, params, acc, cols):
    ax.scatter(p, a, s=240, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(m, (p, a), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=11, color=DARK, fontweight="bold")
ax.set_xscale("log"); ax.set_xlabel("Parameters (log scale)", color=DARK)
ax.set_ylabel("Occupied-cell accuracy", color=DARK); ax.set_ylim(0.0, 1.08)
ax.axhspan(0.99, 1.08, color=LIGHT, zorder=0)
ax.annotate("CNN: 99.96% at ~93x fewer params\nthan yolov8n", (28813, 0.9996),
            xytext=(60000, 0.62), fontsize=10.5, color=MAROON,
            arrowprops=dict(arrowstyle="->", color=MAROON))
ax.set_title("Per-piece accuracy vs model size", color=DARK, fontweight="bold", loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25); plt.tight_layout()
plt.savefig(FIG / "acc_vs_params.png", dpi=150); plt.close()

# Fig 2 — deployment bars
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, vals, ttl, unit in [(axs[0], lat, "Inference latency", "ms / board"),
                            (axs[1], disk, "Model size on disk", "MB")]:
    b = ax.bar(models, vals, color=cols, edgecolor="white")
    ax.set_title(ttl, color=DARK, fontweight="bold", loc="left"); ax.set_ylabel(unit, color=DARK)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:g}", ha="center",
                va="bottom", fontsize=9.5, color=DARK)
plt.tight_layout(); plt.savefig(FIG / "deploy.png", dpi=150); plt.close()

# Fig 3 — sample board + grid
sample = sorted(SAMPLE.glob("*.jpeg"))[:1]
if sample:
    img = Image.open(sample[0]).convert("RGB")
    fig, ax = plt.subplots(figsize=(5.2, 5.2)); ax.imshow(img)
    for i in range(9):
        ax.axhline(i * 50, color=MAROON, lw=1.1, alpha=0.8)
        ax.axvline(i * 50, color=MAROON, lw=1.1, alpha=0.8)
    ax.add_patch(Rectangle((0, 0), 50, 50, fill=False, edgecolor=TEAL, lw=3))
    ax.add_patch(Rectangle((300, 0), 50, 50, fill=False, edgecolor=TEAL, lw=3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("400x400 board -> 8x8 grid -> 50x50 cells", color=DARK,
                 fontweight="bold", fontsize=12)
    plt.tight_layout(); plt.savefig(FIG / "board_grid.png", dpi=150); plt.close()

# Fig 4 — Optuna trials
trials = ["Trial 0\nbase32/adamw\naug .004", "Trial 1\nbase64/SGD\naug .72",
          "Trial 2 (best)\nbase16/adamw\naug .10"]
vals = [0.9976, 0.8636, 0.9995]; tc = [TEAL2, GREY, MAROON]
fig, ax = plt.subplots(figsize=(8, 4.2))
b = ax.bar(range(3), vals, color=tc, edgecolor="white", width=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(trials, fontsize=10)
ax.set_ylabel("Occupied-cell accuracy", color=DARK); ax.set_ylim(0.0, 1.05)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}", ha="center",
            va="bottom", fontsize=11, color=DARK, fontweight="bold")
ax.set_title("Optuna trials: smallest net + AdamW + light aug wins", color=DARK,
             fontweight="bold", loc="left")
ax.annotate("heavy aug + SGD\ncollapses", (1, 0.8636), xytext=(1.0, 0.55), ha="center",
            fontsize=10, color=GREY, arrowprops=dict(arrowstyle="->", color=GREY))
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25); plt.tight_layout()
plt.savefig(FIG / "hpo.png", dpi=150); plt.close()

# Fig 5 — CNN architecture
fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.6)
blocks = [("Input\n3x50x50", "#cfd8dc"), ("Conv16+BN\nReLU+Pool\n16x25", "#bcd3d8"),
          ("Conv32+BN\nReLU+Pool\n32x12", "#9ec3cb"), ("Conv64+BN\nReLU+Pool\n64x6", "#7fb3bd"),
          ("GAP\n64", "#e6c9ce"), ("Dropout\nFC 64", "#d3a9b1"), ("FC -> 13\nsoftmax", MAROON)]
x = 0.2
for i, (t, c) in enumerate(blocks):
    w = 1.45
    ax.add_patch(FancyBboxPatch((x, 0.7), w, 1.2, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=c, ec="white"))
    ax.text(x + w / 2, 1.3, t, ha="center", va="center", fontsize=9.2,
            color="white" if c == MAROON else DARK, fontweight="bold")
    if i < len(blocks) - 1:
        ax.add_patch(FancyArrowPatch((x + w, 1.3), (x + w + 0.15, 1.3),
                                     arrowstyle="-|>", mutation_scale=14, color=GREY))
    x += w + 0.15
ax.text(0.2, 0.25, "28,813 parameters  ·  0.0069 GFLOPs/cell  ·  125 KB on disk",
        fontsize=11, color=MAROON, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "cnn_arch.png", dpi=150, bbox_inches="tight"); plt.close()
print("figures written to", FIG)
