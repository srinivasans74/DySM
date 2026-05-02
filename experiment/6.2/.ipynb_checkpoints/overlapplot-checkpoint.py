import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Config
# ==========================================================
LOG_FILE = "logs/mttkrp_overlap_sweep_fixedwidth.json"
OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

AX_FONTSIZE = 16
LINEWIDTH = 2
TASK_WIDTH = 8  # Each task uses 8 TPCs total

# ==========================================================
# Data Loading (Parsing JSON)
# ==========================================================
if not os.path.exists(LOG_FILE):
    print(f"[Error] Log file {LOG_FILE} not found.")
    exit(1)

with open(LOG_FILE, 'r') as f:
    data = json.load(f)

ov_list, rt1_list, rt2_list = [], [], []
for entry in data["sweep"]:
    ov_list.append(entry["overlap_step"])
    rt1_list.append(entry["task1_avg_rt_ms"])
    rt2_list.append(entry["task2_avg_rt_ms"])

ov = np.array(ov_list)
rt1 = np.array(rt1_list)
rt2 = np.array(rt2_list)

# Calculate Degradation relative to disjoint point (ov=0)
deg1 = (rt1 - rt1[0]) / rt1[0] * 100.0
deg2 = (rt2 - rt2[0]) / rt2[0] * 100.0

# ==========================================================
# Helpers
# ==========================================================
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=AX_FONTSIZE)

# ==========================================================
# Plotting: 1 row x 3 cols
# ==========================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

ax = axs[0]

overlap_cnt = ov.copy()
task_width = 8  # each task always uses 8 TPCs

# Task 1: overlap + non-overlap portion
ax.barh(ov, overlap_cnt,
        hatch='//', edgecolor="black", label="Overlap (shared TPCs)")
ax.barh(ov, task_width - overlap_cnt, left=overlap_cnt,
        hatch='//', edgecolor="black", alpha=0.35, label="Task 1 TPCs")

# Task 2: slightly offset for visibility
ax.barh(ov + 0.25, overlap_cnt,
        hatch='oo', edgecolor="black")
ax.barh(ov + 0.25, task_width - overlap_cnt, left=overlap_cnt,
        hatch='oo', edgecolor="black", alpha=0.35, label="Task 2 TPCs")

# Axis settings
ax.set_xlim(0, task_width)

yticks = np.array([0, 2, 4, 6, 8])
ax.set_yticks(yticks + 0.12)
ax.set_yticklabels([str(v) for v in yticks])

ax.set_xlabel("(a) Overlap size (out of 8 TPCs)", fontsize=AX_FONTSIZE)
ax.set_ylabel("Overlap level (ov)", fontsize=AX_FONTSIZE)

ax.grid(True, axis="x", linestyle=":", alpha=0.6)
style_axis(ax)

# Fix legend location ("up" is invalid → use upper center)
ax.legend(
    fontsize=AX_FONTSIZE - 2,
    frameon=True,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2
)

# --- Col 1: Response Time (Grouped Bars) ---
ax = axs[1]
x_idx = np.arange(len(ov))
width = 0.38
b1 = ax.bar(x_idx - width/2, rt1, width, label="t1", color="#1f77b4", edgecolor="black")
b2 = ax.bar(x_idx + width/2, rt2, width, label="t2", color="#ff7f0e", edgecolor="black")

for r in b1: r.set_hatch("//")
for r in b2: r.set_hatch("\\\\")

ax.set_ylabel("Avg response time (ms)", fontsize=AX_FONTSIZE)
ax.set_xlabel("(b) Overlap", fontsize=AX_FONTSIZE)
ax.set_xticks(x_idx)
ax.set_xticklabels(ov)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
style_axis(ax)

ax = axs[2]
ax.plot(ov, deg1, "-", marker='o', linewidth=LINEWIDTH, color="#1f77b4", label="t1")
ax.plot(ov, deg2, "-.", marker='s', linewidth=LINEWIDTH, color="#ff7f0e", label="t2")
ax.set_ylabel("Degradation (%)", fontsize=AX_FONTSIZE)
ax.set_xlabel("(c) Overlap", fontsize=AX_FONTSIZE)
ax.set_xticks(ov)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
style_axis(ax)

# --- Global Legend for Col 1 & 2 ---
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=AX_FONTSIZE, 
           bbox_to_anchor=(0.75, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.92])
out_path = os.path.join(OUT_DIR, "mttkrp_overlap_style_match.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight")

print("[OK] Saved dynamic plot with matched style to:", out_path)