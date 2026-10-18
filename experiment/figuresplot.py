#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import os
import sys
import csv
from matplotlib import font_manager

# ==========================================================
# Font
# ==========================================================
font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Arial"

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = "logs"
OUT_DIR  = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# Args
# ==========================================================
if len(sys.argv) < 3:
    print("Usage: python3 figures.py <solution_id> <csv_file>")
    sys.exit(1)

solution = int(sys.argv[1])
csv_file = sys.argv[2]

solution_names = {
    3: "Proposed",
    4: "Adhoc",
    5: "STGM"
}
file_name = solution_names.get(solution, f"Sol_{solution}")

# ==========================================================
# Skip first N samples for ALL plots
# ==========================================================
SKIP = 3  # ignore first 10 samples everywhere

# ==========================================================
# Load task count
# ==========================================================
with open(csv_file, "r") as f:
    rows = list(csv.DictReader(f))
num_tasks = len(rows)

# ==========================================================
# Load wall-clock time
# ==========================================================
with open(os.path.join(BASE_DIR, "wallclocktime"), "r") as f:
    wallclocktime = json.load(f).get("wallclocktime", [])

# ==========================================================
# Load task logs
# ==========================================================
tasks_data = {}
for i in range(num_tasks):
    path = os.path.join(BASE_DIR, f"task_{i}")
    if os.path.exists(path):
        with open(path, "r") as f:
            tasks_data[f"task_{i}"] = json.load(f)

if not tasks_data:
    print("[ERROR] No task logs loaded.")
    sys.exit(1)

# ==========================================================
# Plot helpers
# ==========================================================
def add_top_legend(ncol=2):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=ncol,
            frameon=False,
            fontsize=14
        )

def style_ticks():
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=14)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.spines["top"].set_visible(False)

def disable_sci_y():
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)

def safe_len(*arrs):
    """minimum length among non-empty arrays"""
    lens = [len(a) for a in arrs if a is not None]
    return min(lens) if lens else 0

def slice_after_skip(a, skip=SKIP):
    """Return a[skip:] if possible, else []"""
    if not a:
        return []
    if len(a) <= skip:
        return []
    return a[skip:]

def slice_pair(x, y, skip=SKIP):
    """
    Given x, y arrays, skip first 'skip' samples from BOTH,
    then truncate to common length and return (x2, y2).
    """
    x2 = slice_after_skip(x, skip)
    y2 = slice_after_skip(y, skip)
    L = safe_len(x2, y2)
    return x2[:L], y2[:L]

# ==========================================================
# Metric extractor (OUTER LOGS ONLY)
# ==========================================================
def extract_metrics(j):
    return {
        "rtr":  j.get("outerrtr_measured", []),
        "rate": j.get("outercurrent_taskrate", []),
        "exec": j.get("outerexecutiontime", []),
        "resp": j.get("outerresponsetime", []),
        "sms":  j.get("outersms", []),
        "dm":   j.get("outerdeadlinemiss", []),
        "setp": j.get("setpoint", [])  # may be empty, may be len-1
    }

metrics = {k: extract_metrics(v) for k, v in tasks_data.items()}

# ==========================================================
# 1. RTR + Setpoint vs Wall-clock (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6, 3.5))

for name, m in metrics.items():
    x, y = slice_pair(wallclocktime, m["rtr"], SKIP)
    if x and y:
        plt.plot(x,y, '-', linewidth=2, label=f"{name} RTR")

    # setpoint: could be length-1 constant OR same length as rtr
    if m["setp"]:
        if len(m["setp"]) == 1:
            # constant setpoint line over the plotted x-range
            plt.plot(x, [m["setp"][0]] * len(x), '--', linewidth=2, label=f"{name} Setpoint")
        else:
            xs, ys = slice_pair(wallclocktime, m["setp"], SKIP)
            if xs and ys:
                plt.plot(xs, ys, '--', linewidth=2, label=f"{name} Setpoint")

plt.xlabel("Wall-clock Time (s)", fontsize=16, fontweight="bold")
plt.ylabel("RTR", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
style_ticks()
plt.ylim([0,2])
add_top_legend(ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(OUT_DIR, f"{file_name}_RTR.pdf"), dpi=300)
plt.close()

# ==========================================================
# 2. Task Rate vs Wall-clock (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6, 3.5))
for name, m in metrics.items():
    x, y = slice_pair(wallclocktime, m["rate"], SKIP)
    if x and y:
        plt.plot(x, y, linewidth=2, label=name)

plt.xlabel("Wall-clock Time (s)", fontsize=16, fontweight="bold")
plt.ylabel("Task Period (ms)", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
style_ticks()
add_top_legend(ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(OUT_DIR, f"{file_name}_TaskRate.pdf"), dpi=300)
plt.close()

# ==========================================================
# 3. Execution Time vs Wall-clock (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6, 3.5))
for name, m in metrics.items():
    x, y = slice_pair(wallclocktime, m["exec"], SKIP)
    if x and y:
        plt.plot(x, y, linewidth=2, label=name)

plt.xlabel("Wall-clock Time (s)", fontsize=16, fontweight="bold")
plt.ylabel("Execution Time (ms)", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
style_ticks()
disable_sci_y()
add_top_legend(ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(OUT_DIR, f"{file_name}_ExecutionTime.pdf"), dpi=300)
plt.close()

# ==========================================================
# 4. Response Time vs Wall-clock (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6, 3.5))
for name, m in metrics.items():
    x, y = slice_pair(wallclocktime, m["resp"], SKIP)
    if x and y:
        plt.plot(x, y, linewidth=2, label=name)

plt.xlabel("Wall-clock Time (s)", fontsize=16, fontweight="bold")
plt.ylabel("Response Time (ms)", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
style_ticks()
disable_sci_y()
add_top_legend(ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(OUT_DIR, f"{file_name}_ResponseTime.pdf"), dpi=300)
plt.close()

# ==========================================================
# 5. SMs vs Control Period (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6.5, 3.5))
for name, m in metrics.items():
    y = slice_after_skip(m["sms"], SKIP)
    if y:
        x = list(range(SKIP, SKIP + len(y)))  # keep original control-period index
        plt.plot(x, y, linewidth=2, label=name)

plt.xlabel("Control Period", fontsize=16, fontweight="bold")
plt.ylabel("SMs", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
style_ticks()
add_top_legend(ncol=2)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(os.path.join(OUT_DIR, f"{file_name}_SMs.pdf"), dpi=300)
plt.close()

# ==========================================================
# 6. Deadline Miss vs Control Period (skip first 10 samples)
# ==========================================================
plt.figure(figsize=(6.5, 3.5))
plotted = False
for name, m in metrics.items():
    y = slice_after_skip(m["dm"], SKIP)
    if y:
        x = list(range(SKIP, SKIP + len(y)))
        plt.plot(x, y, linewidth=2, label=name)
        plotted = True

if not plotted:
    print("[WARN] No outerdeadlinemiss data found in logs; skipping plot.")
else:
    plt.xlabel("Control Period", fontsize=16, fontweight="bold")
    plt.ylabel("Deadline Miss", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    style_ticks()
    disable_sci_y()
    add_top_legend(ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(OUT_DIR, f"{file_name}_DeadlineMiss.pdf"), dpi=300)
plt.close()

print(f"\n[DONE] All plots generated successfully (skipped first {SKIP} samples).")