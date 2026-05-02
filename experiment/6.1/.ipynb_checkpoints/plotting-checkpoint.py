import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, MultipleLocator
# ==========================================================
# Config
# ==========================================================
SKIP = 10
OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)
SOLUTIONS = [
    ("SMshare", 6, "logs/sol_6/task_0", "logs/sol_6/task_1"),
    ("STGM",    5, "logs/sol_5/task_0", "logs/sol_5/task_1"),
    ("Adhoc",     4, "logs/sol_4/task_0", "logs/sol_4/task_1"),
    ("DySM", 3, "logs/sol_3/task_0", "logs/sol_3/task_1"),
]

FILE_TAG = "stgm_adhoc_proposed_1"
CONST_SETPOINT = 0.80
AX_FONTSIZE = 16
LINEWIDTH = 2

# ==========================================================
# Helpers
# ==========================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_metrics(j):
    return {
        "rtr": j.get("outerrtr_measured", []),
        "dm":  j.get("outerdeadlinemiss", []),
        "sms": j.get("outersms", []),
        "per": j.get("outercurrent_taskrate", []),
    }

def slice_pair(x, y, skip=0):
    L = min(len(x), len(y))
    if L <= skip:
        return [], []
    return x[skip:L], y[skip:L]

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", labelsize=AX_FONTSIZE)

def parse_task_file(path):
    root = load_json(path)
    return extract_metrics(root)

# ==========================================================
# Plot: 3 rows (solutions) x 3 cols (RTR, DM, SMS)
# ==========================================================
fig, axs = plt.subplots(4, 3, figsize=(16, 10), sharex=False)

for row, (name, sol_id, t1_path, t2_path) in enumerate(SOLUTIONS):
    t1 = parse_task_file(t1_path)
    t2 = parse_task_file(t2_path)

    # x-axis based on available RTR samples in task_0
    x = list(range(len(t1["rtr"])))

    # -------------------- Col 0: RTR --------------------
    ax = axs[row, 0]
    x1, r1 = slice_pair(x, t1["rtr"], SKIP)
    _,  r2 = slice_pair(x, t2["rtr"], SKIP)

    if x1 and r1:
        ax.plot(x1, r1, "-", linewidth=LINEWIDTH, label="mm")
    if x1 and r2:
        ax.plot(x1, r2, "-.", linewidth=LINEWIDTH, label="stereodisparity")
    if x1:
        ax.plot(x1, [CONST_SETPOINT]*len(x1), "--", linewidth=LINEWIDTH, label="Setpoint (0.80)")

    ax.set_ylabel("RRT", fontsize=AX_FONTSIZE)
    ax.set_xlabel(f"Control Period\n\n({chr(ord('a') + row*3 + 0)}) {name}", fontsize=AX_FONTSIZE)
    ax.set_ylim([0, 2.5])
    style_axis(ax)

    # -------------------- Col 1: Deadline Miss --------------------
    ax = axs[row, 1]
    x1, d1 = slice_pair(x, t1["dm"], SKIP)
    _,  d2 = slice_pair(x, t2["dm"], SKIP)

    if x1 and d1:
        ax.plot(x1, d1, "-", linewidth=LINEWIDTH, label="mm")
    if x1 and d2:
        ax.plot(x1, d2, "-.", linewidth=LINEWIDTH, label="stereodisparity")

    ax.set_ylabel("Deadline Misses (%)", fontsize=AX_FONTSIZE)
    ax.set_xlabel(f"Control Period\n\n({chr(ord('a') + row*3 + 1)}) {name}", fontsize=AX_FONTSIZE)
    ax.set_ylim([0, 100])
    style_axis(ax)

    # -------------------- Col 2: SMs --------------------
    ax = axs[row, 2]
    x1, s1 = slice_pair(x, t1["sms"], SKIP)
    _,  s2 = slice_pair(x, t2["sms"], SKIP)

    if x1 and s1:
        ax.plot(x1, s1, "-", linewidth=LINEWIDTH, label="mm")
    if x1 and s2:
        ax.plot(x1, s2, "-.", linewidth=LINEWIDTH, label="stereodisparity")

    ax.set_ylabel("SMs", fontsize=AX_FONTSIZE)
    ax.set_xlabel(f"Control Period\n\n({chr(ord('a') + row*3 + 2)}) {name}", fontsize=AX_FONTSIZE)
    style_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True))
    #ax.set_ylim([0, 82])
    #ax.yaxis.set_major_locator(MultipleLocator(2))
# ==========================================================
# One global legend
# ==========================================================
handles, labels = [], []
for ax in axs.ravel():
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll not in labels:
            handles.append(hh)
            labels.append(ll)

fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=AX_FONTSIZE)

plt.tight_layout(rect=[0, 0, 1, 0.93])

out_path = os.path.join(OUT_DIR, f"{FILE_TAG}_3x3_RTR_DM_SMS.pdf")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
#plt.close(fig)

print("[OK] Saved:", out_path)
for name, sol_id, t1_path, t2_path in SOLUTIONS:
    print(f"[OK] {name} (sol_{sol_id}) -> {t1_path}, {t2_path}")