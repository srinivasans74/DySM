import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Config
# ==========================================================
INPUT = "results/sm_mask_overhead.csv"
OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

FILE_TAG = "stream_mask_overhead_1x3.pdf"

# ==========================================================
# Load Data
# ==========================================================
df = pd.read_csv(INPUT)

flips = df["flip"].to_numpy()
set_only_avg_us = df["set_only_avg_us"].to_numpy()
run_sync_avg_us = df["run_sync_avg_us"].to_numpy()
unique_sms = df["unique_sms"].to_numpy()

# Convert microseconds to nanoseconds for plot (a)
set_only_avg_ns = set_only_avg_us * 1000.0

# ==========================================================
# Helpers
# ==========================================================
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def mean_std(x):
    return float(np.mean(x)), float(np.std(x))

# ==========================================================
# Stats
# ==========================================================
m_set, s_set = mean_std(set_only_avg_ns)
m_run, s_run = mean_std(run_sync_avg_us)
m_sms, s_sms = mean_std(unique_sms)

# ==========================================================
# Plot: 1x3
# ==========================================================
fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=False)

# ---------- (a) set-only overhead ----------
ax = axs[0]
ax.bar(flips, set_only_avg_ns)
ax.axhline(m_set, linestyle="--", linewidth=2)
ax.set_ylabel("set_mask overhead (ns)", fontsize=16)
ax.set_xlabel("Flip index\n\n(a) Mask update call", fontsize=16)
ax.set_ylim(0, 100)
style_axis(ax)

# ---------- (b) run+sync overhead ----------
ax = axs[1]
ax.plot(flips, run_sync_avg_us, marker="o", linewidth=2)
ax.axhline(m_run, linestyle="--", linewidth=2)
ax.set_ylabel("Kernel execution time (µs)", fontsize=16)
ax.set_xlabel("Flip index\n\n(b) Observe next dispatch", fontsize=16)
style_axis(ax)

# ---------- (c) SMs used ----------
ax = axs[2]
ax.plot(flips, unique_sms, marker="o", linewidth=2)
ax.axhline(m_sms, linestyle="--", linewidth=2)
ax.set_ylabel("SMs Used", fontsize=16)
ax.set_xlabel("Flip index\n\n(c) Enforced SM allocation", fontsize=16)
style_axis(ax)

for ax in axs:
    ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, FILE_TAG)
fig.savefig(out_path, bbox_inches="tight", dpi=300)

# png_path = os.path.join(OUT_DIR, FILE_TAG.replace(".pdf", ".png"))
# fig.savefig(png_path, bbox_inches="tight", dpi=300)

print("[OK] Saved:", out_path)
