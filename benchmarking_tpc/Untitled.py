#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import json


# In[26]:


import os
import json
import math
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
log_dir = "logs"

if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Directory '{log_dir}' not found. Place your result files there.")

# ---------------- Detect JSON result files ----------------
files = sorted([f for f in os.listdir(log_dir)
                if f.startswith("tpc_results_")])

if not files:
    raise FileNotFoundError("No result JSON files found in 'logs/'. Ensure they are named like 'tpc_results_<workload>.json'.")

# ---------------- Create subplots grid ----------------
n = len(files)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = axes.flatten() if n > 1 else [axes]

# ---------------- Plot for each workload ----------------
for i, fname in enumerate(files):
    ax = axes[i]
    path = os.path.join(log_dir, fname)

    with open(path, "r") as f:
        data = json.load(f)

    workload = fname.replace("tpc_results_", "").replace(".json", "")
    results = data.get("results", [])

    if not results:
        print(f"Skipping empty file: {fname}")
        continue

    tpcs = [r["TPC"] for r in results]
    times = [r["ExecTime_ms"] for r in results]

    ax.plot(tpcs, times, marker='o', linewidth=2, color='tab:blue', label=workload)
    ax.set_title(workload, fontsize=14)
    ax.set_xlabel("#TPC", fontsize=14)
    ax.set_ylabel("Execution Time (ms)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Hide empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

#plt.suptitle("TPC Sweep vs Execution Time (Per Workload)", fontsize=18, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save and show
plot_path = os.path.join(log_dir, "tpc_sweep_all_workloads.pdf")
plt.savefig(plot_path)
plt.show()

print(f" Individual workload subplots saved to '{plot_path}'")


# In[25]:


import os
import json
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
log_dir = "logs"

if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Directory '{log_dir}' not found.")

# ---------------- Detect JSON result files ----------------
files = sorted([f for f in os.listdir(log_dir)
                if f.startswith("tpc_results_")])

if not files:
    raise FileNotFoundError("No result JSON files found in 'logs/'.")

# ---------------- Collect TPC=40 Data ----------------
workloads = []
times_at_40 = []

for fname in files:
    path = os.path.join(log_dir, fname)
    with open(path, "r") as f:
        data = json.load(f)

    workload = fname.replace("tpc_results_", "").replace(".json", "")
    results = data.get("results", [])

    # Find record where TPC == 40
    t40 = next((r for r in results if r["TPC"] == 40), None)
    if t40:
        workloads.append(workload)
        times_at_40.append(t40["ExecTime_ms"])

# ---------------- Plot Bar Chart ----------------
plt.figure(figsize=(10, 6))
bars = plt.bar(workloads, times_at_40, color="tab:orange")

plt.title("Execution Time at TPC = 40", fontsize=16)
plt.xlabel("Workload", fontsize=14)
plt.ylabel("Execution Time (ms)", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Annotate bars
for bar, val in zip(bars, times_at_40):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.2f}", ha='center', va='bottom', fontsize=12)

plt.tight_layout()

# Save and show
out_path = os.path.join(log_dir, "tpc40_exec_times.pdf")
plt.savefig(out_path)
plt.show()

print(f"Bar plot saved to '{out_path}'")


# In[ ]:





# In[ ]:




