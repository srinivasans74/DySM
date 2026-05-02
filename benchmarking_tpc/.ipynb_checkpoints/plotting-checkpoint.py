import os
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
log_dir = "logs"

if not os.path.exists(log_dir):
    raise FileNotFoundError(f"Directory '{log_dir}' not found. Place your result files there.")

files = sorted([f for f in os.listdir(log_dir)
                if f.startswith("tpc_results_")])

if not files:
    raise FileNotFoundError("No result JSON files found in 'logs/'. Ensure they are named like 'tpc_results_<workload>.json'.")

n = len(files)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
axes = axes.flatten() if n > 1 else [axes]

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
    ax.set_title(workload, fontsize=16)
    ax.set_xlabel("TPCs", fontsize=16)
    ax.set_ylabel("Execution Time (ms)", fontsize=16)
    ax.set_xlim([1,10])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    #ax.grid(True, linestyle="--", alpha=0.6)
    #ax.legend(fontsize=14)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

#plt.suptitle("TPC Sweep vs Execution Time (Per Workload)", fontsize=18, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save and show
plot_path = os.path.join("fig1.pdf")
plt.savefig(plot_path)
print(f" Individual workload subplots saved to '{plot_path}'")
