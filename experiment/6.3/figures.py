import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ==========================================================
# Config
# ==========================================================
SKIP = 10
OUT_DIR = "Figures"
os.makedirs(OUT_DIR, exist_ok=True)

setpoints = [0.50, 0.60, 0.70, 0.80]

ROOT_PROPOSED = "3runs"        # DySM / Proposed
ROOT_ADHOC    = "3runs_adhoc"    # Adhoc baseline

TRACE_DIRNAME = "trace_trace_projection_hist_stencil"

TASKS = [0, 1, 2]
TASK_LABELS = ["projection", "histogram", "stencil"]

RTR_KEY  = "outerrtr_measured"
MISS_KEY = "outerdeadlinemiss"

METHODS = [
    ("DySM",  ROOT_PROPOSED),
    ("Adhoc", ROOT_ADHOC),
]

# ==========================================================
# Helpers
# ==========================================================
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _to_1d_array(x):
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, list):
        return np.array(x, dtype=float).reshape(-1)
    return np.array(x, dtype=float).reshape(-1)

def compute_mean_var_from_series(series, skip=SKIP):
    arr = _to_1d_array(series)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size > skip:
        arr = arr[skip:]
    return float(np.mean(arr)), float(np.var(arr, ddof=0))

def find_trace_folder_for_setpoint(root, sp):
    sp_str = f"{sp:.2f}"
    candidates = [
        os.path.join(root, sp_str, TRACE_DIRNAME),
        os.path.join(root, "sp", sp_str, TRACE_DIRNAME),
        os.path.join(root, f"sp{sp_str}", TRACE_DIRNAME),
        os.path.join(root, f"sp_{sp_str}", TRACE_DIRNAME),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    hits = glob.glob(os.path.join(root, "**", f"*{sp_str}*", TRACE_DIRNAME), recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Cannot find trace folder for setpoint {sp_str} under {root}.")

def find_task_file(trace_folder, task_id):
    patterns = [
        os.path.join(trace_folder, f"task_{task_id}.json"),
        os.path.join(trace_folder, f"task{task_id}.json"),
        os.path.join(trace_folder, f"task_{task_id}"),
        os.path.join(trace_folder, f"task{task_id}"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p

    hits = glob.glob(os.path.join(trace_folder, f"task*{task_id}*"))
    hits = [h for h in hits if os.path.isfile(h)]
    if hits:
        return hits[0]

    raise FileNotFoundError(f"Cannot find task file for task {task_id} in {trace_folder}")

def collect_stats_for_root(root):
    num_tasks = len(TASKS)
    num_sps = len(setpoints)

    rtr_mean  = np.zeros((num_tasks, num_sps), dtype=float)
    rtr_var   = np.zeros((num_tasks, num_sps), dtype=float)
    miss_mean = np.zeros((num_tasks, num_sps), dtype=float)
    miss_var  = np.zeros((num_tasks, num_sps), dtype=float)

    for si, sp in enumerate(setpoints):
        trace_folder = find_trace_folder_for_setpoint(root, sp)
        for ti, task_id in enumerate(TASKS):
            task_file = find_task_file(trace_folder, task_id)
            data = _load_json(task_file)

            rtr_mean[ti, si],  rtr_var[ti, si]  = compute_mean_var_from_series(data.get(RTR_KEY), SKIP)
            miss_mean[ti, si], miss_var[ti, si] = compute_mean_var_from_series(data.get(MISS_KEY), SKIP)

    return rtr_mean, rtr_var, miss_mean, miss_var

# ==========================================================
# Plot: 3 DySM bars then 3 Adhoc bars per setpoint (with a gap)
# ==========================================================
def plot_6bars_grouped_by_method(means_by_method, vars_by_method,
                                 ylabel, outpath,
                                 show_errorbars=True,
                                 add_grid=False):

    method_order = ["DySM", "Adhoc"]
    for m in method_order:
        if m not in means_by_method:
            raise KeyError(f"Missing method '{m}' in means_by_method. Found: {list(means_by_method.keys())}")

    T, S = next(iter(means_by_method.values())).shape
    x = np.arange(S)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    width = 0.12
    gap = 0.06

    # 3 bars for each method group
    dysm_offsets  = np.array([-2, -1,  0]) * width
    adhoc_offsets = np.array([ 1,  2,  3]) * width + gap

    method_offsets = {"DySM": dysm_offsets, "Adhoc": adhoc_offsets}

    # workload colors
    workload_colors = ["#4C72B0", "#55A868", "#C44E52"]

    # hatch only for Adhoc
    method_hatches = {"dysm": "", "adhoc": "///"}

    # ---- Plot bars ----
    for mname in method_order:
        for ti in range(T):
            means = means_by_method[mname][ti, :]

            yerr = None
            if show_errorbars and (vars_by_method is not None):
                stds = np.sqrt(np.maximum(vars_by_method[mname][ti, :], 0.0))
                yerr = stds

            bars = ax.bar(
                x + method_offsets[mname][ti],
                means,
                width=width,
                color=workload_colors[ti],
                yerr=yerr,
                capsize=3 if yerr is not None else 0,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
            )

            for b in bars:
                b.set_linewidth(1.2)
                b.set_hatch(method_hatches.get(mname.lower(), ""))

    # ---- Axis formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sp:.2f}" for sp in setpoints], fontsize=14)
    ax.set_xlabel("Set Points", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.spines["top"].set_visible(False)

    if add_grid:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    # ==========================================================
    # Workload legend only (no DySM/Adhoc legend)
    # ==========================================================
    workload_patches = [
        mpatches.Patch(color=workload_colors[i], label=TASK_LABELS[i])
        for i in range(T)
    ]
    ax.legend(
        handles=workload_patches,
        ncol=3, fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=True,
        title_fontsize=14
    )

    # per-setpoint max to place brackets safely
    all_means = np.stack([means_by_method["DySM"], means_by_method["Adhoc"]], axis=0)  # (2,T,S)
    per_sp_max = np.nanmax(all_means, axis=(0, 1))  # (S,)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    text_dy = 0.02 * yr
    br_dy = 0.06 * yr

    for si in range(S):
        y_br = min(per_sp_max[si] + br_dy, y1 - 0.10 * yr)

        # ---- DySM bracket ----
        dysm_left  = x[si] + dysm_offsets.min() - width/2
        dysm_right = x[si] + dysm_offsets.max() + width/2
        #ax.add_patch(FancyArrowPatch(
        #    (dysm_left, y_br), (dysm_right, y_br),
            #arrowstyle="]-[", mutation_scale=15, lw=1.5, color="black"
       # ))
        ax.text(x[si] + np.mean(dysm_offsets), y_br + text_dy, "DySM",
                ha="center", va="bottom", fontsize=14)

        # ---- Adhoc bracket ----
        adhoc_left  = x[si] + adhoc_offsets.min() - width/2
        adhoc_right = x[si] + adhoc_offsets.max() + width/2
 
        ax.text(x[si] + np.mean(adhoc_offsets), y_br + text_dy, "Adhoc",
                ha="center", va="bottom", fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {outpath}")

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    stats = {}
    for mname, root in METHODS:
        rtr_mean, rtr_var, miss_mean, miss_var = collect_stats_for_root(root)
        stats[mname] = {
            "rtr_mean": rtr_mean, "rtr_var": rtr_var,
            "miss_mean": miss_mean, "miss_var": miss_var
        }

    plot_6bars_grouped_by_method(
        means_by_method={m: stats[m]["rtr_mean"] for m, _ in METHODS},
        vars_by_method={m: stats[m]["rtr_var"] for m, _ in METHODS},
        ylabel="Relative Response Time",
        outpath=os.path.join(OUT_DIR, "6_3a.pdf"),
        show_errorbars=True,
        add_grid=True
    )

    plot_6bars_grouped_by_method(
        means_by_method={m: stats[m]["miss_mean"] for m, _ in METHODS},
        vars_by_method=None,
        ylabel="Deadline miss(%)",
        outpath=os.path.join(OUT_DIR, "6_3b.pdf"),
        show_errorbars=False,
        add_grid=False
    )