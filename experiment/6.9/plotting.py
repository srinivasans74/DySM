"""MPC decision overhead for 1–10 tasks, using the paper's 150:200 window."""
import csv
import math
from pathlib import Path
import re
from statistics import mean
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ==========================================================
# Config
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
TASK_COUNTS = tuple(range(1, 11))
START, END = 150, 200  # Zero-based valid decision records, end exclusive.
SETPOINT = 0.80
TIMING = re.compile(
    r"\[CONTROL-TIMING\] solution=(\d+) tasks=(\d+) period=(\d+) decision_us=(\S+)"
)


# ==========================================================
# Load Data and Stats
# ==========================================================
def collect_stats(log_dir, task_counts=TASK_COUNTS):
    records = []
    for count in task_counts:
        point = Path(log_dir) / f"n{count}"
        with (point / "task_set.csv").open(newline="") as f:
            tasks = list(csv.DictReader(f))
        if len(tasks) != count or any(float(t["setpoint"]) != SETPOINT for t in tasks):
            raise ValueError(f"Expected {count} tasks at setpoint 0.80: {point}")
        path = point / "sol_7" / "console.log"
        samples = {}
        for match in TIMING.finditer(path.read_text()):
            solution, measured_count, period = map(int, match.group(1, 2, 3))
            value = float(match.group(4))
            if solution != 7 or measured_count != count:
                raise ValueError(f"Wrong controller/task count in {path}")
            if period in samples or not math.isfinite(value) or value < 0:
                raise ValueError(f"Duplicate/invalid timing record {period} in {path}")
            samples[period] = value
        if any(period not in samples for period in range(END)):
            raise ValueError(
                f"Need valid decision records 0–{END - 1}: {path}. "
                "Check the run or rerun with a longer duration; no figure was generated."
            )
        records.append({"tasks": count, "controller": "MPC", "solution": 7,
                        "setpoint": SETPOINT, "window_start": START,
                        "window_end_exclusive": END, "samples": END - START,
                        "mean_us": mean(samples[i] for i in range(START, END))})
    return records


# ==========================================================
# Plot: MPC decision overhead
# ==========================================================
def overhead_figure(rows, metric, method="MPC"):
    """Single-controller timing in the compact blue-line style of PowerGPU Fig. 17."""
    labels = {'mean_us': 'Mean (µs)', 'median_us': 'Median (µs)',
              'observed_max_us': 'Max. (µs)'}
    fig, ax = plt.subplots(figsize=(1.75, 1.20))
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel(labels[metric])
    ax.spines["top"].set_visible(False)
    ax.set_position([.23, .24, .74, .70])
    selected = sorted((r for r in rows if r['controller'] == method),
                      key=lambda r: int(r['tasks']))
    x = [int(r['tasks']) for r in selected]
    y = [float(r[metric]) for r in selected]
    ax.plot(x, y, color='blue', label=method, linewidth=.6,
            marker='o', markersize=2, markeredgewidth=.3, zorder=3)
    ax.grid(False)
    ax.spines['right'].set_visible(True)
    ax.set_xlim(.5, max(x) + .5)
    ax.set_xticks(range(2, max(x) + 1, 2))
    ax.set_xticks(range(1, max(x) + 1, 2), minor=True)
    ax.set_ylim(0, max(y) * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.set_yticks([y for y in ax.get_yticks() if 0 <= y <= ax.get_ylim()[1]])
    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontsize(6)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2
    ax.tick_params(labelsize=6, width=.35, length=2, pad=1)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(.35)
    return fig

def plot(log_dir):
    records = collect_stats(log_dir)
    # Copy the compact PowerGPU overhead panel from individual_panels.py.
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = overhead_figure(records, "mean_us")
        fig.axes[0].set_ylabel("Execution time (µs)")
        out_dir = BASE_DIR / "Figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / "overhead_mpc"
        for extension in ("pdf",):
            path = stem.with_suffix("." + extension)
            fig.savefig(path, dpi=400, bbox_inches=None)
            print("[OK] Saved:", path)
        plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print("[OK] Saved:", stem.with_suffix(".csv"))


if __name__ == "__main__":
    plot(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "logs")
