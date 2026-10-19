"""Compare 4–7 tasks at setpoints 0.70 and 0.80 using the existing task-log fields."""
import csv
import json
from pathlib import Path
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
TASK_COUNTS = (4, 5, 6, 7)
SOLUTIONS = (("Adhoc", 4), ("DySM-P", 3), ("MPC", 7))
WINDOW = 50  # Same final-50-period window as the current 6.5 bar plots.


# ==========================================================
# Load Data and Stats
# ==========================================================
def collect_stats(log_dir, solution="all", setpoint=0.80):
    methods = [(name, sid) for name, sid in SOLUTIONS
               if solution == "all" or str(sid) == solution]
    if not methods:
        raise ValueError("Use all or controller 4, 3, 7")
    records = []
    for count in TASK_COUNTS:
        point = Path(log_dir) / f"n{count}"
        with (point / "task_set.csv").open(newline="") as f:
            tasks = list(csv.DictReader(f))
        if len(tasks) != count or any(float(t["setpoint"]) != setpoint for t in tasks):
            raise ValueError(f"Expected {count} tasks at setpoint {setpoint:.2f}: {point}")
        for name, sid in methods:
            folder = point / f"sol_{sid}" if solution == "all" else point
            rrt, dmr = [], []
            for task_id in range(count):
                with (folder / f"task_{task_id}").open() as f:
                    data = json.load(f)
                end = len(data["outerrtr_measured"]) - 1
                start = end - WINDOW
                if start < 0 or len(data["outerdeadlinemiss"]) != end + 1:
                    raise ValueError(f"Need 50 aligned periods plus endpoint: {folder}/task_{task_id}")
                for key, values in (("outerrtr_measured", rrt),
                                    ("outerdeadlinemiss", dmr)):
                    series = data[key][start:end]
                    if not series:
                        raise ValueError(f"Insufficient samples: {folder}/task_{task_id}: {key}")
                    values.append(mean(series))
            # Average in time per task, then give every task equal weight.
            records.append({"tasks": count, "controller": name, "solution": sid,
                            "setpoint": setpoint, "mean_rrt": mean(rrt),
                            "mean_dmr_percent": mean(dmr)})
    return records


# ==========================================================
# Plot: independent RRT and DMR panels, copied from 6.5
# ==========================================================
def plot_bars(records, solutions, out_dir, solution, setpoint):
    """Use the same independent bar-panel canvas and style as 6.5."""
    colors = {4: "#0072B2", 3: "#D55E00", 7: "#009E73"}
    hatches = {4: "++++++", 3: "oooooo", 7: "******"}
    labels = [str(n) for n in TASK_COUNTS]
    suffix = "comparison" if solution == "all" else f"sol_{solution}"
    stem = f"task_count_4to7_sp{round(setpoint * 100):03d}_{suffix}"
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": .2}):
        metrics = (("rrt", "Average RRT"), ("dmr_percent", "DMR (%)"))
        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(.49 * 252 / 72.27, 1.40))
            ax.set_position([.22, .18, .74,
                             fig.get_figwidth() * .74 / (2.7125 / 1.925) / 1.40])
            for index, (name, sid) in enumerate(solutions):
                values = [r[f"mean_{metric}"] for r in records if r["controller"] == name]
                positions = [i + (index - (len(solutions) - 1) / 2) * .24
                             for i in range(len(labels))]
                ax.bar(positions, values, width=.24, label=name, color=colors[sid],
                       hatch=hatches[sid], edgecolor="black", linewidth=.25, zorder=3)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlim(-.4, len(labels) - .6)
            ax.set_xlabel("Number of Tasks", labelpad=2)
            ax.set_ylabel(ylabel, labelpad=2)
            if metric == "rrt":
                ax.axhline(setpoint, color="black", linestyle="--", linewidth=.6,
                           label="Set Point", zorder=4)
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            elif metric == "dmr_percent":
                ax.set_ylim(0, max(50.0, min(100.0, max(r["mean_dmr_percent"] for r in records) * 1.05)))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(axis="y", color="#dddddd", linewidth=.2, linestyle="--")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
            ax.tick_params(labelsize=6, width=.35, length=2, pad=1)
            ax.tick_params(axis="x", labelsize=5.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            handles, names = ax.get_legend_handles_labels()
            by_name = dict(zip(names, handles))
            order = [name for name, _ in solutions] + (["Set Point"] if metric == "rrt" else [])
            ax.legend([by_name[name] for name in order], order, frameon=False,
                      loc="lower left", bbox_to_anchor=(-.12, 1.00), ncol=2,
                      fontsize=6, handlelength=1.15, handletextpad=.35,
                      columnspacing=.6, borderaxespad=0, labelspacing=.15)
            for extension in ("pdf",):
                out = out_dir / f"{stem}_{'dmr' if metric == 'dmr_percent' else metric}.{extension}"
                fig.savefig(out, dpi=400, bbox_inches=None)
                print("[OK] Saved:", out)
            plt.close(fig)


def plot(log_dir, solution="all", setpoint=0.80):
    records = collect_stats(log_dir, solution, setpoint)
    methods = [(name, sid) for name, sid in SOLUTIONS
               if solution == "all" or str(sid) == solution]
    out_dir = BASE_DIR / "Figures" / f"sp_{setpoint:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "comparison" if solution == "all" else f"sol_{solution}"
    summary = out_dir / f"task_count_4to7_sp{round(setpoint * 100):03d}_{tag}.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    plot_bars(records, methods, out_dir, solution, setpoint)
    print("[OK] Saved:", summary)


if __name__ == "__main__":
    plot(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "logs/sp_0.80",
         sys.argv[2] if len(sys.argv) > 2 else "all",
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.80)
