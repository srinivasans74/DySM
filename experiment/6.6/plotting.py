"""Plot this experiment's task logs; adapted from 6.1/plotting.py."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE_DIR = Path(__file__).resolve().parent
SKIP = 10  # Same startup omission as 6.1.
def plot(CSV_PATH, LOG_DIR, SOLUTION="all"):
    SOLUTIONS = [("Adhoc", 4), ("DySM-P", 3), ("MPC", 7)]
    with CSV_PATH.open(newline="") as f:
        tasks = list(csv.DictReader(f))
    if not tasks:
        raise ValueError("Empty task CSV")
    if SOLUTION != "all":
        SOLUTIONS = [(name, sid) for name, sid in SOLUTIONS if str(sid) == SOLUTION]
    if not SOLUTIONS:
        raise ValueError("Unsupported solution for this experiment")

    fig, axs = plt.subplots(len(SOLUTIONS), 3, figsize=(16, 4 * len(SOLUTIONS)), squeeze=False)
    colors = plt.get_cmap("tab20")
    metrics = [("outerrtr_measured", "RRT"),
               ("outerdeadlinemiss", "Deadline Misses (%)"), ("outersms", "SMs")]
    for row, (name, sid) in enumerate(SOLUTIONS):
        folder = LOG_DIR / f"sol_{sid}" if SOLUTION == "all" else LOG_DIR
        for task_id, task in enumerate(tasks):
            with (folder / f"task_{task_id}").open() as f:
                data = json.load(f)
            label = "mm" if task["workload"] == "mminterupt" else task["workload"]
            for col, (key, ylabel) in enumerate(metrics):
                values = data[key]
                if len(values) <= SKIP:
                    raise ValueError(f"Insufficient samples: {folder}/task_{task_id}: {key}")
                axs[row, col].plot(range(SKIP, len(values)), values[SKIP:],
                                   color=colors(task_id), linewidth=1, label=label)
        for target in sorted({float(task["setpoint"]) for task in tasks}):
            axs[row, 0].axhline(target, color="black", linestyle="--",
                                label=f"Setpoint ({target:.2f})")
        for col, (_, ylabel) in enumerate(metrics):
            ax = axs[row, col]
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f"Control Period\n({chr(ord('a') + row * 3 + col)}) {name}")
            ax.spines["top"].set_visible(False)
            ax.set_ylim(bottom=0)
        axs[row, 1].set_ylim(0, 100)
        axs[row, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_dir = BASE_DIR / "Figures" / LOG_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "comparison" if SOLUTION == "all" else f"sol_{SOLUTION}"
    out = out_dir / f"runtime_{len(tasks)}tasks_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "all")
    else:
        configs = [BASE_DIR / "logs/sp_0.90/task_set.csv"]
        configs = [config for config in configs if config.is_file()]
        if not configs:
            raise SystemExit("No 0.90 logs found under logs/sp_0.90/")
        for config in configs:
            plot(config, config.parent)
