"""Plot this experiment's task logs; adapted from 6.1/plotting.py."""
import csv
import json
import math
from pathlib import Path
from statistics import mean
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent
SKIP = 10  # Same startup omission as 6.1.
AX_FONTSIZE = 16
LINEWIDTH = 2
SETPOINTS = (0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90)
SOLUTIONS = [("Adhoc", 4), ("DySM-P", 3), ("MPC", 7)]


def summarize_task(data, task, task_id, controller):
    """Final 50 periods, excluding the extra endpoint sample, as in the paper."""
    end = len(data["outerrtr_measured"]) - 1
    start = end - 50
    keys = {"rrt": "outerrtr_measured", "dmr": "outerdeadlinemiss", "sms": "outersms"}
    if start < 0 or any(len(data[key]) != end + 1 for key in keys.values()):
        raise ValueError(f"Need 50 aligned periods plus endpoint: {controller}, task {task_id}")
    return dict(task_id=task_id, workload=task["workload"], controller=controller,
                setpoint=float(task["setpoint"]),
                **{f"mean_{metric}": mean(data[key][start:end]) for metric, key in keys.items()},
                window_start=start, window_end_exclusive=end, samples_per_task=50)


def plot_bars(tasks, solutions, records, out_dir, solution, sweep=False, counts=None):
    """Per-task final-50-period means, using the paper's bar-panel size."""
    colors = {4: "#0072B2", 3: "#D55E00", 7: "#009E73"}
    hatches = {4: "++++++", 3: "oooooo", 7: "******"}
    labels = ["mm" if task["workload"] == "mminterupt" else task["workload"]
              for task in tasks]
    targets = sorted({r["setpoint"] for r in records})
    if sweep:
        labels = [f"{target:.2f}" for target in targets]
    if counts is not None:
        labels = [str(count) for count in counts]
    suffix = "" if solution == "all" else f"_sol_{solution}"
    stem = f"bar_{len(tasks)}tasks" + ("_setpoints" if sweep else "")
    if counts is not None:
        stem = f"task_count_4to8_sp{round(targets[0] * 100):03d}"
    with (out_dir / f"{stem}_summary{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": .2}):
        metrics = (("rrt", "Average RRT"), ("dmr", "DMR (%)")) if sweep else (
            ("rrt", "Average RRT"), ("sms", "Average SMs"))
        for metric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(.49 * 252 / 72.27, 1.40))
            ax.set_position([.22, .18 if sweep else .26, .74,
                             fig.get_figwidth() * .74 / (2.7125 / 1.925) / 1.40
                             - (0 if sweep else .08)])
            for index, (name, sid) in enumerate(solutions):
                values = [r[f"mean_{metric}"] for r in records if r["controller"] == name]
                positions = [i + (index - (len(solutions) - 1) / 2) * .24
                             for i in range(len(labels))]
                ax.bar(positions, values, width=.24, label=name, color=colors[sid],
                       hatch=hatches[sid], edgecolor="black", linewidth=.25, zorder=3)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels if sweep else [
                label if i % 2 == 0 else "\n" + label for i, label in enumerate(labels)])
            ax.set_xlim(-.4, len(labels) - (.6 if sweep else 0))
            ax.set_xlabel("Number of Tasks" if counts is not None else "Set Point" if sweep else "Task", labelpad=2)
            ax.set_ylabel(ylabel, labelpad=2)
            if metric == "rrt":
                if counts is not None:
                    ax.axhline(targets[0], color="black", linestyle="--", linewidth=.6,
                               label="Set Point", zorder=4)
                elif sweep:
                    for i, target in enumerate(targets):
                        ax.plot([i - .4, i + .4], [target, target], color="black",
                                linestyle="--", linewidth=.6, zorder=4,
                                label="Set Point" if i == 0 else "_nolegend_")
                else:
                    for target in targets:
                        ax.axhline(target, color="black", linestyle="--", linewidth=.6,
                                   label="Set Point", zorder=4)
                if counts is not None:
                    ax.set_ylim(bottom=0)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                else:
                    ax.set_ylim(.5 if sweep else .6, 1.0)
                    ax.set_yticks([.5, .6, .7, .8, .9, 1.0] if sweep else [.6, .7, .8, .9, 1.0])
            elif metric == "dmr":
                ax.set_ylim(0, max(50.0, min(100.0, max(r["mean_dmr"] for r in records) * 1.05)))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                ax.set_ylim(bottom=0)
                ax.set_yticks([0, math.ceil(ax.dataLim.y1)])
            ax.grid(axis="y", color="#dddddd", linewidth=.2, linestyle="--")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
            ax.tick_params(labelsize=6, width=.35, length=2, pad=1)
            ax.tick_params(axis="x", labelsize=5.5 if sweep else 4.5)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            handles, names = ax.get_legend_handles_labels()
            by_name = dict(zip(names, handles))
            order = [name for name, _ in solutions] + (["Set Point"] if metric == "rrt" else [])
            ax.legend([by_name[name] for name in order], order, frameon=False,
                      loc="lower left", bbox_to_anchor=(-.12, 1.00 if sweep else 1.02), ncol=2,
                      fontsize=6, handlelength=1.15, handletextpad=.35,
                      columnspacing=.6, borderaxespad=0, labelspacing=.15)
            for extension in ("pdf",):
                out = out_dir / f"{stem}_{metric}{suffix}.{extension}"
                fig.savefig(out, dpi=400, bbox_inches=None)
                print("[OK] Saved:", out)
            plt.close(fig)


def collect_setpoint_records(log_dir, task_count=None):
    """Equal-task means across the complete sweep; one PDF per metric."""
    records = []
    reference = None
    for sp in (SETPOINTS if task_count is None else SETPOINTS[1:]):
        folder = log_dir / f"sp_{sp:.2f}"
        if task_count is not None:
            folder = folder / f"n{task_count}"
        config = folder / "task_set.csv"
        if not config.is_file():
            print(f"[WAIT] Setpoint bars need all selected runs; missing results at {sp:.2f}.")
            return
        with config.open(newline="") as f:
            tasks = list(csv.DictReader(f))
        paths = [folder / f"sol_{sid}" / f"task_{i}"
                 for _, sid in SOLUTIONS for i in range(len(tasks))]
        if not all(path.is_file() for path in paths):
            print(f"[WAIT] Missing task logs at setpoint {sp:.2f}.")
            return
        signature = [{key: value for key, value in task.items() if key != "setpoint"}
                     for task in tasks]
        if not tasks or any(float(task["setpoint"]) != sp for task in tasks):
            raise ValueError(f"Invalid task configuration at setpoint {sp:.2f}: {config}")
        if reference is not None and signature != reference:
            print(f"[WAIT] Task configurations differ across setpoints: {config}")
            return
        reference = signature
        for name, sid in SOLUTIONS:
            per_task = []
            for i, task in enumerate(tasks):
                data = json.loads((folder / f"sol_{sid}" / f"task_{i}").read_text())
                per_task.append(summarize_task(data, task, i, name))
            windows = {(r["window_start"], r["window_end_exclusive"]) for r in per_task}
            if len(windows) != 1:
                raise ValueError(f"Task measurement windows differ: {folder}/sol_{sid}")
            start, end = windows.pop()
            records.append(dict(controller=name, setpoint=sp, tasks=len(tasks),
                                mean_rrt=mean(r["mean_rrt"] for r in per_task),
                                mean_dmr=mean(r["mean_dmr"] for r in per_task),
                                window_start=start, window_end_exclusive=end,
                                samples_per_task=50))
    return tasks, records


def plot_setpoint_bars(log_dir, task_count=None):
    result = collect_setpoint_records(log_dir, task_count)
    if result is None:
        return
    tasks, records = result
    out_dir = BASE_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if task_count is not None:
        out_dir = out_dir / f"n{task_count}"
        out_dir.mkdir(parents=True, exist_ok=True)
    plot_bars(tasks, SOLUTIONS, records, out_dir, "all", sweep=True)


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

    height = max(4, 2.5 * len(SOLUTIONS))
    fig, axs = plt.subplots(len(SOLUTIONS), 3, figsize=(16, height), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    metrics = [("outerrtr_measured", "RRT"),
               ("outerdeadlinemiss", "Deadline Misses (%)"), ("outersms", "SMs")]
    records = []
    for row, (name, sid) in enumerate(SOLUTIONS):
        folder = LOG_DIR / f"sol_{sid}" if SOLUTION == "all" else LOG_DIR
        for task_id, task in enumerate(tasks):
            with (folder / f"task_{task_id}").open() as f:
                data = json.load(f)
            records.append(summarize_task(data, task, task_id, name))
            label = {"mminterupt": "mm", "stereo": "stereodisparity"}.get(
                task["workload"], task["workload"])
            for col, (key, ylabel) in enumerate(metrics):
                values = data[key]
                if len(values) <= SKIP:
                    raise ValueError(f"Insufficient samples: {folder}/task_{task_id}: {key}")
                axs[row, col].plot(range(SKIP, len(values)), values[SKIP:],
                                   color=colors[task_id % len(colors)],
                                   linestyle="-" if task_id % 2 == 0 else "-.",
                                   linewidth=LINEWIDTH, label=label)
        for target in sorted({float(task["setpoint"]) for task in tasks}):
            axs[row, 0].axhline(target, color="black", linestyle="--", linewidth=LINEWIDTH,
                                label=f"Setpoint ({target:.2f})")
        for col, (_, ylabel) in enumerate(metrics):
            ax = axs[row, col]
            ax.set_ylabel(ylabel, fontsize=AX_FONTSIZE)
            ax.set_xlabel(f"Control Period\n\n({chr(ord('a') + row * 3 + col)}) {name}",
                          fontsize=AX_FONTSIZE)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis="both", labelsize=AX_FONTSIZE)
        axs[row, 0].set_ylim(0, 2.5)
        axs[row, 1].set_ylim(0, 100)
        ax = axs[row, 2]
        ax.set_yticks([math.floor(ax.dataLim.y0), math.ceil(ax.dataLim.y1)])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=AX_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.4 * ((len(labels) + 2) // 3) / height])
    try:
        relative = LOG_DIR.resolve().relative_to((BASE_DIR / "logs").resolve())
    except ValueError:
        relative = Path(LOG_DIR.name)
    out_dir = BASE_DIR / "Figures" / relative
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "comparison" if SOLUTION == "all" else f"sol_{SOLUTION}"
    out = out_dir / f"runtime_{len(tasks)}tasks_{tag}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", out)
    plot_bars(tasks, SOLUTIONS, records, out_dir, SOLUTION)
    if SOLUTION == "all":
        if LOG_DIR.name.startswith("n") and LOG_DIR.name[1:].isdigit():
            plot_setpoint_bars(LOG_DIR.parent.parent, len(tasks))
        else:
            plot_setpoint_bars(LOG_DIR.parent)


def plot_task_count_bars(log_dir, setpoint):
    """Reuse the existing final-50 statistics and bar renderer for 4–8 tasks."""
    counts = tuple(range(4, 9))
    records = []
    for count in counts:
        folder = log_dir / f"n{count}"
        config = folder / "task_set.csv"
        paths = [folder / f"sol_{sid}" / f"task_{i}"
                 for _, sid in SOLUTIONS for i in range(count)]
        if not config.is_file() or not all(path.is_file() for path in paths):
            print(f"[WAIT] Task-count bars need complete results: {folder}")
            return
        with config.open(newline="") as f:
            tasks = list(csv.DictReader(f))
        if len(tasks) != count or any(float(t["setpoint"]) != setpoint for t in tasks):
            raise ValueError(f"Unexpected task configuration: {config}")
        for name, sid in SOLUTIONS:
            per_task = []
            for i, task in enumerate(tasks):
                data = json.loads((folder / f"sol_{sid}" / f"task_{i}").read_text())
                per_task.append(summarize_task(data, task, i, name))
            records.append(dict(controller=name, setpoint=setpoint, tasks=count,
                                mean_rrt=mean(r["mean_rrt"] for r in per_task),
                                mean_dmr=mean(r["mean_dmr"] for r in per_task),
                                samples_per_task=50))
    out_dir = BASE_DIR / "Figures" / f"sp_{setpoint:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bars(tasks, SOLUTIONS, records, out_dir, "all", sweep=True, counts=counts)


def draw_setpoint_panel(ax, records, metric):
    """One shared renderer for standalone panels and the paired PDF pages."""
    targets = sorted({r["setpoint"] for r in records})
    colors = {4: "#0072B2", 3: "#D55E00", 7: "#009E73"}
    hatches = {4: "++++++", 3: "oooooo", 7: "******"}
    markers = ("o", "s", "^")
    x = list(range(len(targets)))
    for j, (name, sid) in enumerate(SOLUTIONS):
        values = [next(r[f"mean_{metric}"] for r in records
                       if r["controller"] == name and r["setpoint"] == sp)
                  for sp in targets]
        if metric == "rrt":
            capped = sid == 4 and any(value > 1.0 for value in values)
            shown = [min(value, 1.0) for value in values] if capped else values
            ax.plot(x, shown, color=colors[sid], marker=markers[j],
                    linewidth=.6, markersize=2, markeredgewidth=.3,
                    label=name, zorder=3, clip_on=not capped)
            if capped:
                ax.text(.5, .97, "Adhoc (capped at 1)", transform=ax.transAxes,
                        ha="center", va="top", fontsize=5, color=colors[sid])
        else:
            ax.bar([i + (j - 1) * .24 for i in x], values, width=.24,
                   color=colors[sid], hatch=hatches[sid], edgecolor="black",
                   linewidth=.25, label=name, zorder=3)
    if metric == "rrt":
        ax.plot(x, targets, color="black", linestyle="--", linewidth=.6,
                label="Set Point", zorder=2)
        upper = 1.0
        lower = min(.5, math.floor(min(r["mean_rrt"] for r in records) * 10) / 10)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        ax.set_ylim(0, max(50.0, min(100.0, max(r["mean_dmr"] for r in records) * 1.05)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_xlim(-.4, len(targets) - .6)
    ax.set_xticks(x, labels=[f"{sp:.2f}" for sp in targets])
    ax.set_xlabel("Set Point", labelpad=2)
    ax.set_ylabel("Average RRT" if metric == "rrt" else "DMR (%)", labelpad=2)
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


def plot_setpoint_panels(log_dir):
    """4–8 tasks: ten standalone panels and five paired pages, same statistics."""
    grouped = {}
    for count in range(4, 9):
        result = collect_setpoint_records(log_dir, count)
        if result is None:
            return
        tasks, records = result
        if len(tasks) != count:
            raise ValueError(f"Expected {count} tasks, found {len(tasks)}")
        grouped[count] = records
    out_dir = BASE_DIR / "Figures"
    panels_dir = out_dir / "4-8subfigures"
    panels_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "4-8_setpoints_summary.csv").open("w", newline="") as f:
        rows = [r for records in grouped.values() for r in records]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path = out_dir / "4-8_setpoints_rrt_dmr.pdf"
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": .2}), PdfPages(path) as pdf:
        panel_width = .49 * 252 / 72.27
        width = panel_width * .74
        height = width / (2.7125 / 1.925)
        for count, records in grouped.items():
            for metric in ("rrt", "dmr"):
                fig = plt.figure(figsize=(panel_width, 1.40))
                ax = fig.add_axes([.22, .18, .74, height / 1.40])
                draw_setpoint_panel(ax, records, metric)
                handles, labels = ax.get_legend_handles_labels()
                by_name = dict(zip(labels, handles))
                order = [name for name, _ in SOLUTIONS] + (["Set Point"] if metric == "rrt" else [])
                ax.legend([by_name[name] for name in order], order, frameon=False,
                          loc="lower left", bbox_to_anchor=(-.12, 1), ncol=2,
                          fontsize=6, handlelength=1.15, handletextpad=.35,
                          columnspacing=.6, borderaxespad=0, labelspacing=.15)
                out = panels_dir / f"{count}tasks_{metric}.pdf"
                fig.savefig(out, bbox_inches=None)
                plt.close(fig)
                print("[OK] Saved:", out)
            fig = plt.figure(figsize=(3.5, 1.85))
            for index, metric in enumerate(("rrt", "dmr")):
                left = panel_width * .22 + index * 1.75
                ax = fig.add_axes([left / 3.5, .40 / 1.85, width / 3.5, height / 1.85])
                draw_setpoint_panel(ax, records, metric)
                ax.set_title(f"({chr(97 + index)}) {count} tasks — {metric.upper()}",
                             fontsize=6, pad=4)
            handles, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98),
                       ncol=4, frameon=False, fontsize=6, handlelength=1.15,
                       handletextpad=.35, columnspacing=.8)
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
    print("[OK] Saved:", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--panels":
        plot_setpoint_panels(Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "logs")
    elif len(sys.argv) > 1 and sys.argv[1] == "--taskcounts":
        folder = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "logs/sp_0.80"
        plot_task_count_bars(folder, float(sys.argv[3]) if len(sys.argv) > 3 else .80)
    elif len(sys.argv) > 1 and sys.argv[1] == "--setpoints":
        plot_setpoint_bars(Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "logs",
                           int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif len(sys.argv) > 1:
        plot(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "all")
    else:
        configs = [BASE_DIR / "logs/sp_0.80/task_set.csv"]
        configs = [config for config in configs if config.is_file()]
        if not configs:
            raise SystemExit("No 0.80 logs found under logs/sp_0.80/")
        for config in configs:
            plot(config, config.parent)
