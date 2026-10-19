"""Two-instance MPC inference panels in the existing paper's single-column style."""
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
SKIP = 10
COLORS = ("#0072B2", "#D55E00")


def plot(log_dir):
    log_dir = Path(log_dir)
    run = json.loads((log_dir / "run.json").read_text())
    if run["status"] != "complete":
        raise ValueError("The inference run did not complete; inspect run.json")
    tasks = [json.loads((log_dir / f"task_{i}.json").read_text()) for i in range(2)]
    pulses = json.loads((log_dir / "actuation.json").read_text())
    if not pulses:
        raise ValueError("Missing applied SM records")
    for task in tasks:
        if len(task["windows"]) != run["completed_control_periods"]:
            raise ValueError("Task/control log lengths differ")
    epoch = run["epoch_ns"]
    duration = run["config"]["periods"] * run["config"]["control_period_s"]
    setpoint = run["config"]["setpoint"]
    sms_limit = max(60, 30 * math.ceil(max(max(r["physical_sms"]) for r in pulses) / 30))
    out_dir = BASE_DIR / "Figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Geometry and typography from single_column_runtime_figure() / render_run008.py.
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 6, "axes.labelweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for metric, key, ylabel, ylim, yticks in [
                ("rrt", "rtr", "RRT", (0, 4), [0, 2, 4]),
                ("dmr", "deadline_miss_pct", "DMR (%)", (0, 105), [0, 50, 100]),
                ("sms", None, "SMs", (0, sms_limit), [0, sms_limit / 2, sms_limit])]:
            fig, ax = plt.subplots(figsize=(.32 * 252 / 72.27, 1.10))
            ax.set_position([.29, .30, .68,
                             fig.get_figwidth() * .68 / (2.7125 / 1.925) / 1.10])
            for i, task in enumerate(tasks):
                if metric == "sms":
                    # Keep the existing plot's actual physical allocation.
                    times = [max(0., (r["stream_time_ns"][i] - epoch) / 1e9) for r in pulses]
                    ax.step(times, [r["physical_sms"][i] for r in pulses], where="post",
                            color=COLORS[i], linewidth=.6, label=f"Instance {i + 1}", zorder=3)
                else:
                    windows = task["windows"][SKIP:]
                    times = [(r["planned_end_ns"] - epoch) / 1e9 for r in windows]
                    values = [float("nan") if r[key] is None else r[key] for r in windows]
                    ax.plot(times, values, color=COLORS[i], linewidth=.6,
                            label=f"Instance {i + 1}", zorder=3)
            if metric == "rrt":
                ax.axhline(setpoint, color="black", linestyle="--", linewidth=.6,
                           label="Set Point", zorder=2)
            ax.set(xlabel="Time (s)", ylabel=ylabel, xlim=(0, duration), ylim=ylim)
            # Keep labels in the blank regions; arrows mark logged batch changes.
            if metric == "rrt":
                for batch, label, x, y, tip in [
                        (40, "Batch ↑", .74, 1.17, .96),
                        (20, "Batch ↓", .82, .52, .18)]:
                    times = sorted({(event["planned_ns"] - epoch) / 1e9
                                    for event in run["batch_events"]
                                    if event["batch_size"] == batch})
                    if not times:
                        continue
                    ax.text(x, y, label, transform=ax.transAxes, color="red",
                            fontsize=5, ha="center", va="center", zorder=5)
                    for when in (times[:1] if batch == 40 else times):
                        ax.annotate("", xy=(when, tip),
                                    xytext=(x, y - .05),
                                    xycoords=("data", "axes fraction"),
                                    textcoords="axes fraction",
                                    arrowprops=dict(arrowstyle="->", lw=.6, color="black"),
                                    zorder=5)
            ax.set_xticks([0, duration / 2, duration])
            ax.set_yticks(yticks)
            ax.xaxis.labelpad = ax.yaxis.labelpad = 2
            if metric == "dmr":
                ax.yaxis.labelpad = -1
                ax.yaxis.label.set_linespacing(.9)
            ax.grid(axis="y", color="#dddddd", linewidth=.2, linestyle="--")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
            ax.tick_params(labelsize=6, width=.35, length=2, pad=1)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
            ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.34, 1.085),
                      ncol=2, fontsize=6, handlelength=1.15, handletextpad=.35,
                      columnspacing=.2, borderpad=0, borderaxespad=0, labelspacing=.15)
            for extension in ("pdf",):
                path = out_dir / f"fig_rtx5090_ml_inference_{metric}.{extension}"
                fig.savefig(path, dpi=400, bbox_inches=None)
                print("[OK] Saved:", path)
            plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "logs/sol_7")
