# Comparison under Different Set Points

This experiment evaluates how different closed-loop GPU scheduling strategies perform under varying control set points.

⸻

Compared Approaches

* DySM (proposed)
* Adhoc (heuristic controller)

⸻

Experiment Description

We evaluate three GPU workloads:

* projection
* histogram
* stencil

Each task is assigned a target Relative Response Time (RRT) set point, varied from 0.50 to 0.80.

The system is allowed to reach steady state for each set point, and measurements are taken after transient effects subside.

⸻

Metrics Evaluated

* Average Relative Response Time (RRT)
* Standard deviation of RRT
* Deadline Miss Rate (%)

⸻

Running the Experiment

./experiments.sh

⸻

Output

All results are saved in the Figures/ directory.

Generated Plots

* RRT across different set points
* Deadline miss rate across different set points

Example Output

Figures/rrt_setpoints_6bars_grouped.pdf
Figures/dmr_setpoints_6bars_grouped.pdf

⸻

Expected Behavior

* Adhoc (Heuristic):
    Uses fixed step-size control per task.
    * Cannot consistently converge to desired set points
    * Exhibits oscillations due to independent controllers and lack of coordination
    * May cause SM/TPC overlap, leading to increased contention
* DySM (Proposed):
    Uses a control-theoretic MIMO controller.
    * Accurately drives all tasks toward their target set points
    * Maintains low variance in RRT
    * Minimizes deadline misses across all set points

### Local execution notes

Local RTX 5090 notes: activate `../../../.venv/bin/activate` and run `./experiments.sh`.
The two upstream runners remain independent, each defaulting to 150 seconds.
P sweeps 0.50, 0.60, 0.70, 0.80; Adhoc additionally sweeps 0.90, exactly as upstream.
Raw logs are staged in `logs/`, then archived under `3runs/` (P) and `3runs_adhoc/`.
The unchanged upstream figure compares only the four shared setpoints; Adhoc's
0.90 result remains available in its archive. It is not silently removed.
Aggregate plotting occurs after both sweeps in `experiments.sh`; calls before
all required logs exist are removed. Per-run archiving moves logs only, because
searching across the whole experiment and Scheduler can move unrelated figures.
Failed scheduler runs stop the script. Broad process-name killing is disabled.
Gain generation defaults to `GPU_SMS=170`; it still updates the shared
`Scheduler/gain_matrix.h` and rebuilds the scheduler using the upstream build commands.

### Recorded run status (2026-10-18)

Adhoc completed all five upstream setpoints and P completed all four, at 150
seconds per run. Archived logs and Figures/6_3a.pdf and Figures/6_3b.pdf are
included. Detailed interpretation and validation of controller behavior remain pending.
