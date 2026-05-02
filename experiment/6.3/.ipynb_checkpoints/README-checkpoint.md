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
