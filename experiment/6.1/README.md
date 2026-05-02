## Comparison with Baselines under Runtime Workload Variations

This experiment evaluates how different GPU scheduling strategies handle dynamic workload changes at runtime.

Compared Approaches

* DySM (proposed)
* SMshare (equal partitioning)
* STGM (static allocation)
* Adhoc (heuristic controller)

⸻

# Experiment Description

We evaluate two GPU workloads:

* mm (matrix multiplication)
* stereodisparity

At control period 80, the workload of mm is increased to emulate real-world scenarios such as increased scene complexity (e.g., dense traffic in autonomous driving).
Metrics Evaluated

* Relative Response Time (RRT)
* Deadline Miss Rate
* SM (Streaming Multiprocessor) Allocation Dynamics

⸻

# Running the Experiment
./run_experiment.sh

⸻

Output

All results are saved in the Figures/ directory.
Generated Plots

* RRT over time
* Deadline misses
* SM allocation per task

Example Output
Figures/stgm_adhoc_proposed_1_3x3_RTR_DM_SMS.pdf
⸻

Expected Behavior

* SMshare / STGM (Open-loop):
    Cannot adapt to runtime workload changes → results in deadline misses after control period 80.
* Adhoc (Heuristic):
    Adapts to workload changes but exhibits oscillations due to fixed step size.
* DySM (Proposed):
    Dynamically adjusts SM allocation using a MIMO controller, maintaining RRT near the setpoint with minimal deadline misses.

⸻
Notes
* The workload change is triggered automatically at control period 80.
* No manual intervention is required.
* Ensure GPU access is available before running the experiment.
⸻
