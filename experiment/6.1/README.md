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

### Local execution notes

Local RTX 5090 notes: activate `../../../.venv/bin/activate` from this directory.
Run `./run_experiment.sh` for the four conference controllers (180 seconds each).
The batch selector is `all`, because scheduler mode 7 is MPC in this project.
A single run, e.g. `./run_experiment.sh 180 task_set.csv logs 3`, writes directly
to `logs/`; the batch retains upstream snapshots in `logs/sol_3` through `sol_6`.
`plotting.py` retains the upstream fixed `logs/` paths and requires all four snapshots.
The current scheduler changes the tunable load at duration/2 (90 seconds by default).
Missing single-run execution and log trailing slashes are fixed. The nonexistent
`figuresplot.py` call is removed; the final upstream aggregate plot is retained.
Failed scheduler runs stop the script. Broad process-name killing is disabled.
Gain generation defaults to `GPU_SMS=170`; it still updates the shared
`Scheduler/gain_matrix.h` and rebuilds the scheduler using the upstream build commands.

### Recorded run status (2026-10-18)

All four 180-second runs and the aggregate figure are included. Result validation
remains open: STGM failed to load its relative-path profiles from this entry and
kept initial allocations. The recorded results preserve this behavior; the STGM
profile-loading issue was not changed here.
