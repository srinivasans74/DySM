# 6.5 — Task-count and setpoint sweeps

The merged experiment uses 4–8 fixed-input tasks at setpoints 0.80, 0.82,
0.84, 0.86, 0.88, and 0.90. Controllers are Adhoc (4), P (3), and MPC (7),
400 seconds each. No runtime workload increase is enabled.

Task order: mm, projection, hist, stencil, stereo, hotspot, blackscholes,
reduction. The first seven rows are unchanged. Reduction uses the calibrated
6.9 period (41.617 ms) and model coefficient (e = 70.66974933141243).
The existing gain.py and scheduler build/run commands are retained.

## Run only the remaining experiments

From experiment/6.5 with the existing virtual environment active:

```bash
bash run_remaining.sh 400 task_set.csv logs all
```

This is a copy of run_experiment.sh; only TASK_COUNTS differs (4, 6, 8).
Matching complete controller logs are skipped. Initially, 4 and 6 tasks at
0.80 are complete, leaving 48 controller runs (5 hours 20 minutes plus builds).
Rerunning the command resumes missing/incomplete controllers.

For the complete 4–8 task sweep, with the same skip logic:

```bash
bash run_experiment.sh 400 task_set.csv logs all
```

Both scripts regenerate B/K and rebuild only for task/setpoint groups that
still need a controller run. They retain the current gain settings and 170-SM
budget. Do not run experiments sharing Scheduler concurrently.

## Results and reuse

```text
logs/sp_0.80/n4/task_set.csv
logs/sp_0.80/n4/sol_4/task_0 ...
logs/sp_0.80/n4/sol_3/task_0 ...
logs/sp_0.80/n4/sol_7/task_0 ...
...
logs/sp_0.90/n8/
```

Existing 6.7 results at 0.70 and 0.80 for 4–7 tasks were copied into this
structure. For 0.82–0.90, five-task results came from logs/5tasks and
seven-task results came from the original logs/sp_* directories.
Original locations, including experiment/6.7, are retained.
For the requested 0.80–0.90 sweep, 42 controller runs are reusable and 48 remain.

## Plot without running experiments

```bash
# Time series and per-task bars for one group:
python3 plotting.py logs/sp_0.80/n7/task_set.csv logs/sp_0.80/n7 all
# Fixed task count, compare six setpoints:
python3 plotting.py --setpoints logs 7
# Fixed setpoint, compare 4–8 tasks:
python3 plotting.py --taskcounts logs/sp_0.80 0.80
```

Only PDF images and CSV summaries are generated. Layout, colors, staggered
task names, and two SM tick labels follow the existing 6.5 template.
Statistics remain the final 50 periods excluding the extra endpoint
([350:400] for a 401-sample log), averaging tasks equally.
Incomplete aggregate plots wait for their missing groups.
The legacy flat-log plotting commands remain supported.

## Paired setpoint figures (4–8 tasks)

```bash
python3 plotting.py --panels logs
```

`Figures/4-8subfigures/` contains ten standalone PDFs: an RRT line plot
and a DMR bar plot for each task count. `Figures/4-8_setpoints_rrt_dmr.pdf`
contains five pages, one task set per page (RRT left, DMR right), drawn
with the same panel function. `Figures/4-8_setpoints_summary.csv` records
the final-50-period equal-task means used by both outputs. The RRT axis
has a fixed upper limit of 1.0.
