# 6.7 — Four to seven tasks at setpoints 0.70 and 0.80

This independent runner compares Adhoc (4), DySM-P (3), and MPC (7),
400 seconds each, using the current scheduler and actuator. It does not
launch another experiment automatically.

The template matches the current 6.5 workload order and calibrated e values:

| Task count | Workloads |
|---|---|
| 4 | mm, projection, hist, stencil |
| 5 | Above + stereo |
| 6 | Above + hotspot |
| 7 | Above + blackscholes |

For each setpoint (0.70, 0.80) and task count, the runner takes the first N
rows, sets the common setpoint, regenerates B and K with GPU_SMS=170, and
rebuilds the scheduler. Inputs remain fixed: ordinary mm is used, with no
runtime workload increase. Controller gains, constraints, weights, task
periods and the actuator are not modified by this experiment.

## Run

Wait for the current 6.5 sweep to finish: both experiments share the scheduler
build and GPU. Start this experiment yourself:

```bash
cd ~/ECRTS_original_20261017/ECRTS_25/experiment/6.7
source ../../../.venv/bin/activate
bash run_experiment.sh 400 task_set.csv logs all
```

The complete sweep has 24 runs: 160 minutes plus builds and initialization.
The first argument changes duration; the fourth selects all or 4/3/7.

## Results

```text
logs/sp_0.70/n4/sol_4/task_0
logs/sp_0.70/n5/...
logs/sp_0.70/n6/...
logs/sp_0.70/n7/...
logs/sp_0.80/n4/... through n7/...
Figures/sp_0.70/task_count_4to7_sp070_comparison.pdf
Figures/sp_0.80/task_count_4to7_sp080_comparison.pdf
```

Each PDF retains the existing RRT/DMR two-panel grouped bars, with task count
on the x axis. The corresponding CSV contains the plotted means. Statistics
use the final 50 control-period samples excluding the extra endpoint
([350:400] for 401 records), averaging each task in time and then giving tasks
equal weight, matching the current 6.5 summary convention.

Replot without running the GPU experiment:

```bash
python3 plotting.py logs/sp_0.70 all 0.70
python3 plotting.py logs/sp_0.80 all 0.80
```

Previous n4–n8 logs and their old configuration/scripts are archived under
logs/old_executor/. They are excluded from the new sweep and figures.
