# 6.6 — Fourteen workloads, six setpoints

This experiment keeps all seven tasks from 6.5, in the same order, and appends
seven workloads from `application/`. The added tasks are registered in
`Scheduler/cuda_runner.cpp` and included in `Scheduler/Makefile`.
The runner and plotter follow 6.5; only the runner caption and task CSV differ.
No GPU experiment has been launched as part of this integration.

## Task configuration and model inputs

Rows determine task IDs and the row/column order of both K and B. The same CSV
is used for gain generation and scheduler execution. `task_rate` is the period
in milliseconds, despite the column name.

| Task ID | Workload | Period (ms) | CSV e |
|---|---|---:|---:|
| 0 | mminterupt | 45.330 | blank: existing gain.py table |
| 1 | hotspot | 21.262 | blank: existing gain.py table |
| 2 | hist | 22.692 | blank: existing gain.py table |
| 3 | stencil | 31.500 | blank: existing gain.py table |
| 4 | stereo | 21.256 | blank: existing gain.py table |
| 5 | projection | 39.947 | blank: existing gain.py table |
| 6 | mttkrp | 30.000 | blank: existing gain.py table |
| 7 | blackscholes | 10.000 | 8.24 |
| 8 | reduction | 5.000 | 3.62 |
| 9 | scan | 10.000 | 8.62 |
| 10 | transpose | 1.000 | 0.55 |
| 11 | convolution_separable | 20.000 | 14.68 |
| 12 | nbody | 20.000 | 18.52 |
| 13 | fdtd3d | 10.000 | 7.28 |

The first seven rows preserve 6.5's periods, tunable fields, ranges and model
inputs. See `../6.5/README.md` for their original sources, including the
provisional MTTKRP period. The appended rows retain their periods and raw `e`
values from the previous 6.6 draft, originally copied from
`metrics/runtime_workload_variation_z09_controller_comparison_20260914/raw/rtx5090/z0.90/n14/DySM/task_set.csv`
in the journal workspace. They are inherited inputs, not new measurements of
the current application kernels. The kernel/old-adapter differences are recorded
in `../../application/NEW_WORKLOADS.md`.

`gain.py` is unchanged: its existing `EXEC_TIME_SCALE=10` applies to CSV `e`
values and table values alike. Empty `e` for the first seven tasks deliberately
uses their existing table entries; each appended task has an explicit value,
so none uses the script's unknown-workload fallback. No calibration or controller
retuning is part of this integration.

## Setpoints, controllers and workload change

- Setpoints: 0.80, 0.82, 0.84, 0.86, 0.88, 0.90.
- At each point: Adhoc (4), DySM-P (3), MPC (7), each for 400 seconds.
- Only the existing mminterupt input changes, at 200 seconds: N=1024 to N=1280.
- The template CSV stays at 0.80; each point saves its own CSV with the setpoint changed.
- Default `GPU_SMS=170`; existing initial allocation gives tasks 0–12 12 SMs
  each and task 13 14 SMs. Controller and actuator algorithms are unchanged.

## User launch

From the server shell, activate the existing environment and run:

```bash
cd ~/ECRTS_original_20261017
source .venv/bin/activate
cd ECRTS_25/experiment/6.6
./run_experiment.sh
```

The default sweep is 18 runs, approximately two hours plus build and startup
time. A P-only sweep uses `./run_experiment.sh 400 task_set.csv logs 3`;
use 7 for MPC-only or 4 for Adhoc-only. These still sweep all six setpoints.
Experiments are started by the user, not by the assistant.

Each point regenerates the shared `Scheduler/gain_matrix.h`, cleans/builds the
scheduler, runs from `Scheduler/` (required by Hotspot input paths), and retains
logs by controller. Do not run concurrently with another experiment using that
shared header/binary. This runner is not a resume command; rerunning can replace
results in the selected output directory.

## Logs and figures

```text
logs/sp_0.80/task_set.csv
logs/sp_0.80/sol_4/task_0 ... task_13
logs/sp_0.80/sol_3/task_0 ... task_13
logs/sp_0.80/sol_7/task_0 ... task_13
...
logs/sp_0.90/...
Figures/sp_0.90/runtime_14tasks_comparison.pdf
```

Logs are retained for all six setpoints. Only 0.90 is automatically plotted:
three controller rows and three columns (RRT, deadline misses, SMs), with all
14 task curves. The plot retains 6.1/6.5's first-10-samples omission.
After the default sweep, `python3 plotting.py` replots the 0.90 comparison.
Single-controller runs store live `task_*` files directly under each
`logs/sp_<value>/` and produce `runtime_14tasks_sol_<mode>.pdf` at 0.90.

CPU dispatch checks and compilation do not establish GPU numerical correctness
or experiment performance. No synthetic test data is installed as experiment logs.

## Integration checks (2026-10-18)

- CPU-only dispatcher test: all 14 workload names plus the existing `mm` alias
  called the expected create/run/destroy entry points; CUDA tasks were replaced
  by test stand-ins and no GPU work ran.
- Full scheduler compiled and linked on the server using CUDA 13.2 / sm_120,
  with all seven new CUDA objects. The build log contained no warnings/errors.
- Shared gain_matrix.h regenerated for this 14-task CSV at setpoint 0.80 and
  170 SM; K/B dimensions are 14 x 14. First-seven model inputs match 6.5.
- Plotter checked with temporary synthetic logs: 3 x 3 panels, 14 task curves,
  correct startup omission. Fixtures/output were confined to /tmp.
- Existing controller, gain generator, task source and 6.5 configuration were
  checked unchanged. No scheduler binary or GPU validation workload was executed.

The current shared binary/header are built for 14 tasks. Use the corresponding
experiment runner when switching back to seven tasks so it regenerates/rebuilds
the shared model and binary for that CSV.
