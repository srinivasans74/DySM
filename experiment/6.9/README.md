# 6.9 — MPC controller decision overhead

Measure MPC (scheduler mode 7) for 1–10 concurrent tasks on the RTX 5090.
Adhoc and P are not included in this runner. Experiment 6.4 remains the separate
libsmctrl mask-update overhead experiment. Only the user starts experiments.

## Tasks and measurement

The runner takes the first N rows of task_set.csv in this order:
mm, hotspot, hist, stencil, stereo, projection, mttkrp, blackscholes, reduction,
scan. Periods and model inputs are inherited from the first ten tasks in 6.6.
The existing fixed-input mm entry replaces mminterupt. Setpoint is 0.80 and
inputs remain fixed during the run. Initial allocation and masks use the
existing scheduler. GPU_SMS defaults to 170; gains are regenerated for each N.

With DYSM_MEASURE_DECISION=1, dysm_mpc() in Scheduler/controller.cpp timestamps
the complete decision: mean-command initialization, candidate generation,
resource/safety checks, urgency filtering, model prediction, cost evaluation,
and selection of the next mean SM command. The end timestamp is taken before
printing the timing record. Monitoring/measurement aggregation, output logging,
actuator join/start, mask updates and periodic waits are outside the interval.
The metric is host elapsed time from steady_clock, in microseconds, not thread
CPU-consumption time. Decisions without valid measurements are not recorded.

The timer does not change the MPC actions, constraints, cost weights or model.
Ordinary runs without the environment flag do not call the new timing clocks
or print timing records. P and Adhoc functions are unchanged. Builds retain the
current Makefile flags (including -O0); the measurements describe that build.

## User launch

```bash
cd ~/ECRTS_original_20261017
source .venv/bin/activate
cd ECRTS_25/experiment/6.9
./run_experiment.sh
```

DURATION="${1:-200}" in run_experiment.sh defaults to 200 seconds per run.
There is no extra startup allowance. The current TASK_COUNTS selection is
8, 9, 10; change it to 1–10 for a full rerun. The committed logs contain all
ten task counts. The script requires timing records 0–199; if startup leaves
insufficient valid decisions, it stops rather than silently shortening the
statistical window. Select a longer duration explicitly if needed.

The three arguments are duration, task template and output directory. If a run
needs more time to collect 200 valid records, start a longer sweep into a new
output directory, for example:

```bash
./run_experiment.sh 400 task_set.csv logs_400
```

Extra records are retained but excluded from the fixed statistical window.
This runner is not a resume command. Reusing an output directory overwrites
that run's console.log and task results; use a new directory to retain old runs.
As in the other experiments, this script regenerates the shared gain header
and rebuilds scheduler. Do not run it concurrently with another experiment
that shares that build. It runs from Scheduler/ for Hotspot's input paths.

## Logs and plots

```text
logs/n1/task_set.csv
logs/n1/sol_7/console.log
logs/n1/sol_7/task_0
...
logs/n10/task_set.csv
logs/n10/sol_7/console.log
logs/n10/sol_7/task_0 ... task_9
Figures/overhead_mpc.pdf
Figures/overhead_mpc.csv
```

Each timing line includes solution=7, tasks=N, the zero-based valid decision
index (period), and decision_us. The figure uses the paper's 200-decision
protocol: take the arithmetic mean of indices [150,200), exactly 50 decisions,
for each task count. It does not take the last 50 records of an arbitrarily
longer run. Full records 0–199 must be present, finite, nonnegative and unique.
The CSV records task count, controller, setpoint, window, sample count and mean.

Only the required line style from the paper's individual_panels.py is included
in plotting.py: blue line/markers, bold axes and ticks, no grid. It does not
import the paper archive or packaging code. Arial is used when available,
with DejaVu Sans as a fallback. The y-axis scales to the new data.

After a completed sweep, regenerate the single MPC figure with:

```bash
python3 plotting.py
# For the longer sweep above:
python3 plotting.py logs_400
```

No formal experiment was launched while preparing this code. Static checks,
CPU-only model generation/compilation, and temporary synthetic-log plots do
not constitute overhead results.

## BlackScholes period calibration (2026-10-18)

The BlackScholes row in this experiment now uses the application-sized workload
(2,097,152 options, 20 kernel launches per job) benchmarked with the current
Scheduler CUDA compilation settings, including -O0 -G -g.
Source: `benchmarking_tpc/logs/rtx5090_20261018_173426/tpc_results_blackscholes` and `coefficients.csv`.

The selected reference allocation is 9 TPCs (18 SMs), near the equal share for
ten tasks on 170 SMs. This is an explicit calibration choice, not a recovered
reference allocation from the original paper. It stays fixed across task counts.
The measured 100-job mean at this allocation is 14.126117100 ms. Following the
multiplier in `../6.3/3tasktrace.py`, the period is
`1.05 * 14.126117100 = 14.832422955 ms`, stored as `14.832` ms in `task_set.csv`.
This implies an isolated reference RRT of approximately 0.952, not the 0.80
control target; concurrent behavior still requires measurement.

The new fit is `time_ms = 124.802464312156/TPC + 0.320693501095`.
The CSV model input is `e = a_ms_sm / 10 = 24.960492862431273` for gain.py's existing
execution-time scaling. The runner regenerates B and K for each task count
from this template. Only the BlackScholes row was updated in this calibration step.

## Reduction period calibration (2026-10-18)

Source: `benchmarking_tpc/logs/rtx5090_20261018_173922/tpc_results_reduction` and `coefficients.csv`.
The benchmark uses the application workload (2,097,152 floats, 256 threads per
block, 12 complete array-to-scalar reductions per job) and Scheduler CUDA
compilation settings, including -O0 -G -g.

Use the same fixed 9-TPC (18-SM) reference and 1.05 period multiplier as above.
The measured 100-job mean is 39.635587500 ms, giving
`1.05 * 39.635587500 = 41.617366875 ms`, stored as `41.617` ms.
The fit is `time_ms = 353.348746657062/TPC + 0.264573502339`;
`e = a_ms_sm / 10 = 70.66974933141243`. Only the Reduction row was updated in this step.
The runner will regenerate B and K from the updated template; existing logs
and the shared gain header were not overwritten during calibration.

## Scan period calibration (2026-10-18)

Source: `benchmarking_tpc/logs/rtx5090_20261018_174332/tpc_results_scan` and `coefficients.csv`.
The benchmark uses the application workload (8 arrays of 262,144 elements,
12 full three-stage scans per job) and Scheduler CUDA compilation settings,
including -O0 -G -g.

Use the same fixed 9-TPC (18-SM) reference and 1.05 period multiplier as above.
The measured 100-job mean is 20.125228100 ms, giving
`1.05 * 20.125228100 = 21.131489505 ms`, stored as `21.131` ms.
The fit is `time_ms = 174.526293801652/TPC + 0.400814170202`;
`e = a_ms_sm / 10 = 34.905258760330454`. Only the Scan row was updated in this step.
All three calibrated task rows are now in this experiment's template. Other
experiments and existing run logs retain their original configurations.
