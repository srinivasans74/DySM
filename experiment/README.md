# Conference experiments 6.1–6.4

Upstream: https://github.com/srinivasans74/DySM/tree/fee7b4b718f92710ff43fae0353cef481a64de5d/experiment

Use the upstream experiment definitions and separate runners. Local changes are
limited to RTX 5090 execution and identified script errors; see each README's
local execution notes. Existing P/MPC controller code and application kernels
are retained. Activate `../.venv/bin/activate` from the ECRTS_25 project root.

| Directory | Entry | Data | Figures |
|---|---|---|---|
| 6.1 | `./run_experiment.sh` | `logs/sol_3`–`logs/sol_6` (batch) | `Figures/` |
| 6.2 | `./run.sh` | `logs/` | `Figures/` |
| 6.3 | `./experiments.sh` | `3runs/`, `3runs_adhoc/` (staging: `logs/`) | `Figures/` |
| 6.4 | `./run.sh` | `results/` | `Figures/` |

6.1 defaults to 180 seconds per controller; 6.3 defaults to 150 seconds per
setpoint. 6.3 preserves upstream P's four and Adhoc's five setpoints; the original
figure compares the four common values. 6.1's batch selector is `all`, reserving
scheduler mode 7 for MPC. Keep the default `logs` path for upstream 6.1 plotting.

6.1/6.3 regenerate the shared gain header and rebuild the scheduler. Regenerate
for your chosen task CSV before returning to other P/MPC experiments.

Complete 6.2/6.4 measurements are retained. Full-length 6.1 (four controllers,
180 seconds each) and 6.3 (five Adhoc and four P runs, 150 seconds each) have now
completed and generated their figures. STGM profile lookup now resolves from
the scheduler executable's directory. Its 6.1 run completed again with successful
initialization, and the corresponding logs and aggregate figure were updated.
Detailed review of 6.3 remains pending.

## Directory organization

The experiment root contains only `6.1`–`6.9` and this index. Legacy root-level
runners, traces, logs, notebooks and executables were removed; their cleanup
backup was subsequently deleted at the user's request. Existing 6.1–6.4 results and scripts are retained. 6.5 uses seven existing workloads and sweeps 0.80, 0.82, 0.84, 0.86, 0.88 and
0.90 with Adhoc/P/MPC. All setpoints retain logs; only 0.80 is automatically plotted for 6.5.
See its README for task-period sources. 6.6 keeps those seven tasks and appends
blackscholes, reduction, scan, transpose, convolution_separable, nbody and fdtd3d.
6.6 uses the same six setpoints and Adhoc/P/MPC order; its automatic plot remains 0.90. The added
tasks are registered and compiled; inherited periods/model inputs are documented
in 6.6/README.md. No 6.6 experiment has been launched by this integration.
The assistant-launched 6.5 sweep was stopped at the user request. Retained logs
are partial, not completed experiment results. All future experiments are started
by the user; the assistant prepares code and launch instructions only.
6.6 has not run.
6.7 compares 4, 5, 6, 7 and 8 tasks at fixed setpoint 0.80 with Adhoc/P/MPC,
using fixed inputs and the existing mm entry (no runtime input change).
Its runner defaults to 150 seconds per controller and produces average RRT/DMR
versus task count. Scripts are prepared; no 6.7 experiment has run.
6.8 runs two ResNet-50 instances with the current MPC via ./run_experiment.sh:
480 s, 50 ms task periods, 1 s control, setpoint 0.80 and 170 SMs. The retained
instances increase batch size at 200/280 s and both restore it at 360 s. Logs
are in logs/sol_7; plotting.py writes the RRT/DMR/applied-SM figure in Figures/.
This is prepared code, not a completed GPU run; see 6.8/README.md for dependencies
and the inherited two-by-two ML model.

6.9 measures MPC decision overhead for 1–10 tasks, separate from 6.4 mask overhead.
Run ./run_experiment.sh inside 6.9; default duration is 220 seconds per task count (to collect 200 valid decisions).
It requires valid decision records 0–199 and averages [150,200), producing
Figures/overhead_mpc.pdf and .csv. Only MPC is enabled. No 6.9 experiment has
been launched during preparation; see its README for the timing boundary.
