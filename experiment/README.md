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
completed and generated their figures. Runner/plot completion does not establish
controller correctness: 6.1's STGM profile-loading fallback remains unresolved,
and detailed review of 6.3 is pending.

## Directory organization

The experiment root contains only `6.1`–`6.8` and this index. Legacy root-level
runners, traces, logs, notebooks and executables were removed; their cleanup
backup was subsequently deleted at the user's request. Existing 6.1–6.4 results and scripts are retained. 6.5–6.7 remain
reserved directories; 6.8 contains the existing ML entry and its dependencies,
without changes to its experiment parameters or a claim of completed reproduction.
