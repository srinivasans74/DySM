# 6.8 — Two ResNet-50 instances with MPC

This implements Srini's meeting request to move the existing ML experiment into
this codebase and use two inference instances. The user selected removal of the
original instance whose batch increased at 120 seconds. The remaining original
instances 2 and 3 are now labelled Instance 1 and Instance 2.

## Experiment settings

- Two concurrent ResNet-50 instances, separate CUDA streams in one process.
- RTX 5090: 85 TPCs / 170 SMs; initial allocation is 84/86 SMs, using the current
  scheduler's whole-TPC distribution rule.
- Task period 50 ms, control period 1 second, RRT setpoint 0.80.
- 480 seconds after both instances have prepared and warmed up.
- Initial batch size 20 for both instances.
- At 200 seconds, Instance 1 changes to batch 40.
- At 280 seconds, Instance 2 changes to batch 40.
- At 360 seconds, both return to batch 20.

All inference tasks start at one common epoch. The staggered events are batch
changes, not staggered task launches. Batch-20 and batch-40 inputs are prepared
and warmed up before the measured run, following the existing ML worker.
Only MPC is run here. There is no implicit Adhoc/P sweep.

## Implementation and provenance

The four Python modules reuse the existing DySM-main/execution/pytorch ML
implementation: ResNet-50 FP32 inference, periodic releases, CUDA event execution
time, completed-job response time/deadline misses, shared start epoch and safe
shutdown. Old campaign/model-identification commands and dependencies are not
copied into this experiment. Plotting remains a local plotting.py entry.

controller.py builds native_bridge.cpp in a temporary directory. It extracts
only the active MPC decision block, assign_overlap allocator and delta-sigma
actuator from this project's Scheduler/controller.cpp. Their formulas, actions,
safety checks and weights are not rewritten in Python. The bridge passes model
coefficients in RRT/SM directly, without the old bridge's SM-to-TPC model scaling.
Current -O0 compilation is retained. The generated libdysm_pytorch.so and
build.json stay in 6.8. The shared scheduler, gain header, Makefile and other
experiment directories are not modified by this build.

The 128-bit libsmctrl_set_stream_mask_ext ABI is called through C++, not a
64-bit ctypes mask. Submission locks keep a forward's kernel enqueue sequence
from racing with multiword stream-mask updates. Actual physical SM/mask pulses
are saved by the existing actuator bridge for plotting.

MODEL_B in Parameters.py is the rows/columns 1 and 2 submatrix of the existing
three-instance ML model in:
DySM-main/build/rtx5090/pytorch/model-identified-20261008/model_receipt.json.
The source field is deployed_plant_B, with units RRT/SM. This retains the model
entries corresponding to the two kept instances; it is not a newly identified
two-instance model. The matrix and original instance IDs are saved in run.json.
No new model identification or GPU validation has been performed during setup.

## User launch

The runner reuses the server's existing PyTorch environment and local weights;
it does not download models or install packages during an experiment.

```bash
cd ~/ECRTS_original_20261017/ECRTS_25/experiment/6.8
./run_experiment.sh
```

Defaults in run_experiment.sh:

- DURATION=480 seconds (first positional argument).
- OUTPUT_DIR=logs/sol_7 (second positional argument).
- PYTHON=$HOME/.venvs/dysm-pytorch-20261008/bin/python3, the existing ML environment.
- PLOT_PYTHON=../../../.venv/bin/python3, the current project's plotting environment.
- RESNET50_WEIGHTS=$HOME/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth.

PYTHON, PLOT_PYTHON and RESNET50_WEIGHTS can be overridden through environment
variables on a different machine. Weights are the existing local ResNet-50
state_dict, copied into the standard cache; they are not part of the source repo.
The input image remains n02099601_3004.jpg from the original experiment.
libsmctrl is resolved from this project's libsmctrl/libsmctrl.so.

Use a new output directory for another run, e.g.:

```bash
./run_experiment.sh 480 logs/sol_7_run2
```

Existing log directories are rejected rather than overwritten. Only the user
starts GPU experiments. Do not run this simultaneously with another GPU
experiment when collecting comparison results.

## Logs and figure

```text
logs/sol_7/run.json
logs/sol_7/task_0.json
logs/sol_7/task_1.json
logs/sol_7/actuation.json
Figures/resnet50_2tasks_mpc_RRT_DMR_SMS.pdf
```

run.json records configuration, batch events, completion status and model/build
provenance. Each task file contains control-period windows and completed jobs.
Response time is completion minus scheduled release; RRT is mean response time
in the window divided by 50 ms; DMR is the fraction of completed jobs in the
window whose response exceeds 50 ms. Empty windows are null, not fabricated
zero latency. actuation.json records the applied SM counts and 128-bit masks.

The figure has three columns (RRT, DMR and actual applied SMs), each with two
instance curves. It follows the existing 6.1 plotting organization and skips
the first ten measurement windows for RRT/DMR. Regenerate it using:

```bash
../../../.venv/bin/python3 plotting.py logs/sol_7
```

CPU-only protocol tests, bridge compilation, mock-mask tests and synthetic-log
rendering were performed during preparation. They are not GPU inference or
hardware-mask correctness results. The user-run experiment is still outstanding.
