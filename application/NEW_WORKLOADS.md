# New workload source and integration boundary

The seven application folders below are registered in Scheduler/cuda_runner.cpp,
included in Scheduler/Makefile, and selected after the existing seven tasks in
experiment/6.6/task_set.csv. Their source and input definitions are unchanged
by this integration. No experiment or GPU correctness test has been launched.

Each folder has one .cu and one .h, with a task class and create/run/destroy
entry points matching existing application folders. Initialization allocates an
instance-owned CUDA stream, buffers and events and applies p->getMask(task_id).
The periodic loop, CPU pinning, release clock, response-time definition and
measurement locking follow application/hist/hist.cu. In particular, as in that
template, preemption_ts is set after the release wait; this is not a new
finish-minus-nominal-release response-time definition. Every timed job refreshes
the current mask. Pure GPU time is measured with CUDA events; response time is
pre-launch delay plus that GPU time. Job resets precede the GPU timing window.
No task chooses its own SM count or uses the old gpu_partition interface.

## Provenance

The old DySM-main cudaSamples.cu is a concept-derived adapter with reduced or
incomplete algorithms; it is not imported wholesale. Computational kernels here
come from NVIDIA cuda-samples v13.2update, commit
`aeab82ff306b12ae980f40109d9b0152f77b1f12`. NVIDIA copyright and BSD license text
are retained in each .cu. Each file identifies its upstream path and adaptations.
The wrapper is adapted from the current project's Histogram task, not from the
old fixed-TPC runtime. Kernel helpers are translation-unit-local; mutable device
coefficients and scratch storage belong to each task instance.

- `blackscholes`: `cpp/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh`. Upstream float2 kernel retained, including both call and put outputs. Valid floating-point input arrays replace byte-filled floats.
- `reduction`: `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`. Unchanged upstream reduce0 kernel, repeatedly reducing partial sums to one scalar. Per-instance scratch buffers.
- `scan`: `cpp/2_Concepts_and_Techniques/scan/scan.cu`. Upstream exclusive-scan kernels and three-stage large-array dispatch retained. Eight independent arrays of 262144 elements; scratch buffer is instance-owned, not upstream static d_Buf.
- `transpose`: `cpp/6_Performance/transpose/transpose.cu`. Unchanged upstream transposeNoBankConflicts, including the complete tile loops. Matching 32x16 block and 32x32 tile.
- `convolution_separable`: `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu`. Upstream row and column kernels. Only coefficient access changes from device-global c_Kernel to an instance-owned buffer parameter, so simultaneous instances do not overwrite coefficients.
- `nbody`: `cpp/5_Domain_Specific/nbody/bodysystemcuda.cu`. Upstream all-pairs tiled acceleration and integration specialized to float. Fixed softeningSquared=0.001, single GPU, no OpenGL. Every release starts from the same saved initial positions and zero velocities. It is more work than the old 512-interaction approximation.
- `fdtd3d`: `cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh`. Upstream radius-4 kernel and halo layout. Coefficients are instance-owned instead of device-global stencil. Twenty ping-pong timesteps per release; both buffers have initialized halos, and every release resets to the saved input.

## Current input definitions (not calibrated experiment settings)

| Workload | Input / complete job |
|---|---|
| blackscholes | 2,097,152 options; both call and put; 20 repetitions |
| reduction | 2,097,152 floats; full reduction to one scalar; 12 repetitions |
| scan | 8 arrays × 262,144 uints; batched exclusive scan; 12 repetitions |
| transpose | 2048 × 2048; full transpose; 8 repetitions |
| convolution_separable | 2048 × 1024; radius 8; row then column; 12 repetitions |
| nbody | 65,536 bodies; all-pairs interaction and integration; 4 timesteps |
| fdtd3d | 128³ interior plus radius-4 halo; 20 timesteps |

Sizes/repetition counts retain the old adapter's broad input scale (BlackScholes
uses the old recipe's explicit 20 repeats). Complete algorithms now do different
work: old block-only scan, partial reduction, 512-neighbor NBody, one-dimensional
convolution and radius-1 FDTD timings are not equivalent. These inputs are named
class constants for review; no deadline, period, execution-time model value or
performance claim is assigned here. In particular, full all-pairs NBody may
require a different size/period than its old approximation. Existing task source,
controllers, gain generator and resource allocation remain unchanged. The 6.6 CSV
now contains these tasks and records inherited model inputs; see its README.

Kernel numerical correctness and CUDA-runtime behavior still require user-started
validation; dispatch/build integration alone does not establish these properties. Compile-only checks
are not numerical or timing validation. No self-running test/benchmark entry is
installed by this addition.

## Checks performed for this addition

- Public create/run/destroy signatures checked against the existing task API.
- All seven .cu files compiled locally with CUDA 12.6, C++17, O0/G, sm_89.
- All seven objects linked together with a temporary mask stub; the 21 public
  C entry points were verified. The resulting check executable was NOT run.
- Kernel bodies compared to the stated upstream source, allowing only the
  documented float specialization and per-instance coefficient substitutions.
- Full scheduler integration compiled and linked successfully on the server with
  CUDA 13.2 / sm_120, including all seven new CUDA objects. CPU-only dispatch
  checks passed for all new and existing workload names. The scheduler binary
  was not executed; GPU numerical/runtime validation remains pending.
