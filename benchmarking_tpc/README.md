# TPC benchmarks on RTX 5090

This is the existing benchmark suite adapted to the project's CUDA 13.2 /
RTX 5090 setup. It uses the existing `../libsmctrl/libsmctrl.so`; it does not
rebuild that library or modify the scheduler, controllers, or gain files.

## Run

Wait for other GPU experiments to finish, activate the project's Python venv
(NumPy is needed for fitting), then run:

```bash
cd ~/ECRTS_original_20261017/ECRTS_25/benchmarking_tpc
source ../../.venv/bin/activate
./benchmark.sh
```

The default list is `mm projection hist hotspot stencil stereo_disparity mttkrp`.
Each is compiled for `sm_120` and run sequentially. To select a subset:

```bash
./benchmark.sh mm stereo_disparity
```

`mm` uses the existing mapped-memory `mm1.cu`, which produces the same
`TPC` / `ExecTime_ms` schema as the other workload sweeps. The separate
transfer experiment is still available as `./benchmark.sh mmhost`; its H2D,
kernel, and D2H timings are not substituted into the mapped-mm fit.
Existing `pathfinder` and `srad` can also be selected explicitly.

Compilation without starting any GPU workload:

```bash
./benchmark.sh --build-only
```

`NVCC` defaults to `/usr/local/cuda/bin/nvcc`; `CUDA_ARCH` defaults to `sm_120`.

## Scope of the adaptation

- Detect the TPC count using the existing libsmctrl API (85 on the 170-SM 5090).
- Sweep `1..TPC_count`. There is no misleading zero-TPC sample or 41-TPC cap.
- Use a 128-bit disable mask on an explicit CUDA stream. Kernel launches and
  timing events use that same stream; counts above 64 remain distinguishable.
- Keep the original kernel bodies, input sizes, and repetition counts.
- Report CUDA failures instead of saving failed launches as valid measurements.
- For mapped mm, record the stop event immediately after the launch, then wait
  for that event, so a host-side device synchronization gap is not timed.
- For the optional mmhost experiment, enqueue transfers and timing events on
  the same stream as its kernel.

## Results and coefficients

Each invocation creates `logs/rtx5090_YYYYMMDD_HHMMSS/`, containing one console
log and one timing JSON per selected workload, followed by `coefficients.csv`.
Old logs are preserved. `BENCHMARK_LOG_DIR` may specify another **new** directory.
The raw JSON includes `total_tpcs` and the existing `results` array.

The fitting script uses the inverse-resource form already explored in the
notebook, separately for each workload:

```text
time_ms = a_ms_tpc / TPC + b_ms
        = a_ms_sm  / SM  + b_ms     (SM = 2 * TPC on this testbed)
```

The CSV also reports R-squared and RMSE. These are fitted coefficients, not
direct measurements of controller coupling. Inspect fit quality before using
them; a poor fit is not repaired by labeling the coefficients as calibrated.
To repeat fitting without running CUDA:

```bash
python3 fit_coefficients.py logs/rtx5090_TIMESTAMP mm projection hist hotspot stencil stereodisparity mttkrp
```

Use the existing standalone benchmark workloads, following the author’s
instruction to run this suite on the new testbed. The proposed switch to
application-backed measurement was withdrawn and was never deployed.

This script does **not** automatically replace `EXECUTION_TIME_TABLE` in
`Scheduler/gain.py`. Its output is the standalone benchmark fit described above.
The author’s coefficient-to-gain mapping still needs to be checked: `gain.py`
multiplies its `e` input by `EXEC_TIME_SCALE` and uses SM units. These fit columns
are not automatically interchangeable with that input.

The hardware adaptation has been checked by compilation and CPU-only mask /
synthetic-data tests. No new GPU measurements or hardware mask validation
were performed during preparation; those require the user-started run.


## Additional workloads for 6.9 task counts 8–10

Select the three added standalone sweeps explicitly (the default seven are unchanged):

```bash
bash benchmark.sh blackscholes reduction scan
```

These new sweeps copy the current application kernels and complete-job dispatch:
BlackScholes uses 2,097,152 options with 20 repeats; Reduction uses 2,097,152
float elements with 12 complete array-to-scalar reductions; Scan uses eight
262,144-element arrays with 12 complete three-stage scans. Their random seeds
match task IDs 7, 8, and 9 in the current 6.9 template. Application files are
not changed. The original seven standalone benchmarks keep their existing inputs.

Each new sweep measures 100 complete jobs per TPC count after one warm-up,
using CUDA events on the masked stream. Allocation and input transfers are
outside the timed region. The existing benchmark compiler settings apply;
these are standalone benchmark results, not scheduler runtime measurements.
The runner fits each result automatically. For the current gain.py convention,
copy a_ms_sm / 10 into the matching task CSV's e field before generating B/K.
No coefficients are written into experiment CSVs automatically.

Compile-only validation (no GPU jobs):

```bash
bash benchmark.sh --build-only blackscholes reduction scan
```
