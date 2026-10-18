## Experiment 6.4: Overhead Measurement of libsmctrl

This experiment measures the runtime overhead of using the `libsmctrl` library for dynamic SM/TPC partitioning on NVIDIA GPUs. Specifically, it evaluates the cost of updating stream masks and the impact on kernel execution. The experiment corresponds to **Section 6.4 (Overhead Evaluation)** in the paper and can be used to reproduce **Figure 6**.

### Description

The experiment performs the following:

- Applies different SM/TPC masks using `libsmctrl`
- Measures:
  - Mask-setting overhead (CPU-side latency)
  - Kernel execution + synchronization time
- Repeats mask updates multiple times to compute average overhead
- Logs results and generates plots automatically

### Running the Experiment
Make the script executable and run:

```bash
chmod +x run.sh
./run.sh
```

### Local execution notes

Local RTX 5090 notes: activate `../../../.venv/bin/activate` and run `./run.sh`.
The CUDA source and plotting script are byte-for-byte upstream versions.
CSV output is `results/sm_mask_overhead.csv`; figures remain in `Figures/`.
Only the runner's CUDA architecture and environment paths are adapted.
The existing measured CSV was moved back from `logs/` to `results/`.
