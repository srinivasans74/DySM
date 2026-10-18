## Overlap Experiment

This experiment evaluates how overlapping TPC allocations across concurrent GPU tasks impact task response times. This is used to reproduce **Figure 4** in the paper.

We run two periodic GPU tasks implemented as tensor-computation kernels (dense 3-way tensor operations). Each task is released every 30 ms from a dedicated CPU thread and launched on its own CUDA stream.

- Task 1 is fixed to TPCs 0–7  
- Task 2 initially runs on a disjoint range (TPCs 8–15)  

We then gradually shift Task 2’s allocation leftward, increasing the number of shared TPCs from 0 to 8. At full overlap, both tasks execute on the same TPC set (0–7).

For each configuration, 10 task releases are executed, and the average response time is measured.

### Running the Experiment

To run this experiment:

```bash
./run.sh

Output

* The generated figure is saved in the Figures/ directory
* The plot illustrates how increasing overlap affects task response times
```

### Local execution notes

Local RTX 5090 notes: activate `../../../.venv/bin/activate` and run `./run.sh`.
The experiment keeps upstream dimensions 1024 x 512 x 512, rank 32, 8 TPCs per
task, overlap 0..8, and 10 releases per point. The only CUDA-source changes allow
85 detected TPCs and avoid an out-of-range 64-bit shift; only TPCs 0..15 are used.
Compilation targets sm_120 with C++17 and pthread support. Logs remain in `logs/`.
