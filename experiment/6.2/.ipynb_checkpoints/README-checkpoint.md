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