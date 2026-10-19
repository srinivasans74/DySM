# DySM

## Current ECRTS_25 snapshot — RTX 5090

Branch: `ecrts25-mpc`. The runtime uses 170 SMs / 85 TPCs, the existing
128-bit stream-mask API, and CUDA 13.2 support. Controller modes are P (3),
Adhoc (4), STGM (5), SMshare (6), and MPC (7).

The actuator retains each task's TPC positions after initial placement:
shrinking releases excess TPCs; growing uses free TPCs before sharing.
MPC retains its one-SM candidate step and uses action-cost coefficient 0.004.
Gain scaling is 0.005 and coupling is 0.15; output gains are no longer capped
at 10. Regenerate B/K and rebuild for each task configuration. The committed
header is the final 6.5 configuration: eight tasks, setpoint 0.90, 170 SMs.

Seven workload adapters have been added: BlackScholes, Reduction, Scan,
Transpose, ConvolutionSeparable, NBody, and FDTD3D. Benchmarking supports
all 85 TPCs and fits execution time as a/TPC + b. The added BlackScholes,
Reduction, and Scan benchmarks match their application job sizes.

### Experiments and results — 2026-10-19

- 6.1–6.4: conference experiments and recorded results retained.
- 6.5: completed 4–8 tasks × six setpoints (0.80–0.90, step 0.02) ×
  Adhoc/P/MPC: 90 runs, 400 seconds each, fixed workload inputs.
- 6.6: fourteen-task runner/configuration prepared; retains earlier model
  inputs and is not presented as a completed validated experiment.
- 6.7: earlier 4–7 task runs at 0.70/0.80 retained; overlapping results
  are reused by the merged 6.5 sweep.
- 6.8: two ResNet-50 instances with the current MPC/actuator adapter;
  480 control periods, batch increases at 200/280 s and decrease at 360 s.
- 6.9: recorded MPC decision overhead for 1–10 tasks. The runner currently
  selects 8–10; set TASK_COUNTS explicitly for a full rerun. Default duration
  is 200 seconds, with no extra startup allowance.

6.5 statistics use the final 50 control periods excluding the endpoint
([350:400] for a 401-sample log), averaging tasks equally. Generate its
standalone RRT-line/DMR-bar panels and five-page paired PDF with:

```bash
cd experiment/6.5
python3 plotting.py --panels logs
```

Outputs are in `Figures/4-8subfigures/` and
`Figures/4-8_setpoints_rrt_dmr.pdf`. RRT is displayed with upper limit 1;
Adhoc values above 1 are explicitly labelled "capped at 1". CSV statistics
retain the original values. Figures are PDF only.

Activate a Python environment with numpy, scipy, matplotlib and pandas.
With CUDA 13.2 at /usr/local/cuda:

```bash
make -C libsmctrl -B libsmctrl.so CUDA=/usr/local/cuda
GPU_SMS=170 python3 Scheduler/gain.py experiment/6.5/task_set.csv Scheduler/gain_matrix.h
make -C Scheduler -B -j4
export LD_LIBRARY_PATH="$PWD/libsmctrl:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

See each experiment's README for its runner. Runs share the GPU and scheduler
build, so execute them sequentially. Sources, canonical measurements and PDF
figures are included; local binaries, caches and duplicate backup archives are
excluded. Existing repository setup documents are retained.

## Abstract

Many of today’s real-time embedded systems increasingly rely on GPUs for AI-related computing. However, existing GPU scheduling solutions for spatially shared GPU systems are mostly open-loop and depend on worst-case execution time (WCET) estimation for offline schedulability analysis. Such approaches cannot adapt to runtime workload variations. While adaptive scheduling techniques have been proposed to handle execution-time variability, prior work primarily targets CPUs or time-slicing GPUs, where only one task executes on the GPU at a time. In contrast, spatial sharing enables concurrent kernel execution via Streaming Multiprocessor (SM) partitioning, allowing more efficient GPU utilization. This motivates the need for adaptive solutions tailored to spatially shared GPU systems.

This artifact implements DySM, a closed-loop response-time control algorithm for spatially shared GPUs in soft real-time systems. DySM dynamically adjusts SM allocations at runtime to regulate task response times with low overhead. To model GPU resource contention among tasks, we derive a multi-input multi-output (MIMO) system model that captures how SM allocation affects the response times of multiple tasks. Based on this model, DySM is designed using feedback control theory to ensure system stability and control accuracy. Experimental results on an NVIDIA GPU testbed demonstrate that DySM outperforms state-of-the-art approaches by maintaining real-time guarantees under workload variations. Compared to the best-performing baseline, DySM reduces the deadline miss ratio by up to 90.93%.

## Python Requirements

All Python dependencies used in the experiments are listed in [`requirements.txt`](./requirements.txt). Please go through the [REQUIREMENTS.md](./REQUIREMENTS.md) since sometimes we need to adjust the gain parameters.
You can install them using:
```bash
pip install -r requirements.txt
```
To make all .sh files in the repository executable, run the following command from the repository root:
find . -type f -name "*.sh" -exec chmod +x {} +
---

## Experiments

To run the experiments, please refer to [`INSTALL.md`](./INSTALL.md), which contains complete setup and usage instructions.
All experimental results, including overhead measurements, are available in this repository. Each experiment includes fully-commented code and plotting scripts. Various workloads can be tested using the provided configurations.

---

## Repository Structure

The repository is organized as follows:
- `applications/` – workloads used in the experimental evaluation  
- `Scheduler/` – Implementation of the scheduling framework, including controller, monitoring, and runtime components  
- `experiment/` – All experiments  

Other documentation files available in this repository:
- [`INSTALL.md`](./INSTALL.md) – Step-by-step installation guide  
- [`REQUIREMENTS.md`](./REQUIREMENTS.md) – Additional dependencies and hardware notes  
- [`LICENSE`](./LICENSE) – Licensing information  



## Citation
If you use this code or dataset in your work, please cite the following papers:

```bibtex
@article{subramaniyan2025fcgpu,
  author    = {Srinivasan Subramaniyan and Xiaorui Wang},
  title     = {FC-GPU: Feedback Control GPU Scheduling for Real-time Embedded Systems},
  journal   = {ACM Transactions on Embedded Computing Systems (TECS)},
  year      = {2025},
  month     = sep,
  publisher = {ACM},
  note      = {To appear. Presented at EMSOFT 2025, Taipei, Taiwan}
}

@inproceedings{subramaniyan2026dysm,
  author    = {Srinivasan Subramaniyan and Xiaorui Wang},
  title     = {DySM: Dynamic Scaling of GPU Streaming Multiprocessor in Spatially Shared Real-Time Embedded GPU Systems},
  booktitle = {Proceedings of the 38th European Conference on Real-Time Systems (ECRTS)},
  year      = {2026},
  publisher = {Schloss Dagstuhl -- Leibniz-Zentrum für Informatik},
  series    = {Leibniz International Proceedings in Informatics (LIPIcs)}
}

