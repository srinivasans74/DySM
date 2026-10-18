# DySM

## Current ECRTS_25 snapshot — RTX 5090

This branch retains the existing P and MPC controllers and uses the detected
170 SMs / 85 TPCs on the RTX 5090, with the existing 128-bit stream-mask API and
CUDA 13.2 support. The branch is `ecrts25-mpc`; the runtime uses all 170 SMs.
Two tasks start at 84/86 SMs.

The committed gain header is the latest generated configuration: three tasks
(`projection`, `hist`, `stencil`) from experiment 6.3, with a 170-SM model budget.
Regenerate gains and rebuild whenever the task configuration changes. The MPC
candidate step is 1 SM and its action-cost coefficient is currently zero.
Mode 3 is P, 4 Adhoc, 5 STGM, 6 SMshare, and 7 MPC.

Activate your Python environment and install `numpy scipy matplotlib pandas`.
With CUDA 13.2 at `/usr/local/cuda`, build the library and scheduler from source:

```bash
make -C libsmctrl -B libsmctrl.so CUDA=/usr/local/cuda
GPU_SMS=170 python3 Scheduler/gain.py \
  experiment/6.3/3traces/trace_projection_hist_stencil.csv Scheduler/gain_matrix.h
make -C Scheduler -B -j4
export LD_LIBRARY_PATH="$PWD/libsmctrl:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

Run each conference experiment from its own directory; see
[`experiment/README.md`](experiment/README.md). The 6.1 batch selector is `all`.
6.1 and 6.3 regenerate the shared gain header, so run them sequentially.
Build products are excluded; measurement logs and generated figures are included.

### Experiment status as of 2026-10-18

- 6.1: all four 180-second runs and the aggregate figure are recorded. Result
  validation remains open: STGM's relative profile path fails from this entry,
  leaving initial allocations, and P's generated 2-task gain clips to all 10s.
  These behaviors are preserved, not presented as validated baseline results.
- 6.2: complete overlap sweep and figure are recorded.
- 6.3: Adhoc's five and P's four 150-second runs completed; both figures are
  included. Detailed interpretation/review of the results remains pending.
- 6.4: complete mask-overhead measurements and figure are recorded.
- 6.5–6.7: reserved for the journal-extension experiments; not implemented here.
- 6.8: existing ML entry, its local modules and input image are organized here.
  The requested two-task experiment has not been implemented or validated.

The conference runners preserve upstream task settings, separate P/Adhoc
scripts, build commands and plotting calculations, with only documented
execution fixes and hardware support. Legacy files outside the numbered
experiment directories were removed. No P/MPC algorithm changes are introduced
by this publishing step.

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

