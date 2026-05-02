# DySM

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

