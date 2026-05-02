# INSTALL

## Overview
This repository contains code used to reproduce the main figures in our research paper.
### Directory Summary

| Directory       | Purpose                                            | Figure     |
|----------------|----------------------------------------------------|-----------  |
| `benchmarking_tpc/`     |Impact of SM scaling on GPU execution time.   | Figure 1    |
| `experiment/`  | All experiments | Figures 3-6|

---

## Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```
### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```
Hardware requirements are listed in `REQUIREMENTS.md`.

---

## GPU Setup
./activate_libsmctrl.sh

---
## Reproducing Key Figures
### Figure 1 – Modelling Results

```bash
cd benchmarking_tpc
./benchmark.sh
```
This will generate and save `fig1.pdf`.

---

## Figure 3-6

The `Experiments/` folder contains all benchmark workloads with the FC-GPU controller embedded. To run these:
1. Navigate to the directory:
    ```bash
    cd experiment
    ```

2. Follow the steps in the provided `README.md`.
The results should be reproducible as shown in the paper.

## Important Note on GPU Tuning

**If you are using a GPU different from the one used for initial development and tuning, the controller gains (`a11` through `a44`) will likely need to be re-tuned.** These gains are critical for the stability of the control system and are highly dependent on the underlying hardware's performance characteristics. We recommend performing a stability analysis (as per your system design) to determine the optimal gain values for your specific GPU to ensure the controller operates effectively and maintains real-time guarantees. Failure to re-tune these gains on different hardware may lead to performance degradation or instability.

