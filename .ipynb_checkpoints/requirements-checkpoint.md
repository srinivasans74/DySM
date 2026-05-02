# System Requirements

This document describes the system-level configuration required to build, run, and evaluate the artifact. The artifact supports NVIDIA CUDA-based systems.

---

## Operating System and Kernel

- OS: Ubuntu 20.04.6 LTS
- Linux Kernel: 5.15.0-139-generic
- Architecture: x86_64

---

## Toolchain

- GCC version: 9.4.0
- CMake version: 3.19.6
- glibc version: 2.31
- make version: 4.2.1
- pthreads: Built-in via glibc

---

## GPU Acceleration Support

### NVIDIA CUDA Setup

- NVIDIA Driver version: 560.35.03
- CUDA Toolkit version: 11.6.124
- GPU: Use `nvidia-smi` to confirm your GPU

### Validation Commands
nvcc –version
nvidia-smi

---

## libsmctrl Requirement

This artifact depends on the **libsmctrl** library for hardware-level SM/TPC partitioning on NVIDIA GPUs.

- Source: http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/
- Original paper:
  J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", RTAS 2023

### Important Notes

- A working version of **libsmctrl is already included/downloaded in this repository**
- You do NOT need to manually clone it

### Build Instructions
To compile libsmctrl, simply run:
./activate_libsmctrl.sh

This script builds the library and prepares it for use with the artifact.

### Requirements

- CUDA Toolkit installed (nvcc available)
- GCC installed

---

## Python Environment

A complete list of Python libraries is provided in `requirements.txt`.

To install:
python3 -m venv venv
source venv/bin/activate
pip install –upgrade pip
pip install -r requirements.txt

### Minimal requirement for plotting
pip3 install matplotlib

