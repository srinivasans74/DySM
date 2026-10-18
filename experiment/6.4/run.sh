#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/../../libsmctrl:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
set -e

mkdir -p results Figures
nvcc -arch=sm_120 streamoverhead.cu -o sm_mask_overhead \
    -I ../../libsmctrl \
    -L ../../libsmctrl \
    -lsmctrl \
    -std=c++17

./sm_mask_overhead

python3 plotting.py
