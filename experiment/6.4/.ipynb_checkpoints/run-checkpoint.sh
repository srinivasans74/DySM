#!/bin/bash
set -e

mkdir -p results Figures
nvcc streamoverhead.cu -o sm_mask_overhead \
    -I ../../libsmctrl \
    -L ../../libsmctrl \
    -lsmctrl \
    -std=c++17

./sm_mask_overhead

python3 plotting.py
