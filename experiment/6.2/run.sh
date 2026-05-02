#!/bin/bash

rm -f overlap
nvcc -o overlap overlap.cu -I../../libsmctrl/ -L../../libsmctrl/ -lsmctrl

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../../libsmctrl/
    
    ./overlap
    python3 overlapplot.py
else
    echo "Compilation failed. Check errors above."
    exit 1
fi