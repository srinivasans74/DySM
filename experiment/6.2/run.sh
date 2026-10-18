#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/../../libsmctrl:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg

rm -f overlap
nvcc -arch=sm_120 -std=c++17 -Xcompiler -pthread -o overlap overlap.cu -I../../libsmctrl/ -L../../libsmctrl/ -lsmctrl

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../../libsmctrl/
    
    ./overlap || exit $?
    python3 overlapplot.py
else
    echo "Compilation failed. Check errors above."
    exit 1
fi
