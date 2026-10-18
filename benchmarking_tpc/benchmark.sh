#!/bin/bash


#!/bin/bash
set -e  # stop on first error

echo "=============================="
echo "CUDA Benchmark Suite Execution"
echo "=============================="

# Library path
LIB_PATH="../libsmctrl"

# Helper to compile and run benchmarks
run_bench() {
    local name=$1
    local src=$2
    local flags=$3

    echo
    echo ">>> Building ${name}.cu ..."
    nvcc -o ${name} ${src} -I ${LIB_PATH} -L ${LIB_PATH} -lsmctrl ${flags}

    echo ">>> Running ${name} ..."
    ./${name}
}

# -----------------------------------------
# List of all benchmarks
# -----------------------------------------

# run_bench "mm"                "mm.cu"                        ""
# run_bench "projection"        "projection.cu"                 ""
# run_bench "hist"              "hist.cu"                       ""
# run_bench "hotspot"           "hotspot.cu"                    ""
# run_bench "stencil"           "stencil.cu"                    "--extended-lambda -O3"
# run_bench "stereo_disparity"  "stereo_disparity.cu"           ""
# run_bench "mttkrp"  "mttkrp.cu"           ""
run_bench "mm" "mmhost.cu" ""








echo
echo "=============================="
echo "All CUDA benchmarks completed!"
echo "=============================="
