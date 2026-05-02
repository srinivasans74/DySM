#!/bin/bash
set -e

echo "=============================="
echo "CUDA Benchmark Suite Execution"
echo "=============================="

LIB_PATH="../libsmctrl"

# -----------------------------------------
# List of benchmarks
# -----------------------------------------
BENCH_NAMES=(
    "mm"
    "projection"
    "hist"
    "stencil"
    "stereo_disparity"
    "mttkrp"
)

BENCH_SRCS=(
    "mm1.cu"
    "projection.cu"
    "hist.cu"
    "stencil.cu"
    "stereo_disparity.cu"
    "mttkrp.cu"
)

BENCH_FLAGS=(
    ""
    ""
    ""
    "--extended-lambda -O3"
    ""
    ""
)

# -----------------------------------------
# Build all executables
# -----------------------------------------
echo
echo ">>> Building all benchmarks..."
for i in "${!BENCH_NAMES[@]}"; do
    name=${BENCH_NAMES[$i]}
    src=${BENCH_SRCS[$i]}
    flags=${BENCH_FLAGS[$i]}

    echo "Building ${name}..."
    nvcc -o ${name} ${src} -I ${LIB_PATH} -L ${LIB_PATH} -lsmctrl ${flags}
done

# -----------------------------------------
# Run all executables
# -----------------------------------------
echo
echo ">>> Running all benchmarks..."
for name in "${BENCH_NAMES[@]}"; do
    echo "Running ${name}..."
    ./${name}
done

# -----------------------------------------
# Plot results
# -----------------------------------------
echo
echo ">>> Generating plots..."
python3 plotting.py

echo
echo "=============================="
echo "All CUDA benchmarks completed!"
echo "=============================="