#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
LIB_PATH="${BASE_DIR}/../libsmctrl"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
CUDA_ARCH="${CUDA_ARCH:-sm_120}"
export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"

# Optional compile-only mode never starts a CUDA workload.
BUILD_ONLY=0
if [[ "${1:-}" == --build-only ]]; then
    BUILD_ONLY=1
    shift
fi
WORKLOADS=("$@")
if ((${#WORKLOADS[@]} == 0)); then
    WORKLOADS=(mm projection hist hotspot stencil stereo_disparity mttkrp)
fi
for name in "${WORKLOADS[@]}"; do
    case "$name" in
        mm|projection|hist|hotspot|stencil|stereo_disparity|mttkrp|mmhost|pathfinder|srad|blackscholes|reduction|scan) ;;
        *) echo "Unknown benchmark: $name" >&2; exit 1 ;;
    esac
done

if ((BUILD_ONLY == 0)); then
    export BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-${BASE_DIR}/logs/rtx5090_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$(dirname "$BENCHMARK_LOG_DIR")"
    mkdir "$BENCHMARK_LOG_DIR"
    echo "[INFO] Results: $BENCHMARK_LOG_DIR"
fi

RESULTS=()
for name in "${WORKLOADS[@]}"; do
    src="${name}.cu"
    result="$name"
    flags=()
    case "$name" in
        mm) src=mm1.cu ;;
        # Match Scheduler/Makefile for the application-sized BlackScholes, Reduction and Scan jobs.
        blackscholes|reduction|scan) flags=(-O0 -G -g -m64 --default-stream per-thread
                            --extended-lambda --expt-relaxed-constexpr
                            -Xcompiler -fno-omit-frame-pointer) ;;
        stencil) flags=(--extended-lambda -O3) ;;
        stereo_disparity) result=stereodisparity ;;
    esac
    echo "[BUILD] $name ($src, $CUDA_ARCH)"
    "$NVCC" -std=c++17 -arch="$CUDA_ARCH" -o "$name" "$src" \
        -I "$LIB_PATH" -L "$LIB_PATH" -lsmctrl \
        -Xlinker -rpath -Xlinker "$LIB_PATH" "${flags[@]}"
    if ((BUILD_ONLY == 0)); then
        echo "[RUN] $name"
        "./$name" 2>&1 | tee "${BENCHMARK_LOG_DIR}/${name}.log"
        # mmhost measures transfers separately; it is not the mapped-mm fit.
        if [[ "$name" != mmhost ]]; then RESULTS+=("$result"); fi
    fi
done

if ((BUILD_ONLY == 0 && ${#RESULTS[@]} > 0)); then
    python3 "$BASE_DIR/fit_coefficients.py" "$BENCHMARK_LOG_DIR" "${RESULTS[@]}"
fi
