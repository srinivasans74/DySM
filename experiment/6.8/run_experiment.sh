#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
DURATION="${1:-480}"
OUTPUT_DIR="${2:-logs/sol_7}"
PYTHON="${PYTHON:-$HOME/.venvs/dysm-pytorch-20261008/bin/python3}"
PLOT_PYTHON="${PLOT_PYTHON:-${ROOT_DIR}/../.venv/bin/python3}"
export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${ROOT_DIR}/libsmctrl:${LD_LIBRARY_PATH:-}"
export RESNET50_WEIGHTS="${RESNET50_WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth}"

# ------------------------------------------------------------
# STEP 0: Checks
# ------------------------------------------------------------
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[FATAL] Use a fresh log directory: $OUTPUT_DIR"
    exit 1
fi
if [[ ! -f "$RESNET50_WEIGHTS" ]]; then
    echo "[FATAL] Missing ResNet-50 weights: $RESNET50_WEIGHTS"
    exit 1
fi
"$PYTHON" -c 'from importlib.metadata import version; print({n: version(n) for n in ("torch", "torchvision", "numpy", "scikit-image")})'
"$PLOT_PYTHON" -c 'import matplotlib'

# ------------------------------------------------------------
# STEP 1: Build the current MPC adapter (CPU only)
# ------------------------------------------------------------
"$PYTHON" controller.py

# ------------------------------------------------------------
# STEP 2: Run two ResNet-50 instances with MPC
# ------------------------------------------------------------
"$PYTHON" scheduler.py "$DURATION" "$OUTPUT_DIR"

# ------------------------------------------------------------
# STEP 3: Plot
# ------------------------------------------------------------
"$PLOT_PYTHON" plotting.py "$OUTPUT_DIR"
