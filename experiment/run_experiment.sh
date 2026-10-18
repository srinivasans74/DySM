#!/usr/bin/env bash
# ============================================================
# RTAS_25 Experiment Runner (Smart Build via make) - x86_64 friendly
# - SOLUTION=6 => run 3,4,5 sequentially
# - DO NOT delete logs/
# - After each run, copy logs/ into logs/sol_<sol>/
# ============================================================

set -euo pipefail

kill_processes() {
    echo "Stopping any running processes..."
    local pids
    pids="$(ps aux | grep -E './t1|./t2|./server' | grep -v grep | awk '{print $2}' || true)"
    if [[ -n "${pids}" ]]; then
        echo "${pids}" | xargs kill -9 2>/dev/null || true
    fi
}
kill_processes

# --- Paths ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$BASE_DIR")"
SCHEDULER_DIR="${ROOT_DIR}/Scheduler"
SCHEDULER_BIN="${SCHEDULER_DIR}/scheduler"
GAIN_SCRIPT="${SCHEDULER_DIR}/gain.py"
GAIN_HEADER="${SCHEDULER_DIR}/gain_matrix.h"

DEFAULT_CSV="task_set.csv"
DEFAULT_LOGDIR="logs"

# --- Arguments ---
DURATION="${1:-180}"
CSV_FILE="${2:-$DEFAULT_CSV}"
OUTPUT_DIR="${3:-$DEFAULT_LOGDIR}"
SOLUTION="${4:-7}"

CSV_PATH="${BASE_DIR}/${CSV_FILE}"
LOG_DIR="${BASE_DIR}/${OUTPUT_DIR}"

echo "============================================================"
echo " RTAS_25 EXPERIMENT RUNNER "
echo "============================================================"
echo "[INFO] Selected solution : ${SOLUTION}"
echo "[INFO] Host architecture : $(uname -m)"
echo "[INFO] Base dir          : ${BASE_DIR}"
echo "[INFO] Scheduler dir     : ${SCHEDULER_DIR}"
echo "[INFO] Log dir           : ${LOG_DIR}"
echo "============================================================"

# ------------------------------------------------------------
# STEP 0: Checks
# ------------------------------------------------------------
if [[ ! -f "$CSV_PATH" ]]; then
    echo "[FATAL] CSV missing: $CSV_PATH"
    exit 1
fi

# ------------------------------------------------------------
# STEP 1: Generate Gain Matrix
# ------------------------------------------------------------
echo "[INFO] Regenerating gain matrix..."
if [[ -f "$GAIN_SCRIPT" ]]; then
    python3 "$GAIN_SCRIPT" "$CSV_PATH" "$GAIN_HEADER"
    echo "[OK] gain_matrix.h regenerated."
else
    echo "[FATAL] Missing gain.py at: $GAIN_SCRIPT"
    exit 1
fi

# ------------------------------------------------------------
# STEP 2: Build scheduler
# ------------------------------------------------------------
echo "[INFO] Building scheduler..."
(
  cd "$SCHEDULER_DIR"
  make clean
  make -j128
)
echo "[OK] Build finished."

# ------------------------------------------------------------
# STEP 3: File Checks + mkdir logs
# ------------------------------------------------------------
if [[ ! -f "$SCHEDULER_BIN" ]]; then
    echo "[FATAL] Scheduler binary missing: $SCHEDULER_BIN"
    exit 1
fi

echo "[DIAGNOSTIC] Binary architecture:"
file "$SCHEDULER_BIN"

mkdir -p "$LOG_DIR"

# Helper: snapshot logs + figures into a solution folder
snapshot_artifacts() {
    local sol="$1"
    local dest="${LOG_DIR}/sol_${sol}"
    mkdir -p "$dest"

    # Copy logs (everything inside LOG_DIR except sol_* folders)
    shopt -s dotglob nullglob
    for item in "${LOG_DIR}/"*; do
        [[ -e "$item" ]] || continue
        local base
        base="$(basename "$item")"
        # skip previous snapshots
        if [[ "$base" == sol_* ]]; then
            continue
        fi
        cp -R "$item" "$dest/" 2>/dev/null || true
    done
    shopt -u dotglob nullglob

    # Copy figures generated under BASE_DIR (optional but usually wanted)
    find "$BASE_DIR" -maxdepth 3 -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.svg" \) \
        -print0 2>/dev/null \
      | xargs -0 -I{} cp "{}" "$dest/" 2>/dev/null || true

    echo "[OK] Snapshot saved to: $dest"
}

# Helper: clear "live" logs in LOG_DIR without deleting snapshots
# (This keeps per-run snapshots clean; comment out if you want accumulation.)
clear_live_logs() {
    shopt -s dotglob nullglob
    for item in "${LOG_DIR}/"*; do
        [[ -e "$item" ]] || continue
        local base
        base="$(basename "$item")"
        if [[ "$base" == sol_* ]]; then
            continue
        fi
        rm -rf "$item"
    done
    shopt -u dotglob nullglob
}
cleanup_live_logs_after_snapshot() {
    shopt -s dotglob nullglob
    for item in "${LOG_DIR}/"*; do
        [[ -e "$item" ]] || continue
        base="$(basename "$item")"
        # keep snapshots
        if [[ "$base" == sol_* ]]; then
            continue
        fi
        rm -rf "$item"
    done
    shopt -u dotglob nullglob
}

# ------------------------------------------------------------
# STEP 4: Run
# ------------------------------------------------------------
if [[ "$SOLUTION" -eq 7 ]]; then
    echo "============================================================"
    echo "[INFO] SOLUTION=6 → Running solutions 3, 4, and 5 sequentially..."
    echo "============================================================"

    for sol in 6 3 4 5 ; do
        echo "---------------------------------------------"
        echo "[INFO] Running solution ${sol} ..."
        echo "---------------------------------------------"

        # Optional: start clean live logs for this solution (but keep snapshots)
        clear_live_logs

        set +e
        env LD_LIBRARY_PATH="${SCHEDULER_DIR}:${LD_LIBRARY_PATH:-}" \
            "$SCHEDULER_BIN" "$CSV_PATH" "$DURATION" "$LOG_DIR" "$sol"
        status=$?
        set -e

        sleep 2

        echo "[INFO] Plotting results for solution ${sol}..."
        # If your plotting expects CSV filename only, keep "$CSV_FILE".
        # If it expects full path, use "$CSV_PATH".
        python3 "${BASE_DIR}/figuresplot.py" "$sol" "$CSV_FILE" || true

        snapshot_artifacts "$sol"
        cleanup_live_logs_after_snapshot

        if [[ $status -ne 0 ]]; then
            echo "[WARN] Solution ${sol} failed with status=${status} (continuing)."
        fi
    done

    echo "============================================================"
    echo "[DONE] All solutions completed."
    echo "Snapshots are under: ${LOG_DIR}/sol_3 , sol_4 , sol_5"
    echo "============================================================"
    exit 0
fi

# Single-solution run
echo "[INFO] Starting experiment..."
echo "---------------------------------------------"
echo " CSV File    : $CSV_PATH"
echo " Duration    : $DURATION s"
echo " Output Dir  : $LOG_DIR"
echo " Scheduler   : $SCHEDULER_BIN"
echo " Solution    : $SOLUTION"
echo "---------------------------------------------"

clear_live_logs

set +e
env LD_LIBRARY_PATH="${SCHEDULER_DIR}:${LD_LIBRARY_PATH:-}" \
    "$SCHEDULER_BIN" "$CSV_PATH" "$DURATION" "$LOG_DIR" "$SOLUTION"
STATUS=$?
set -e

echo "[INFO] Plotting results..."
python3 "${BASE_DIR}/figuresplot.py" "$SOLUTION" "$CSV_FILE" || true

snapshot_artifacts "$SOLUTION"

if [[ $STATUS -eq 0 ]]; then
    echo "[OK] Experiment completed successfully."
else
    echo "[ERROR] Experiment failed (status=$STATUS)."
fi

echo "============================================================"