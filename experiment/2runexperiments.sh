#!/usr/bin/env bash
# ============================================================
# RTAS_25 Experiment Runner (Sweep setpoints + traces)
# - For each setpoint: run 2tasktrace.py -> produces ~15 CSV traces in traces/
# - For each CSV in traces/: run gain.py -> build -> run scheduler -> plot -> archive
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

# -------------------------
# Paths / defaults
# -------------------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$BASE_DIR")"
SCHEDULER_DIR="${ROOT_DIR}/Scheduler"
SCHEDULER_BIN="${SCHEDULER_DIR}/scheduler"
GAIN_SCRIPT="${SCHEDULER_DIR}/gain.py"
GAIN_HEADER="${SCHEDULER_DIR}/gain_matrix.h"

# Args:
#   $1 duration (sec)
#   $2 logdir name (relative to BASE_DIR)
#   $3 solution (int). If 6 => run 3,4,5
DURATION="${1:-90}"
DEFAULT_LOGDIR="${2:-logs}"
SOLUTION="${3:-3}"

echo "============================================================"
echo " RTAS_25 EXPERIMENT RUNNER (SETPOINT SWEEP) "
echo "============================================================"
echo "[INFO] Host architecture : $(uname -m)"
echo "[INFO] Duration          : ${DURATION}s"
echo "[INFO] Default log dir   : ${DEFAULT_LOGDIR}"
echo "[INFO] Solution          : ${SOLUTION}"
echo "============================================================"

# -------------------------
# Output structure
# -------------------------
RUNS_DIR="${BASE_DIR}/2runs"
mkdir -p "${RUNS_DIR}"

# sweep setpoints
SETPOINTS=(0.70 0.75 0.80 0.85 0.90)

# Move run artifacts (logs + ALL figures) into destination
move_artifacts() {
    local dest="$1"
    mkdir -p "${dest}"

    # Move logs/*
    if [[ -d "${BASE_DIR}/${DEFAULT_LOGDIR}" ]]; then
        shopt -s nullglob
        mv "${BASE_DIR}/${DEFAULT_LOGDIR}/"* "${dest}/" 2>/dev/null || true
        shopt -u nullglob
    fi

    # Move ALL figures generated under BASE_DIR, but DON'T touch 2runs/
    # (Increase maxdepth if your figures are deeper than 3 levels.)
    find "${BASE_DIR}" \
        -path "${RUNS_DIR}" -prune -o \
        -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.svg" \) \
        -print0 2>/dev/null \
    | xargs -0 -I{} mv "{}" "${dest}/" 2>/dev/null || true
}

# -------------------------
# Main sweep
# -------------------------
for sp in "${SETPOINTS[@]}"; do
    echo
    echo "============================================================"
    echo "[SWEEP] Setpoint = ${sp}"
    echo "============================================================"

    # 1) Generate traces for this setpoint
    echo "[INFO] Generating traces via 2tasktrace.py..."
    python3 "${BASE_DIR}/2tasktrace.py" "${sp}"

    # 2) Collect CSVs in traces/
    TRACES_DIR="${BASE_DIR}/traces"
    if [[ ! -d "${TRACES_DIR}" ]]; then
        echo "[FATAL] traces/ directory not found at: ${TRACES_DIR}"
        exit 1
    fi

    shopt -s nullglob
    trace_files=( "${TRACES_DIR}"/*.csv )
    shopt -u nullglob

    if [[ ${#trace_files[@]} -eq 0 ]]; then
        echo "[FATAL] No .csv traces found in ${TRACES_DIR} after setpoint ${sp}"
        exit 1
    fi

    # 3) For each trace CSV, run pipeline
    for CSV_PATH in "${trace_files[@]}"; do
        trace_name="$(basename "${CSV_PATH}")"

        echo
        echo "------------------------------------------------------------"
        echo "[TRACE] ${trace_name}"
        echo "------------------------------------------------------------"

        # Destination: 2runs/sp_<setpoint>/trace_<filename-without-ext>/
        sp_dir="${RUNS_DIR}/sp_${sp}"
        out_dir="${sp_dir}/trace_${trace_name%.csv}"
        mkdir -p "${out_dir}"

        # Full trace path you asked to pass: "${CSV_PATH}/filename" equivalent
        # (CSV_PATH is already the full file path; we also build it explicitly:)
        TRACE_FULL_PATH="${TRACES_DIR}/${trace_name}"

        # 3a) Generate Gain Matrix from THIS trace CSV
        echo "[INFO] Regenerating gain matrix from: ${TRACE_FULL_PATH}"
        if [[ -f "${GAIN_SCRIPT}" ]]; then
            python3 "${GAIN_SCRIPT}" "${TRACE_FULL_PATH}" "${GAIN_HEADER}"
            echo "[OK] gain_matrix.h regenerated."
        else
            echo "[FATAL] Missing gain.py at: ${GAIN_SCRIPT}"
            exit 1
        fi

        # 3b) Build scheduler
        echo "[INFO] Building scheduler..."
        ( cd "${SCHEDULER_DIR}" && make clean && make -j128 )
        echo "[OK] Build finished."

        # 3c) Checks
        if [[ ! -f "${SCHEDULER_BIN}" ]]; then
            echo "[ERROR] Scheduler binary missing: ${SCHEDULER_BIN}"
            exit 1
        fi
        echo "[DIAGNOSTIC] Binary architecture:"
        file "${SCHEDULER_BIN}"

        if [[ ! -f "${TRACE_FULL_PATH}" ]]; then
            echo "[ERROR] Trace/CSV missing: ${TRACE_FULL_PATH}"
            exit 1
        fi

        # 3d) Run
        mkdir -p "${BASE_DIR}/${DEFAULT_LOGDIR}"

        if [[ "${SOLUTION}" -eq 6 ]]; then
            echo "============================================================"
            echo "[INFO] SOLUTION=6 → Running solutions 3, 4, and 5 sequentially..."
            echo "============================================================"
            for sol in 3 4 5; do
                echo "---------------------------------------------"
                echo "[INFO] Running solution ${sol} ..."
                echo "---------------------------------------------"

                set +e
                env LD_LIBRARY_PATH="${SCHEDULER_DIR}:${LD_LIBRARY_PATH:-}" \
                    "${SCHEDULER_BIN}" "${TRACE_FULL_PATH}" "${DURATION}" "${BASE_DIR}/${DEFAULT_LOGDIR}" "${sol}"
                status=$?
                set -e

                echo "[INFO] Plotting results for solution ${sol}..."
                # PASS FULL TRACE PATH (not just filename)
                python3 "${BASE_DIR}/figures.py" "${sol}" "${TRACE_FULL_PATH}" || true

                if [[ $status -ne 0 ]]; then
                    echo "[WARN] Solution ${sol} failed with status=${status} (continuing)."
                fi
                sleep 2
            done
        else
            echo "[INFO] Starting experiment..."
            echo "---------------------------------------------"
            echo " Trace File  : ${TRACE_FULL_PATH}"
            echo " Duration    : ${DURATION} s"
            echo " Log Dir     : ${BASE_DIR}/${DEFAULT_LOGDIR}"
            echo " Scheduler   : ${SCHEDULER_BIN}"
            echo " Solution    : ${SOLUTION}"
            echo "---------------------------------------------"

            set +e
            env LD_LIBRARY_PATH="${SCHEDULER_DIR}:${LD_LIBRARY_PATH:-}" \
                "${SCHEDULER_BIN}" "${TRACE_FULL_PATH}" "${DURATION}" "${BASE_DIR}/${DEFAULT_LOGDIR}" "${SOLUTION}"
            status=$?
            set -e

            echo "[INFO] Plotting results..."
            # PASS FULL TRACE PATH (your requested form)
            python3 "${BASE_DIR}/figures.py" "${SOLUTION}" "${TRACE_FULL_PATH}" || true

            if [[ $status -eq 0 ]]; then
                echo "[INFO] Experiment completed successfully."
            else
                echo "[ERROR] Experiment failed (status=$status)."
            fi
        fi

        echo "[INFO] Archiving artifacts into: ${out_dir}"
        move_artifacts "${out_dir}"

        echo "[OK] Done trace ${trace_name} @ setpoint ${sp}"
    done
done

echo
echo "============================================================"
echo "[DONE] Sweep complete. Results in: ${RUNS_DIR}"
echo "============================================================"