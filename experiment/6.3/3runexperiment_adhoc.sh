#!/usr/bin/env bash
# ============================================================
# RTAS_25 Experiment Runner (Sweep setpoints + traces)
# - For each setpoint: run 2tasktrace.py -> produces ~15 CSV traces in traces/
# - For each CSV in traces/: run gain.py -> build -> run scheduler -> plot -> archive
# ============================================================

set -euo pipefail

# Do not kill unrelated experiments by matching process names.

# -------------------------
# Paths / defaults
# -------------------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
export GPU_SMS="${GPU_SMS:-170}"
export MPLBACKEND=Agg
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
SCHEDULER_DIR="${ROOT_DIR}/Scheduler"
export LD_LIBRARY_PATH="${ROOT_DIR}/libsmctrl:${LD_LIBRARY_PATH:-}"
APPLICATION_DIR="${ROOT_DIR}/application"
SCHEDULER_BIN="${SCHEDULER_DIR}/scheduler"
GAIN_SCRIPT="${SCHEDULER_DIR}/gain.py"
GAIN_HEADER="${SCHEDULER_DIR}/gain_matrix.h"

DURATION="${1:-150}"
DEFAULT_LOGDIR="${2:-logs}"
SOLUTION="${3:-4}"

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
RUNS_DIR="${BASE_DIR}/3runs_adhoc"
mkdir -p "${RUNS_DIR}"

# sweep setpoints
SETPOINTS=(0.50 0.60 0.70 0.80 0.90)

# Move run artifacts (logs + ALL figures) into destination
move_artifacts() {
    local dest="$1"
    mkdir -p "$dest"

    # 1) Move logs/*
    if [[ -d "${BASE_DIR}/${DEFAULT_LOGDIR}" ]]; then
        shopt -s nullglob
        mv "${BASE_DIR}/${DEFAULT_LOGDIR}/"* "$dest/" 2>/dev/null || true
        shopt -u nullglob
    fi

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
    python3 "${BASE_DIR}/3tasktrace.py" "${sp}"

    # 2) Collect CSVs in traces/
    TRACES_DIR="${BASE_DIR}/3traces"
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
                    "${SCHEDULER_BIN}" "${TRACE_FULL_PATH}" "${DURATION}" "${BASE_DIR}/${DEFAULT_LOGDIR}/" "${sol}"
                status=$?
                set -e

                if [[ $status -ne 0 ]]; then
                    echo "[ERROR] Solution ${sol} failed with status=${status}"
                    exit "$status"
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
                "${SCHEDULER_BIN}" "${TRACE_FULL_PATH}" "${DURATION}" "${BASE_DIR}/${DEFAULT_LOGDIR}/" "${SOLUTION}"
            status=$?
            set -e

            if [[ $status -eq 0 ]]; then
                echo "[INFO] Experiment completed successfully."
            else
                echo "[ERROR] Experiment failed (status=$status)."
                exit "$status"
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
