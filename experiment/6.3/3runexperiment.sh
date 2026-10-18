#!/usr/bin/env bash
set -euo pipefail

# Do not kill unrelated experiments by matching process names.

# -------------------------
# Paths / defaults
# -------------------------
#BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
SOLUTION="${3:-3}"

TRACES_DIR="${BASE_DIR}/3traces"
RUNS_DIR="${BASE_DIR}/3runs"
LOG_DIR="${BASE_DIR}/${DEFAULT_LOGDIR}"

echo "[DEBUG] BASE_DIR=${BASE_DIR}"
echo "[DEBUG] ROOT_DIR=${ROOT_DIR}"
echo "[DEBUG] SCHEDULER_DIR=${SCHEDULER_DIR}"
echo "[DEBUG] APPLICATION_DIR=${APPLICATION_DIR}"
echo "[DEBUG] TRACES_DIR=${TRACES_DIR}"
echo "[DEBUG] LOG_DIR=${LOG_DIR}"

if [[ ! -d "${SCHEDULER_DIR}" ]]; then
    echo "[FATAL] Scheduler dir not found: ${SCHEDULER_DIR}"
    exit 1
fi

if [[ ! -d "${APPLICATION_DIR}" ]]; then
    echo "[FATAL] Application dir not found: ${APPLICATION_DIR}"
    exit 1
fi

if [[ ! -f "${GAIN_SCRIPT}" ]]; then
    echo "[FATAL] gain.py not found: ${GAIN_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo " RTAS_25 EXPERIMENT RUNNER (SETPOINT SWEEP) "
echo "============================================================"
echo "[INFO] Host architecture : $(uname -m)"
echo "[INFO] Duration          : ${DURATION}s"
echo "[INFO] Default log dir   : ${DEFAULT_LOGDIR}"
echo "[INFO] Solution          : ${SOLUTION}"
echo "============================================================"

mkdir -p "${RUNS_DIR}"

SETPOINTS=(0.50 0.60 0.70 0.80)

move_artifacts() {
    local dest="$1"
    mkdir -p "$dest"

    if [[ -d "${LOG_DIR}" ]]; then
        shopt -s nullglob
        mv "${LOG_DIR}/"* "$dest/" 2>/dev/null || true
        shopt -u nullglob
    fi

}

run_scheduler_once() {
    local trace_path="$1"
    local sol="$2"

    mkdir -p "${LOG_DIR}"

    # Run from ROOT_DIR so relative workload input paths still work.
    (
        cd "${ROOT_DIR}"
        env LD_LIBRARY_PATH="${SCHEDULER_DIR}:${LD_LIBRARY_PATH:-}" \
            "${SCHEDULER_BIN}" "${trace_path}" "${DURATION}" "${LOG_DIR}/" "${sol}"
    )
}

# -------------------------
# Main sweep
# -------------------------
for sp in "${SETPOINTS[@]}"; do
    echo
    echo "============================================================"
    echo "[SWEEP] Setpoint = ${sp}"
    echo "============================================================"

    echo "[INFO] Cleaning old traces..."
    rm -rf "${TRACES_DIR}"
    mkdir -p "${TRACES_DIR}"

    echo "[INFO] Generating traces via 3tasktrace.py..."
    python3 "${BASE_DIR}/3tasktrace.py" "${sp}"

    if [[ ! -d "${TRACES_DIR}" ]]; then
        echo "[FATAL] 3traces directory not found: ${TRACES_DIR}"
        exit 1
    fi

    shopt -s nullglob
    trace_files=( "${TRACES_DIR}"/*.csv )
    shopt -u nullglob

    if [[ ${#trace_files[@]} -eq 0 ]]; then
        echo "[FATAL] No .csv traces found in ${TRACES_DIR} after setpoint ${sp}"
        exit 1
    fi

    for CSV_PATH in "${trace_files[@]}"; do
        trace_name="$(basename "${CSV_PATH}")"
        TRACE_FULL_PATH="${TRACES_DIR}/${trace_name}"

        echo
        echo "------------------------------------------------------------"
        echo "[TRACE] ${trace_name}"
        echo "------------------------------------------------------------"

        sp_dir="${RUNS_DIR}/sp_${sp}"
        out_dir="${sp_dir}/trace_${trace_name%.csv}"
        mkdir -p "${out_dir}"

        echo "[INFO] Cleaning log dir..."
        rm -rf "${LOG_DIR}"
        mkdir -p "${LOG_DIR}"

        echo "[INFO] Regenerating gain matrix from: ${TRACE_FULL_PATH}"
        python3 "${GAIN_SCRIPT}" "${TRACE_FULL_PATH}" "${GAIN_HEADER}"
        echo "[OK] gain_matrix.h regenerated."

        echo "[INFO] Building scheduler..."
        ( cd "${SCHEDULER_DIR}" && make clean && make -j"$(nproc)" )
        echo "[OK] Build finished."

        if [[ ! -f "${SCHEDULER_BIN}" ]]; then
            echo "[ERROR] Scheduler binary missing: ${SCHEDULER_BIN}"
            exit 1
        fi

        if [[ ! -f "${TRACE_FULL_PATH}" ]]; then
            echo "[ERROR] Trace CSV missing: ${TRACE_FULL_PATH}"
            exit 1
        fi

        echo "[DIAGNOSTIC] Binary architecture:"
        file "${SCHEDULER_BIN}"

        if [[ "${SOLUTION}" -eq 6 ]]; then
            echo "============================================================"
            echo "[INFO] SOLUTION=6 → Running solutions 3, 4, and 5 sequentially..."
            echo "============================================================"

            for sol in 3 4 5; do
                echo "---------------------------------------------"
                echo "[INFO] Running solution ${sol}"
                echo " Trace File  : ${TRACE_FULL_PATH}"
                echo " Duration    : ${DURATION}s"
                echo " Log Dir     : ${LOG_DIR}"
                echo " Scheduler   : ${SCHEDULER_BIN}"
                echo " Working Dir : ${ROOT_DIR}"
                echo "---------------------------------------------"

                rm -rf "${LOG_DIR}"
                mkdir -p "${LOG_DIR}"

                set +e
                run_scheduler_once "${TRACE_FULL_PATH}" "${sol}"
                status=$?
                set -e

                sol_out_dir="${out_dir}/sol_${sol}"
                echo "[INFO] Archiving artifacts into: ${sol_out_dir}"
                move_artifacts "${sol_out_dir}"

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
            echo " Duration    : ${DURATION}s"
            echo " Log Dir     : ${LOG_DIR}"
            echo " Scheduler   : ${SCHEDULER_BIN}"
            echo " Working Dir : ${ROOT_DIR}"
            echo " Solution    : ${SOLUTION}"
            echo "---------------------------------------------"

            set +e
            run_scheduler_once "${TRACE_FULL_PATH}" "${SOLUTION}"
            status=$?
            set -e

            if [[ $status -eq 0 ]]; then
                echo "[INFO] Experiment completed successfully."
            else
                echo "[ERROR] Experiment failed with status=${status}."
                exit "$status"
            fi

            echo "[INFO] Archiving artifacts into: ${out_dir}"
            move_artifacts "${out_dir}"
        fi

        echo "[OK] Done trace ${trace_name} @ setpoint ${sp}"
    done
done

echo
echo "============================================================"
echo "[DONE] Sweep complete. Results in: ${RUNS_DIR}"
echo "============================================================"
