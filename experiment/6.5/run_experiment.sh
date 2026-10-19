#!/usr/bin/env bash
set -euo pipefail
# Do not kill unrelated experiments by matching process names.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
export GPU_SMS="${GPU_SMS:-170}"
export MPLBACKEND=Agg
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
SCHEDULER_DIR="${ROOT_DIR}/Scheduler"
export LD_LIBRARY_PATH="${ROOT_DIR}/libsmctrl:${LD_LIBRARY_PATH:-}"
SCHEDULER_BIN="${SCHEDULER_DIR}/scheduler"
GAIN_SCRIPT="${SCHEDULER_DIR}/gain.py"
GAIN_HEADER="${SCHEDULER_DIR}/gain_matrix.h"

DEFAULT_CSV="task_set.csv"
DEFAULT_LOGDIR="logs"
DURATION="${1:-400}"
CSV_FILE="${2:-$DEFAULT_CSV}"
OUTPUT_DIR="${3:-$DEFAULT_LOGDIR}"
SOLUTION="${4:-all}"

case "$CSV_FILE" in
    /*) CSV_PATH="$CSV_FILE" ;;
    *) CSV_PATH="${BASE_DIR}/${CSV_FILE}" ;;
esac
case "$OUTPUT_DIR" in
    /*) LOG_DIR="$OUTPUT_DIR" ;;
    *) LOG_DIR="${BASE_DIR}/${OUTPUT_DIR}" ;;
esac
case "$SOLUTION" in
    all|4|3|7) ;;
    *) echo "[FATAL] Use all or controller 4 3 7 for this experiment."; exit 1 ;;
esac

echo "============================================================"
echo " DySM 6.5 EXPERIMENT RUNNER "
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

# Check the current dispatch table before changing the shared gain header/build.
python3 - "$CSV_PATH" "${SCHEDULER_DIR}/cuda_runner.cpp" <<'PY_CHECK'
import csv
from pathlib import Path
import re
import sys
source = Path(sys.argv[2]).read_text()
source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
supported = set(re.findall(r'workload\s*==\s*"([^"\n]+)"', source))
with open(sys.argv[1], newline="") as f:
    missing = sorted({row["workload"] for row in csv.DictReader(f)} - supported)
if missing:
    sys.exit("[FATAL] Workloads not connected in cuda_runner.cpp: " + ", ".join(missing))
PY_CHECK

run_setpoint() {
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

    # Helper: snapshot logs into a solution folder
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
            if [[ "$base" == sol_* || "$base" == task_set.csv ]]; then
                continue
            fi
            cp -R "$item" "$dest/" 2>/dev/null || true
        done
        shopt -u dotglob nullglob

        echo "[OK] Snapshot saved to: $dest"
}

clear_live_logs() {
    shopt -s dotglob nullglob
    for item in "${LOG_DIR}/"*; do
        [[ -e "$item" ]] || continue
        local base
        base="$(basename "$item")"
        if [[ "$base" == sol_* || "$base" == task_set.csv ]]; then
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
        if [[ "$base" == sol_* || "$base" == task_set.csv ]]; then
            continue
        fi
        rm -rf "$item"
    done
    shopt -u dotglob nullglob
}

# ------------------------------------------------------------
# STEP 4: Run
# ------------------------------------------------------------
    echo "============================================================"
    echo "[INFO] Running solutions ${SOLUTIONS_TO_RUN[*]} sequentially..."
    echo "============================================================"

    for sol in "${SOLUTIONS_TO_RUN[@]}"; do
        echo "---------------------------------------------"
        echo "[INFO] Running solution ${sol} ..."
        echo "---------------------------------------------"

        # Optional: start clean live logs for this solution (but keep snapshots)
        clear_live_logs

        set +e
        ( cd "$SCHEDULER_DIR"
          "$SCHEDULER_BIN" "$CSV_PATH" "$DURATION" "$LOG_DIR/" "$sol" )
        status=$?
        set -e

        sleep 2

        snapshot_artifacts "$sol"
        cleanup_live_logs_after_snapshot

        if [[ $status -ne 0 ]]; then
            echo "[ERROR] Solution ${sol} failed with status=${status}"
            exit "$status"
        fi
    done

    echo "============================================================"
    echo "[DONE] All solutions completed."
    echo "Snapshots are under: ${LOG_DIR}/sol_4 , ${LOG_DIR}/sol_3 , ${LOG_DIR}/sol_7"
    echo "============================================================"
    if [[ "$SOLUTION" == all ]]; then
        python3 "$BASE_DIR/plotting.py" "$CSV_PATH" "$LOG_DIR"
    else
        python3 "$BASE_DIR/plotting.py" "$CSV_PATH" "$LOG_DIR/sol_$SOLUTION" "$SOLUTION"
    fi
}
TASK_COUNTS=(4 5 6 7 8)
SETPOINTS=(0.80 0.82 0.84 0.86 0.88 0.90)
TEMPLATE_CSV="$CSV_PATH"
SWEEP_LOG_DIR="$LOG_DIR"
for sp in "${SETPOINTS[@]}"; do
    POINT_LOG_DIR="${SWEEP_LOG_DIR}/sp_${sp}"
    for count in "${TASK_COUNTS[@]}"; do
        LOG_DIR="${POINT_LOG_DIR}/n${count}"
        CSV_PATH="${LOG_DIR}/task_set.csv"
        # Keep matching complete results; emit only controllers still needed.
        pending="$(python3 - "$TEMPLATE_CSV" "$CSV_PATH" "$count" "$sp" "$DURATION" "$SOLUTION" <<'PY_CSV'
import csv
import json
from pathlib import Path
import sys
template, destination, count, sp, duration, solution = sys.argv[1:]
count = int(count)
path = Path(destination)
with open(template, newline="") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)[:count]
if len(rows) != count or any(row["workload"] == "mminterupt" for row in rows):
    sys.exit("[FATAL] Need enough fixed-input tasks in the template.")
for row in rows:
    row["setpoint"] = sp
if path.exists():
    with path.open(newline="") as f:
        if list(csv.DictReader(f)) != rows:
            sys.exit(f"[FATAL] Existing task configuration differs: {path}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
pending = []
for sid in ([4, 3, 7] if solution == "all" else [int(solution)]):
    folder = path.parent / f"sol_{sid}"
    complete = len(list(folder.glob("task_*"))) == count
    try:
        for i in range(count):
            data = json.loads((folder / f"task_{i}").read_text())
            complete = complete and all(len(data.get(key, [])) == int(duration) + 1
                for key in ("outerrtr_measured", "outerdeadlinemiss", "outersms"))
    except (OSError, ValueError):
        complete = False
    if not complete:
        pending.append(str(sid))
print(" ".join(pending))
PY_CSV
)"
        if [[ -z "$pending" ]]; then
            echo "[SKIP] Complete: tasks=$count, setpoint=$sp"
            continue
        fi
        read -r -a SOLUTIONS_TO_RUN <<< "$pending"
        echo "[SWEEP] Tasks = $count, setpoint = $sp, remaining solutions = $pending"
        run_setpoint
    done
    if [[ "$SOLUTION" == all ]]; then
        python3 "$BASE_DIR/plotting.py" --taskcounts "$POINT_LOG_DIR" "$sp"
    fi
done
