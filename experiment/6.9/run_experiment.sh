#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"
SCHEDULER_DIR="${ROOT_DIR}/Scheduler"
SCHEDULER_BIN="${SCHEDULER_DIR}/scheduler"
GAIN_SCRIPT="${SCHEDULER_DIR}/gain.py"
GAIN_HEADER="${SCHEDULER_DIR}/gain_matrix.h"
export GPU_SMS="${GPU_SMS:-170}"
export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${ROOT_DIR}/libsmctrl:${LD_LIBRARY_PATH:-}"

DURATION="${1:-200}"
CSV_FILE="${2:-task_set.csv}"
OUTPUT_DIR="${3:-logs}"
TASK_COUNTS=(8 9 10)
SOLUTION=7
case "$CSV_FILE" in
    /*) CSV_PATH="$CSV_FILE" ;;
    *) CSV_PATH="${BASE_DIR}/${CSV_FILE}" ;;
esac
case "$OUTPUT_DIR" in
    /*) LOG_DIR="$OUTPUT_DIR" ;;
    *) LOG_DIR="${BASE_DIR}/${OUTPUT_DIR}" ;;
esac

echo "============================================================"
echo " DySM 6.9 MPC DECISION OVERHEAD "
echo " Tasks: 1–10 | Setpoint: 0.80 | Duration: $DURATION s per run"
echo " Statistics: valid decision records [150, 200)"
echo "============================================================"

# ------------------------------------------------------------
# STEP 0: Checks
# ------------------------------------------------------------
python3 - "$CSV_PATH" "${SCHEDULER_DIR}/cuda_runner.cpp" <<'PY_CHECK'
import csv
from pathlib import Path
import re
import sys
with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) < 10:
    sys.exit("[FATAL] The task template must contain at least 10 rows.")
if any(row["workload"] == "mminterupt" for row in rows[:10]):
    sys.exit("[FATAL] Use the fixed-input mm entry for this experiment.")
source = Path(sys.argv[2]).read_text()
source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
supported = set(re.findall(r'workload\s*==\s*"([^"\n]+)"', source))
missing = sorted({row["workload"] for row in rows[:10]} - supported)
if missing:
    sys.exit("[FATAL] Workloads not connected in cuda_runner.cpp: " + ", ".join(missing))
PY_CHECK

for count in "${TASK_COUNTS[@]}"; do
    POINT_DIR="${LOG_DIR}/n${count}"
    RUN_LOG_DIR="${POINT_DIR}/sol_${SOLUTION}"
    mkdir -p "$RUN_LOG_DIR"

    # ------------------------------------------------------------
    # STEP 1: Generate the task-count configuration
    # ------------------------------------------------------------
    TASK_CSV="${POINT_DIR}/task_set.csv"
    python3 - "$CSV_PATH" "$TASK_CSV" "$count" <<'PY_CSV'
import csv
import sys
with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(f)
    fields, rows = reader.fieldnames, list(reader)[:int(sys.argv[3])]
for row in rows:
    row["setpoint"] = "0.80"
with open(sys.argv[2], "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
PY_CSV

    # ------------------------------------------------------------
    # STEP 2: Generate Gain Matrix
    # ------------------------------------------------------------
    echo "[SWEEP] Tasks = $count, solution = $SOLUTION"
    python3 "$GAIN_SCRIPT" "$TASK_CSV" "$GAIN_HEADER"

    # ------------------------------------------------------------
    # STEP 3: Build scheduler
    # ------------------------------------------------------------
    (
        cd "$SCHEDULER_DIR"
        make clean
        make -j128
    )

    # ------------------------------------------------------------
    # STEP 4: Run MPC and retain the timing log
    # ------------------------------------------------------------
    (
        cd "$SCHEDULER_DIR"
        DYSM_MEASURE_DECISION=1 "$SCHEDULER_BIN" \
            "$TASK_CSV" "$DURATION" "$RUN_LOG_DIR/" "$SOLUTION"
    ) 2>&1 | tee "$RUN_LOG_DIR/console.log"

    # Stop before the next task count if 200 valid records were not collected.
    python3 - "$LOG_DIR" "$count" <<'PY_TIMING'
import sys
from plotting import collect_stats
row = collect_stats(sys.argv[1], (int(sys.argv[2]),))[0]
print(f"[OK] {row['tasks']} tasks: mean MPC decision time = {row['mean_us']:.6f} us")
PY_TIMING
done

# ------------------------------------------------------------
# STEP 5: Plot
# ------------------------------------------------------------
python3 "$BASE_DIR/plotting.py" "$LOG_DIR"
