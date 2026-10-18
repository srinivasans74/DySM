#!/usr/bin/env python3
import csv
import sys
import itertools
import os

# -----------------------------------------------------
# Baseline workload configuration
# -----------------------------------------------------
workload_executiontime = {
    'mm': 63.3664,
    'projection': 28.045,
    'hist': 21.611,
    'stereo': 18.61757,
    'stencil': 30,
}


workload_rates = {w: 0.75*t for w, t in workload_executiontime.items()}

# -----------------------------------------------------
# Input: setpoint only
# -----------------------------------------------------
if len(sys.argv) != 2:
    print("Usage: python3 script.py <setpoint>")
    sys.exit(1)

try:
    setpoint = float(sys.argv[1])
except ValueError:
    print("Setpoint must be a float")
    sys.exit(1)

# -----------------------------------------------------
# Prepare output directory
# -----------------------------------------------------
TRACE_DIR = "traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# -----------------------------------------------------
# Generate ALL 2-task combinations
# -----------------------------------------------------
workloads = list(workload_rates.keys())
pairs = list(itertools.combinations(workloads, 2))

for w1, w2 in pairs:
    filename = f"trace_{w1}_{w2}.csv"
    filepath = os.path.join(TRACE_DIR, filename)

    rows = []
    task_id = 1

    for workload in (w1, w2):
        task_rate = workload_rates[workload]

        # Keep your original logic
        tunable = "tunable" if workload == "mminterupt" else "nontunable"
        task_range = 0

        rows.append([
            task_id,
            workload,
            f"{task_rate:.3f}",
            f"{setpoint:.2f}",
            tunable,
            task_range
        ])
        task_id += 1

    # Write per-pair trace
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sl_no",
            "workload",
            "task_rate",
            "setpoint",
            "tunable/nontunable",
            "range (%)"
        ])
        writer.writerows(rows)

    print(f"Generated {filepath}")

# -----------------------------------------------------
# Summary
# -----------------------------------------------------
print("\n-------------------------------------------------------")
print(f"Generated {len(pairs)} traces in '{TRACE_DIR}/'")
print(f"Setpoint = {setpoint}")
print("-------------------------------------------------------")