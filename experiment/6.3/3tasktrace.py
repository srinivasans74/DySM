#!/usr/bin/env python3
import csv
import sys
import itertools
import os

# -----------------------------------------------------
# Baseline workload configuration
# -----------------------------------------------------
workload_executiontime = {
    'projection': 38.045,
    'hist': 21.611,
    'stencil': 30,
}

workload_rates = {w: 1.05 * t for w, t in workload_executiontime.items()}

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
# Prepare output directory (CHANGED)
# -----------------------------------------------------
TRACE_DIR = "3traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# -----------------------------------------------------
# Generate ALL 3-task combinations (CHANGED)
# -----------------------------------------------------
workloads = list(workload_rates.keys())
triples = list(itertools.combinations(workloads, 3))

for w1, w2, w3 in triples:
    filename = f"trace_{w1}_{w2}_{w3}.csv"
    filepath = os.path.join(TRACE_DIR, filename)

    rows = []
    task_id = 1

    for workload in (w1, w2, w3):
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

    # Write per-triplet trace
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
print(f"Generated {len(triples)} traces in '{TRACE_DIR}/'")
print(f"Setpoint = {setpoint}")
print("-------------------------------------------------------")