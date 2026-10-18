#!/usr/bin/env python3
import csv
import os
import random
import sys

# -----------------------------------------------------
# Baseline workload configuration


workload_executiontime = {
    'mminterupt': 50.3664,
    #'mm': 83.3664,
    #'projection': 38.045,
    #'hist': 71.611,
    #"hotspot": 20.250,
    'stereo': 23.61757,
    #'stencil': 50,
}

workload_rates = {w: 0.90* t for w, t in workload_executiontime.items()}
# -----------------------------------------------------
# User input: number of tasks
# -----------------------------------------------------
if len(sys.argv) > 1:
    try:
        num_tasks = int(sys.argv[1])
        setpoint=float(sys.argv[2])
        if num_tasks <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer for number of tasks.")
        sys.exit(1)
else:
    try:
        num_tasks = int(input("Enter number of tasks: "))
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        sys.exit(1)
workload_setpoint = {w: setpoint for w in workload_executiontime.keys()}

# -----------------------------------------------------
# Generate randomized unique workloads
# -----------------------------------------------------
rows = []
workload_list = list(workload_rates.keys())

for i in range(1, num_tasks + 1):
    workload = workload_list[(i - 1) % len(workload_list)]

    # Shuffle after completing a full cycle through all workloads
    if i % len(workload_list) == 0:
        random.shuffle(workload_list)

    task_rate = workload_rates[workload]
    setpoint = workload_setpoint[workload]
    if workload =="mminterupt":
        tunable ="tunable"
    else:
        tunable ="nontunable"


    # Random range only if tunable
    #task_range = random.randint(30,50) if tunable == "tunable" else 0
    task_range = random.randint(15,30) if tunable == "tunable" else 0

    rows.append([
        i,
        workload,
        f"{task_rate:.3f}",
        f"{setpoint:.2f}",
        tunable,
        task_range
    ])

# -----------------------------------------------------
# Write results to CSV file
# -----------------------------------------------------
output_file = "task_set.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sl_no", "workload", "task_rate", "setpoint", "tunable/nontunable", "range (%)"])
    writer.writerows(rows)

# -----------------------------------------------------
# Print summary
# -----------------------------------------------------
print(f"\nGenerated '{output_file}' with {num_tasks} tasks in {os.getcwd()}")
print("-------------------------------------------------------")
for r in rows:
    print(r)
