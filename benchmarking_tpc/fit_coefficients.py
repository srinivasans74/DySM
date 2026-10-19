#!/usr/bin/env python3
"""Fit each measured workload to time_ms = a / TPC + b; never modify gain.py."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fit_workload(path):
    data = json.loads(path.read_text())
    results = data["results"]
    tpcs = np.asarray([row["TPC"] for row in results], dtype=float)
    times = np.asarray([row["ExecTime_ms"] for row in results], dtype=float)
    count = int(data["total_tpcs"])
    if count < 2 or not np.array_equal(tpcs, np.arange(1, count + 1)):
        raise ValueError(f"{path}: incomplete or unordered TPC sweep")
    if not np.all(np.isfinite(times) & (times > 0)):
        raise ValueError(f"{path}: invalid execution times")
    a, b = np.polyfit(1.0 / tpcs, times, 1)
    residual = times - (a / tpcs + b)
    ss_total = np.sum((times - times.mean()) ** 2)
    return {
        "workload": path.name[len("tpc_results_"):],
        "total_tpcs": count,
        "a_ms_tpc": float(a),
        "b_ms": float(b),
        # For this testbed SM = 2 * TPC, so a/TPC = (2*a)/SM.
        "a_ms_sm": float(2 * a),
        "r2": float(1 - np.sum(residual ** 2) / ss_total) if ss_total > 0 else "",
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("workloads", nargs="+", help="Result suffixes, e.g. mm hist stereodisparity")
    args = parser.parse_args()
    rows = [fit_workload(args.log_dir / ("tpc_results_" + name)) for name in args.workloads]
    output = args.log_dir / "coefficients.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['workload']}: time_ms = {row['a_ms_tpc']:.6g}/TPC "
              f"+ {row['b_ms']:.6g}, R2={row['r2']}, RMSE={row['rmse_ms']:.6g} ms")
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
