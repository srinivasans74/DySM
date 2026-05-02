import csv
import os
import numpy as np
from pathlib import Path
from numpy.linalg import matrix_rank
from scipy.signal import place_poles
'''
# ============================
# ============================
POLES = [0.5]        # resized to N automatically
# GAIN_SCALE = 0.5
# EXEC_TIME_SCALE = 5.0   
EXEC_TIME_SCALE = 10
GAIN_SCALE = 0.75
# ============================
# Default execution times
# ============================
EXECUTION_TIME_TABLE = {
    "mm":         0.901,
    "hotspot":    0.250,
    "hist":       0.524,
    "stencil":    1.120,
    "mttrkp":     0.101,
    "stereo":     0.150,
    "projection": 0.372,
}

# ----------------------------
# GPU SM discovery (best-effort)
# ----------------------------
def get_gpu_sms(default_sms=80):
    try:
        from numba import cuda
        dev = cuda.get_current_device()
        for attr in ("MULTIPROCESSOR_COUNT", "multiprocessor_count"):
            if hasattr(dev, attr):
                val = int(getattr(dev, attr))
                if val > 0:
                    return val
        if hasattr(dev, "get_device_attributes"):
            attrs = dev.get_device_attributes()
            for k, v in attrs.items():
                if "MULTIPROCESSOR_COUNT" in str(k):
                    return int(v)
    except Exception:
        pass

    env = os.getenv("GPU_SMS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)

    return int(default_sms)

# ----------------------------
# CSV loader
# ----------------------------
def load_tasks(csv_path, exec_scale):
    workloads, p_vals, e_vals = [], [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wl = (row.get("workload") or "").strip().lower()
            if not wl:
                continue

            p = float(row.get("task_rate", "0"))
            if p <= 0:
                raise ValueError(f"Invalid task_rate for workload '{wl}'")

            e_raw = (row.get("e", "") or "").strip()
            base_e = float(e_raw) if e_raw else EXECUTION_TIME_TABLE.get(wl, 1.0)

            e = base_e * exec_scale  # <<< SCALE HERE

            workloads.append(wl)
            p_vals.append(p)
            e_vals.append(e)

    return workloads, np.array(p_vals), np.array(e_vals)

# ----------------------------
# Control utilities
# ----------------------------
def check_controllability(A, B):
    N = A.shape[0]
    ctrb = B
    Ak = np.eye(N)
    for _ in range(1, N):
        Ak = Ak @ A
        ctrb = np.hstack((ctrb, Ak @ B))
    return matrix_rank(ctrb) == N

def place_gain(A, B, poles, scale):
    if not check_controllability(A, B):
        raise RuntimeError("System not controllable.")
    poles = np.clip(np.array(poles), -0.99, 0.99)
    result = place_poles(A, B, poles)
    return result.gain_matrix * scale

# ----------------------------
# System model
# ----------------------------
def build_A_diag(e_vec, p_vec, dmax):
    denom = p_vec * dmax
    if np.any(denom <= 0):
        raise ValueError("Invalid p_i * dmax")
    return np.diag(e_vec / denom)

# ----------------------------
# Save C++ header
# ----------------------------
def save_as_cpp(K, out_path):
    N = K.shape[0]
    guard = "GAIN_MATRIX_H_"

    with open(out_path, "w") as f:
        f.write("// Auto-generated single-loop gain matrix\n")
        f.write(f"// Execution time scaled by x{EXEC_TIME_SCALE}\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"constexpr int NUM_TASKS = {N};\n\n")
        f.write("constexpr float gain[NUM_TASKS][NUM_TASKS] = {\n")
        for i in range(N):
            row = ", ".join(f"{np.abs(K[i,j]):.6f}f" for j in range(N))
            f.write(f"    {{{row}}}{',' if i != N-1 else ''}\n")
        f.write("};\n\n#endif\n")

# ----------------------------
# Main
# ----------------------------
def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 gain_poleplacement_singleloop.py <csv_file> <output.h>")
        sys.exit(1)

    csv_path, out_path = sys.argv[1], sys.argv[2]

    workloads, p_vals, e_vals = load_tasks(csv_path, EXEC_TIME_SCALE)
    N = len(workloads)
    if N == 0:
        raise RuntimeError("No valid workloads found")

    sms = get_gpu_sms()
    dmax = 1.0 / sms

    poles = np.resize(np.array(POLES), N)

    print("=============================================")
    print(f"[INFO] Workloads: {workloads}")
    print(f"[INFO] p-values: {p_vals}")
    print(f"[INFO] e-values (scaled x{EXEC_TIME_SCALE}): {e_vals}")
    print(f"[INFO] SMs: {sms}, dmax = {dmax:.6f}")
    print(f"[INFO] Poles: {poles}")
    print("=============================================")

    A = build_A_diag(e_vals, p_vals, dmax)
    B = np.eye(N)

    K = place_gain(A, B, poles, GAIN_SCALE)

    print("[INFO] A =\n", A)
    print("[INFO] K =\n", K)

    save_as_cpp(K, out_path)
    print("[DONE] Gain matrix written to", out_path)

if __name__ == "__main__":
    main()
'''
import csv
import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy.signal import place_poles

POLES = [0.8]
EXEC_TIME_SCALE = 10.0
GAIN_SCALE = 0.01
GAIN_LIMIT = 20.0

EXECUTION_TIME_TABLE = {
    "mm":         0.901,
    "hotspot":    0.250,
    "hist":       0.524,
    "stencil":    1.120,
    "mttrkp":     0.101,
    "stereo":     0.150,
    "projection": 0.372,
}

def get_gpu_sms(default_sms=80):
    env = os.getenv("GPU_SMS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return int(default_sms)

def load_tasks(csv_path, exec_scale):
    workloads, p_vals, e_vals = [], [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            wl = (row.get("workload") or "").strip().lower()
            if not wl:
                continue

            p = float(row.get("task_rate", "0"))
            if p <= 0:
                raise ValueError(f"Invalid task_rate for workload '{wl}'")

            e_raw = (row.get("e", "") or "").strip()
            base_e = float(e_raw) if e_raw else EXECUTION_TIME_TABLE.get(wl, 1.0)

            workloads.append(wl)
            p_vals.append(p)
            e_vals.append(base_e * exec_scale)

    return workloads, np.array(p_vals), np.array(e_vals)

def check_controllability(A, B):
    n = A.shape[0]
    ctrb = B
    Ak = np.eye(n)

    for _ in range(1, n):
        Ak = Ak @ A
        ctrb = np.hstack((ctrb, Ak @ B))

    return matrix_rank(ctrb) == n

def build_A_diag(e_vec, p_vec, total_sms):
    """
    Full N x N plant approximation.

    r_i ~= e_i / (p_i * s_i)

    Linearized:
        r(k) = A r(k-1) + B delta_s(k)

    Here A is diagonal sensitivity magnitude around operating point.
    """
    N = len(e_vec)
    s_op = total_sms / N

    alpha = -e_vec / (p_vec * s_op * s_op)

    # Use integrator plant: x(k) = x(k-1) + B delta_s
    A = np.eye(N)
    B = np.diag(alpha)

    return A, B

def save_as_cpp(K_cpp, out_path):
    num_tasks = K_cpp.shape[0]
    guard = "GAIN_MATRIX_H_"

    with open(out_path, "w") as f:
        f.write("// Auto-generated full N x N MIMO gain matrix\n")
        f.write(f"// gain dimension: ({num_tasks} x {num_tasks})\n")
        f.write(f"// Execution time scaled by x{EXEC_TIME_SCALE}\n")
        f.write("// Use with: outerdelta[i] = -sum_j gain[i][j] * outererror[j]\n\n")

        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"constexpr int NUM_TASKS = {num_tasks};\n")
        f.write(f"constexpr int NUM_INPUTS = {num_tasks};\n\n")

        f.write("constexpr float gain[NUM_TASKS][NUM_TASKS] = {\n")
        for i in range(num_tasks):
            row = ", ".join(f"{K_cpp[i, j]:.9f}f" for j in range(num_tasks))
            comma = "," if i != num_tasks - 1 else ""
            f.write(f"    {{{row}}}{comma}\n")
        f.write("};\n\n#endif\n")

def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 gain_poleplacement_full_mimo.py <csv_file> <output.h>")
        sys.exit(1)

    csv_path, out_path = sys.argv[1], sys.argv[2]

    workloads, p_vals, e_vals = load_tasks(csv_path, EXEC_TIME_SCALE)
    N = len(workloads)

    if N == 0:
        raise RuntimeError("No valid workloads found")

    total_sms = get_gpu_sms()

    A, B = build_A_diag(e_vals, p_vals, total_sms)

    poles = np.resize(np.array(POLES, dtype=float), N)
    poles = np.clip(poles, -0.99, 0.99)

    if not check_controllability(A, B):
        raise RuntimeError("Full N x N system is not controllable")

    placed = place_poles(A, B, poles)
    K = placed.gain_matrix * GAIN_SCALE

    # C++ does outerdelta = -gain * outererror.
    # Keep signs. Do NOT abs().
    K_cpp = K

    K_cpp = np.clip(K_cpp, -GAIN_LIMIT, GAIN_LIMIT)

    print("=============================================")
    print(f"[INFO] Workloads: {workloads}")
    print(f"[INFO] periods: {p_vals}")
    print(f"[INFO] e-values scaled x{EXEC_TIME_SCALE}: {e_vals}")
    print(f"[INFO] total_sms: {total_sms}")
    print(f"[INFO] A shape: {A.shape}")
    print(f"[INFO] B shape: {B.shape}")
    print(f"[INFO] poles: {poles}")
    print("[INFO] B =\n", B)
    print("[INFO] K =\n", K)
    print("[INFO] C++ gain clipped =\n", K_cpp)
    print("=============================================")

    save_as_cpp(np.abs(K_cpp), out_path)
    print("[DONE] Gain matrix written to", out_path)

if __name__ == "__main__":
    main()

