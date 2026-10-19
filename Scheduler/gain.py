import csv
import os
import numpy as np
from numpy.linalg import matrix_rank
from scipy.signal import place_poles

# ============================
# Tuning knobs
# ============================
POLES = [0.6]
EXEC_TIME_SCALE = 10.0
GAIN_SCALE = 0.005
GAIN_LIMIT = float("inf")  # No limit on gain values for C++ output
#GAIN_LIMIT = 10.0
COUPLING = 0.15

EXECUTION_TIME_TABLE = {
    "mm":         0.901,
    "mminterupt": 0.901,
    "hotspot":    0.250,
    "hist":       0.524,
    "stencil":    1.120,
    "mttrkp":     0.101,
    "mttkrp":     0.101,
    "stereo":     0.150,
    "stereodisparity": 0.150,
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
    ctrb = B.copy()
    Ak = np.eye(n)

    for _ in range(1, n):
        Ak = Ak @ A
        ctrb = np.hstack((ctrb, Ak @ B))

    return matrix_rank(ctrb) == n

def build_A_B_coupled(e_vec, p_vec, total_sms, coupling=0.25):
    """
    Full coupled N x N plant.

    State:
        r(k) = r(k-1) + B * delta_s(k)

    Diagonal:
        changing task i's SMs affects task i's RRT.

    Off-diagonal:
        approximate shared-resource coupling.
        This produces real N x N MIMO gains.

    alpha_i < 0 because increasing SMs decreases RRT.
    """
    N = len(e_vec)
    s_op = total_sms / N

    alpha = -e_vec / (p_vec * s_op * s_op)

    A = np.eye(N)
    B = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                B[i, j] = alpha[i]
            else:
                # If task j takes more SMs, task i effectively loses capacity.
                # Since alpha_i < 0, this off-diagonal becomes positive.
                B[i, j] = -coupling * alpha[i] / max(1, N - 1)

    return A, B

def save_as_cpp(K_cpp, B, out_path, csv_path, total_sms):
    num_tasks = K_cpp.shape[0]
    guard = "GAIN_MATRIX_H_"

    with open(out_path, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"constexpr int NUM_TASKS = {num_tasks};\n")
        f.write(f"constexpr int NUM_INPUTS = {num_tasks};\n\n")

        f.write("constexpr float gain[NUM_TASKS][NUM_TASKS] = {\n")
        for i in range(num_tasks):
            row = ", ".join(f"{K_cpp[i, j]:.9f}f" for j in range(num_tasks))
            comma = "," if i != num_tasks - 1 else ""
            f.write(f"    {{{row}}}{comma}\n")
        f.write("};\n\n")
        f.write(f"// Model source CSV: {os.path.abspath(csv_path)}\n")
        f.write(f"// Model operating total SMs: {total_sms}\n")
        f.write("// Same plant used for gain above: delta_RRT = model_B * delta_SM.\n")
        f.write("constexpr double model_B[NUM_TASKS][NUM_TASKS] = {\n")
        for i in range(num_tasks):
            row = ", ".join(f"{B[i, j]:.17g}" for j in range(num_tasks))
            comma = "," if i != num_tasks - 1 else ""
            f.write(f"    {{{row}}}{comma}\n")
        f.write("};\n\n#endif\n")

def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 gain_poleplacement_full_coupled_mimo.py <csv_file> <output.h>")
        sys.exit(1)

    csv_path, out_path = sys.argv[1], sys.argv[2]

    workloads, p_vals, e_vals = load_tasks(csv_path, EXEC_TIME_SCALE)
    N = len(workloads)

    if N == 0:
        raise RuntimeError("No valid workloads found")

    total_sms = get_gpu_sms()

    A, B = build_A_B_coupled(e_vals, p_vals, total_sms, COUPLING)

    poles = np.resize(np.array(POLES, dtype=float), N)
    poles = np.clip(poles, -0.99, 0.99)

    if not check_controllability(A, B):
        raise RuntimeError("Full coupled N x N system is not controllable")

    placed = place_poles(A, B, poles)
    K = placed.gain_matrix * GAIN_SCALE

    # C++ does:
    #   outerdelta = -gain * outererror
    #
    # Keep signs. Do NOT use np.abs().
    K_cpp = np.clip(K, -GAIN_LIMIT, GAIN_LIMIT)

    print("=============================================")
    print(f"[INFO] Workloads: {workloads}")
    print(f"[INFO] periods: {p_vals}")
    print(f"[INFO] e-values scaled x{EXEC_TIME_SCALE}: {e_vals}")
    print(f"[INFO] total_sms: {total_sms}")
    print(f"[INFO] coupling: {COUPLING}")
    print(f"[INFO] poles: {poles}")
    print(f"[INFO] A shape: {A.shape}")
    print(f"[INFO] B shape: {B.shape}")
    print("[INFO] B =\n", B)
    print("[INFO] raw K =\n", K)
    print("[INFO] clipped C++ gain =\n", K_cpp)
    print("=============================================")

    save_as_cpp(np.abs(K_cpp), B, out_path, csv_path, total_sms)
    print("[DONE] Gain matrix written to", out_path)

if __name__ == "__main__":
    main()
