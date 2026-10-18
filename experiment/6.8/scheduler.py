# scheduler_experiments.py
import os
import time
import threading
import json
import sys

from controller import Controller
from inferworkload import PeriodicInferenceTask
from Parameters import Parameters

def dump_logs_to_json(p: Parameters, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ---------- meta info ----------
    meta = {
        "tasks": p.tasks,
        "task_period_s": list(p.task_period_s),
        "saved_at_unix": time.time(),
    }

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------- per-task logs ----------
    for i in range(p.tasks):
        filename = os.path.join(output_dir, f"task_{i}.json")

        with p.locks[i]:
            data = {
                # control-period averaged values
                "rtr": list(p.filertr[i]),                       # RTR per control period
                "sms": list(p.filesms[i]),                       # SMs applied per control period
                "inference_latency_ms": list(p.fileinferencelatency[i]),  # avg ET per control period
                "response_time_ms": list(p.fileresponsetime[i]),          # avg RT per control period

                # NEW: deadline miss percentage per control period
                "deadline_miss_pct": list(p.filedeadlinemiss[i]),

                # mask history
                "tpc_masks_hex": [f"0x{m:016x}" for m in p.tpc_masks[i]],
            }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[logs] wrote {filename}")

def _unique_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    k = 1
    while True:
        d = os.path.join(base_dir, f"run{k:03d}")
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            return d
        k += 1


def _maybe_init_batch_trace(p: Parameters):
    # optional: track batch size changes over time in the shared header
    if not hasattr(p, "batch_size_trace"):
        p.batch_size_trace = [[] for _ in range(p.tasks)]


def _set_batch_size(p: Parameters, task_id: int, new_bs: int, t_elapsed: float):
    with p.locks[task_id]:
        p.batch_size[task_id] = int(new_bs)
        if hasattr(p, "batch_size_trace"):
            p.batch_size_trace[task_id].append({"t": float(t_elapsed), "batch": int(new_bs)})


def run_one_experiment(
    *,
    num_tasks: int,
    duration_s: float,
    controlperiod_s: float,
    period_s: float,
    batchsize_init: int,
    total_tpcs: int,
    solution: str,                 # "adhoc" or "smscaling"
    setpoint: float,
    kp: float,
    out_dir: str,
    verbose: bool = True,
    dynamic_batch_schedule=None,   # list of (t_sec, task_id, new_bs)
):
    stop_event = threading.Event()

    # fresh shared header each experiment (clears old logs)
    p = Parameters(tasks=num_tasks)
    _maybe_init_batch_trace(p)

    for i in range(num_tasks):
        p.task_period_s[i] = float(period_s)
        p.batch_size[i] = int(batchsize_init)
        p.tpc_masks[i].append(0)  # start with "disable none"

    # start inference threads
    tpcs_per_task = max(1, total_tpcs // num_tasks)
    threads = []
    for i in range(num_tasks):
        task = PeriodicInferenceTask(
            task_id=i,
            batch_size=batchsize_init,     # initial only; runtime changes via p.batch_size[i]
            period_s=period_s,
            total_tpcs=tpcs_per_task,
            params=p,
            stop_event=stop_event,
            img_path="n02099601_3004.jpg",
            duration_s=duration_s,
            verbose=False,  # set True if you want per-batch-change prints from tasks
        )
        t = threading.Thread(target=task.run, daemon=True)
        t.start()
        threads.append(t)

    # controller thread
    ctrl = Controller(
        duration=duration_s,
        controlperiod=controlperiod_s,
        params=p,
        stop_event=stop_event,
        solution=solution,
        setpoint=setpoint,
        kp=kp,
        total_tpcs=total_tpcs,
        verbose=verbose,
    )
    tctrl = threading.Thread(target=ctrl.run, daemon=True)
    tctrl.start()

    # scheduler loop to apply runtime batch-size changes
    start = time.time()
    sched = list(dynamic_batch_schedule or [])
    sched.sort(key=lambda x: x[0])
    next_idx = 0

    for i in range(num_tasks):
        _set_batch_size(p, i, batchsize_init, 0.0)

    while True:
        elapsed = time.time() - start
        if elapsed >= duration_s:
            break

        while next_idx < len(sched) and elapsed >= sched[next_idx][0]:
            t_sec, task_id, new_bs = sched[next_idx]
            _set_batch_size(p, task_id, new_bs, elapsed)
            print(f"[scheduler] t={elapsed:6.2f}s: task{task_id} batch_size -> {new_bs}", flush=True)
            next_idx += 1

        time.sleep(0.02)

    # stop everything
    stop_event.set()
    tctrl.join(timeout=3)
    for t in threads:
        t.join(timeout=3)

    # save logs
    dump_logs_to_json(p, output_dir=out_dir)
    print(f"[DONE] saved -> {out_dir}")


def experiment_sweep_setpoints(num_tasks: int):
    duration_s = 100.0
    controlperiod_s = 2.0
    period_s = 0.040
    batchsize = 20
    total_tpcs = 41

    setpoints = [0.60, 0.70, 0.80, 0.90]

    # ADHOC sweep
    for sp in setpoints:
        base = os.path.join("logs", "sweep", f"adhoc_sp{sp:.2f}")
        out_dir = _unique_run_dir(base)
        run_one_experiment(
            num_tasks=num_tasks,
            duration_s=duration_s,
            controlperiod_s=controlperiod_s,
            period_s=period_s,
            batchsize_init=batchsize,
            total_tpcs=total_tpcs,
            solution="adhoc",
            setpoint=sp,
            kp=0.0,
            out_dir=out_dir,
            verbose=True,
        )

    for sp in setpoints:
        base = os.path.join("logs", "sweep", f"proposed_sp{sp:.2f}")
        out_dir = _unique_run_dir(base)
        run_one_experiment(
            num_tasks=num_tasks,
            duration_s=duration_s,
            controlperiod_s=controlperiod_s,
            period_s=period_s,
            batchsize_init=batchsize,
            total_tpcs=total_tpcs,
            solution="smscaling",
            setpoint=sp,
            kp=4.0,  # tune if needed
            out_dir=out_dir,
            verbose=True,
        )


# def experiment_dynamic_batch_proposed(num_tasks: int):
#     duration_s = 200.0
#     controlperiod_s = 2.0
#     period_s = 0.050
#     batchsize = 20
#     total_tpcs = 41

#     dynamic_schedule = [
#         (40.0, 0, 40),    # t=40: increase batchsize of task 1
#         (80.0, 1, 40),    # t=80: increase batchsize of task 2
#         (160.0, 0, 20),   # t=160: revert task 1 back to 20
#         (160.0, 1, 20),   # t=160: revert task 2 back to 20
#     ]

#     base = os.path.join("logs", "dynamic", "proposed_batch_changes")
#     out_dir = _unique_run_dir(base)

#     run_one_experiment(
#         num_tasks=num_tasks,
#         duration_s=duration_s,
#         controlperiod_s=controlperiod_s,
#         period_s=period_s,
#         batchsize_init=batchsize,
#         total_tpcs=total_tpcs,
#         solution="smscaling",
#         setpoint=0.70,
#         kp=4.5,
#         out_dir=out_dir,
#         verbose=True,
#         dynamic_batch_schedule=dynamic_schedule,
#     )
    
def experiment_dynamic_batch_proposed(num_tasks: int):
    duration_s = 240.0
    controlperiod_s = 1.0
    period_s = 0.030
    batchsize = 20
    total_tpcs = 41

    dynamic_schedule = [
        (60.0, 0, 40),    # t=60: increase batchsize of task 1
        (100.0, 1, 40),    # t=100: increase batchsize of task 2
        (140.0, 2, 40),   # t=140: revert task 1 back to 20
        (180.0, 0, 20),   # t=180: revert task 2 back to 20
        (180.0, 1, 20),   # t=180: revert task 2 back to 20
        (180.0, 2, 20),   # t=180: revert task 2 back to 20

    ]

    # ----------------------------
    # Run ADHOC
    # ----------------------------
    base_adhoc = os.path.join("logs", "dynamic", "adhoc_batch_changes")
    out_dir_adhoc = _unique_run_dir(base_adhoc)

    run_one_experiment(
        num_tasks=num_tasks,
        duration_s=duration_s,
        controlperiod_s=controlperiod_s,
        period_s=period_s,
        batchsize_init=batchsize,
        total_tpcs=total_tpcs,
        solution="adhoc",
        setpoint=0.70,     # keep same setpoint for fair comparison
        kp=0,
        out_dir=out_dir_adhoc,
        verbose=True,
        dynamic_batch_schedule=dynamic_schedule,
    )

    # ----------------------------
    # Run SMSCALING (proposed)
    # ----------------------------
    base_prop = os.path.join("logs", "dynamic", "proposed_batch_changes")
    out_dir_prop = _unique_run_dir(base_prop)

    run_one_experiment(
        num_tasks=num_tasks,
        duration_s=duration_s,
        controlperiod_s=controlperiod_s,
        period_s=period_s,
        batchsize_init=batchsize,
        total_tpcs=total_tpcs,
        solution="smscaling",
        setpoint=0.70,
        kp=4.0,
        out_dir=out_dir_prop,
        verbose=True,
        dynamic_batch_schedule=dynamic_schedule,
    )


if __name__ == "__main__":
    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    #experiment_sweep_setpoints(num_tasks)
    experiment_dynamic_batch_proposed(num_tasks)
    print("All experiments completed.")