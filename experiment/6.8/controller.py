# # controller.py
# import time
# import threading
# from typing import List, Optional


# def mean(xs: List[float]) -> float:
#     return sum(xs) / len(xs) if xs else 0.0


# def clamp(x, lo, hi):
#     return lo if x < lo else hi if x > hi else x


# def assign_simple_contiguous(total_tpcs: int, tpcs_per_task: List[int]) -> List[int]:
#     """
#     Simple non-overlap allocator:
#     gives each task a contiguous block, wraps if not enough.
#     Returns disable masks (bit=1 disabled, bit=0 enabled).
#     """
#     all_bits = (1 << total_tpcs) - 1
#     masks: List[int] = []
#     start = 0
#     for need in tpcs_per_task:
#         need = max(1, min(int(need), total_tpcs))
#         enabled = 0
#         for k in range(need):
#             enabled |= (1 << ((start + k) % total_tpcs))
#         start = (start + need) % total_tpcs
#         disable = (~enabled) & all_bits
#         masks.append(disable)
#     return masks


# class Controller:
#     def __init__(
#         self,
#         duration: float,
#         controlperiod: float,
#         params,
#         stop_event: threading.Event,
#         solution: str = "adhoc",                   # "adhoc" or "smscaling"
#         setpoint: float = 0.90,                   # scalar setpoint for all tasks (RTR target)
#         setpoints: Optional[List[float]] = None,  # OR per-task setpoints
#         total_tpcs: int = 82,
#         sms_per_tpc: int = 2,
#         minsms: int = 2,
#         maxsms: Optional[int] = None,
#         sm_step: int = 5,                         # for adhoc
#         eps: float = 0.0,                         # deadband
#         kp: float = 5.0,                          # proportional gain for smscaling
#         verbose: bool = True,
#     ):
#         self.duration = float(duration)
#         self.controlperiod = float(controlperiod)
#         self.p = params
#         self.stop_event = stop_event
#         self.solution = solution.lower().strip()
#         self.verbose = bool(verbose)

#         self.N = int(self.p.tasks)
#         self.total_tpcs = int(total_tpcs)
#         self.sms_per_tpc = int(sms_per_tpc)
#         self.minsms = int(minsms)
#         self.maxsms = int(maxsms) if maxsms is not None else (self.total_tpcs * self.sms_per_tpc)

#         self.sm_step = int(sm_step)
#         self.eps = float(eps)
#         self.kp = float(kp)

#         # ---- Setpoints ----
#         if setpoints is not None:
#             if len(setpoints) != self.N:
#                 raise ValueError(f"setpoints length must be {self.N}")
#             self.setpoints = [float(x) for x in setpoints]
#         else:
#             self.setpoints = [float(setpoint)] * self.N

#         # ---- Actuator state (SMs) ----
#         start_sms = max(
#             self.minsms,
#             (self.maxsms // max(1, self.N)) // self.sms_per_tpc * self.sms_per_tpc
#         )
#         self.sms_current = [start_sms for _ in range(self.N)]

#         # error[i] = setpoint[i] - measured_rtr[i]
#         self.outererror = [0.0] * self.N

#         # consume new samples since last control period
#         self._last_idx_resp = [0] * self.N
#         self._last_idx_et = [0] * self.N

#         # debug/print buffers
#         self._avg_rt_ms = [0.0] * self.N
#         self._avg_et_ms = [0.0] * self.N
#         self._rtr_meas = [0.0] * self.N
#         self._dm_pct = [0.0] * self.N  # deadline miss percentage per control period
#         self._delta_sm = [0.0] * self.N
#         self._mask_now = [0] * self.N
#         self._sms_old = [0] * self.N
#         self._sms_new = [0] * self.N

#     def stop(self):
#         self.stop_event.set()

#     def _measure_and_log(self):
#         """
#         Each control period:
#           - take NEW RT samples since last tick
#           - RTR = avg_rt / period_ms   (rt/period)
#           - deadline miss % = (#rt > period_ms) / total * 100
#           - error = setpoint - rtr
#           - append rtr + deadline miss % into params
#         """
#         for i in range(self.N):
#             # Snapshot sizes + config under lock (safe)
#             with self.p.locks[i]:
#                 resp_list = self.p.responsetime[i]         # RT samples (ms)
#                 et_list = self.p.inferencelatency[i]       # ET samples (ms)
#                 task_period_s = self.p.task_period_s[i] if self.p.task_period_s[i] > 0 else self.controlperiod
#                 mask_now = self.p.tpc_masks[i][-1] if self.p.tpc_masks[i] else 0

#                 # Capture *current lengths* for slicing outside lock
#                 resp_len = len(resp_list)
#                 et_len = len(et_list)

#             # ---- Slice NEW windows (no lock needed; list append is atomic, but slice is safe enough here) ----
#             start_rt = self._last_idx_resp[i]
#             window_rt = resp_list[start_rt:resp_len]
#             self._last_idx_resp[i] = resp_len

#             start_et = self._last_idx_et[i]
#             window_et = et_list[start_et:et_len]
#             self._last_idx_et[i] = et_len

#             period_ms = task_period_s * 1000.0

#             if window_rt and period_ms > 1e-9:
#                 avg_rt = mean(window_rt)
#                 rtr_meas = avg_rt / period_ms

#                 misses = sum(1 for rt in window_rt if rt > period_ms)
#                 dm_pct = (misses / len(window_rt)) * 100.0
#             else:
#                 avg_rt = 0.0
#                 rtr_meas = 0.0
#                 dm_pct = 0.0

#             avg_et = mean(window_et) if window_et else 0.0

#             # Append results under lock
#             with self.p.locks[i]:
#                 self.p.rtr[i].append(rtr_meas)
#                 self.p.deadlinemiss[i].append(dm_pct)

#             self.outererror[i] = float(self.setpoints[i]) - float(rtr_meas)

#             # store for print/log
#             self._avg_rt_ms[i] = avg_rt
#             self._avg_et_ms[i] = avg_et
#             self._rtr_meas[i] = rtr_meas
#             self._dm_pct[i] = dm_pct
#             self._mask_now[i] = mask_now
#             self._delta_sm[i] = 0.0

#     def _adhoc(self):
#         """
#         Step controller:
#           if RTR < setpoint -> remove SMs
#           if RTR > setpoint -> add SMs
#         """
#         for i in range(self.N):
#             self._sms_old[i] = self.sms_current[i]

#             rtr = self._rtr_meas[i]
#             setp = float(self.setpoints[i])

#             sm_old = self.sms_current[i]
#             sm_new = sm_old

#             if rtr < setp - self.eps:
#                 sm_new = sm_old - self.sm_step
#             elif rtr > setp + self.eps:
#                 sm_new = sm_old + self.sm_step

#             sm_new = clamp(sm_new, self.minsms, self.maxsms)
#             sm_new = (sm_new // self.sms_per_tpc) * self.sms_per_tpc
#             sm_new = max(self.sms_per_tpc, sm_new)

#             self.sms_current[i] = sm_new
#             self._sms_new[i] = sm_new
#             self._delta_sm[i] = float(sm_new - sm_old)

#     def _smscaling(self):
#         """
#         Proportional controller:
#           delta_sm = -kp * (setpoint - rtr)
#         """
#         for i in range(self.N):
#             self._sms_old[i] = self.sms_current[i]

#             err = self.outererror[i]  # setpoint - rtr
#             delta = -self.kp * err
#             sm_old = self.sms_current[i]
#             sm_new = int(round(sm_old + delta))

#             sm_new = clamp(sm_new, self.minsms, self.maxsms)
#             sm_new = (sm_new // self.sms_per_tpc) * self.sms_per_tpc
#             sm_new = max(self.sms_per_tpc, sm_new)

#             self.sms_current[i] = sm_new
#             self._sms_new[i] = sm_new
#             self._delta_sm[i] = float(sm_new - sm_old)

#     def _publish_masks(self):
#         """
#         Convert sms_current -> requested TPCs -> disable masks.
#         Append history into:
#           p.num_sms[i], p.tpc_masks[i]
#         """
#         tpcs_req: List[int] = []
#         for sm in self.sms_current:
#             sm = max(self.sms_per_tpc, (int(sm) // self.sms_per_tpc) * self.sms_per_tpc)
#             tpcs_req.append(max(1, min(self.total_tpcs, sm // self.sms_per_tpc)))

#         disable_masks = assign_simple_contiguous(self.total_tpcs, tpcs_req)

#         for i in range(self.N):
#             applied_sms = tpcs_req[i] * self.sms_per_tpc
#             with self.p.locks[i]:
#                 self.p.num_sms[i].append(applied_sms)
#                 self.p.tpc_masks[i].append(disable_masks[i])

#             self._mask_now[i] = disable_masks[i]

#     def _print_status(self, control_period_idx: int, elapsed_s: float):
#         if not self.verbose:
#             return

#         print(f"\n[outerloop] Control period {control_period_idx} | Time = {elapsed_s:.1f}s")
#         for i in range(self.N):
#             period_ms = (self.p.task_period_s[i] if self.p.task_period_s[i] > 0 else self.controlperiod) * 1000.0

#             last_rt = self._avg_rt_ms[i]
#             last_et = self._avg_et_ms[i]
#             last_rtr = self._rtr_meas[i]
#             dm = self._dm_pct[i]
#             err = self.outererror[i]
#             dd = self._delta_sm[i]
#             sm_old = self._sms_old[i]
#             sm_new = self._sms_new[i]
#             mask = self._mask_now[i]
#             mask_hex = f"0x{mask:016x}"

#             # file buffers (if you want)
#             with self.p.locks[i]:
#                 if hasattr(self.p, "filertr"):
#                     self.p.filertr[i].append(last_rtr)
#                 if hasattr(self.p, "fileinferencelatency"):
#                     self.p.fileinferencelatency[i].append(last_et)
#                 if hasattr(self.p, "fileresponsetime"):
#                     self.p.fileresponsetime[i].append(last_rt)
#                 if hasattr(self.p, "filesms"):
#                     self.p.filesms[i].append(sm_old)
#                 if hasattr(self.p, "filedeadlinemiss"):
#                     self.p.filedeadlinemiss[i].append(dm)

#             print(
#                 f"Task {i + 1}"
#                 f" | RT={last_rt:.3f}ms"
#                 f" | ET={last_et:.3f}ms"
#                 f" | Period={period_ms:.3f}ms"
#                 f" | RTR={last_rtr:.4f}"
#                 f" | DM={dm:.2f}%"
#                 f" | Setpoint={self.setpoints[i]:.4f}"
#                 f" | Error={err:.4f}"
#                 f" | Δd={dd:.3f}"
#                 f" | Current SMs={sm_old}"
#                 f" | New SMs={sm_new}"
#                 f" | Current Mask={mask_hex}"
#             )

#     def run(self):
#         start = time.time()
#         next_t = time.monotonic()
#         control_period_idx = 0

#         while not self.stop_event.is_set() and (time.time() - start) < self.duration:
#             now = time.monotonic()
#             if now < next_t:
#                 time.sleep(next_t - now)

#             elapsed_s = time.time() - start

#             # 1) measure + compute error = setpoint - measured_rtr
#             self._measure_and_log()

#             # 2) actuation
#             if self.solution == "adhoc":
#                 self._adhoc()
#             elif self.solution == "smscaling":
#                 self._smscaling()
#             else:
#                 raise ValueError(f"Unknown solution: {self.solution}")

#             # 3) print/log
#             self._print_status(control_period_idx, elapsed_s)

#             # 4) publish masks + sms history
#             self._publish_masks()

#             control_period_idx += 1
#             next_t += self.controlperiod
# controller.py

# controller.py
# controller.py
import time
import threading
from typing import List, Optional, Tuple
import math


# ============================================================
# Helpers
# ============================================================
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _popcount(x: int) -> int:
    # Python 3.8-safe popcount (no int.bit_count())
    return bin(x).count("1")


# ============================================================
# Overlap allocator (Python version of assign_overlap)
# ============================================================
def _build_ring_mask(start: int, length: int, total_tpcs: int) -> int:
    if length <= 0 or total_tpcs <= 0:
        return 0
    length = min(length, total_tpcs)
    start = start % total_tpcs
    m = 0
    for k in range(length):
        idx = (start + k) % total_tpcs
        m |= (1 << idx)
    return m


def _find_free_run(free_mask: int, length: int, total_tpcs: int) -> int:
    if length <= 0:
        return -1
    length = min(length, total_tpcs)
    if _popcount(free_mask) < length:
        return -1

    for start in range(total_tpcs):
        ok = True
        for k in range(length):
            idx = (start + k) % total_tpcs
            if ((free_mask >> idx) & 1) == 0:
                ok = False
                break
        if ok:
            return start
    return -1


def _pick_any_free(free_mask: int, need: int, total_tpcs: int) -> int:
    sel = 0
    for i in range(total_tpcs):
        if need <= 0:
            break
        if ((free_mask >> i) & 1) == 1:
            sel |= (1 << i)
            need -= 1
    return sel


def assign_overlap_masks(
    total_tpcs: int,
    tpcs_req: List[int],
    priorities: List[float],
    max_overlap_per_tpc: int = 2,
    max_borrow_per_task: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Build disable masks with overlap allowed.

    Returns:
      disable_masks[i]   bit=1 disabled, bit=0 enabled
      enabled_counts[i]  actual number of enabled TPCs per task
    """
    N = len(tpcs_req)
    if N == 0 or total_tpcs <= 0:
        return [], []

    all_bits = (1 << total_tpcs) - 1
    max_overlap_per_tpc = max(1, int(max_overlap_per_tpc))
    if max_borrow_per_task is None:
        max_borrow_per_task = total_tpcs
    max_borrow_per_task = max(0, int(max_borrow_per_task))

    order = list(range(N))
    order.sort(key=lambda i: (float(priorities[i]), int(tpcs_req[i])), reverse=True)

    enabled_masks = [0] * N
    share = [0] * total_tpcs

    def compute_free_mask() -> int:
        m = 0
        for t in range(total_tpcs):
            if share[t] == 0:
                m |= (1 << t)
        return m & all_bits

    def can_use(t: int) -> bool:
        return share[t] < max_overlap_per_tpc

    def add_tpc(task_i: int, t: int):
        if ((enabled_masks[task_i] >> t) & 1) == 0:
            enabled_masks[task_i] |= (1 << t)
            share[t] += 1

    # PASS A: free-only
    for i in order:
        need = max(1, min(int(tpcs_req[i]), total_tpcs))
        free_mask = compute_free_mask()

        start = _find_free_run(free_mask, need, total_tpcs)
        if start >= 0:
            sel = _build_ring_mask(start, need, total_tpcs) & free_mask
        else:
            sel = _pick_any_free(free_mask, need, total_tpcs)

        for t in range(total_tpcs):
            if ((sel >> t) & 1) == 1:
                add_tpc(i, t)

    # PASS B: overlap/borrow
    for i in order:
        need = max(1, min(int(tpcs_req[i]), total_tpcs))
        got = _popcount(enabled_masks[i])
        short = need - got
        if short <= 0:
            continue

        borrowed = 0
        base = (i * total_tpcs) // max(1, N)

        for k in range(2 * total_tpcs):
            if short <= 0 or borrowed >= max_borrow_per_task:
                break
            t = (base + k) % total_tpcs
            if ((enabled_masks[i] >> t) & 1) == 1:
                continue
            if not can_use(t):
                continue
            add_tpc(i, t)
            short -= 1
            borrowed += 1

        for t in range(total_tpcs):
            if short <= 0 or borrowed >= max_borrow_per_task:
                break
            if ((enabled_masks[i] >> t) & 1) == 1:
                continue
            if not can_use(t):
                continue
            add_tpc(i, t)
            short -= 1
            borrowed += 1

    disable_masks: List[int] = []
    enabled_counts: List[int] = []
    for i in range(N):
        enabled = enabled_masks[i] & all_bits
        disable = (~enabled) & all_bits
        disable_masks.append(disable)
        enabled_counts.append(_popcount(enabled))
    return disable_masks, enabled_counts


# ============================================================
# Controller
# ============================================================
class Controller:
    def __init__(
        self,
        duration: float,
        controlperiod: float,
        params,
        stop_event: threading.Event,
        solution: str = "adhoc",                   # "adhoc" or "smscaling"
        setpoint: float = 0.90,
        setpoints: Optional[List[float]] = None,
        total_tpcs: int = 41,
        sms_per_tpc: int = 2,
        minsms: int = 2,
        maxsms: Optional[int] = None,
        sm_step: float = 2.0,                      # step size in SMs (will be quantized)
        rtr_deadband: float = 0.03,                # if |setpoint - rtr| <= deadband, hold
        kp: float = 5.0,
        slice_ms: int = 10,
        max_overlap_per_tpc: int = 2,
        max_borrow_per_task: Optional[int] = None,
        verbose: bool = True,
        freeze_delta_sigma_inside_deadband: bool = False,  # optional stronger "freeze"
    ):
        self.duration = float(duration)
        self.controlperiod = float(controlperiod)
        self.p = params
        self.stop_event = stop_event
        self.solution = solution.lower().strip()
        self.verbose = bool(verbose)

        self.N = int(self.p.tasks)
        self.total_tpcs = int(total_tpcs)
        self.sms_per_tpc = int(sms_per_tpc)
        self.minsms = int(minsms)
        self.maxsms = int(maxsms) if maxsms is not None else (self.total_tpcs * self.sms_per_tpc)

        self.sm_step = float(sm_step)
        self.rtr_deadband = float(rtr_deadband)
        self.kp = float(kp)

        self.slice_ms = max(1, int(slice_ms))
        self.max_overlap_per_tpc = max(1, int(max_overlap_per_tpc))
        self.max_borrow_per_task = max_borrow_per_task
        self.freeze_delta_sigma_inside_deadband = bool(freeze_delta_sigma_inside_deadband)

        if setpoints is not None:
            if len(setpoints) != self.N:
                raise ValueError(f"setpoints length must be {self.N}")
            self.setpoints = [float(x) for x in setpoints]
        else:
            self.setpoints = [float(setpoint)] * self.N

        # Start SMs (even share), quantized to sms_per_tpc
        start_sms = max(
            self.minsms,
            (self.maxsms // max(1, self.N)) // self.sms_per_tpc * self.sms_per_tpc
        )
        start_sms = max(self.sms_per_tpc, (start_sms // self.sms_per_tpc) * self.sms_per_tpc)

        self.sms_current = [int(start_sms) for _ in range(self.N)]

        # For proposed (smscaling + delta-sigma)
        self._sm_acc = [float(start_sms) for _ in range(self.N)]
        self._ds_acc_tpc = [0.0 for _ in range(self.N)]

        self.outererror = [0.0] * self.N

        # debug/print buffers
        self._avg_rt_ms = [0.0] * self.N
        self._avg_et_ms = [0.0] * self.N
        self._rtr_meas = [0.0] * self.N
        self._dm_pct = [0.0] * self.N
        self._delta_sm = [0.0] * self.N
        self._mask_now = [0] * self.N
        self._sms_old = [0] * self.N
        self._sms_new = [0] * self.N

        # actuation thread for delta-sigma
        self._act_thread: Optional[threading.Thread] = None
        self._stop_act = threading.Event()
        self._act_lock = threading.Lock()

    def stop(self):
        self.stop_event.set()
        self._stop_actuation_thread()

    # ------------------------------------------------------------
    # Measurement (RTR + Deadline Miss %), with buffer clearing
    # ------------------------------------------------------------
    def _measure_and_log(self):
        """
        Each control period:
          - copy ALL RT and ET samples accumulated since last control tick (under lock)
          - clear the RT/ET buffers so next period starts fresh
          - RTR = avg(RT) / period_ms
          - DM% = count(RT/period_ms > 1) / count(RT) * 100
        """
        for i in range(self.N):
            with self.p.locks[i]:
                task_period_s = self.p.task_period_s[i] if self.p.task_period_s[i] > 0 else self.controlperiod
                mask_now = self.p.tpc_masks[i][-1] if self.p.tpc_masks[i] else 0

                window_rt = list(self.p.responsetime[i])       # response time samples (ms)
                window_et = list(self.p.inferencelatency[i])   # execution time samples (ms)

                # IMPORTANT: clear per-control-period measurement buffers
                self.p.responsetime[i].clear()
                self.p.inferencelatency[i].clear()

            period_ms = task_period_s * 1000.0

            if window_rt and period_ms > 1e-9:
                avg_rt = mean(window_rt)
                rtr_meas = avg_rt / period_ms

                misses = sum(1 for rt in window_rt if (rt / period_ms) > 1.0)  # deadline miss if rt/period > 1
                dm_pct = (misses / len(window_rt)) * 100.0
            else:
                avg_rt = 0.0
                rtr_meas = 0.0
                dm_pct = 0.0

            avg_et = mean(window_et) if window_et else 0.0

            with self.p.locks[i]:
                self.p.rtr[i].append(rtr_meas)
                self.p.deadlinemiss[i].append(dm_pct)

            self.outererror[i] = float(self.setpoints[i]) - float(rtr_meas)

            self._avg_rt_ms[i] = avg_rt
            self._avg_et_ms[i] = avg_et
            self._rtr_meas[i] = rtr_meas
            self._dm_pct[i] = dm_pct
            self._mask_now[i] = mask_now
            self._delta_sm[i] = 0.0

    # ------------------------------------------------------------
    # Adhoc controller (step + deadband)
    # ------------------------------------------------------------
    def _adhoc(self):
        """
        Step controller with deadband:
          if |error| <= deadband: hold
          else:
            rtr < setpoint: decrease SMs
            rtr > setpoint: increase SMs
        """
        for i in range(self.N):
            self._sms_old[i] = int(self.sms_current[i])

            rtr = float(self._rtr_meas[i])
            setp = float(self.setpoints[i])
            err = setp - rtr

            sm_old = int(self.sms_current[i])
            sm_new = sm_old

            if abs(err) <= self.rtr_deadband:
                sm_new = sm_old
            else:
                if rtr < setp:
                    sm_new = sm_old - self.sm_step
                else:
                    sm_new = sm_old + self.sm_step

            sm_new = int(round(clamp(sm_new, self.minsms, self.maxsms)))
            sm_new = (sm_new // self.sms_per_tpc) * self.sms_per_tpc
            sm_new = max(self.sms_per_tpc, sm_new)

            self.sms_current[i] = sm_new
            self._sms_new[i] = sm_new
            self._delta_sm[i] = float(sm_new - sm_old)

    # ------------------------------------------------------------
    # Proposed controller target update (P + deadband), return changed flag
    # ------------------------------------------------------------
    def _smscaling_compute_target(self) -> bool:
        """
        Proportional controller with deadband:
          if |error| <= deadband: do not change target at all
          else: target_sm += -kp * (setpoint - rtr)

        Returns:
          changed = True if any task is outside deadband (so we should update actuation)
        """
        changed = False
        for i in range(self.N):
            self._sms_old[i] = int(self.sms_current[i])

            rtr = float(self._rtr_meas[i])
            setp = float(self.setpoints[i])
            err = setp - rtr

            if abs(err) <= self.rtr_deadband:
                delta_sm = 0.0
            else:
                delta_sm = -self.kp * err
                changed = True

            self._sm_acc[i] += float(delta_sm)
            self._sm_acc[i] = float(clamp(self._sm_acc[i], self.minsms, self.maxsms))

            sm_rounded = int(round(self._sm_acc[i]))
            sm_rounded = int(clamp(sm_rounded, self.minsms, self.maxsms))
            sm_quant = (sm_rounded // self.sms_per_tpc) * self.sms_per_tpc
            sm_quant = max(self.sms_per_tpc, sm_quant)

            self._sms_new[i] = sm_quant
            self._delta_sm[i] = float(sm_quant - self._sms_old[i])

        return changed

    # ------------------------------------------------------------
    # Delta-sigma actuation for proposed solution
    # ------------------------------------------------------------
    def _actuation_for_period_delta_sigma(self, period_s: float):
        end_t = time.monotonic() + float(period_s)
        tick_s = self.slice_ms / 1000.0

        while (not self._stop_act.is_set()) and (time.monotonic() < end_t):
            chosen_tpcs = [1] * self.N
            priorities = [max(0.0, float(self.outererror[i])) for i in range(self.N)]

            for i in range(self.N):
                x_tpc = float(self._sm_acc[i]) / float(self.sms_per_tpc)

                min_tpc = max(1.0, float(self.minsms) / float(self.sms_per_tpc))
                max_tpc = max(min_tpc, float(self.maxsms) / float(self.sms_per_tpc))
                x_tpc = float(clamp(x_tpc, min_tpc, max_tpc))

                L = int(math.floor(x_tpc))
                U = L + 1

                L = int(clamp(L, int(math.ceil(min_tpc)), int(math.floor(max_tpc))))
                U = int(clamp(U, int(math.ceil(min_tpc)), int(math.floor(max_tpc))))

                if U <= L:
                    chosen = L
                else:
                    f = (x_tpc - float(L)) / float(U - L)
                    f = float(clamp(f, 0.0, 1.0))
                    self._ds_acc_tpc[i] += f

                    if self._ds_acc_tpc[i] >= 1.0:
                        chosen = U
                        self._ds_acc_tpc[i] -= 1.0
                    else:
                        chosen = L

                chosen_tpcs[i] = max(1, min(int(chosen), self.total_tpcs))

            disable_masks, enabled_counts = assign_overlap_masks(
                total_tpcs=self.total_tpcs,
                tpcs_req=chosen_tpcs,
                priorities=priorities,
                max_overlap_per_tpc=self.max_overlap_per_tpc,
                max_borrow_per_task=self.max_borrow_per_task,
            )

            with self._act_lock:
                for i in range(self.N):
                    applied_sms = enabled_counts[i] * self.sms_per_tpc
                    with self.p.locks[i]:
                        self.p.num_sms[i].append(int(applied_sms))
                        self.p.tpc_masks[i].append(int(disable_masks[i]))

                    self.sms_current[i] = int(applied_sms)
                    self._mask_now[i] = int(disable_masks[i])

            now = time.monotonic()
            if now + tick_s >= end_t:
                break
            time.sleep(tick_s)

    def _start_new_actuation_period(self, period_s: float):
        self._stop_actuation_thread()
        self._stop_act.clear()
        self._act_thread = threading.Thread(
            target=self._actuation_for_period_delta_sigma,
            args=(period_s,),
            daemon=True,
        )
        self._act_thread.start()

    def _stop_actuation_thread(self):
        if self._act_thread and self._act_thread.is_alive():
            self._stop_act.set()
            self._act_thread.join(timeout=1.0)
        self._act_thread = None

    # ------------------------------------------------------------
    # Print / logging
    # ------------------------------------------------------------
    def _print_status(self, control_period_idx: int, elapsed_s: float):
        if not self.verbose:
            return

        print(f"\n[outerloop] Control period {control_period_idx} | Time = {elapsed_s:.1f}s")
        for i in range(self.N):
            period_ms = (self.p.task_period_s[i] if self.p.task_period_s[i] > 0 else self.controlperiod) * 1000.0

            last_rt = self._avg_rt_ms[i]
            last_et = self._avg_et_ms[i]
            last_rtr = self._rtr_meas[i]
            dm = self._dm_pct[i]
            err = self.outererror[i]
            dd = self._delta_sm[i]
            sm_old = self._sms_old[i]
            sm_new = self._sms_new[i]
            mask = self._mask_now[i]
            mask_hex = f"0x{mask:016x}"

            with self.p.locks[i]:
                if hasattr(self.p, "filertr"):
                    self.p.filertr[i].append(last_rtr)
                if hasattr(self.p, "fileinferencelatency"):
                    self.p.fileinferencelatency[i].append(last_et)
                if hasattr(self.p, "fileresponsetime"):
                    self.p.fileresponsetime[i].append(last_rt)
                if hasattr(self.p, "filesms"):
                    self.p.filesms[i].append(int(sm_old))
                if hasattr(self.p, "filedeadlinemiss"):
                    self.p.filedeadlinemiss[i].append(float(dm))

            print(
                f"Task {i + 1}"
                f" | RT={last_rt:.3f}ms"
                f" | ET={last_et:.3f}ms"
                f" | Period={period_ms:.3f}ms"
                f" | RTR={last_rtr:.4f}"
                f" | DM={dm:.2f}%"
                f" | Setpoint={self.setpoints[i]:.4f}"
                f" | Error={err:.4f}"
                f" | DeltaSM={dd:.3f}"
                f" | Current SMs={sm_old}"
                f" | New SMs={sm_new}"
                f" | Current Mask={mask_hex}"
            )

    # ------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------
    def run(self):
        start = time.time()
        next_t = time.monotonic()
        control_period_idx = 0

        # Start delta-sigma thread only for proposed solution
        if self.solution == "smscaling":
            self._start_new_actuation_period(self.controlperiod)

        while not self.stop_event.is_set() and (time.time() - start) < self.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)

            elapsed_s = time.time() - start

            # 1) measure (and clear RT/ET buffers)
            self._measure_and_log()

            # 2) control + actuation
            if self.solution == "adhoc":
                self._adhoc()

                tpcs_req = [
                    max(1, min(self.total_tpcs, int(sm) // self.sms_per_tpc))
                    for sm in self.sms_current
                ]
                priorities = [max(0.0, float(self.outererror[i])) for i in range(self.N)]

                disable_masks, enabled_counts = assign_overlap_masks(
                    total_tpcs=self.total_tpcs,
                    tpcs_req=tpcs_req,
                    priorities=priorities,
                    max_overlap_per_tpc=self.max_overlap_per_tpc,
                    max_borrow_per_task=self.max_borrow_per_task,
                )

                for i in range(self.N):
                    applied_sms = enabled_counts[i] * self.sms_per_tpc
                    with self.p.locks[i]:
                        self.p.num_sms[i].append(int(applied_sms))
                        self.p.tpc_masks[i].append(int(disable_masks[i]))
                    self._mask_now[i] = int(disable_masks[i])

            elif self.solution == "smscaling":
                changed = self._smscaling_compute_target()

                # This is the key change you asked for:
                # if within +/- deadband, do not change anything (do not restart DS)
                if changed:
                    self._start_new_actuation_period(self.controlperiod)
                else:
                    if self.freeze_delta_sigma_inside_deadband:
                        self._stop_actuation_thread()

            else:
                raise ValueError(f"Unknown solution: {self.solution}")

            # 3) print/log
            self._print_status(control_period_idx, elapsed_s)

            control_period_idx += 1
            next_t += self.controlperiod

        self._stop_actuation_thread()