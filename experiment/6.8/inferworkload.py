import ctypes
import time
import threading
from typing import Optional

import numpy as np
import torch
import torchvision.models as models
from skimage import io
from skimage.transform import resize

# -------- libsmctrl (stream mask) --------
LIB_PATH = "/home/pacs/Srini/libsmctrl/libsmctrl.so"
libsmctrl = ctypes.CDLL(LIB_PATH)
libsmctrl.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
libsmctrl.libsmctrl_set_stream_mask.restype = None


def set_stream_mask(stream: torch.cuda.Stream, disable_mask: int):
    handle = ctypes.c_void_p(stream.cuda_stream)
    libsmctrl.libsmctrl_set_stream_mask(handle, ctypes.c_uint64(disable_mask))


class PeriodicInferenceTask:
    def __init__(
        self,
        task_id: int,
        batch_size: int,
        period_s: float,
        total_tpcs: int,              # kept for your interface (not required here)
        params,                       # Parameters instance (shared header)
        stop_event: threading.Event,
        img_path: str,
        duration_s: Optional[float] = None,  # stop after duration_s seconds if provided
        warmup_iters: int = 10,
        jobs: int = 1,                # number of forwards per period (averaged)
        guard_ns: int = 50_000,        # 50us hybrid sleep guard (optional)
        verbose: bool = True,
    ):
        self.task_id = int(task_id)
        self.period_s = float(period_s)
        self.total_tpcs = int(total_tpcs)
        self.p = params
        self.stop_event = stop_event
        self.img_path = img_path
        self.duration_s = float(duration_s) if duration_s is not None else None
        self.warmup_iters = int(warmup_iters)
        self.jobs = int(jobs)
        self.guard_ns = int(guard_ns)
        self.verbose = bool(verbose)

        # torch / cuda objects
        self.stream = torch.cuda.Stream()
        self.model = models.resnet18(weights="DEFAULT").eval().cuda()

        # runtime batch size state
        self._current_bs: Optional[int] = None
        self.inp: Optional[torch.Tensor] = None

        # ensure shared header has initial batch size
        with self.p.locks[self.task_id]:
            # if Parameters has p.batch_size list
            if hasattr(self.p, "batch_size"):
                self.p.batch_size[self.task_id] = int(batch_size)

        # build initial input
        self._ensure_input_matches_runtime_bs(force=True)

        # events for GPU timing
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_stop = torch.cuda.Event(enable_timing=True)

        self._warmup()
        
        
    def _warmup(self, timed: bool = True, timed_iters: int = 20):
        """
        Warm up CUDA + model so first measured iteration isn't inflated.
        - Applies current mask once.
        - Runs warmup_iters untimed.
        - Optionally runs timed_iters timed and prints median.
        """
        self._ensure_input_matches_runtime_bs(force=True)

        # 1) Apply current mask once before warmup
        with self.p.locks[self.task_id]:
            current_mask = self.p.tpc_masks[self.task_id][-1] if self.p.tpc_masks[self.task_id] else 0

        try:
            set_stream_mask(self.stream, current_mask)
        except Exception as e:
            if self.verbose:
                print(f"[InferenceTask {self.task_id}] mask set failed during warmup: {e}", flush=True)

        # Make sure mask takes effect before warmup kernels
        self.stream.synchronize()

        # 2) Untimed warmup forwards (compile kernels, autotune, caching)
        with torch.no_grad(), torch.cuda.stream(self.stream):
            for _ in range(max(1, self.warmup_iters)):
                _ = self.model(self.inp)

        # Ensure warmup is done globally
        torch.cuda.synchronize()

        # 3) Optional timed warmup to sanity-check latency
        if timed:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            samples_ms = []

            with torch.no_grad(), torch.cuda.stream(self.stream):
                for _ in range(max(1, timed_iters)):
                    ev0.record(self.stream)
                    _ = self.model(self.inp)
                    ev1.record(self.stream)
                    self.stream.synchronize()
                    samples_ms.append(float(ev0.elapsed_time(ev1)))

            torch.cuda.synchronize()

            if self.verbose and samples_ms:
                samples_ms.sort()
                med = samples_ms[len(samples_ms) // 2]
                p90 = samples_ms[int(0.9 * (len(samples_ms) - 1))]
                print(
                    f"[InferenceTask {self.task_id}] warmup done | "
                    f"median={med:.3f}ms p90={p90:.3f}ms | "
                    f"mask=0x{current_mask:016x} bs={self._current_bs}",
                    flush=True
                )

    # ---------- helpers ----------
    def _make_batch(self, img_path: str, batch_size: int) -> torch.Tensor:
        img = resize(io.imread(img_path), (224, 224))
        img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        batch = np.repeat(img, batch_size, axis=0).astype(np.float32)
        return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().cuda()

    def _get_desired_batch_size(self) -> int:
        # Default to current if Parameters has no batch_size
        if not hasattr(self.p, "batch_size"):
            return int(self._current_bs or 1)

        with self.p.locks[self.task_id]:
            bs = int(self.p.batch_size[self.task_id])
        return max(1, bs)

    def _ensure_input_matches_runtime_bs(self, force: bool = False):
        """
        Runtime support: if p.batch_size[task_id] changes, rebuild input tensor.
        Called once per inference period (cheap check, rebuild only on change).
        """
        desired_bs = self._get_desired_batch_size()

        if (not force) and (self._current_bs == desired_bs) and (self.inp is not None):
            return

        # rebuild input for new batch size
        self.inp = self._make_batch(self.img_path, desired_bs)
        self._current_bs = desired_bs

        if self.verbose:
            print(f"[InferenceTask {self.task_id}] batch_size now {desired_bs}", flush=True)

   

    def _now_ns(self) -> int:
        return time.monotonic_ns()

    def _sleep_until_abs_hybrid(self, target_ns: int):
        # hybrid sleep: sleep until (target-guard) then spin
        while True:
            now = self._now_ns()
            diff = target_ns - now
            if diff <= 0:
                return
            if diff > self.guard_ns:
                pre_ns = target_ns - self.guard_ns
                sleep_s = max(0.0, (pre_ns - now) / 1e9)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            else:
                # spin
                pass

    # ---------- one period measurement ----------
    def _run_once_and_measure(self, release_ns: int):
        """
        Measures:
          preemption_ms = (actual_start - release_time) in ms (scheduler / jitter)
          exec_ms       = GPU time for forward
          response_ms   = preemption_ms + exec_ms
        Appends into p.<metric>[task_id]
        """
        self._ensure_input_matches_runtime_bs(force=False)

        # "actual start" (just before we launch)
        start_ns = self._now_ns()
        preemption_ms = max(0.0, (start_ns - release_ns) / 1e6)

        # get current mask (controller may have appended a new one)
        with self.p.locks[self.task_id]:
            current_mask = self.p.tpc_masks[self.task_id][-1] if self.p.tpc_masks[self.task_id] else 0

        # re-apply mask each period (like your C++)
        set_stream_mask(self.stream, current_mask)
        self.stream.synchronize()

        # run inference with GPU timing on stream
        with torch.no_grad(), torch.cuda.stream(self.stream):
            self.ev_start.record(self.stream)
            for _ in range(self.jobs):
                _ = self.model(self.inp)
            self.ev_stop.record(self.stream)
            self.stream.synchronize()

        exec_ms_total = float(self.ev_start.elapsed_time(self.ev_stop))
        exec_ms = exec_ms_total / max(1, self.jobs)
        response_ms = preemption_ms + exec_ms

        rtr = response_ms / (self.period_s * 1000.0) if self.period_s > 0 else 0.0

        # append metrics into shared header
        with self.p.locks[self.task_id]:
            self.p.inferencelatency[self.task_id].append(exec_ms)
            self.p.responsetime[self.task_id].append(response_ms)
            self.p.rtr[self.task_id].append(rtr)

            # DO NOT append num_sms here if controller owns it.
            # If you really want a placeholder, keep it, but it will mix streams.
            # self.p.num_sms[self.task_id].append(self.total_tpcs)

            self.p.intermediateinferencelatency[self.task_id].append(exec_ms)
            self.p.intermediateresponsetime[self.task_id].append(response_ms)
            self.p.intermediatertr[self.task_id].append(rtr)

    # ---------- main loop ----------
    def run(self):
        """
        Runs periodically until:
          - stop_event is set, OR
          - duration_s exceeded (if provided)
        """
        start_wall = time.time()
        start_ns = self._now_ns()
        next_release_ns = start_ns  # first release "now"

        while not self.stop_event.is_set():
            if self.duration_s is not None and (time.time() - start_wall) >= self.duration_s:
                break

            self._sleep_until_abs_hybrid(next_release_ns)
            if self.stop_event.is_set():
                break

            self._run_once_and_measure(release_ns=next_release_ns)

            period_ns = int(round(self.period_s * 1e9))
            next_release_ns += period_ns

        try:
            self.stream.synchronize()
        except Exception:
            pass

    def close(self):
        self.model = None
        self.inp = None
        self.ev_start = None
        self.ev_stop = None
        self.stream = None
        torch.cuda.synchronize()