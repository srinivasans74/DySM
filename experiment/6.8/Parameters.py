"""Completed-job records, adapted from Srini's PyTorch experiment structure."""
from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

# Inherited ML plant, RRT/SM: keep original instances 2 and 3 (indices 1, 2).
# This is a reduced inherited model, not a new two-instance identification.
MODEL_B = [[-0.028319982458333327, -0.0014145018541666706], [-0.0018402428958333415, -0.027112935291666657]]


@dataclass(frozen=True)
class JobSample:
    release_ns: int
    completion_ns: int
    gpu_time_ms: float
    batch_size: int
    enqueue_ns: Optional[int] = None


class Parameters:
    def __init__(self, tasks, period_s):
        if tasks <= 0 or not math.isfinite(period_s) or period_s <= 0:
            raise ValueError('Positive task count and period required')
        self.period_ns = round(period_s * 1e9)
        self.jobs = [[] for _ in range(tasks)]
        self.cursor = [0] * tasks
        self.locks = [threading.Lock() for _ in range(tasks)]
        self.batch_size = [20] * tasks

    def record_job(self, task_id, sample):
        if (sample.completion_ns < sample.release_ns
                or not math.isfinite(sample.gpu_time_ms) or sample.gpu_time_ms < 0):
            raise ValueError('Invalid completed-job timing')
        with self.locks[task_id]:
            self.jobs[task_id].append(sample)

    def take_window(self, task_id):
        with self.locks[task_id]:
            samples = self.jobs[task_id][self.cursor[task_id]:]
            self.cursor[task_id] = len(self.jobs[task_id])
        count = len(samples)
        response_ns = [s.completion_ns - s.release_ns for s in samples]
        misses = sum(t > self.period_ns for t in response_ns)
        mean_ns = sum(response_ns) / count if count else None
        return dict(sample_count=count, deadline_miss_count=misses,
                    response_time_ms=mean_ns / 1e6 if count else None,
                    rtr=mean_ns / self.period_ns if count else None,
                    inference_latency_ms=sum(s.gpu_time_ms for s in samples) / count if count else None,
                    deadline_miss_pct=100.0 * misses / count if count else None)

    def set_batch(self, task_id, batch, clock=time.monotonic_ns):
        with self.locks[task_id]:
            requested = clock()
            self.batch_size[task_id] = batch
            return requested

    def get_batch(self, task_id):
        with self.locks[task_id]:
            return self.batch_size[task_id]
