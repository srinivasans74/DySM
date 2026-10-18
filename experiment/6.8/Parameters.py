# Parameters.py
from dataclasses import dataclass, field
from typing import List
import threading

DataType = float

@dataclass
class Parameters:
    tasks: int

    # per-task configuration
    task_period_s: List[float] = field(default_factory=list)  # seconds
    batch_size: List[int] = field(default_factory=list)

    # per-task sample streams (written by inference threads)
    responsetime: List[List[DataType]] = field(default_factory=list)       # RT samples (ms)
    inferencelatency: List[List[DataType]] = field(default_factory=list)   # ET samples (ms)

    # per-task controller outputs (written by controller)
    rtr: List[List[DataType]] = field(default_factory=list)                # RTR per control period
    deadlinemiss: List[List[float]] = field(default_factory=list)          # DM% per control period
    num_sms: List[List[int]] = field(default_factory=list)                 # SMs per control period
    tpc_masks: List[List[int]] = field(default_factory=list)               # disable masks per control period

    # optional intermediate dumps (if you still want them)
    intermediateresponsetime: List[List[DataType]] = field(default_factory=list)
    intermediateinferencelatency: List[List[DataType]] = field(default_factory=list)
    intermediatertr: List[List[DataType]] = field(default_factory=list)

    # optional "file" buffers for later JSON/CSV plotting
    filertr: List[List[DataType]] = field(default_factory=list)
    fileinferencelatency: List[List[DataType]] = field(default_factory=list)
    fileresponsetime: List[List[DataType]] = field(default_factory=list)
    filesms: List[List[int]] = field(default_factory=list)
    filedeadlinemiss: List[List[float]] = field(default_factory=list)

    # one lock per task
    locks: List[threading.Lock] = field(default_factory=list)

    def __post_init__(self):
        n = int(self.tasks)

        if not self.task_period_s:
            self.task_period_s = [0.0] * n

        if not self.batch_size:
            self.batch_size = [0] * n

        # Streams from inference threads
        if not self.responsetime:
            self.responsetime = [[] for _ in range(n)]
        if not self.inferencelatency:
            self.inferencelatency = [[] for _ in range(n)]

        # Controller outputs
        if not self.rtr:
            self.rtr = [[] for _ in range(n)]
        if not self.deadlinemiss:
            self.deadlinemiss = [[] for _ in range(n)]
        if not self.num_sms:
            self.num_sms = [[] for _ in range(n)]
        if not self.tpc_masks:
            self.tpc_masks = [[] for _ in range(n)]

        # Optional intermediate
        if not self.intermediateresponsetime:
            self.intermediateresponsetime = [[] for _ in range(n)]
        if not self.intermediateinferencelatency:
            self.intermediateinferencelatency = [[] for _ in range(n)]
        if not self.intermediatertr:
            self.intermediatertr = [[] for _ in range(n)]

        # Optional file buffers
        if not self.filertr:
            self.filertr = [[] for _ in range(n)]
        if not self.fileinferencelatency:
            self.fileinferencelatency = [[] for _ in range(n)]
        if not self.fileresponsetime:
            self.fileresponsetime = [[] for _ in range(n)]
        if not self.filesms:
            self.filesms = [[] for _ in range(n)]
        if not self.filedeadlinemiss:
            self.filedeadlinemiss = [[] for _ in range(n)]

        # Locks
        if not self.locks:
            self.locks = [threading.Lock() for _ in range(n)]