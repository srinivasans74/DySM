"""Periodic ResNet-50 worker, adapted from Srini's inferworkload.py.

Keeps the uploaded skimage resize/float32 input construction. CUDA and model
weights are loaded only by prepare(), not at import or construction time.
"""
from pathlib import Path
import time

from Parameters import JobSample


class PeriodicInferenceTask:
    def __init__(self, task_id, config, params, controller, stop_event, image_path, weights_path):
        self.task_id, self.config, self.params = task_id, config, params
        self.controller, self.stop_event = controller, stop_event
        self.image_path, self.weights_path = Path(image_path), Path(weights_path)
        self.stream = self.model = self.ev_start = self.ev_stop = None
        self.inputs = {}
        self.stream_handle = None
        self.device_info = {}

    def prepare(self):
        import ctypes as ct
        import numpy as np
        import torch
        import torchvision
        from skimage import io
        from skimage.transform import resize

        self.config.validate()
        if not self.weights_path.is_file() or not self.image_path.is_file():
            raise FileNotFoundError('Local ResNet-50 state_dict and input image are required')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; GPU preparation has not been completed')
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        library = ct.CDLL(str(Path(self.config.libsmctrl_path).resolve()))
        library.libsmctrl_set_stream_mask_ext  # Reject the old 64-bit-only library.
        query = library.libsmctrl_get_tpc_info_cuda
        query.argtypes = [ct.POINTER(ct.c_uint32), ct.c_int]
        count = ct.c_uint32()
        if query(ct.byref(count), device) or count.value != self.config.total_tpcs:
            raise ValueError('Actual GPU TPC count differs from the model/configuration')
        if props.multi_processor_count != 2 * self.config.total_tpcs:
            raise ValueError('Expected two SMs per TPC')
        self.device_info = dict(name=props.name, device=device, total_tpcs=count.value,
                                sms=props.multi_processor_count,
                                compute_capability=[props.major, props.minor],
                                torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                                cuda_runtime=torch.version.cuda)
        # Keep FP32 execution explicit; no TF32/autocast or implicit downloads.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.stream = torch.cuda.Stream(device=device)
        self.stream_handle = int(self.stream.cuda_stream)
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(torch.load(str(self.weights_path), map_location='cpu', weights_only=True))
        image = resize(io.imread(str(self.image_path)), (224, 224))
        image = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
        if image.shape != (1, 224, 224, 3):
            raise ValueError('Expected an RGB input image')
        with torch.no_grad(), torch.cuda.stream(self.stream):
            self.model = model.float().eval().cuda(device)
            for batch_size in (20, 40):
                batch = np.repeat(image, batch_size, axis=0).astype(np.float32)
                self.inputs[batch_size] = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().cuda(device)
                for _ in range(10):
                    self.model(self.inputs[batch_size])
        self.stream.synchronize()
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_stop = torch.cuda.Event(enable_timing=True)

    def _run_once_and_measure(self, release_ns):
        batch = self.params.get_batch(self.task_id)
        torch = self.torch
        # libsmctrl writes several mask words: a forward cannot overlap that write.
        with self.controller.submission(self.task_id), torch.no_grad(), torch.cuda.stream(self.stream):
            enqueue = time.monotonic_ns()
            self.ev_start.record(self.stream)
            output = self.model(self.inputs[batch])
            self.ev_stop.record(self.stream)
        # Synchronization is deliberately OUTSIDE the mask/enqueue lock.
        self.stream.synchronize()
        completion = time.monotonic_ns()
        elapsed = float(self.ev_start.elapsed_time(self.ev_stop))
        self.params.record_job(self.task_id, JobSample(release_ns, completion, elapsed, batch, enqueue))
        del output

    def run(self, epoch_ns):
        release = epoch_ns
        period = round(self.config.task_period_s * 1e9)
        while not self.stop_event.is_set():
            remaining = (release - time.monotonic_ns()) / 1e9
            if remaining > 0 and self.stop_event.wait(remaining):
                break
            if self.stop_event.is_set():
                break
            self._run_once_and_measure(release)
            # Retain overdue releases rather than silently dropping jobs.
            release += period

    def close(self):
        try:
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            self.model = self.ev_start = self.ev_stop = None
            self.inputs.clear()
            self.stream = None
