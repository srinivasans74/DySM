"""Python ABI for current MPC/allocator/actuator code; no CUDA at import time."""
import ctypes as ct
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parents[1]


def _function(source, name):
    # Mask comments/strings to find the active function and balanced braces.
    tokens = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"')
    masked = tokens.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), source)
    match = re.search(r"^[ \t]*void\s+" + re.escape(name) + r"\s*\(", masked, re.M)
    if match is None:
        raise ValueError("Missing controller function: " + name)
    begin = masked.index("{", match.end())
    level = 1
    end = begin + 1
    while level:
        level += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[match.start():end]


def build_native(output=BASE_DIR / "libdysm_pytorch.so", root=ROOT):
    """Compile in a temporary directory; never change the shared scheduler build."""
    root, output = Path(root), Path(output)
    source_path = root / "Scheduler/controller.cpp"
    source = source_path.read_text()
    mpc = _function(source, "dysm_mpc")
    begin = mpc.index("        // 2) Previous MEAN command")
    end = mpc.index("        const auto& target_sms = command_sms;", begin)
    decision = mpc[begin:end]
    allocator = _function(source, "assign_overlap").replace(
        "void assign_overlap(", "void baseline_assign_overlap(", 1)
    actuator = "\n".join(_function(source, name) for name in (
        "actuation_outerloop_for_period", "start_new_actuation_period", "stop_actuation_thread"))
    bridge = BASE_DIR / "native_bridge.cpp"
    with tempfile.TemporaryDirectory(prefix="dysm68-build-") as folder:
        folder = Path(folder)
        for name, data in (("frozen_decision.inc", decision),
                           ("frozen_allocator.inc", allocator), ("frozen_actuator.inc", actuator)):
            (folder / name).write_text(data)
        library = folder / output.name
        command = ["g++", "-std=c++17", "-O0", "-g", "-shared", "-fPIC", "-pthread",
                   "-Wall", "-Wextra", "-I"+str(folder), str(bridge), "-o", str(library), "-ldl"]
        subprocess.run(command, check=True)
        output.write_bytes(library.read_bytes())
    receipt = dict(controller_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                   decision_sha256=hashlib.sha256(decision.encode()).hexdigest(),
                   library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   compiler_flags="-std=c++17 -O0 -g -shared -fPIC -pthread",
                   core="Current Scheduler/controller.cpp MPC, allocator and delta-sigma actuator")
    output.with_name("build.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return output


class Controller:
    def __init__(self, library, b_sm, budget_sms):
        self.n = len(b_sm)
        if (not self.n or any(len(row) != self.n for row in b_sm)
                or any(not math.isfinite(v) for row in b_sm for v in row)
                or budget_sms != int(budget_sms) or budget_sms % 2
                or not 2*self.n <= budget_sms <= 256):
            raise ValueError('Invalid model dimensions, values, or SM budget')
        self.budget_sms = int(budget_sms)
        self.library = Path(library).resolve()
        self.lib = ct.CDLL(str(self.library))
        self.handle = ct.c_void_p()
        self.lib.dysm_create.argtypes = [ct.c_int,ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_void_p)]
        self.lib.dysm_step.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_double),
                                     ct.POINTER(ct.c_double),ct.POINTER(ct.c_int),ct.POINTER(ct.c_double)]
        self.lib.dysm_destroy.argtypes = [ct.c_void_p]
        self.lib.dysm_destroy.restype = None
        self.lib.dysm_last_error.restype = ct.c_char_p
        self.lib.dysm_bind_streams.argtypes = [ct.c_void_p,ct.POINTER(ct.c_size_t),ct.c_char_p]
        self.lib.dysm_publish.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int),ct.POINTER(ct.c_double),ct.c_double]
        self.lib.dysm_begin_submit.argtypes = [ct.c_void_p,ct.c_int]
        self.lib.dysm_end_submit.argtypes = [ct.c_void_p,ct.c_int]
        self.lib.dysm_stop.argtypes = [ct.c_void_p]
        self.lib.dysm_copy_actuation_log.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.POINTER(ct.c_size_t)]
        flat = (ct.c_double*(self.n*self.n))(*(x for row in b_sm for x in row))
        self._check(self.lib.dysm_create(self.n,self.budget_sms,flat,ct.byref(self.handle)))

    def _check(self, status):
        if status:
            raise ValueError(self.lib.dysm_last_error().decode())

    def _vector(self, values, kind):
        if len(values) != self.n or any(not math.isfinite(v) for v in values):
            raise ValueError('Invalid controller vector')
        if kind is ct.c_int and any(v != int(v) for v in values):
            raise ValueError('SM commands must be integers')
        return (kind*self.n)(*values)

    def decide(self, mean_sms, rrt, target):
        if not self.handle:
            raise ValueError('Controller is closed')
        current = self._vector(mean_sms,ct.c_int)
        reference = self._vector(target,ct.c_double)
        if len(rrt) != self.n or any(v is not None and (not math.isfinite(v) or v <= 0) for v in rrt):
            raise ValueError('Invalid RRT samples')
        if any(v is None for v in rrt):
            if min(mean_sms)<2 or sum(mean_sms)>self.budget_sms or any(not 0<t<1 for t in target):
                raise ValueError('Invalid hold command')
            return dict(mean_sms=list(mean_sms),predicted_rrt=None)
        measured = self._vector(rrt,ct.c_double)
        out, prediction = (ct.c_int*self.n)(), (ct.c_double*self.n)()
        self._check(self.lib.dysm_step(self.handle,current,measured,reference,out,prediction))
        return dict(mean_sms=list(out),predicted_rrt=list(prediction))

    def bind_streams(self, handles, libsmctrl):
        if len(handles) != self.n or any(not isinstance(h,int) or h<=0 for h in handles):
            raise ValueError('Invalid CUDA stream handles')
        self._check(self.lib.dysm_bind_streams(self.handle,(ct.c_size_t*self.n)(*handles),
                                             str(Path(libsmctrl).resolve()).encode()))

    def publish(self, mean_sms, errors, duration_s):
        self._check(self.lib.dysm_publish(self.handle,self._vector(mean_sms,ct.c_int),
                                         self._vector(errors,ct.c_double),duration_s))

    @contextmanager
    def submission(self, task_id):
        self._check(self.lib.dysm_begin_submit(self.handle,task_id))
        try:
            yield
        finally:
            self._check(self.lib.dysm_end_submit(self.handle,task_id))

    def stop(self):
        if self.handle:
            self._check(self.lib.dysm_stop(self.handle))

    def actuation_log(self):
        size=ct.c_size_t()
        self._check(self.lib.dysm_copy_actuation_log(self.handle,None,0,ct.byref(size)))
        buffer=ct.create_string_buffer(size.value)
        self._check(self.lib.dysm_copy_actuation_log(self.handle,buffer,len(buffer),ct.byref(size)))
        return json.loads(buffer.value)

    def close(self):
        if self.handle:
            self.lib.dysm_destroy(self.handle)
            self.handle = ct.c_void_p()


if __name__ == "__main__":
    print(build_native())
