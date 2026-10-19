"""Two periodic ResNet-50 instances driven by the current project MPC.

Adapted from Srini's Pytorch_experiment/scheduler.py. Importing this module
does not import PyTorch, initialize CUDA, or load a GPU masking library.
"""
from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parents[1]
TASKS = 2


@dataclass
class RunConfig:
    task_period_s: float = .05
    control_period_s: float = 1.
    periods: int = 480
    setpoint: float = .80
    total_tpcs: int = 85
    model_name: str = 'resnet50'
    libsmctrl_path: str = ''

    def validate(self):
        if (self.task_period_s != .05 or self.control_period_s != 1.
                or self.model_name != 'resnet50' or not 0 < self.setpoint < 1
                or not isinstance(self.periods, int) or self.periods <= 0
                or not isinstance(self.total_tpcs, int) or self.total_tpcs != 85):
            raise ValueError('Expected ResNet-50, 50 ms tasks, 1 s control, and a valid budget/reference')


def batch_schedule(config):
    # Keep workload changes fixed in wall time when comparing control periods.
    second = 1_000_000_000
    return [(200*second, 0, 40), (280*second, 1, 40)] + [
        (360*second, i, 20) for i in range(TASKS)]


def initial_sms(total_tpcs, count):
    # Same whole-TPC distribution as Scheduler/scheduler.cpp.
    return [2 * (total_tpcs // count + (i >= count - total_tpcs % count))
            for i in range(count)]


def run_one_experiment(config, controller, tasks, output, *, clock=time.monotonic_ns, wait_until=None,
                       provenance=None):
    config.validate()
    if len(tasks) != TASKS or [t.task_id for t in tasks] != list(range(TASKS)):
        raise ValueError('Exactly two ordered tasks are required')
    state, stop = tasks[0].params, tasks[0].stop_event
    if any(t.params is not state or t.stop_event is not stop for t in tasks):
        raise ValueError('Tasks must share the same parameters and stop event')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    errors = queue.Queue()
    threads = []
    windows = [[] for _ in tasks]
    events, pulses = [], []
    result = dict(status='failed', config=asdict(config), completed_control_periods=0,
                  epoch_ns=None, provenance=provenance or {})
    mean = initial_sms(config.total_tpcs, TASKS)
    reference = [config.setpoint] * TASKS
    failure = None

    def worker(task, epoch):
        try:
            task.run(epoch)
        except BaseException as exc:
            errors.put('Task {}: {}: {}'.format(task.task_id, type(exc).__name__, exc))
            stop.set()

    def wait(target):
        if wait_until is not None:
            wait_until(target)
        else:
            while not stop.is_set():
                remaining = (target - clock()) / 1e9
                if remaining <= 0:
                    break
                stop.wait(remaining)

    try:
        for task in tasks:
            task.prepare()
        result['devices'] = [getattr(t, 'device_info', {}) for t in tasks]
        controller.bind_streams([t.stream_handle for t in tasks], config.libsmctrl_path)
        controller.publish(mean, [0.] * TASKS, config.control_period_s)
        epoch = result['epoch_ns'] = clock()
        for task in tasks:
            thread = threading.Thread(target=worker, args=(task, epoch), name='inference-'+str(task.task_id))
            threads.append(thread)
            thread.start()
        schedule = batch_schedule(config)
        previous_sample_ns = [epoch] * TASKS
        for k in range(1, config.periods + 1):
            boundary = epoch + round(k * config.control_period_s * 1e9)
            wait(boundary)
            if stop.is_set():
                raise RuntimeError('Inference stopped before the final control boundary')
            rows = []
            for i in range(TASKS):
                row = state.take_window(i)
                sampled = clock()
                row.update(control_period=k, planned_end_ns=boundary,
                           window_start_ns=previous_sample_ns[i], window_end_ns=sampled,
                           mean_sms_command=mean[i])
                previous_sample_ns[i] = sampled
                windows[i].append(row)
                rows.append(row)
            result['completed_control_periods'] = k
            print(f'[MPC] Control period {k}: RRT={[row["rtr"] for row in rows]}, SM command={mean}', flush=True)
            if k == config.periods:
                break
            for relative, i, batch in schedule:
                if relative == boundary - epoch:
                    requested = state.set_batch(i, batch, clock)
                    events.append(dict(task_id=i, batch_size=batch, planned_ns=epoch+relative,
                                       requested_ns=requested))
            rrt = [row['rtr'] for row in rows]
            decision = controller.decide(mean, rrt, reference)
            mean = decision['mean_sms']
            for i, row in enumerate(rows):
                row['next_mean_sms_command'] = mean[i]
                row['predicted_rrt'] = (decision['predicted_rrt'][i]
                                        if decision['predicted_rrt'] is not None else None)
            controller.publish(mean, [0. if r is None else z-r for z, r in zip(reference, rrt)],
                               config.control_period_s)
    except BaseException as exc:
        failure = exc
    finally:
        stop.set()
        for thread in threads:
            thread.join()
        # Workers synchronize in-flight work before releasing their stream/model.
        try:
            controller.stop()
        except BaseException as exc:
            failure = failure or exc
        try:
            pulses = controller.actuation_log()
        except BaseException as exc:
            failure = failure or exc
        for task in tasks:
            try:
                task.close()
            except BaseException as exc:
                failure = failure or exc
        try:
            controller.close()
        except BaseException as exc:
            failure = failure or exc
    worker_errors = []
    while not errors.empty():
        worker_errors.append(errors.get_nowait())
    if worker_errors:
        failure = RuntimeError('; '.join(worker_errors))
    for event in events:
        event['first_enqueue_ns'] = next((job.enqueue_ns for job in state.jobs[event['task_id']]
            if job.batch_size == event['batch_size'] and job.enqueue_ns is not None
            and job.enqueue_ns >= event['requested_ns']), None)
    result.update(status='failed' if failure else 'complete',
                  error=str(failure) if failure else None, batch_events=events)
    for i in range(TASKS):
        # Include all drained completions, even those after the final sampling boundary.
        payload = dict(task_id=i, windows=windows[i], jobs=[asdict(s) for s in state.jobs[i]])
        (output/('task_{}.json'.format(i))).write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    (output/'actuation.json').write_text(json.dumps(pulses, indent=2, allow_nan=False)+'\n')
    (output/'run.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if failure:
        raise failure
    return result



def main():
    from controller import Controller
    from inferworkload import PeriodicInferenceTask
    from Parameters import Parameters, MODEL_B
    parser = argparse.ArgumentParser()
    parser.add_argument("duration", type=int, nargs="?", default=480)
    parser.add_argument("output", type=Path, nargs="?", default=BASE_DIR / "logs/sol_7")
    parser.add_argument("--weights", type=Path, default=Path(os.environ.get(
        "RESNET50_WEIGHTS", str(Path.home()/".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"))))
    args = parser.parse_args()
    config = RunConfig(periods=args.duration, libsmctrl_path=str(ROOT/"libsmctrl/libsmctrl.so"))
    config.validate()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"ResNet-50 weights missing: {args.weights}")
    state, stop = Parameters(TASKS, config.task_period_s), threading.Event()
    controller = Controller(BASE_DIR/"libdysm_pytorch.so", MODEL_B, 2*config.total_tpcs)
    tasks = [PeriodicInferenceTask(i, config, state, controller, stop,
                                  BASE_DIR/"n02099601_3004.jpg", args.weights)
             for i in range(TASKS)]
    provenance = dict(model_B=MODEL_B, model_units="RRT/SM",
                      model_source="Existing three-instance ML model, rows/columns 1 and 2",
                      original_instance_ids=[2, 3],
                      build=json.loads((BASE_DIR/"build.json").read_text()))
    run_one_experiment(config, controller, tasks, args.output, provenance=provenance)


if __name__ == "__main__":
    main()
