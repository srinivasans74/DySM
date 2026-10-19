/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Matrix transpose
// Transposes a row-major matrix using shared-memory tiles.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/6_Performance/transpose/transpose.cu
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Unchanged upstream transposeNoBankConflicts, including the complete tile loops. Matching 32x16
// block and 32x32 tile.
#include "transpose.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include "../../libsmctrl/libsmctrl.h"

// Controlled externally
extern std::atomic<bool> gpu_running;

// ============================================================
// Utility helpers
// ============================================================
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            std::cerr << "[CUDA] " << #call << ": " << cudaGetErrorString(e) << "\n";              \
            std::terminate();                                                                      \
        }                                                                                          \
    } while (0)

static inline timespec ts_now_raw()
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t;
}
static inline timespec ts_add(const timespec &base, long long ns)
{
    timespec t = base;
    t.tv_sec += ns / 1000000000LL;
    t.tv_nsec += ns % 1000000000LL;
    if (t.tv_nsec >= 1000000000L) {
        t.tv_sec++;
        t.tv_nsec -= 1000000000L;
    }
    return t;
}
static inline long long ts_diff_ns(const timespec &a, const timespec &b)
{
    return (a.tv_sec - b.tv_sec) * 1000000000LL + (a.tv_nsec - b.tv_nsec);
}

static inline void sleep_until_abs_hybrid_raw(const timespec &target)
{
    const long long guard_ns = 50'000; // 50 µs
    while (true) {
        timespec now = ts_now_raw();
        long long diff = ts_diff_ns(target, now);
        if (diff <= 0)
            break;
        if (diff > guard_ns) {
            // Sleep until slightly before target
            timespec pre = ts_add(target, -guard_ns);
            clock_nanosleep(CLOCK_MONOTONIC_RAW, TIMER_ABSTIME, &pre, nullptr);
        } else {
            // Busy-spin for the last few microseconds
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
#else
            std::this_thread::yield();
#endif
        }
    }
}

// ============================================================
// CUDA Kernel — Transpose
// ============================================================
namespace
{
namespace cg = cooperative_groups;
#define TILE_DIM 32
#define BLOCK_ROWS 16
__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

} // namespace

// ============================================================
// Transpose class
// ============================================================
class Transpose
{
  public:
    Transpose(int id, parameters *params) : task_id(id), p(params) {}
    void taskInit();
    void runContinuous();
    void taskFinish();

  private:
    void kernelOnceMeasure();
    void launchJob(); // One job includes all configured repeats or time steps.

    // Shared scheduler state; each instance owns the CUDA resources below.
    int task_id;
    parameters *p;
    cudaStream_t stream = nullptr;
    cudaEvent_t st{}, sp{}; // Start/stop events for the complete GPU job.
    timespec preemption_ts{}; // Timestamp taken after the release wait, as in Histogram.

    // Work per release; repetitions/time steps are included in the measured GPU time.
    // Width and height must be multiples of TILE_DIM for this launch.
    static constexpr int WIDTH = 2048, HEIGHT = 2048, REPEATS = 8;

    // Pinned input and separate device input/output matrices.
    float *h_in = nullptr, *d_in = nullptr, *d_out = nullptr;
};

// ============================================================
// Initialization
// ============================================================
void Transpose::taskInit()
{
    std::cout << "[Transpose] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t count = static_cast<size_t>(WIDTH) * HEIGHT, bytes = count * sizeof(float);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < count; ++i)
        h_in[i] = static_cast<float>(i);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  width=" << WIDTH << " height=" << HEIGHT << " repeats=" << REPEATS << "\n";
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // Warm-up run
    launchJob();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================
// GPU job dispatch
// ============================================================
void Transpose::launchJob()
{
    // Transpose the same input REPEATS times into a separate output matrix.
    const dim3 block(TILE_DIM, BLOCK_ROWS), grid(WIDTH / TILE_DIM, HEIGHT / TILE_DIM);
    for (int i = 0; i < REPEATS; ++i)
        transposeNoBankConflicts<<<grid, block, 0, stream>>>(d_out, d_in, WIDTH, HEIGHT);
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void Transpose::kernelOnceMeasure()
{
    // Delay since the timestamp taken after the release wait (not nominal release lateness).
    long long pre_ns = ts_diff_ns(ts_now_raw(), preemption_ts);
    if (pre_ns < 0)
        pre_ns = 0;
    const float pre_ms = static_cast<float>(pre_ns * 1e-6);
    // Re-apply mask in case controller changed it
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    // -------- GPU execution time --------
    CUDA_CHECK(cudaEventRecord(st, stream));
    launchJob();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(sp, stream));
    CUDA_CHECK(cudaEventSynchronize(sp));
    float exec_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&exec_ms, st, sp));
    // RT = pre_ms + exec_ms
    const float response_ms = pre_ms + exec_ms;
    // Publish milliseconds under the same task lock used by the monitor.
    std::lock_guard<std::mutex> lock(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = exec_ms; // pure GPU time
    p->responsetimes[task_id] = response_ms; // preemption + GPU time
}

// ============================================================
// Periodic loop (standalone-style schedule)
// ============================================================
void Transpose::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[Transpose] Periodic loop started (Task " << task_id << ")\n";
    timespec next_release = ts_now_raw();
    while (gpu_running.load()) {
        // Read the current period in seconds; the scheduler may update it.
        const double period_s = p->period[task_id];
        const long long period_ns = static_cast<long long>(std::llround(period_s * 1e9));
        sleep_until_abs_hybrid_raw(next_release);
        // mark the time just before we "launch" the kernel
        preemption_ts = ts_now_raw();
        // measure RT + ET like standalone
        kernelOnceMeasure();
        // Advance from the scheduled release, not from job completion.
        next_release = ts_add(next_release, period_ns);
    }
    std::cout << "[Transpose] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Transpose::taskFinish()
{
    std::cout << "[Transpose] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *transpose_create(int task_id, parameters *p)
{
    auto *task = new Transpose(task_id, p);
    task->taskInit();
    return task;
}
void transpose_run(void *handle)
{
    static_cast<Transpose *>(handle)->runContinuous();
}
void transpose_destroy(void *handle)
{
    auto *task = static_cast<Transpose *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
