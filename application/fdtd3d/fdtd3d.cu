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

// 3D finite differences
// Updates a 3D field with a radius-4 stencil and fixed halo values.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Upstream radius-4 kernel and halo layout. Coefficients are instance-owned instead of
// device-global stencil. Twenty ping-pong timesteps per release; both buffers have initialized
// halos, and every release resets to the saved input.
#include "fdtd3d.h"
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
// CUDA Kernel — FDTD3D
// ============================================================
namespace
{
namespace cg = cooperative_groups;
#define RADIUS 4
#define k_blockDimX 32
#define k_blockDimMaxY 16
__global__ void FiniteDifferencesKernel(float *output, const float *input, const int dimx,
                                        const int dimy, const int dimz, const float *coefficients)
{
    bool validr = true;
    bool validw = true;
    const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ltidx = threadIdx.x;
    const int ltidy = threadIdx.y;
    const int workx = blockDim.x;
    const int worky = blockDim.y;
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

    const int stride_y = dimx + 2 * RADIUS;
    const int stride_z = stride_y * (dimy + 2 * RADIUS);

    int inputIndex = 0;
    int outputIndex = 0;

    // Advance inputIndex to start of inner volume
    inputIndex += RADIUS * stride_y + RADIUS;

    // Advance inputIndex to target element
    inputIndex += gtidy * stride_y + gtidx;

    float infront[RADIUS];
    float behind[RADIUS];
    float current;

    const int tx = ltidx + RADIUS;
    const int ty = ltidy + RADIUS;

    // Check in bounds
    if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
        validr = false;

    if ((gtidx >= dimx) || (gtidy >= dimy))
        validw = false;

    // Preload the "infront" and "behind" data
    for (int i = RADIUS - 2; i >= 0; i--) {
        if (validr)
            behind[i] = input[inputIndex];

        inputIndex += stride_z;
    }

    if (validr)
        current = input[inputIndex];

    outputIndex = inputIndex;
    inputIndex += stride_z;

    for (int i = 0; i < RADIUS; i++) {
        if (validr)
            infront[i] = input[inputIndex];

        inputIndex += stride_z;
    }

// Step through the xy-planes
#pragma unroll 9

    for (int iz = 0; iz < dimz; iz++) {
        // Advance the slice (move the thread-front)
        for (int i = RADIUS - 1; i > 0; i--)
            behind[i] = behind[i - 1];

        behind[0] = current;
        current = infront[0];
#pragma unroll 4

        for (int i = 0; i < RADIUS - 1; i++)
            infront[i] = infront[i + 1];

        if (validr)
            infront[RADIUS - 1] = input[inputIndex];

        inputIndex += stride_z;
        outputIndex += stride_z;
        cg::sync(cta);

        // Note that for the work items on the boundary of the problem, the
        // supplied index when reading the halo (below) may wrap to the
        // previous/next row or even the previous/next xy-plane. This is
        // acceptable since a) we disable the output write for these work
        // items and b) there is at least one xy-plane before/after the
        // current plane, so the access will be within bounds.

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < RADIUS) {
            tile[ltidy][tx] = input[outputIndex - RADIUS * stride_y];
            tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
        }

        // Halo left & right
        if (ltidx < RADIUS) {
            tile[ty][ltidx] = input[outputIndex - RADIUS];
            tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
        }

        tile[ty][tx] = current;
        cg::sync(cta);

        // Compute the output value
        float value = coefficients[0] * current;
#pragma unroll 4

        for (int i = 1; i <= RADIUS; i++) {
            value += coefficients[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                                        tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (validw)
            output[outputIndex] = value;
    }
}

} // namespace

// ============================================================
// FDTD3D class
// ============================================================
class FDTD3D
{
  public:
    FDTD3D(int id, parameters *params) : task_id(id), p(params) {}
    void taskInit();
    void runContinuous();
    void taskFinish();

  private:
    void kernelOnceMeasure();
    void prepareJob();
    void launchJob(); // One job includes all configured repeats or time steps.

    // Shared scheduler state; each instance owns the CUDA resources below.
    int task_id;
    parameters *p;
    cudaStream_t stream = nullptr;
    cudaEvent_t st{}, sp{}; // Start/stop events for the complete GPU job.
    timespec preemption_ts{}; // Timestamp taken after the release wait, as in Histogram.

    // Work per release; repetitions/time steps are included in the measured GPU time.
    // DIM is the interior size; OUTER_DIM includes RADIUS halo cells on each face.
    static constexpr int DIM = 128, OUTER_DIM = DIM + 2 * RADIUS, TIMESTEPS = 20;
    static constexpr size_t COUNT = static_cast<size_t>(OUTER_DIM) * OUTER_DIM * OUTER_DIM;

    // Saved initial field, two alternating halo-padded fields, and per-instance coefficients.
    float *h_in = nullptr, *d_initial = nullptr, *d_in = nullptr, *d_out = nullptr,
          *d_coefficients = nullptr;
    // Center weight, then weights for the six axial neighbors at each distance 1..4.
    float h_coefficients[RADIUS + 1]{0.4f, 0.025f, 0.025f, 0.025f, 0.025f};
};

// ============================================================
// Initialization
// ============================================================
void FDTD3D::taskInit()
{
    std::cout << "[FDTD3D] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t bytes = COUNT * sizeof(float);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    // Zero all halo cells; randomize only the interior volume below.
    std::fill(h_in, h_in + COUNT, 0.0f);
    // Reproducible input for this task ID, reused across controller comparisons.
    std::mt19937 gen(5489u + task_id);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int z = RADIUS; z < RADIUS + DIM; ++z)
        for (int y = RADIUS; y < RADIUS + DIM; ++y)
            for (int x = RADIUS; x < RADIUS + DIM; ++x)
                h_in[(static_cast<size_t>(z) * OUTER_DIM + y) * OUTER_DIM + x] = dist(gen);
    CUDA_CHECK(cudaMalloc(&d_initial, bytes));
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_coefficients, sizeof(h_coefficients)));
    CUDA_CHECK(cudaMemcpyAsync(d_initial, h_in, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_coefficients, h_coefficients, sizeof(h_coefficients),
                               cudaMemcpyHostToDevice, stream));
    std::cout << "  inner_volume=" << DIM << "^3 radius=" << RADIUS << " timesteps=" << TIMESTEPS
              << "\n";
    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // Warm-up run
    prepareJob();
    launchJob();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================
// Per-job input reset - queued before the timing start event
// ============================================================
void FDTD3D::prepareJob()
{
    // Reset both buffers so halos remain initialized when input/output roles alternate.
    CUDA_CHECK(
        cudaMemcpyAsync(d_in, d_initial, COUNT * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_out, d_initial, COUNT * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

// ============================================================
// GPU job dispatch
// ============================================================
void FDTD3D::launchJob()
{
    // Advance TIMESTEPS, exchanging input/output roles after each stencil update.
    const dim3 block(k_blockDimX, 8), grid(DIM / k_blockDimX, DIM / 8);
    for (int i = 0; i < TIMESTEPS; ++i) {
        FiniteDifferencesKernel<<<grid, block, 0, stream>>>(d_out, d_in, DIM, DIM, DIM,
                                                            d_coefficients);
        std::swap(d_in, d_out);
    }
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void FDTD3D::kernelOnceMeasure()
{
    // Delay since the timestamp taken after the release wait (not nominal release lateness).
    long long pre_ns = ts_diff_ns(ts_now_raw(), preemption_ts);
    if (pre_ns < 0)
        pre_ns = 0;
    const float pre_ms = static_cast<float>(pre_ns * 1e-6);
    // Re-apply mask in case controller changed it
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    // Per-job input resets, like Histogram's bin reset, precede the timed work.
    prepareJob();
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
void FDTD3D::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[FDTD3D] Periodic loop started (Task " << task_id << ")\n";
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
    std::cout << "[FDTD3D] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void FDTD3D::taskFinish()
{
    std::cout << "[FDTD3D] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_initial));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_coefficients));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *fdtd3d_create(int task_id, parameters *p)
{
    auto *task = new FDTD3D(task_id, p);
    task->taskInit();
    return task;
}
void fdtd3d_run(void *handle)
{
    static_cast<FDTD3D *>(handle)->runContinuous();
}
void fdtd3d_destroy(void *handle)
{
    auto *task = static_cast<FDTD3D *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
