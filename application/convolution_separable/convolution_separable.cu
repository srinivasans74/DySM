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

// Separable 2D convolution
// Filters an image along rows, then columns, using a radius-8 kernel.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Upstream row and column kernels. Only coefficient access changes from device-global c_Kernel to
// an instance-owned buffer parameter, so simultaneous instances do not overwrite coefficients.
#include "convolution_separable.h"
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
// CUDA Kernel — ConvolutionSeparable
// ============================================================
namespace
{
namespace cg = cooperative_groups;
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1
__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch,
                                      const float *kernel)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[ROWS_BLOCKDIM_Y]
                           [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    // Offset to the left halo edge
    const int baseX =
        (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

// Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

// Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
         i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    // Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += kernel[KERNEL_RADIUS - j] *
                   s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW, int imageH,
                                         int pitch, const float *kernel)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float
        s_Data[COLUMNS_BLOCKDIM_X]
              [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY =
        (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

// Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

// Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
         i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    // Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += kernel[KERNEL_RADIUS - j] *
                   s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

} // namespace

// ============================================================
// ConvolutionSeparable class
// ============================================================
class ConvolutionSeparable
{
  public:
    ConvolutionSeparable(int id, parameters *params) : task_id(id), p(params) {}
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
    // Image dimensions must match the row/column tile coverage below.
    static constexpr int WIDTH = 2048, HEIGHT = 1024, REPEATS = 12;

    // Pinned image; device input, row intermediate, output and filter coefficients.
    float *h_in = nullptr, *d_in = nullptr, *d_rows = nullptr, *d_out = nullptr,
          *d_kernel = nullptr;
    float h_kernel[KERNEL_LENGTH]{};
};

// ============================================================
// Initialization
// ============================================================
void ConvolutionSeparable::taskInit()
{
    std::cout << "[ConvolutionSeparable] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t count = static_cast<size_t>(WIDTH) * HEIGHT, bytes = count * sizeof(float);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    // Reproducible input for this task ID, reused across controller comparisons.
    std::mt19937 gen(5489u + task_id);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i)
        h_in[i] = dist(gen);
    // Normalized 17-tap filter; coefficients belong to this task instance.
    for (float &value : h_kernel)
        value = 1.0f / KERNEL_LENGTH;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_rows, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, sizeof(h_kernel)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice, stream));
    std::cout << "  width=" << WIDTH << " height=" << HEIGHT << " radius=" << KERNEL_RADIUS
              << " repeats=" << REPEATS << "\n";
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
void ConvolutionSeparable::launchJob()
{
    // Each repetition filters d_in -> d_rows -> d_out on this stream.
    const dim3 row_block(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    const dim3 row_grid(WIDTH / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), HEIGHT / ROWS_BLOCKDIM_Y);
    const dim3 col_block(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    const dim3 col_grid(WIDTH / COLUMNS_BLOCKDIM_X,
                        HEIGHT / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    for (int i = 0; i < REPEATS; ++i) {
        // Horizontal pass writes the intermediate image.
        convolutionRowsKernel<<<row_grid, row_block, 0, stream>>>(d_rows, d_in, WIDTH, HEIGHT,
                                                                  WIDTH, d_kernel);
        // Vertical pass consumes that intermediate; stream ordering preserves the dependency.
        convolutionColumnsKernel<<<col_grid, col_block, 0, stream>>>(d_out, d_rows, WIDTH, HEIGHT,
                                                                     WIDTH, d_kernel);
    }
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void ConvolutionSeparable::kernelOnceMeasure()
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
void ConvolutionSeparable::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[ConvolutionSeparable] Periodic loop started (Task " << task_id << ")\n";
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
    std::cout << "[ConvolutionSeparable] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void ConvolutionSeparable::taskFinish()
{
    std::cout << "[ConvolutionSeparable] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *convolution_create(int task_id, parameters *p)
{
    auto *task = new ConvolutionSeparable(task_id, p);
    task->taskInit();
    return task;
}
void convolution_run(void *handle)
{
    static_cast<ConvolutionSeparable *>(handle)->runContinuous();
}
void convolution_destroy(void *handle)
{
    auto *task = static_cast<ConvolutionSeparable *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
