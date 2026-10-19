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

// Black-Scholes option pricing
// Prices European call and put options from stock, strike and expiry arrays.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Upstream float2 kernel retained, including both call and put outputs. Valid floating-point input
// arrays replace byte-filled floats.
#include "blackscholes.h"
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
// CUDA Kernel — BlackScholes
// ============================================================
namespace
{
namespace cg = cooperative_groups;
__device__ inline float cndGPU(float d)
{
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float cnd =
        RSQRT2PI * __expf(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                           float S, // Stock price
                                           float X, // Option strike
                                           float T, // Option years
                                           float R, // Riskless rate
                                           float V  // Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    expRT = __expf(-R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128) __global__
    void BlackScholesGPU(float2 *__restrict d_CallResult, float2 *__restrict d_PutResult,
                         float2 *__restrict d_StockPrice, float2 *__restrict d_OptionStrike,
                         float2 *__restrict d_OptionYears, float Riskfree, float Volatility,
                         int optN)
{
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculating 2 options per thread to increase ILP (instruction level
    // parallelism)
    if (opt < (optN / 2)) {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(callResult1, putResult1, d_StockPrice[opt].x, d_OptionStrike[opt].x,
                            d_OptionYears[opt].x, Riskfree, Volatility);
        BlackScholesBodyGPU(callResult2, putResult2, d_StockPrice[opt].y, d_OptionStrike[opt].y,
                            d_OptionYears[opt].y, Riskfree, Volatility);
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
    }
}

} // namespace

// ============================================================
// BlackScholes class
// ============================================================
class BlackScholes
{
  public:
    BlackScholes(int id, parameters *params) : task_id(id), p(params) {}
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
    static constexpr int NUM_OPTIONS = 2 * 1024 * 1024;
    static constexpr int REPEATS = 20;

    // Pinned stock/strike/expiry inputs; device inputs and separate call/put outputs.
    float *h_stock = nullptr, *h_strike = nullptr, *h_years = nullptr;
    float *d_stock = nullptr, *d_strike = nullptr, *d_years = nullptr;
    float *d_call = nullptr, *d_put = nullptr;
};

// ============================================================
// Initialization
// ============================================================
void BlackScholes::taskInit()
{
    std::cout << "[BlackScholes] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t bytes = NUM_OPTIONS * sizeof(float);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_stock, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_strike, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_years, bytes, cudaHostAllocDefault));
    // Reproducible input for this task ID, reused across controller comparisons.
    std::mt19937 gen(5489u + task_id);
    std::uniform_real_distribution<float> stocks(5.0f, 30.0f), strikes(1.0f, 100.0f),
        years(0.25f, 10.0f);
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        h_stock[i] = stocks(gen);
        h_strike[i] = strikes(gen);
        h_years[i] = years(gen);
    }
    CUDA_CHECK(cudaMalloc(&d_stock, bytes));
    CUDA_CHECK(cudaMalloc(&d_strike, bytes));
    CUDA_CHECK(cudaMalloc(&d_years, bytes));
    CUDA_CHECK(cudaMalloc(&d_call, bytes));
    CUDA_CHECK(cudaMalloc(&d_put, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_stock, h_stock, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_strike, h_strike, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_years, h_years, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  options=" << NUM_OPTIONS << " repeats=" << REPEATS << "\n";
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
void BlackScholes::launchJob()
{
    // Price all options REPEATS times; every pass reads the same input.
    // float2 packs two options per thread; risk-free rate = 0.02, volatility = 0.30.
    for (int i = 0; i < REPEATS; ++i)
        BlackScholesGPU<<<(NUM_OPTIONS / 2 + 127) / 128, 128, 0, stream>>>(
            reinterpret_cast<float2 *>(d_call), reinterpret_cast<float2 *>(d_put),
            reinterpret_cast<float2 *>(d_stock), reinterpret_cast<float2 *>(d_strike),
            reinterpret_cast<float2 *>(d_years), 0.02f, 0.30f, NUM_OPTIONS);
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void BlackScholes::kernelOnceMeasure()
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
void BlackScholes::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[BlackScholes] Periodic loop started (Task " << task_id << ")\n";
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
    std::cout << "[BlackScholes] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void BlackScholes::taskFinish()
{
    std::cout << "[BlackScholes] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_stock));
    CUDA_CHECK(cudaFree(d_strike));
    CUDA_CHECK(cudaFree(d_years));
    CUDA_CHECK(cudaFree(d_call));
    CUDA_CHECK(cudaFree(d_put));
    CUDA_CHECK(cudaFreeHost(h_stock));
    CUDA_CHECK(cudaFreeHost(h_strike));
    CUDA_CHECK(cudaFreeHost(h_years));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *blackscholes_create(int task_id, parameters *p)
{
    auto *task = new BlackScholes(task_id, p);
    task->taskInit();
    return task;
}
void blackscholes_run(void *handle)
{
    static_cast<BlackScholes *>(handle)->runContinuous();
}
void blackscholes_destroy(void *handle)
{
    auto *task = static_cast<BlackScholes *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
