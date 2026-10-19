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

// Batched exclusive prefix sum
// Computes an exclusive prefix sum independently for each array in the batch.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/2_Concepts_and_Techniques/scan/scan.cu
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Upstream exclusive-scan kernels and three-stage large-array dispatch retained. Eight independent
// arrays of 262144 elements; scratch buffer is instance-owned, not upstream static d_Buf.
#include "scan.h"
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
// CUDA Kernel — Scan
// ============================================================
namespace
{
namespace cg = cooperative_groups;
using uint = unsigned int;
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation
// and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size,
                                      cg::thread_block cta)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size,
                                      cg::thread_block cta)
{
    return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size,
                                       cg::thread_block cta)
{
    // Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    // Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size,
                                       cg::thread_block cta)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    uint4 idata4 = d_Src[pos];

    // Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

    // Write back
    d_Dst[pos] = odata4;
}

// Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src, uint N,
                                     uint arrayLength)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    // Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load top elements
    // Convert results of bottom-level scan back to inclusive
    uint idata = 0;

    if (pos < N)
        idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
                d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    // Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength, cta);

    // Avoid out-of-bound access
    if (pos < N) {
        d_Buf[pos] = odata;
    }
}

// Final step of large-array scan: combine basic inclusive scan with exclusive
// scan of top elements of input arrays
__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        buf = d_Buffer[blockIdx.x];
    }

    cg::sync(cta);

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

} // namespace

// ============================================================
// Scan class
// ============================================================
class Scan
{
  public:
    Scan(int id, parameters *params) : task_id(id), p(params) {}
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
    // Each array has 256 blocks of 1024 values; prefixes restart at each array.
    static constexpr unsigned int ARRAY_LENGTH = 256 * 1024;
    static constexpr unsigned int BATCH_SIZE = 8;
    static constexpr unsigned int NUM_ELEMS = ARRAY_LENGTH * BATCH_SIZE;
    static constexpr int REPEATS = 12;

    // Pinned input, device input/output, and one offset per bottom-level block.
    unsigned int *h_in = nullptr, *d_in = nullptr, *d_out = nullptr, *d_block_sums = nullptr;
};

// ============================================================
// Initialization
// ============================================================
void Scan::taskInit()
{
    std::cout << "[Scan] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t bytes = NUM_ELEMS * sizeof(unsigned int);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    // Reproducible input for this task ID, reused across controller comparisons.
    std::mt19937 gen(5489u + task_id);
    std::uniform_int_distribution<unsigned int> dist(0, 15);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = dist(gen);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(
        cudaMalloc(&d_block_sums, (NUM_ELEMS / (4 * THREADBLOCK_SIZE)) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  arrays=" << BATCH_SIZE << " array_length=" << ARRAY_LENGTH
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
void Scan::launchJob()
{
    // Scan each block, scan block totals within each array, then add block offsets.
    const unsigned int blocks = NUM_ELEMS / (4 * THREADBLOCK_SIZE);
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        // 1. Exclusive scan within each 1024-value block.
        scanExclusiveShared<<<blocks, THREADBLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<uint4 *>(d_out), reinterpret_cast<uint4 *>(d_in),
            4 * THREADBLOCK_SIZE);
        // 2. Exclusive scan of block totals, separately for each input array.
        scanExclusiveShared2<<<(blocks + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE, THREADBLOCK_SIZE,
                               0, stream>>>(d_block_sums, d_out, d_in, blocks,
                                            ARRAY_LENGTH / (4 * THREADBLOCK_SIZE));
        // 3. Add the preceding-block sum to every element of each block.
        uniformUpdate<<<blocks, THREADBLOCK_SIZE, 0, stream>>>(reinterpret_cast<uint4 *>(d_out),
                                                               d_block_sums);
    }
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void Scan::kernelOnceMeasure()
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
void Scan::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[Scan] Periodic loop started (Task " << task_id << ")\n";
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
    std::cout << "[Scan] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Scan::taskFinish()
{
    std::cout << "[Scan] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *scan_create(int task_id, parameters *p)
{
    auto *task = new Scan(task_id, p);
    task->taskInit();
    return task;
}
void scan_run(void *handle)
{
    static_cast<Scan *>(handle)->runContinuous();
}
void scan_destroy(void *handle)
{
    auto *task = static_cast<Scan *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
