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

// All-pairs N-body simulation
// Updates particle positions and velocities using all-pairs gravitational interactions.
//
// Kernel source:
// https://github.com/NVIDIA/cuda-samples/blob/aeab82ff306b12ae980f40109d9b0152f77b1f12/cpp/5_Domain_Specific/nbody/bodysystemcuda.cu
// Task lifecycle, timing and mask protocol follow application/hist/hist.cu.
// Upstream all-pairs tiled acceleration and integration specialized to float. Fixed
// softeningSquared=0.001, single GPU, no OpenGL. Every release starts from the same saved initial
// positions and zero velocities. It is more work than the old 512-interaction approximation.
#include "nbody.h"
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
// CUDA Kernel — NBody
// ============================================================
namespace
{
namespace cg = cooperative_groups;
template <class T> struct SharedMemory {
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += 0.001f;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}
__device__ float3 computeBodyAccel(float4 bodyPos, float4 *positions, int numTiles,
                                   cg::thread_block cta)
{
    float4 *sharedPos = SharedMemory<float4>();

    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}
__global__ void integrateBodies(float4 *__restrict__ newPos, float4 *__restrict__ oldPos,
                                float4 *vel, unsigned int deviceOffset,
                                unsigned int deviceNumBodies, float deltaTime, float damping,
                                int numTiles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies) {
        return;
    }

    float4 position = oldPos[deviceOffset + index];

    float3 accel = computeBodyAccel(position, oldPos, numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in
    // bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    float4 velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index] = velocity;
}

} // namespace

// ============================================================
// NBody class
// ============================================================
class NBody
{
  public:
    NBody(int id, parameters *params) : task_id(id), p(params) {}
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
    // NUM_BODIES must be a multiple of THREADS: the tiled kernel loads full blocks.
    static constexpr int NUM_BODIES = 65536, THREADS = 256, TIMESTEPS = 4;

    // Saved initial positions, two alternating position buffers, and in-place velocities.
    float4 *h_positions = nullptr, *d_initial = nullptr, *d_positions = nullptr, *d_next = nullptr,
           *d_velocity = nullptr;
};

// ============================================================
// Initialization
// ============================================================
void NBody::taskInit()
{
    std::cout << "[NBody] Init Task " << task_id << "\n";
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
    const size_t bytes = NUM_BODIES * sizeof(float4);
    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_positions, bytes, cudaHostAllocDefault));
    // Reproducible input for this task ID, reused across controller comparisons.
    std::mt19937 gen(5489u + task_id);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // float4.xyz stores position; .w stores mass (total mass = 1).
    for (int i = 0; i < NUM_BODIES; ++i)
        h_positions[i] = make_float4(dist(gen), dist(gen), dist(gen), 1.0f / NUM_BODIES);
    CUDA_CHECK(cudaMalloc(&d_initial, bytes));
    CUDA_CHECK(cudaMalloc(&d_positions, bytes));
    CUDA_CHECK(cudaMalloc(&d_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_velocity, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_initial, h_positions, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  bodies=" << NUM_BODIES << " timesteps=" << TIMESTEPS << "\n";
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
void NBody::prepareJob()
{
    // Start each release from saved positions and zero velocity, without accumulated drift.
    CUDA_CHECK(cudaMemcpyAsync(d_positions, d_initial, NUM_BODIES * sizeof(float4),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_velocity, 0, NUM_BODIES * sizeof(float4), stream));
}

// ============================================================
// GPU job dispatch
// ============================================================
void NBody::launchJob()
{
    // Advance TIMESTEPS from the reset state; each step reads all particles.
    for (int i = 0; i < TIMESTEPS; ++i) {
        // Time step = 0.001; damping = 0.995. Position buffers exchange roles below.
        integrateBodies<<<NUM_BODIES / THREADS, THREADS, THREADS * sizeof(float4), stream>>>(
            d_next, d_positions, d_velocity, 0, NUM_BODIES, 0.001f, 0.995f, NUM_BODIES / THREADS);
        std::swap(d_positions, d_next);
    }
}

// ============================================================
// Single job dispatch with standalone-style timing
//   RT = preemption_delay + GPU_execution_time
// ============================================================
void NBody::kernelOnceMeasure()
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
void NBody::runContinuous()
{
    // Thread pinning
    cpu_set_t cs;
    CPU_ZERO(&cs);
    const int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
        perror("pthread_setaffinity_np");
    std::cout << "[NBody] Periodic loop started (Task " << task_id << ")\n";
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
    std::cout << "[NBody] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void NBody::taskFinish()
{
    std::cout << "[NBody] Cleaning Task " << task_id << "\n";
    // Wait for queued GPU work before freeing its buffers and stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    CUDA_CHECK(cudaFree(d_initial));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_velocity));
    CUDA_CHECK(cudaFreeHost(h_positions));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {
void *nbody_create(int task_id, parameters *p)
{
    auto *task = new NBody(task_id, p);
    task->taskInit();
    return task;
}
void nbody_run(void *handle)
{
    static_cast<NBody *>(handle)->runContinuous();
}
void nbody_destroy(void *handle)
{
    auto *task = static_cast<NBody *>(handle);
    task->taskFinish();
    delete task;
}
} // extern "C"
