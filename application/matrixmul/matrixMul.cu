// matrixMul.cu — MATCHED to your “working” structure, BUT:
// A) Uses steady_clock + sleep_until like the working code (no clock_nanosleep)
// B) Keeps termination condition: while (gpu_running.load(...))
// C) Uses explicit cudaMemcpyAsync + cudaStreamSynchronize (Option B) — NOT mapped hostalloc
// D) No shared memory / no async thread / no files
//
// Metrics:
// - ET: kernel-only time via CUDA events
// - RT: (finish - actual_release) where actual_release is taken immediately after sleep_until returns
//       (this matches the behavior of your working code more closely; it cannot go negative)

#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>

#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"

extern std::atomic<bool> gpu_running;

using steady_clock = std::chrono::steady_clock;

// ============================================================
// Utility
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call);  \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA] " << #call << " failed: " \
                  << cudaGetErrorString(_e) << " (" << (int)_e << ")\n"; \
        std::terminate(); \
    } \
} while(0)

// ============================================================
// Kernel
// ============================================================
__global__ void matMulKernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

// ============================================================
// MatrixMul class
// ============================================================
class MatrixMul {
public:
    MatrixMul(int id, parameters* p_)
        : task_id(id), p(p_), N(1024),
          h_A(nullptr), h_B(nullptr), h_C(nullptr),
          d_A(nullptr), d_B(nullptr), d_C(nullptr),
          stream(nullptr), ev_start(nullptr), ev_stop(nullptr)
    {}

    void taskInit();
    void runContinuous();   // runs until gpu_running == false
    void taskFinish();

private:
    void kernelOnceMeasure(const steady_clock::time_point& actual_release_tp);

    int task_id;
    parameters* p;
    int N;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    cudaStream_t stream;
    cudaEvent_t  ev_start, ev_stop;
    dim3 block, grid;
};

// ============================================================
// Init
// ============================================================
void MatrixMul::taskInit()
{
    std::cout << "[MatrixMul] Init Task " << task_id << "\n";

    CUDA_CHECK(cudaStreamCreate(&stream));
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));

    const size_t elems = (size_t)N * (size_t)N;
    const size_t bytes = elems * sizeof(float);

    // Host buffers: pinned (good for memcpyAsync), NOT mapped
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_C, bytes, cudaHostAllocDefault));

    // Device buffers
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    for (size_t i = 0; i < elems; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    block = dim3(32, 32);
    grid  = dim3((N + block.x - 1) / block.x,
                 (N + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm-up: H2D + kernel + sync
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));
    matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "[MatrixMul] Task " << task_id << " initialized.\n";
}

// ============================================================
// One job (ET + RT) — “working code style”
//   RT is measured from ACTUAL release (after sleep_until returns), not scheduled timestamp.
//   This avoids negative RT and matches your working program’s measurement behavior.
// ============================================================
void MatrixMul::kernelOnceMeasure(const steady_clock::time_point& actual_release_tp)
{
    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    auto finish_tp = steady_clock::now();
    float rt_ms = std::chrono::duration<float, std::milli>(finish_tp - actual_release_tp).count();
    // Stream-ordered transfers
   // CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
   // CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));
   CUDA_CHECK(cudaEventRecord(ev_start, stream));
    matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    //CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    float et_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&et_ms, ev_start, ev_stop));
    std::lock_guard<std::mutex> lk(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = et_ms;
    p->responsetimes[task_id]  = et_ms+rt_ms;
}

// ============================================================
// Periodic loop — MATCHED structure:
// - steady_clock
// - sleep_until(next_release)
// - keep termination condition: while(gpu_running.load())
// - update next_release using current p->period[task_id] (in seconds) each iteration
// ============================================================
void MatrixMul::runContinuous()
{
    auto next_release = steady_clock::now();

    while (gpu_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_until(next_release);
        auto actual_release = steady_clock::now();
        libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));
        kernelOnceMeasure(actual_release);
        double ps = p->period[task_id];
        long long period_us = (long long)llround(ps * 1e6);
        next_release += std::chrono::microseconds(period_us);
    }
}

// ============================================================
// Cleanup
// ============================================================
void MatrixMul::taskFinish()
{
    std::cout << "[MatrixMul] Cleaning Task " << task_id << "\n";

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {

void* matrixMul_create(int task_id, parameters* p) {
    auto* mm = new MatrixMul(task_id, p);
    mm->taskInit();
    return mm;
}

void matrixMul_run(void* handle) {
    static_cast<MatrixMul*>(handle)->runContinuous();
}

void matrixMul_destroy(void* handle) {
    auto* mm = static_cast<MatrixMul*>(handle);
    mm->taskFinish();
    delete mm;
}

} // extern "C"