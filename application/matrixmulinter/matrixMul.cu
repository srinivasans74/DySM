// matrixMul.cu
// Adds an interrupt() that:
//   1) stops the current periodic execution safely
//   2) doubles N (N = N * 2)
//   3) re-initializes buffers/grid/events
//   4) restarts the periodic loop (continues runContinuous)
//
// Design:
// - interrupt() is called from another thread.
// - runContinuous() is the worker thread.
// - We use atomic flags + condition_variable to wake sleep_until early.
// - Restart happens INSIDE the worker thread (safe CUDA resource handling).

#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"

extern std::atomic<bool> gpu_running;

using steady_clock = std::chrono::steady_clock;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call);  \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA] " << #call << " failed: " \
                  << cudaGetErrorString(_e) << " (" << (int)_e << ")\n"; \
        std::terminate(); \
    } \
} while(0)

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

class MatrixMul {
public:
    MatrixMul(int id, parameters* p_)
        : task_id(id), p(p_), N(1024),
          h_A(nullptr), h_B(nullptr), h_C(nullptr),
          d_A(nullptr), d_B(nullptr), d_C(nullptr),
          stream(nullptr), ev_start(nullptr), ev_stop(nullptr),
          restart_requested(false), stop_requested(false)
    {}

    void taskInit();
    void runContinuous();   // worker loop
    void taskFinish();

    // Called from controller/other thread:
    // - requests current loop to restart with N = N*2
    void interrupt();

private:
    void kernelOnceMeasure(const steady_clock::time_point& actual_release_tp);

    // Restart machinery (worker thread performs restart safely)
    void request_restart();           // sets flags and wakes thread
    bool consume_restart_request();   // worker consumes flag
    bool consume_stop_request();      // worker consumes flag
    void do_restart();               // worker: sync, free, N*=2, init

    void freeResourcesNoThrow();      // internal: free if allocated
    void initResources();            // internal: allocate/init (like taskInit, but assumes stream exists)

    int task_id;
    parameters* p;
    int N;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    cudaStream_t stream;
    cudaEvent_t  ev_start, ev_stop;
    dim3 block, grid;

    // synchronization for interrupt/restart
    std::atomic<bool> restart_requested;
    std::atomic<bool> stop_requested;
    std::mutex cv_mtx;
    std::condition_variable cv;
};

// -------------------- init helpers --------------------
void MatrixMul::initResources()
{
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    const size_t elems = (size_t)N * (size_t)N;
    const size_t bytes = elems * sizeof(float);

    // Host buffers: pinned (good for memcpyAsync)
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

    // Warm-up (H2D optional if you keep data resident; leaving as you had)
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));
    matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void MatrixMul::freeResourcesNoThrow()
{
    // destroy events
    if (ev_start) { cudaEventDestroy(ev_start); ev_start = nullptr; }
    if (ev_stop)  { cudaEventDestroy(ev_stop);  ev_stop  = nullptr; }

    // device
    if (d_A) { cudaFree(d_A); d_A = nullptr; }
    if (d_B) { cudaFree(d_B); d_B = nullptr; }
    if (d_C) { cudaFree(d_C); d_C = nullptr; }

    // host
    if (h_A) { cudaFreeHost(h_A); h_A = nullptr; }
    if (h_B) { cudaFreeHost(h_B); h_B = nullptr; }
    if (h_C) { cudaFreeHost(h_C); h_C = nullptr; }
}

// -------------------- public init/finish --------------------
void MatrixMul::taskInit()
{
    std::cout << "[MatrixMul] Init Task " << task_id << " (N=" << N << ")\n";

    CUDA_CHECK(cudaStreamCreate(&stream));
    initResources();

    std::cout << "[MatrixMul] Task " << task_id << " initialized.\n";
}

void MatrixMul::taskFinish()
{
    std::cout << "[MatrixMul] Cleaning Task " << task_id << "\n";

    // signal stop and wake worker if sleeping
    stop_requested.store(true, std::memory_order_release);
    cv.notify_all();

    // NOTE: taskFinish() is called after runContinuous returns in your current design.
    // So we just free what exists:
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    freeResourcesNoThrow();

    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// -------------------- metrics --------------------
void MatrixMul::kernelOnceMeasure(const steady_clock::time_point& actual_release_tp)
{
    auto finish_tp = steady_clock::now();
    float rt_ms = std::chrono::duration<float, std::milli>(finish_tp - actual_release_tp).count();

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float et_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&et_ms, ev_start, ev_stop));

    std::lock_guard<std::mutex> lk(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = et_ms;
    p->responsetimes[task_id]  = et_ms + rt_ms;
}

// -------------------- restart control --------------------
void MatrixMul::request_restart()
{
    restart_requested.store(true, std::memory_order_release);
    cv.notify_all(); // wake sleep
}

bool MatrixMul::consume_restart_request()
{
    return restart_requested.exchange(false, std::memory_order_acq_rel);
}

bool MatrixMul::consume_stop_request()
{
    return stop_requested.load(std::memory_order_acquire);
}

void MatrixMul::do_restart()
{
    // Worker-thread only
    std::cout << "[MatrixMul] Task " << task_id << " restarting... (old N=" << N << ")\n";

    // Ensure GPU work is done before freeing
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free old buffers/events
    freeResourcesNoThrow();

    // Double N (guard if you want)
    int Ni=ceil(N/0.25);
    N = ceil(N * 1.25);
   // N=N+Ni;
    std::cout<<"******************************************** \n";
    std::cout<<"New N = <<\t "<<N;
    std::cout<<"******************************************** \n";
    std::cout<<"******************************************** \n";
    std::cout<<"******************************************** \n";
    // Re-init resources for new N
    initResources();

    std::cout << "[MatrixMul] Task " << task_id << " restart complete. (new N=" << N << ")\n";
}

// Called from controller thread:
// “terminate the task and make N=Nx2 restart runContinuous()”
void MatrixMul::interrupt()
{
    // We do NOT touch CUDA resources here (unsafe from other thread).
    // We just request restart; worker thread will do it safely.
    request_restart();
}

// -------------------- periodic loop --------------------
void MatrixMul::runContinuous()
{
    auto next_release = steady_clock::now();

    while (gpu_running.load(std::memory_order_relaxed) && !consume_stop_request()) {

        // Wait until next_release OR until interrupt/stop wakes us
        {
            std::unique_lock<std::mutex> lk(cv_mtx);
            cv.wait_until(lk, next_release, [&]{
                return consume_stop_request() || restart_requested.load(std::memory_order_acquire)
                       || !gpu_running.load(std::memory_order_relaxed);
            });
        }

        if (!gpu_running.load(std::memory_order_relaxed) || consume_stop_request())
            break;

        // If restart requested, do it and reset timing base
        if (consume_restart_request()) {
            do_restart();
            next_release = steady_clock::now(); // restart schedule from "now"
            continue;
        }

        // Normal period execution
        auto actual_release = steady_clock::now();
        libsmctrl_set_stream_mask(stream, p->masks[task_id]);
        kernelOnceMeasure(actual_release);

        double ps = p->period[task_id]; // seconds
        long long period_us = (long long)llround(ps * 1e6);
        next_release += std::chrono::microseconds(period_us);
    }

    std::cout << "[MatrixMul] Task " << task_id << " runContinuous() exit.\n";
}

// -------------------- C API --------------------
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

// If you want to trigger from C side:
void matrixMul_interrupt(void* handle) {
    static_cast<MatrixMul*>(handle)->interrupt();
}

} // extern "C"