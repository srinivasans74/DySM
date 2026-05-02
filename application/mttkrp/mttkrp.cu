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
// MTTKRP Kernel (mode-0)
// X: [I,J,K], A: [J,R], B: [K,R], M: [I,R]
// M(i,r) = sum_{j,k} X(i,j,k) * A(j,r) * B(k,r)
// ============================================================
__global__ void mttkrp_mode0_kernel(const float* __restrict__ X,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ M,
                                   int I, int J, int K, int R)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < I && r < R) {
        float acc = 0.0f;

        // Flattening:
        // X index: ((i*J + j)*K + k)
        for (int j = 0; j < J; ++j) {
            float a_jr = A[j * R + r];
            #pragma unroll 1
            for (int k = 0; k < K; ++k) {
                float x_ijk = X[(i * J + j) * K + k];
                float b_kr  = B[k * R + r];
                acc += x_ijk * a_jr * b_kr;
            }
        }

        M[i * R + r] = acc;
    }
}

// ============================================================
// MTTKRP class
// ============================================================
class MTTKRP {
public:
    MTTKRP(int id, parameters* p_)
        : task_id(id), p(p_),
          I(64), J(64), K(64), R(32),
          h_X(nullptr), h_A(nullptr), h_B(nullptr), h_M(nullptr),
          d_X(nullptr), d_A(nullptr), d_B(nullptr), d_M(nullptr),
          stream(nullptr), ev_start(nullptr), ev_stop(nullptr)
    {}

    void taskInit();
    void runContinuous();   // runs until gpu_running == false
    void taskFinish();

private:
    void kernelOnceMeasure(const steady_clock::time_point& actual_release_tp);

    int task_id;
    parameters* p;

    // Tensor dims + rank
    int I, J, K, R;

    // Host (pinned) buffers
    float *h_X, *h_A, *h_B, *h_M;

    // Device buffers
    float *d_X, *d_A, *d_B, *d_M;

    cudaStream_t stream;
    cudaEvent_t  ev_start, ev_stop;
    dim3 block, grid;
};

// ============================================================
// Init
// ============================================================
void MTTKRP::taskInit()
{
    std::cout << "[MTTKRP] Init Task " << task_id << "\n";

    CUDA_CHECK(cudaStreamCreate(&stream));
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    const size_t x_elems = (size_t)I * (size_t)J * (size_t)K;
    const size_t a_elems = (size_t)J * (size_t)R;
    const size_t b_elems = (size_t)K * (size_t)R;
    const size_t m_elems = (size_t)I * (size_t)R;

    const size_t x_bytes = x_elems * sizeof(float);
    const size_t a_bytes = a_elems * sizeof(float);
    const size_t b_bytes = b_elems * sizeof(float);
    const size_t m_bytes = m_elems * sizeof(float);

    CUDA_CHECK(cudaHostAlloc((void**)&h_X, x_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, a_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, b_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_M, m_bytes, cudaHostAllocDefault));

    // Device buffers
    CUDA_CHECK(cudaMalloc((void**)&d_X, x_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_M, m_bytes));

    // Initialize inputs (simple deterministic values)
    for (size_t idx = 0; idx < x_elems; ++idx) h_X[idx] = 1.0f;
    for (size_t idx = 0; idx < a_elems; ++idx) h_A[idx] = 1.0f;
    for (size_t idx = 0; idx < b_elems; ++idx) h_B[idx] = 1.0f;
    for (size_t idx = 0; idx < m_elems; ++idx) h_M[idx] = 0.0f;

    // Threads cover (r, i)
    block = dim3(32, 8);
    grid  = dim3((R + block.x - 1) / block.x,
                 (I + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm-up: explicit H2D + kernel + D2H + sync (matches option C style)
    CUDA_CHECK(cudaMemcpyAsync(d_X, h_X, x_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, a_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, b_bytes, cudaMemcpyHostToDevice, stream));
    mttkrp_mode0_kernel<<<grid, block, 0, stream>>>(d_X, d_A, d_B, d_M, I, J, K, R);
    CUDA_CHECK(cudaMemcpyAsync(h_M, d_M, m_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[MTTKRP] Task " << task_id << " initialized.\n";
}


void MTTKRP::kernelOnceMeasure(const steady_clock::time_point& actual_release_tp)
{
    const size_t x_bytes = (size_t)I * (size_t)J * (size_t)K * sizeof(float);
    const size_t a_bytes = (size_t)J * (size_t)R * sizeof(float);
    const size_t b_bytes = (size_t)K * (size_t)R * sizeof(float);
    const size_t m_bytes = (size_t)I * (size_t)R * sizeof(float);

    auto finish_tp = steady_clock::now();
    float rt_ms = std::chrono::duration<float, std::milli>(finish_tp - actual_release_tp).count();

    CUDA_CHECK(cudaMemcpyAsync(d_X, h_X, x_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, a_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, b_bytes, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    mttkrp_mode0_kernel<<<grid, block, 0, stream>>>(d_X, d_A, d_B, d_M, I, J, K, R);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_M, d_M, m_bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float et_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&et_ms, ev_start, ev_stop));

    std::lock_guard<std::mutex> lk(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = et_ms;
    p->responsetimes[task_id]  = et_ms + rt_ms;
}


void MTTKRP::runContinuous()
{
    auto next_release = steady_clock::now();

    while (gpu_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_until(next_release);
        auto actual_release = steady_clock::now();

        libsmctrl_set_stream_mask(stream, p->masks[task_id]);
        kernelOnceMeasure(actual_release);

        double ps = p->period[task_id];
        long long period_us = (long long)llround(ps * 1e6);
        next_release += std::chrono::microseconds(period_us);
    }
}


void MTTKRP::taskFinish()
{
    std::cout << "[MTTKRP] Cleaning Task " << task_id << "\n";

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_M));

    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_M));

    CUDA_CHECK(cudaStreamDestroy(stream));
}


extern "C" {

void* mttkrp_create(int task_id, parameters* p) {
    auto* t = new MTTKRP(task_id, p);
    t->taskInit();
    return t;
}

void mttkrp_run(void* handle) {
    static_cast<MTTKRP*>(handle)->runContinuous();
}

void mttkrp_destroy(void* handle) {
    auto* t = static_cast<MTTKRP*>(handle);
    t->taskFinish();
    delete t;
}

} // extern "C"