// projection.cu — MATCHED to MatrixMul timing & structure
// - steady_clock + sleep_until
// - RT measured from ACTUAL release
// - No clock_nanosleep / no async thread
// - Same gpu_running termination
// - CUDA event based ET
// - malloc + cudaMalloc (NO cudaHostAlloc / cudaHostGetDevicePointer)

#include "projection.h"
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <cmath>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <future>
#include <mutex>
#include <random>
#include <vector>

#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"
#include "../common/include/support.h"

extern std::atomic<bool> gpu_running;

using steady_clock = std::chrono::steady_clock;

// ============================================================
// Utilities
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
// Dummy CPU work
// ============================================================
static inline bool isPrime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i * i <= n; ++i)
        if (n % i == 0) return false;
    return true;
}

static inline std::vector<int> computePrimes(int limit) {
    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i)
        if (isPrime(i)) primes.push_back(i);
    return primes;
}

// ============================================================
// CUDA Kernel
// ============================================================
__global__ void projectPoints(float *d_input, float *d_output,
                              int num_points, float f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int loopIters = 2500;

    if (idx < num_points) {
        float x = d_input[3 * idx + 0];
        float y = d_input[3 * idx + 1];
        float z = d_input[3 * idx + 2];

        #pragma unroll 1
        for (int i = 0; i < loopIters; i++) {
            x = sinf(x) + cosf(y);
            y = cosf(y) + sinf(x);
            z = tanf(z) + cosf(x);
        }

        float scale = f / (f + z);
        d_output[2 * idx + 0] = x * scale;
        d_output[2 * idx + 1] = y * scale;
    }
}

// ============================================================
// Projection class
// ============================================================
class Projection {
public:
    Projection(int id, parameters* params)
        : task_id(id), p(params),
          num_points(16834),
          focal_length(1024),
          h_input(nullptr), h_output(nullptr),
          d_input(nullptr), d_output(nullptr),
          stream(nullptr), ev_start(nullptr), ev_stop(nullptr) {}

    void taskInit();
    void runContinuous();
    void taskFinish();

private:
    void kernelOnceMeasure(const steady_clock::time_point& actual_release);

    int task_id;
    parameters* p;
    int   num_points;
    float focal_length;

    float *h_input, *h_output;
    float *d_input, *d_output;

    cudaStream_t stream;
    cudaEvent_t  ev_start, ev_stop;
};

// ============================================================
// Initialization
// ============================================================
void Projection::taskInit()
{
    std::cout << "[Projection] Init Task " << task_id << "\n";

    CUDA_CHECK(cudaStreamCreate(&stream));
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    // malloc instead of cudaHostAlloc
    h_input  = (float*)malloc(sizeof(float) * 3 * num_points);
    h_output = (float*)malloc(sizeof(float) * 2 * num_points);

    CUDA_CHECK(cudaMalloc((void**)&d_input,
        sizeof(float) * 3 * num_points));
    CUDA_CHECK(cudaMalloc((void**)&d_output,
        sizeof(float) * 2 * num_points));

    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-100.f, 100.f);

    for (int i = 0; i < 3 * num_points; ++i)
        h_input[i] = dist(rng);
    for (int i = 0; i < 2 * num_points; ++i)
        h_output[i] = 0.0f;

    CUDA_CHECK(cudaMemcpyAsync(
        d_input, h_input,
        sizeof(float) * 3 * num_points,
        cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_output, h_output,
        sizeof(float) * 2 * num_points,
        cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int threads = 256;
    int blocks  = (num_points + threads - 1) / threads;

    projectPoints<<<blocks, threads, 0, stream>>>(
        d_input, d_output, num_points, focal_length);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[Projection] Task " << task_id << " initialized.\n";
}

// ============================================================
// One job — MATCHED RT/ET semantics
// ============================================================
void Projection::kernelOnceMeasure(
    const steady_clock::time_point& actual_release)
{
    auto finish = steady_clock::now();
    float rt_ms =
        std::chrono::duration<float, std::milli>(finish - actual_release).count();

    volatile auto primes = computePrimes(num_points);
    (void)primes;

    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    int threads = 256;
    int blocks  = (num_points + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    projectPoints<<<blocks, threads, 0, stream>>>(
        d_input, d_output, num_points, focal_length);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float et_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&et_ms, ev_start, ev_stop));

    std::lock_guard<std::mutex> lk(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = et_ms;
    p->responsetimes[task_id]  = et_ms + rt_ms;
}

// ============================================================
// Periodic loop — IDENTICAL to MatrixMul
// ============================================================
void Projection::runContinuous()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

    std::cout << "[Projection] Periodic loop started (Task "
              << task_id << ", CPU " << core << ")\n";

    auto next_release = steady_clock::now();

    while (gpu_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_until(next_release);
        auto actual_release = steady_clock::now();

        kernelOnceMeasure(actual_release);

        double ps = p->period[task_id];
        long long us = (long long)llround(ps * 1e6);
        next_release += std::chrono::microseconds(us);
    }

    std::cout << "[Projection] Task " << task_id << " stopped.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Projection::taskFinish()
{
    std::cout << "[Projection] Cleaning Task " << task_id << "\n";

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// C API
// ============================================================
extern "C" {

void* projection_create(int task_id, parameters* p)
{
    auto* proj = new Projection(task_id, p);
    proj->taskInit();
    return proj;
}

void projection_run(void* handle)
{
    static_cast<Projection*>(handle)->runContinuous();
}

void projection_destroy(void* handle)
{
    auto* proj = static_cast<Projection*>(handle);
    proj->taskFinish();
    delete proj;
}

} // extern "C"