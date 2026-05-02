// histogram.cu
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
#include <time.h>
#include <sys/mman.h>
#include <mutex>
#include <random>

#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"

// Controlled externally
extern std::atomic<bool> gpu_running;

// ============================================================
// Utility helpers
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call);  \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA] " << #call << " failed: " \
                  << cudaGetErrorString(_e) << " (" << (int)_e << ")\n"; \
        std::terminate(); \
    } \
} while(0)

static inline timespec ts_now_raw() {
    timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t;
}
static inline timespec ts_add(const timespec& base, long long ns) {
    timespec t = base;
    t.tv_sec  += ns / 1000000000LL;
    t.tv_nsec += ns % 1000000000LL;
    if (t.tv_nsec >= 1000000000L) { t.tv_sec++; t.tv_nsec -= 1000000000L; }
    return t;
}
static inline long long ts_diff_ns(const timespec& a, const timespec& b) {
    return (a.tv_sec - b.tv_sec) * 1000000000LL + (a.tv_nsec - b.tv_nsec);
}

static inline void sleep_until_abs_hybrid_raw(const timespec& target)
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
            __asm__ __volatile__ ("yield" ::: "memory");
#else
            std::this_thread::yield();
#endif
        }
    }
}

// ============================================================
// CUDA Kernel — Histogram
// ============================================================
__global__ void histogram_kernel(const unsigned int* __restrict__ input,
                                 unsigned int* __restrict__ bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins)
{
    // Repeat a few times to simulate heavier GPU load
    for (int iter = 0; iter < 60; ++iter) {
        unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        for (unsigned int i = tid; i < num_elements; i += stride) {
            unsigned int val = input[i];
            if (val < num_bins)
                atomicAdd(&bins[val], 1);
        }
    }
}

// ============================================================
// Histogram class
// ============================================================
class Histogram {
public:
    Histogram(int id, parameters* params)
        : task_id(id), p(params),
          NUM_ELEMS(1u << 20), // 1M elements
          NUM_BINS(4096),
          h_in(nullptr), h_bins(nullptr),
          d_in(nullptr), d_bins(nullptr),
          stream(nullptr)
    {}

    void taskInit();
    void runContinuous();
    void taskFinish();

private:
    void kernelOnceMeasure();          // standalone-style timing
    void pin_to_core(int core);
    void prepare_input();

    int task_id;
    parameters* p;
    static constexpr int JOBS   = 1;

    const unsigned int NUM_ELEMS;
    const unsigned int NUM_BINS;

    unsigned int* h_in;
    unsigned int* h_bins;
    unsigned int* d_in;
    unsigned int* d_bins;

    cudaStream_t stream;
    cudaEvent_t  st, sp;
    dim3 block, grid;

    timespec preemption_ts;            // like preemptionlaunch in standalone
};

// ============================================================
// Thread pinning
// ============================================================
void Histogram::pin_to_core(int core)
{
    if (core < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    std::cout << "[Histogram] Task " << task_id
              << " pinned to CPU " << core << "\n";
}

// ============================================================
// Input preparation
// ============================================================
void Histogram::prepare_input()
{
    // Fill input data with uniform distribution in [0, NUM_BINS)
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist(0U, NUM_BINS - 1U);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = dist(gen);
}

// ============================================================
// Initialization
// ============================================================
void Histogram::taskInit()
{
    std::cout << "[Histogram] Init Task " << task_id
              << " (" << NUM_ELEMS << " elems, " << NUM_BINS << " bins)\n";

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Apply the configured SM/TPC mask
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    // Host (pinned) + Device buffers
    CUDA_CHECK(cudaHostAlloc(&h_in,   NUM_ELEMS * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_bins, NUM_BINS  * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d_in,   NUM_ELEMS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS  * sizeof(unsigned int)));

    // Initialize inputs/outputs
    prepare_input();
    std::memset(h_bins, 0, NUM_BINS * sizeof(unsigned int));

    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, NUM_ELEMS * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Launch geometry
    const unsigned int BLOCK_SIZE = 512;
    block = dim3(BLOCK_SIZE);
    unsigned int grid_size = (NUM_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid  = dim3(grid_size);

    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // Warm-up run
    CUDA_CHECK(cudaMemsetAsync(d_bins, 0, NUM_BINS * sizeof(unsigned int), stream));
    histogram_kernel<<<grid, block, 0, stream>>>(d_in, d_bins, NUM_ELEMS, NUM_BINS);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[Histogram] Task " << task_id << " initialized.\n";
}

// ============================================================
// ============================================================
void Histogram::kernelOnceMeasure()
{
    // -------- preemption delay (like pre_ms in standalone code) --------
    timespec now = ts_now_raw();
    long long pre_ns = ts_diff_ns(now, preemption_ts);
    if (pre_ns < 0) pre_ns = 0;
    float pre_ms = static_cast<float>(pre_ns * 1e-6f);

    // Re-apply mask in case controller changed it
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    // Clear bins
    CUDA_CHECK(cudaMemsetAsync(d_bins, 0, NUM_BINS * sizeof(unsigned int), stream));

    // -------- GPU execution time --------
    CUDA_CHECK(cudaEventRecord(st, stream));
    histogram_kernel<<<grid, block, 0, stream>>>(d_in, d_bins, NUM_ELEMS, NUM_BINS);
    CUDA_CHECK(cudaEventRecord(sp, stream));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float exec_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&exec_ms, st, sp));
    exec_ms = exec_ms / JOBS;

    // Same semantic as your standalone: RT = pre_ms + exec_ms
    float response_ms = pre_ms + exec_ms;

    {
        std::lock_guard<std::mutex> lock(p->task_mutexes[task_id]);
        p->executiontimes[task_id] = exec_ms;     // pure GPU time
        p->responsetimes[task_id]  = response_ms; // preemption + GPU time
    }
}

// ============================================================
// Periodic loop (standalone-style schedule)
// ============================================================
void Histogram::runContinuous()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) {
        perror("pthread_setaffinity_np");
    } else {
        std::cout << "[Histogram] Task " << task_id
                  << " pinned to CPU " << core << "\n";
    }

    std::cout << "[Histogram] Periodic loop started (Task " << task_id << ")\n";

    timespec next_release = ts_now_raw();

    while (gpu_running.load()) {
        double period_s  = p->period[task_id];
        long long period_ns = static_cast<long long>(std::llround(period_s * 1e9));

        sleep_until_abs_hybrid_raw(next_release);
        preemption_ts = ts_now_raw();
        kernelOnceMeasure();

        next_release = ts_add(next_release, period_ns);
    }

    std::cout << "[Histogram] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Histogram::taskFinish()
{
    std::cout << "[Histogram] Cleaning Task " << task_id << std::endl;
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    if (h_in)   CUDA_CHECK(cudaFreeHost(h_in));
    if (h_bins) CUDA_CHECK(cudaFreeHost(h_bins));
    if (d_in)   CUDA_CHECK(cudaFree(d_in));
    if (d_bins) CUDA_CHECK(cudaFree(d_bins));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {

void* hist_create(int task_id, parameters* p) {
    auto* h = new Histogram(task_id, p);
    h->taskInit();
    return h;
}
void  hist_run(void* handle) {
    static_cast<Histogram*>(handle)->runContinuous();
}
void  hist_destroy(void* handle) {
    auto* h = static_cast<Histogram*>(handle);
    h->taskFinish();
    delete h;
}

} // extern "C"