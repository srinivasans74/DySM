// stencil3d.cu
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
#include <vector>

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
            // Original x86 pause
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            // FIX: Use ARM's equivalent to PAUSE/YIELD
            __asm__ __volatile__ ("yield" ::: "memory");
#else
            // Fallback for other/unknown architectures
            std::this_thread::yield();
#endif
        }
    }
}











// ============================================================
// CUDA Kernel — 3D Stencil
// ============================================================
#define NX 256
#define NY 256
#define NZ 128
#define BSIZE 16
#define XTILE 16
#define HEAVY 4

__global__ void stencil3d_kernel(const float * __restrict__ psi,
                                 float * __restrict__ npsi,
                                 const float * __restrict__ sigmaX,
                                 const float * __restrict__ sigmaY,
                                 const float * __restrict__ sigmaZ,
                                 int nx, int ny, int nz)
{
    __shared__ float tile[4][BSIZE][BSIZE];

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int ox = XTILE * blockIdx.x;
    int oy = (BSIZE - 2) * blockIdx.y;
    int oz = (BSIZE - 2) * blockIdx.z;

    int maxX = nx - 1 - ox;
    int maxY = ny - 1 - oy;
    int maxZ = nz - 1 - oz;
    int nLast_x = (XTILE + 1 < maxX ? XTILE + 1 : maxX);
    int nLast_y = (BSIZE - 1 < maxY ? BSIZE - 1 : maxY);
    int nLast_z = (BSIZE - 1 < maxZ ? BSIZE - 1 : maxZ);
    if (tj > nLast_y || tk > nLast_z) return;

    auto idx = [=] __device__(int x, int y, int z) {
        return (size_t)x * ny * nz + (size_t)y * nz + (size_t)z;
    };

    tile[1][tj][tk] = psi[idx(ox + 0, oy + tj, oz + tk)];
    tile[2][tj][tk] = psi[idx(ox + 1, oy + tj, oz + tk)];
    __syncthreads();

    int pi = 0, ci = 1, ni = 2, ti;
    for (int rep = 0; rep < HEAVY; rep++) {
        for (int i = 1; i < nLast_x; ++i) {
            tile[ni][tj][tk] = psi[idx(ox + i + 1, oy + tj, oz + tk)];
            __syncthreads();

            float c = tile[ci][tj][tk];
            float l = tile[pi][tj][tk > 0 ? tk - 1 : 0];
            float r = tile[pi][tj][tk < nLast_z ? tk + 1 : nLast_z];
            float sum = c + l + r
                      + sigmaX[idx(ox + i, oy + tj, oz + tk)]
                      + sigmaY[idx(ox + i, oy + tj, oz + tk)]
                      + sigmaZ[idx(ox + i, oy + tj, oz + tk)];

            npsi[idx(ox + i, oy + tj, oz + tk)] = sum;

            __syncthreads();
            ti = pi; pi = ci; ci = ni; ni = ti;
        }
    }
}

// ============================================================
// Stencil3D Class
// ============================================================
class Stencil3D {
public:
    Stencil3D(int id, parameters* params)
        : task_id(id), p(params),
          NX_(NX), NY_(NY), NZ_(NZ),
          d_psi(nullptr), d_npsi(nullptr),
          d_sigX(nullptr), d_sigY(nullptr), d_sigZ(nullptr),
          stream(nullptr)
    {}

    void taskInit();
    void runContinuous();
    void taskFinish();

private:
    void kernelOnceMeasure(const timespec& release_ts, const timespec& launch_ts);
    void setup_rt_thread();
    void pin_to_core(int core);

    int task_id;
    parameters* p;

    const int NX_, NY_, NZ_;
  

    float *d_psi, *d_npsi;
    float *d_sigX, *d_sigY, *d_sigZ;

    cudaStream_t stream;
    cudaEvent_t  st, sp;
    dim3 gridDim, blockDim;
};

// ============================================================
// Real-time helpers
// ============================================================
void Stencil3D::setup_rt_thread()
{
    mlockall(MCL_CURRENT | MCL_FUTURE);
    sched_param sp{}; sp.sched_priority = 95;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
}
void Stencil3D::pin_to_core(int core)
{
    if (core < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// ============================================================
// Initialization
// ============================================================
void Stencil3D::taskInit()
{
    std::cout << "[Stencil3D] Init Task " << task_id << " (" << NX_ << "x" << NY_ << "x" << NZ_ << ")\n";

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreate(&stream));

    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));

    size_t vol = (size_t)NX_ * NY_ * NZ_;
    size_t bytes = vol * sizeof(float);

    std::vector<float> h_psi(vol, 1.0f);
    std::vector<float> h_sigX(vol, 0.1f);
    std::vector<float> h_sigY(vol, 0.2f);
    std::vector<float> h_sigZ(vol, 0.3f);

    CUDA_CHECK(cudaMalloc(&d_psi, bytes));
    CUDA_CHECK(cudaMalloc(&d_npsi, bytes));
    CUDA_CHECK(cudaMalloc(&d_sigX, bytes));
    CUDA_CHECK(cudaMalloc(&d_sigY, bytes));
    CUDA_CHECK(cudaMalloc(&d_sigZ, bytes));

    CUDA_CHECK(cudaMemcpyAsync(d_psi, h_psi.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sigX, h_sigX.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sigY, h_sigY.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sigZ, h_sigZ.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int bdimx = (NX_ - 2) / XTILE + ((NX_ - 2) % XTILE != 0);
    int bdimy = (NY_ - 2) / (BSIZE - 2) + ((NY_ - 2) % (BSIZE - 2) != 0);
    int bdimz = (NZ_ - 2) / (BSIZE - 2) + ((NZ_ - 2) % (BSIZE - 2) != 0);
    gridDim  = dim3(bdimx, bdimy, bdimz);
    blockDim = dim3(BSIZE, BSIZE);

    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // Warm-up
    stencil3d_kernel<<<gridDim, blockDim, 0, stream>>>(d_psi, d_npsi, d_sigX, d_sigY, d_sigZ, NX_, NY_, NZ_);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[Stencil3D] Task " << task_id << " initialized.\n";
}

// ============================================================
// One execution period
// ============================================================
void Stencil3D::kernelOnceMeasure(const timespec& release_ts, const timespec& launch_ts)
{
    long long pre_ns = ts_diff_ns(launch_ts, release_ts);
    if (pre_ns < 0) pre_ns = 0;
    float pre_ms = static_cast<float>(pre_ns * 1e-6f);

    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));

    CUDA_CHECK(cudaEventRecord(st, stream));
    stencil3d_kernel<<<gridDim, blockDim, 0, stream>>>(d_psi, d_npsi, d_sigX, d_sigY, d_sigZ, NX_, NY_, NZ_);
    CUDA_CHECK(cudaEventRecord(sp, stream));
    CUDA_CHECK(cudaEventSynchronize(sp));

    float exec_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&exec_ms, st, sp));

    constexpr float EPS_MS = 0.020f;
    float pre_disp     = (pre_ms < EPS_MS) ? 0.0f : pre_ms;
    float response_ms  = exec_ms + pre_disp;

    {
        std::lock_guard<std::mutex> lock(p->task_mutexes[task_id]);
        p->executiontimes[task_id] = exec_ms;
        p->responsetimes[task_id]  = response_ms;
    }
}

// ============================================================
// Periodic loop
// ============================================================
void Stencil3D::runContinuous()
{
    //mlockall(MCL_CURRENT | MCL_FUTURE);
    cpu_set_t cs;
    CPU_ZERO(&cs);
    int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) {
        perror("pthread_setaffinity_np");
    } else {
        std::cout << "[Stencil] Task " << task_id
                  << " pinned to CPU " << core << "\n";
    }
    //sched_param sp_prio{};
    //sp_prio.sched_priority = 99;
    //pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp_prio);
    std::cout << "[Stencil3D] Periodic loop started (Task " << task_id << ")\n";


    timespec next_release = ts_now_raw();

    while (gpu_running.load()) {
        double period_s = p->period[task_id];
        sleep_until_abs_hybrid_raw(next_release);

        timespec release_ts = next_release;
        timespec launch_ts  = ts_now_raw();
        kernelOnceMeasure(release_ts, launch_ts);

        long long add_ns = static_cast<long long>(std::llround(period_s * 1e9));
        next_release = ts_add(next_release, add_ns);
    }

    std::cout << "[Stencil3D] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Stencil3D::taskFinish()
{
    std::cout << "[Stencil3D] Cleaning Task " << task_id << std::endl;
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    if (d_psi)  CUDA_CHECK(cudaFree(d_psi));
    if (d_npsi) CUDA_CHECK(cudaFree(d_npsi));
    if (d_sigX) CUDA_CHECK(cudaFree(d_sigX));
    if (d_sigY) CUDA_CHECK(cudaFree(d_sigY));
    if (d_sigZ) CUDA_CHECK(cudaFree(d_sigZ));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {

void* stencil_create(int task_id, parameters* p) {
    auto* s = new Stencil3D(task_id, p);
    s->taskInit();
    return s;
}
void stencil_run(void* handle) {
    static_cast<Stencil3D*>(handle)->runContinuous();
}
void stencil_destroy(void* handle) {
    auto* s = static_cast<Stencil3D*>(handle);
    s->taskFinish();
    delete s;
}

} // extern "C"
