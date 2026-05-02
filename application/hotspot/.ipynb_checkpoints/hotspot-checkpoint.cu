// hotspot.cu
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
#include <cstdio>

// Project headers
#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"

// Controlled externally
extern std::atomic<bool> gpu_running;

// ============================================================
// Utility helpers
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
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
// ============================================================
static inline void fatal(const char* s) {
    std::fprintf(stderr, "error: %s\n", s);
    std::terminate();
}
static void read_grid_file(float *vect, int rows, int cols, const char *file) {
    FILE *fp = std::fopen(file, "r");
    if (!fp) fatal("could not open input file");
    char buf[256];
    float val;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!std::fgets(buf, sizeof(buf), fp)) fatal("not enough lines in file");
            if (std::sscanf(buf, "%f", &val) != 1) fatal("invalid file format");
            vect[i * cols + j] = val;
        }
    }
    std::fclose(fp);
}

// ============================================================
// ============================================================
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void calculate_temp(
    int iteration, const float * __restrict__ power,
    const float * __restrict__ temp_src, float * __restrict__ temp_dst,
    int grid_cols, int grid_rows, int border_cols, int border_rows,
    float Cap, float Rx, float Ry, float Rz, float step)
{
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

    const float step_div_Cap = step / Cap;
    const float Rx_1 = 1.0f / Rx, Ry_1 = 1.0f / Ry, Rz_1 = 1.0f / Rz;
    const float amb_temp = 80.0f;

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const int small_block_rows = BLOCK_SIZE - iteration * 2;
    const int small_block_cols = BLOCK_SIZE - iteration * 2;

    const int blkY    = small_block_rows * by - border_rows;
    const int blkX    = small_block_cols * bx - border_cols;
    const int blkYmax = blkY + BLOCK_SIZE - 1;
    const int blkXmax = blkX + BLOCK_SIZE - 1;

    const int yidx = blkY + ty;
    const int xidx = blkX + tx;
    const int index = grid_cols * yidx + xidx;

    if (yidx >= 0 && yidx < grid_rows && xidx >= 0 && xidx < grid_cols) {
        temp_on_cuda[ty][tx]  = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }
    __syncthreads();

    const int validYmin = (blkY < 0) ? -blkY : 0;
    const int validYmax = (blkYmax > grid_rows - 1)
                          ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1)
                          : BLOCK_SIZE - 1;
    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > grid_cols - 1)
                          ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1)
                          : BLOCK_SIZE - 1;

    for (int i = 0; i < iteration; ++i) {
        bool computed =
            (tx >= i + 1 && tx <= BLOCK_SIZE - i - 2 &&
             ty >= i + 1 && ty <= BLOCK_SIZE - i - 2 &&
             tx >= validXmin && tx <= validXmax &&
             ty >= validYmin && ty >= validYmax);

        if (computed) {
            float N = temp_on_cuda[ty - 1][tx];
            float S = temp_on_cuda[ty + 1][tx];
            float E = temp_on_cuda[ty][tx + 1];
            float W = temp_on_cuda[ty][tx - 1];

            temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (
                power_on_cuda[ty][tx] +
                (S + N - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
                (E + W - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
                (amb_temp - temp_on_cuda[ty][tx]) * Rz_1
            );
        }
        __syncthreads();

        if (i == iteration - 1) break;

        if (computed)
            temp_on_cuda[ty][tx] = temp_t[ty][tx];

        __syncthreads();
    }

    if (yidx >= 0 && yidx < grid_rows && xidx >= 0 && xidx < grid_cols) {
        temp_dst[index] = temp_t[ty][tx];
    }
}

// ============================================================
// Hotspot class
// ============================================================
class Hotspot {
public:
    Hotspot(int id, parameters* params)
        : task_id(id), p(params),
          GRID_ROWS(512), GRID_COLS(512),
          TOTAL_ITER(60), PYRAMID_HEIGHT(2),
          // FIX: Path adjusted for execution from the Scheduler directory 
          // to project root, then into application/data/hotspot.
          TEMP_FILE("../application/data/hotspot/temp_512"), 
          POWER_FILE("../application/data/hotspot/power_512"), 
          OUT_FILE("data/output_hotspot.out"),
          power_d(nullptr), temp_d0(nullptr), temp_d1(nullptr),
          stream(nullptr)
    {}

    void taskInit();
    void runContinuous();
    void taskFinish();

private:
    void kernelOnceMeasure(const timespec& release_ts, const timespec& launch_ts);
    void setup_rt_thread();
    void pin_to_core(int core);
    void load_inputs();
    void compute_constants_and_grid();

    // Config
    int task_id;
    parameters* p;

    // Problem sizes / params
    const int GRID_ROWS, GRID_COLS;
    const int TOTAL_ITER, PYRAMID_HEIGHT;

    // Files (can be adapted to parameters if needed)
    const char* TEMP_FILE;
    const char* POWER_FILE;
    const char* OUT_FILE;

    // Host buffers
    std::vector<float> h_temp;
    std::vector<float> h_power;
    std::vector<float> h_out;

    // Device buffers
    float* power_d;
    float* temp_d0;
    float* temp_d1;

    // Physical constants (from original code)
    const float t_chip       = 0.0005f;
    const float chip_h       = 0.016f;
    const float chip_w       = 0.016f;
    const float amb_temp     = 80.0f;
    const float SPEC_HEAT_SI = 1.75e6f;
    const float K_SI         = 100.0f;
    const float FACTOR_CHIP  = 0.5f;
    const float MAX_PD       = 3.0e6f;
    const int   EXPAND_RATE  = 2;

    // Derived constants
    float grid_h = 0.0f, grid_w = 0.0f;
    float Cap = 0.0f, Rx = 0.0f, Ry = 0.0f, Rz = 0.0f, step = 0.0f;
    int borderCols = 0, borderRows = 0;
    int smallBlockCol = 0, smallBlockRow = 0;
    int blockCols = 0, blockRows = 0;

    // Launch resources
    cudaStream_t stream;
    cudaEvent_t  st{}, sp{};
    dim3 dimBlock, dimGrid;
};

// ============================================================
// RT helpers
// ============================================================
void Hotspot::setup_rt_thread()
{
    mlockall(MCL_CURRENT | MCL_FUTURE);
    sched_param sp{}; sp.sched_priority = 95;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
}
void Hotspot::pin_to_core(int core)
{
    if (core < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// ============================================================
// Inputs / constants / grid
// ============================================================
void Hotspot::load_inputs()
{
    const int size = GRID_ROWS * GRID_COLS;
    h_temp.resize(size);
    h_power.resize(size);
    h_out.resize(size);

    read_grid_file(h_temp.data(),  GRID_ROWS, GRID_COLS, TEMP_FILE);
    read_grid_file(h_power.data(), GRID_ROWS, GRID_COLS, POWER_FILE);
}
void Hotspot::compute_constants_and_grid()
{
    grid_h = chip_h / GRID_ROWS;
    grid_w = chip_w / GRID_COLS;

    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_w * grid_h;
    Rx  = grid_w       / (2.0f * K_SI * t_chip * grid_h);
    Ry  = grid_h       / (2.0f * K_SI * t_chip * grid_w);
    Rz  = t_chip       / (K_SI * grid_h * grid_w);

    const float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    step = 0.001f / max_slope; // PRECISION = 0.001

    borderCols = (PYRAMID_HEIGHT) * EXPAND_RATE / 2;
    borderRows = (PYRAMID_HEIGHT) * EXPAND_RATE / 2;
    smallBlockCol = BLOCK_SIZE - (PYRAMID_HEIGHT) * EXPAND_RATE;
    smallBlockRow = BLOCK_SIZE - (PYRAMID_HEIGHT) * EXPAND_RATE;

    blockCols = GRID_COLS / smallBlockCol + ((GRID_COLS % smallBlockCol == 0) ? 0 : 1);
    blockRows = GRID_ROWS / smallBlockRow + ((GRID_ROWS % smallBlockRow == 0) ? 0 : 1);

    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid  = dim3(blockCols, blockRows);
}

// ============================================================
// Initialization
// ============================================================
void Hotspot::taskInit()
{
    std::cout << "[Hotspot] Init Task " << task_id
              << " (" << GRID_ROWS << "x" << GRID_COLS
              << ", iters=" << TOTAL_ITER << ", H=" << PYRAMID_HEIGHT << ")\n";

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    int loPrio=0, hiPrio=0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPrio, &hiPrio));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Apply configured SM/TPC mask for this stream
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    // Host input load
    load_inputs();

    // Device buffers
    const int size = GRID_ROWS * GRID_COLS;
    CUDA_CHECK(cudaMalloc(&power_d, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp_d0, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temp_d1, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(power_d, h_power.data(), size * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(temp_d0, h_temp.data(),  size * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    compute_constants_and_grid();

    CUDA_CHECK(cudaEventCreate(&st));
    CUDA_CHECK(cudaEventCreate(&sp));

    // Warm-up run to reduce first-iteration jitter
    {
        int src = 0, dst = 1;
        for (int t = 0; t < TOTAL_ITER; t += PYRAMID_HEIGHT) {
            int tmp = src; src = dst; dst = tmp;
            calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(
                min(PYRAMID_HEIGHT, TOTAL_ITER - t),
                power_d, (src ? temp_d1 : temp_d0), (dst ? temp_d1 : temp_d0),
                GRID_COLS, GRID_ROWS, borderCols, borderRows,
                Cap, Rx, Ry, Rz, step
            );
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::cout << "[Hotspot] Task " << task_id << " initialized.\n";
}

// ============================================================
// Single period: launch + measure
// ============================================================
void Hotspot::kernelOnceMeasure(const timespec& release_ts, const timespec& launch_ts)
{
    long long pre_ns = ts_diff_ns(launch_ts, release_ts);
    if (pre_ns < 0) pre_ns = 0;
    float pre_ms = static_cast<float>(pre_ns * 1e-6f);

    // Reapply mask if a controller updates it dynamically
    libsmctrl_set_stream_mask(stream, p->masks[task_id]);

    CUDA_CHECK(cudaEventRecord(st, stream));

    int src = 0, dst = 1;
    for (int t = 0; t < TOTAL_ITER; t += PYRAMID_HEIGHT) {
        int tmp = src; src = dst; dst = tmp;
        calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(
            min(PYRAMID_HEIGHT, TOTAL_ITER - t),
            power_d, (src ? temp_d1 : temp_d0), (dst ? temp_d1 : temp_d0),
            GRID_COLS, GRID_ROWS, borderCols, borderRows,
            Cap, Rx, Ry, Rz, step
        );
    }

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
void Hotspot::runContinuous()
{
    //mlockall(MCL_CURRENT | MCL_FUTURE);
    cpu_set_t cs;
    CPU_ZERO(&cs);
    int core = task_id % std::thread::hardware_concurrency();
    CPU_SET(core, &cs);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) {
        perror("pthread_setaffinity_np");
    } else {
        std::cout << "[Hotspot] Task " << task_id
                  << " pinned to CPU " << core << "\n";
    }
    //sched_param sp_prio{};
    //sp_prio.sched_priority = 99;
    //pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp_prio);
    std::cout << "[Hotspot] Periodic loop started (Task " << task_id << ")\n";

    //setup_rt_thread();
    //pin_to_core(task_id % std::thread::hardware_concurrency());

    timespec next_release = ts_now_raw();

    while (gpu_running.load()) {
        const double period_s = p->period[task_id];

        sleep_until_abs_hybrid_raw(next_release);

        timespec release_ts = next_release; // intended release
        timespec launch_ts  = ts_now_raw(); // actual timestamp before enqueue
        kernelOnceMeasure(release_ts, launch_ts);

        long long add_ns = static_cast<long long>(std::llround(period_s * 1e9));
        next_release = ts_add(next_release, add_ns);
    }

    std::cout << "[Hotspot] Task " << task_id << " stopped cleanly.\n";
}

// ============================================================
// Cleanup
// ============================================================
void Hotspot::taskFinish()
{
    std::cout << "[Hotspot] Cleaning Task " << task_id << std::endl;
    CUDA_CHECK(cudaEventDestroy(st));
    CUDA_CHECK(cudaEventDestroy(sp));
    if (power_d) CUDA_CHECK(cudaFree(power_d));
    if (temp_d0) CUDA_CHECK(cudaFree(temp_d0));
    if (temp_d1) CUDA_CHECK(cudaFree(temp_d1));
    if (stream)  CUDA_CHECK(cudaStreamDestroy(stream));

    // Uncomment if you need file output in the periodic version:
    // {
    //     const int size = GRID_ROWS * GRID_COLS;
    //     h_out.resize(size);
    //     CUDA_CHECK(cudaMemcpy(h_out.data(), temp_d0, size*sizeof(float), cudaMemcpyDeviceToHost));
    //     FILE *fp = std::fopen(OUT_FILE, "w");
    //     if (fp) {
    //         for (int i = 0; i < GRID_ROWS; ++i)
    //             for (int j = 0; j < GRID_COLS; ++j)
    //                 std::fprintf(fp, "%d\t%g\n", i * GRID_COLS + j, h_out[i * GRID_COLS + j]);
    //         std::fclose(fp);
    //     }
    // }
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {

void* hotspot_create(int task_id, parameters* p) {
    auto* hs = new Hotspot(task_id, p);
    hs->taskInit();
    return hs;
}
void  hotspot_run(void* handle) {
    static_cast<Hotspot*>(handle)->runContinuous();
}
void  hotspot_destroy(void* handle) {
    auto* hs = static_cast<Hotspot*>(handle);
    hs->taskFinish();
    delete hs;
}

} // extern "C"
