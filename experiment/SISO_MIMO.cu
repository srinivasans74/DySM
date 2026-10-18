#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstdlib>

#include "libsmctrl.h"   // provides libsmctrl_set_stream_mask

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)


#ifndef STREAM_MASK_IS_DISABLE_MASK
#define STREAM_MASK_IS_DISABLE_MASK 1
#endif

// Limit flips to only the valid lower bits (avoid ~mask setting random high bits)
static uint64_t g_valid_tpc_bits = ~0ULL; // set in main if you know #TPCs

static inline uint64_t maybe_flip_for_disable_mask(uint64_t enabled_mask)
{
#if STREAM_MASK_IS_DISABLE_MASK
    // convert enabled-mask (1=enable) -> disable-mask (1=disable)
    return (~enabled_mask) & g_valid_tpc_bits;
#else
    // pass enabled mask directly (1=enable)
    return enabled_mask & g_valid_tpc_bits;
#endif
}

// ==========================================================
// Mask helpers: enable TPCs in [lo..hi] inclusive (enabled-mask semantics)
// ==========================================================
static uint64_t make_enabled_mask_range(int lo, int hi)
{
    if (hi < lo) return 0ULL;
    if (lo < 0) lo = 0;
    if (hi > 63) hi = 63;

    uint64_t m = 0ULL;
    for (int i = lo; i <= hi; ++i) m |= (1ULL << i);
    return m;
}

static void apply_stream_mask(cudaStream_t stream, uint64_t enabled_mask)
{
    uint64_t lib_mask = maybe_flip_for_disable_mask(enabled_mask);

    libsmctrl_set_stream_mask(stream, lib_mask);
   
}

static double avg_ms(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / static_cast<double>(v.size());
}

__global__ void mttkrp_mode0_dense(
    const float* __restrict__ X,   // [I * J * K]
    const float* __restrict__ B,   // [J * R]
    const float* __restrict__ C,   // [K * R]
    float* __restrict__ A,         // [I * R]
    int I, int J, int K, int R)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < I && r < R) {
        float acc = 0.0f;
        // Iterate over J and K dimensions to reduce
        for (int j = 0; j < J; ++j) {
            float bjr = B[j * R + r];
            int base_idx = (i * J + j) * K;
            for (int k = 0; k < K; ++k) {
                acc += X[base_idx + k] * bjr * C[k * R + r];
            }
        }
        A[i * R + r] = acc;
    }
}

// ==========================================================
// Periodic Task Runner
// Response time = CPU release -> GPU completion (event sync)
// ==========================================================
struct TaskConfig {
    int task_id;
    int iterations;
    int period_ms;
    cudaStream_t stream;

    int I, J, K, R;
    const float* d_X;
    const float* d_B;
    const float* d_C;
    float* d_A;
};

static void run_periodic_task(const TaskConfig cfg, std::vector<double>& out_rts_ms)
{
    out_rts_ms.clear();
    out_rts_ms.reserve(cfg.iterations);

    dim3 block(16, 16);
    dim3 grid((cfg.R + block.x - 1) / block.x,
              (cfg.I + block.y - 1) / block.y);

    cudaEvent_t done;
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

    auto next_release = Clock::now();

    for (int it = 0; it < cfg.iterations; ++it) {
        next_release += std::chrono::milliseconds(cfg.period_ms);
        std::this_thread::sleep_until(next_release);

        auto t0 = Clock::now();

        mttkrp_mode0_dense<<<grid, block, 0, cfg.stream>>>(
            cfg.d_X, cfg.d_B, cfg.d_C, cfg.d_A, cfg.I, cfg.J, cfg.K, cfg.R);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(done, cfg.stream));
        CUDA_CHECK(cudaEventSynchronize(done));

        auto t1 = Clock::now();
        out_rts_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    CUDA_CHECK(cudaEventDestroy(done));
}

// ==========================================================
// Experiment runner (two tasks, two streams, per-stream masks)
// ==========================================================
struct ExperimentSpec {
    std::string label;
    uint64_t task1_enabled_mask; // (1=enabled in our internal representation)
    uint64_t task2_enabled_mask;
    std::string json_path;
};

struct Result {
    double task1_avg_rt_ms;
    double task2_avg_rt_ms;
};

static void ensure_logs_dir()
{
#if defined(_WIN32)
    // ignore
#else
    std::system("mkdir -p logs");
#endif
}

static Result run_experiment(const ExperimentSpec& spec,
                             int I, int J, int K, int R,
                             int period_ms, int iters)
{
    const size_t x_elems = static_cast<size_t>(I) * J * K;
    const size_t b_elems = static_cast<size_t>(J) * R;
    const size_t c_elems = static_cast<size_t>(K) * R;
    const size_t a_elems = static_cast<size_t>(I) * R;

    // Host init
    std::vector<float> h_X(x_elems, 1.0f);
    std::vector<float> h_B(b_elems, 1.0f);
    std::vector<float> h_C(c_elems, 1.0f);

    // Device buffers
    float *d_X=nullptr, *d_B=nullptr, *d_C=nullptr, *d_A1=nullptr, *d_A2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, x_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, a_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), x_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_elems * sizeof(float), cudaMemcpyHostToDevice));

    // Streams (isolated)
    cudaStream_t s1=nullptr, s2=nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    // Apply per-stream masks (lib expects: 1=DISABLE)
    apply_stream_mask(s1, spec.task1_enabled_mask);
    apply_stream_mask(s2, spec.task2_enabled_mask);

    // Clear outputs
    CUDA_CHECK(cudaMemsetAsync(d_A1, 0, a_elems * sizeof(float), s1));
    CUDA_CHECK(cudaMemsetAsync(d_A2, 0, a_elems * sizeof(float), s2));
    CUDA_CHECK(cudaDeviceSynchronize());

    TaskConfig t1{
        .task_id=1, .iterations=iters, .period_ms=period_ms, .stream=s1,
        .I=I,.J=J,.K=K,.R=R, .d_X=d_X,.d_B=d_B,.d_C=d_C, .d_A=d_A1
    };
    TaskConfig t2{
        .task_id=2, .iterations=iters, .period_ms=period_ms, .stream=s2,
        .I=I,.J=J,.K=K,.R=R, .d_X=d_X,.d_B=d_B,.d_C=d_C, .d_A=d_A2
    };

    std::vector<double> rts1, rts2;

    std::thread th1(run_periodic_task, t1, std::ref(rts1));
    std::thread th2(run_periodic_task, t2, std::ref(rts2));
    th1.join();
    th2.join();

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A1));
    CUDA_CHECK(cudaFree(d_A2));

    return Result{avg_ms(rts1), avg_ms(rts2)};
}

static void save_json(const ExperimentSpec& spec,
                      const Result& res,
                      int I,int J,int K,int R,
                      int period_ms,int iters)
{
    std::ofstream f(spec.json_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open JSON: " << spec.json_path << "\n";
        return;
    }

    auto tohex16 = [](uint64_t v){
        std::ostringstream oss;
        oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
        return oss.str();
    };

    // Also log the actually-applied lib mask (disable-mask)
    uint64_t t1_applied = maybe_flip_for_disable_mask(spec.task1_enabled_mask);
    uint64_t t2_applied = maybe_flip_for_disable_mask(spec.task2_enabled_mask);

    f << "{\n";
    f << "  \"label\": \"" << spec.label << "\",\n";
    f << "  \"period_ms\": " << period_ms << ",\n";
    f << "  \"iterations\": " << iters << ",\n";
    f << "  \"tensor_dims\": {\"I\": " << I << ", \"J\": " << J << ", \"K\": " << K << "},\n";
    f << "  \"rank_R\": " << R << ",\n";
    f << "  \"mask_bit1_means\": \"DISABLE\",\n";
    f << "  \"valid_tpc_bits\": \"" << tohex16(g_valid_tpc_bits) << "\",\n";
    f << "  \"task1_enabled_mask\": \"" << tohex16(spec.task1_enabled_mask) << "\",\n";
    f << "  \"task2_enabled_mask\": \"" << tohex16(spec.task2_enabled_mask) << "\",\n";
    f << "  \"task1_applied_disable_mask\": \"" << tohex16(t1_applied) << "\",\n";
    f << "  \"task2_applied_disable_mask\": \"" << tohex16(t2_applied) << "\",\n";
    f << "  \"task1_avg_response_time_ms\": " << std::fixed << std::setprecision(6) << res.task1_avg_rt_ms << ",\n";
    f << "  \"task2_avg_response_time_ms\": " << std::fixed << std::setprecision(6) << res.task2_avg_rt_ms << "\n";
    f << "}\n";
}

// ==========================================================
// Main
// ==========================================================
int main()
{
    ensure_logs_dir();

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "[Two-Task Periodic MTTKRP + per-stream TPC Masking]\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Mask convention USED: bit=1 disables that TPC for the stream\n\n";

    const int I = 512, J = 512, K = 512;
    const int R = 16;

    const int PERIOD_MS = 30;
    const int ITERS = 10;

    // EXP1: Task1 [0..9], Task2 [6..15]  (overlap)
    ExperimentSpec exp1{
        .label = "EXP1_OVERLAP_T1_0_9__T2_6_15",
        .task1_enabled_mask = make_enabled_mask_range(0, 9),
        .task2_enabled_mask = make_enabled_mask_range(6, 15),
        .json_path = "logs/exp1_overlap_0_9__6_15.json"
    };

    // EXP2: Task1 [0..9], Task2 [11..20] (disjoint)
    ExperimentSpec exp2{
        .label = "EXP2_DISJOINT_T1_0_9__T2_11_20",
        .task1_enabled_mask = make_enabled_mask_range(0, 9),
        .task2_enabled_mask = make_enabled_mask_range(11, 20),
        .json_path = "logs/exp2_disjoint_0_9__11_20.json"
    };

    // ============================
    // NEW: run both experiments 10 times and print final averages
    // ============================
    const int RUNS = 10;

    double exp1_t1_sum = 0.0, exp1_t2_sum = 0.0;
    double exp2_t1_sum = 0.0, exp2_t2_sum = 0.0;

    for (int run = 1; run <= RUNS; ++run) {
        std::cout << "============================\n";
        std::cout << "RUN " << run << "/" << RUNS << "\n";
        std::cout << "============================\n";

        // ---- EXP1
        {
            ExperimentSpec e = exp1;
            e.json_path = "logs/exp1_overlap_0_9__6_15_run" + std::to_string(run) + ".json";

            std::cout << "=== Running " << e.label << " ===\n";
            auto r = run_experiment(e, I, J, K, R, PERIOD_MS, ITERS);
            save_json(e, r, I, J, K, R, PERIOD_MS, ITERS);

            std::cout << "Task1 avg RT: " << std::fixed << std::setprecision(3) << r.task1_avg_rt_ms
                      << " ms | Task2 avg RT: " << r.task2_avg_rt_ms << " ms\n";
            std::cout << "Saved: " << e.json_path << "\n\n";

            exp1_t1_sum += r.task1_avg_rt_ms;
            exp1_t2_sum += r.task2_avg_rt_ms;
        }

        // ---- EXP2
        {
            ExperimentSpec e = exp2;
            e.json_path = "logs/exp2_disjoint_0_9__11_20_run" + std::to_string(run) + ".json";

            std::cout << "=== Running " << e.label << " ===\n";
            auto r = run_experiment(e, I, J, K, R, PERIOD_MS, ITERS);
            save_json(e, r, I, J, K, R, PERIOD_MS, ITERS);

            std::cout << "Task1 avg RT: " << std::fixed << std::setprecision(3) << r.task1_avg_rt_ms
                      << " ms | Task2 avg RT: " << r.task2_avg_rt_ms << " ms\n";
            std::cout << "Saved: " << e.json_path << "\n\n";

            exp2_t1_sum += r.task1_avg_rt_ms;
            exp2_t2_sum += r.task2_avg_rt_ms;
        }
    }

    const double exp1_t1_final = exp1_t1_sum / RUNS;
    const double exp1_t2_final = exp1_t2_sum / RUNS;
    const double exp2_t1_final = exp2_t1_sum / RUNS;
    const double exp2_t2_final = exp2_t2_sum / RUNS;

    std::cout << "=====================================\n";
    std::cout << "FINAL AVERAGES ACROSS " << RUNS << " RUNS\n";
    std::cout << "=====================================\n";
    std::cout << exp1.label << "\n";
    std::cout << "  Task1 final avg RT: " << std::fixed << std::setprecision(3) << exp1_t1_final << " ms\n";
    std::cout << "  Task2 final avg RT: " << std::fixed << std::setprecision(3) << exp1_t2_final << " ms\n\n";

    std::cout << exp2.label << "\n";
    std::cout << "  Task1 final avg RT: " << std::fixed << std::setprecision(3) << exp2_t1_final << " ms\n";
    std::cout << "  Task2 final avg RT: " << std::fixed << std::setprecision(3) << exp2_t2_final << " ms\n";

    return 0;
}