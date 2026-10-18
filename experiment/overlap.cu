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
#include <algorithm>

#include "libsmctrl.h"   // libsmctrl_set_stream_mask

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

// ==========================================================
// Mask convention in YOUR setup: bit=1 DISABLES TPC
// ==========================================================
#ifndef STREAM_MASK_IS_DISABLE_MASK
#define STREAM_MASK_IS_DISABLE_MASK 1
#endif

static uint64_t g_valid_tpc_bits = ~0ULL; // set in main if NUM_TPCS known

static inline uint64_t maybe_flip_for_disable_mask(uint64_t enabled_mask)
{
#if STREAM_MASK_IS_DISABLE_MASK
    return (~enabled_mask) & g_valid_tpc_bits; // 1=disabled
#else
    return enabled_mask & g_valid_tpc_bits;    // 1=enabled
#endif
}

static uint64_t make_enabled_mask_range(int lo, int hi)
{
    if (hi < lo) return 0ULL;
    lo = std::max(lo, 0);
    hi = std::min(hi, 63);

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

// ==========================================================
// Dense MTTKRP kernel (mode-0), simple baseline
// X: I x J x K, B: J x R, C: K x R, A: I x R
// ==========================================================
__global__ void mttkrp_mode0_dense(
    const float* __restrict__ X,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ A,
    int I, int J, int K, int R)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < I && r < R) {
        float acc = 0.0f;
        for (int j = 0; j < J; ++j) {
            float bjr = B[j * R + r];
            int base = (i * J + j) * K;
            for (int k = 0; k < K; ++k) {
                acc += X[base + k] * bjr * C[k * R + r];
            }
        }
        A[i * R + r] = acc;
    }
}

// ==========================================================
// Periodic launcher (response time = CPU release -> GPU done)
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

static void ensure_logs_dir()
{
#if defined(_WIN32)
    // ignore
#else
    std::system("mkdir -p logs");
#endif
}

static void save_json_sweep(const std::string& path,
                            const std::string& gpu_name,
                            int sm_count,
                            int I,int J,int K,int R,
                            int period_ms,int iters,
                            int t1_lo,int t1_hi,
                            int t2_width,
                            const std::vector<int>& shifts,
                            const std::vector<double>& t1_avg,
                            const std::vector<double>& t2_avg)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open JSON: " << path << "\n";
        return;
    }

    auto tohex16 = [](uint64_t v){
        std::ostringstream oss;
        oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
        return oss.str();
    };

    f << "{\n";
    f << "  \"gpu\": \"" << gpu_name << "\",\n";
    f << "  \"sm_count\": " << sm_count << ",\n";
    f << "  \"tensor_dims\": {\"I\": " << I << ", \"J\": " << J << ", \"K\": " << K << "},\n";
    f << "  \"rank_R\": " << R << ",\n";
    f << "  \"period_ms\": " << period_ms << ",\n";
    f << "  \"iterations\": " << iters << ",\n";
    f << "  \"mask_bit1_means\": \"DISABLE\",\n";
    f << "  \"valid_tpc_bits\": \"" << tohex16(g_valid_tpc_bits) << "\",\n";
    f << "  \"task1_range\": [" << t1_lo << ", " << t1_hi << "],\n";
    f << "  \"task2_width\": " << t2_width << ",\n";
    f << "  \"sweep\": [\n";

    for (size_t i = 0; i < shifts.size(); ++i) {
        int shift = shifts[i];
        int t2_lo = t1_hi + 1 - shift;       // shift=0 => disjoint (10-19 when t1=0-9, width=10)
        int t2_hi = t2_lo + (t2_width - 1);

        uint64_t t1_en = make_enabled_mask_range(t1_lo, t1_hi);
        uint64_t t2_en = make_enabled_mask_range(t2_lo, t2_hi);
        uint64_t t1_dis = maybe_flip_for_disable_mask(t1_en);
        uint64_t t2_dis = maybe_flip_for_disable_mask(t2_en);

        int overlap_lo = std::max(t1_lo, t2_lo);
        int overlap_hi = std::min(t1_hi, t2_hi);
        int overlap = (overlap_hi >= overlap_lo) ? (overlap_hi - overlap_lo + 1) : 0;

        f << "    {"
          << "\"shift\": " << shift << ", "
          << "\"task2_range\": [" << t2_lo << ", " << t2_hi << "], "
          << "\"overlap_tpcs\": " << overlap << ", "
          << "\"task1_enabled_mask\": \"" << tohex16(t1_en) << "\", "
          << "\"task2_enabled_mask\": \"" << tohex16(t2_en) << "\", "
          << "\"task1_applied_disable_mask\": \"" << tohex16(t1_dis) << "\", "
          << "\"task2_applied_disable_mask\": \"" << tohex16(t2_dis) << "\", "
          << "\"task1_avg_rt_ms\": " << std::fixed << std::setprecision(6) << t1_avg[i] << ", "
          << "\"task2_avg_rt_ms\": " << std::fixed << std::setprecision(6) << t2_avg[i]
          << "}";

        if (i + 1 != shifts.size()) f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";
}

// One sweep point: run two periodic tasks with specified enabled masks
static void run_one_point(int I,int J,int K,int R, int period_ms,int iters,
                          uint64_t t1_enabled, uint64_t t2_enabled,
                          double& out_t1_avg, double& out_t2_avg)
{
    const size_t x_elems = static_cast<size_t>(I) * J * K;
    const size_t b_elems = static_cast<size_t>(J) * R;
    const size_t c_elems = static_cast<size_t>(K) * R;
    const size_t a_elems = static_cast<size_t>(I) * R;

    std::vector<float> h_X(x_elems, 1.0f);
    std::vector<float> h_B(b_elems, 1.0f);
    std::vector<float> h_C(c_elems, 1.0f);

    float *d_X=nullptr, *d_B=nullptr, *d_C=nullptr, *d_A1=nullptr, *d_A2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, x_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, a_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), x_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_elems * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s1=nullptr, s2=nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    apply_stream_mask(s1, t1_enabled);
    apply_stream_mask(s2, t2_enabled);

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

    out_t1_avg = avg_ms(rts1);
    out_t2_avg = avg_ms(rts2);
}

int main()
{
    ensure_logs_dir();

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "[MTTKRP Two-Stream TPC Sweep]\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Mask convention: bit=1 DISABLES TPC (per-stream)\n\n";

    // Optional: bound to real TPC count to avoid disabling non-existent bits.
    // If you know your GPU has NUM_TPCS TPCs, set it here.
    // Example: RTX3090 often has 28 TPCs (varies by SKU).
    // const int NUM_TPCS = 28;
    // g_valid_tpc_bits = (NUM_TPCS >= 64) ? ~0ULL : ((1ULL << NUM_TPCS) - 1ULL);

    // Workload size (tune)
    const int I = 512, J = 512, K = 512;
    const int R = 16;

    // Periodic releases
    const int PERIOD_MS = 30;
    const int ITERS = 10;

    // Task1 fixed window
    const int T1_LO = 0;
    const int T1_HI = 9;
    const int WIDTH = (T1_HI - T1_LO + 1); // 10 TPCs

    // Sweep "shift" amount:
    // shift=0  => Task2 = 10..19  (disjoint)
    // shift=1  => Task2 = 9..18   (overlap 1)
    // shift=2  => Task2 = 8..17   (overlap 2)
    // ...
    // shift=10 => Task2 = 0..9    (full overlap 10)
    const std::vector<int> SHIFTS = {0,1,2,3,4,5,6,7,8,9,10};

    std::vector<double> t1_avgs, t2_avgs;
    t1_avgs.reserve(SHIFTS.size());
    t2_avgs.reserve(SHIFTS.size());

    uint64_t t1_enabled = make_enabled_mask_range(T1_LO, T1_HI);

    std::cout << "Task1 fixed TPCs: [" << T1_LO << "," << T1_HI << "]\n";
    std::cout << "Sweeping Task2 window of width " << WIDTH << " by shifting left (increasing overlap)\n\n";

    for (int shift : SHIFTS) {
        int t2_lo = (T1_HI + 1) - shift;
        int t2_hi = t2_lo + (WIDTH - 1);

        uint64_t t2_enabled = make_enabled_mask_range(t2_lo, t2_hi);

        int overlap_lo = std::max(T1_LO, t2_lo);
        int overlap_hi = std::min(T1_HI, t2_hi);
        int overlap = (overlap_hi >= overlap_lo) ? (overlap_hi - overlap_lo + 1) : 0;

        double a1=0.0, a2=0.0;
        run_one_point(I,J,K,R, PERIOD_MS, ITERS, t1_enabled, t2_enabled, a1, a2);

        t1_avgs.push_back(a1);
        t2_avgs.push_back(a2);

        std::cout << "shift=" << std::setw(2) << shift
                  << " | Task2=[" << std::setw(2) << t2_lo << "," << std::setw(2) << t2_hi << "]"
                  << " | overlap=" << std::setw(2) << overlap
                  << " | Task1 avg RT=" << std::fixed << std::setprecision(3) << a1 << " ms"
                  << " | Task2 avg RT=" << std::fixed << std::setprecision(3) << a2 << " ms\n";
    }

    const std::string out_json = "logs/mttkrp_sweep_task1_0_9_task2_shift.json";
    save_json_sweep(out_json, prop.name, prop.multiProcessorCount,
                    I,J,K,R, PERIOD_MS, ITERS,
                    T1_LO, T1_HI, WIDTH,
                    SHIFTS, t1_avgs, t2_avgs);

    std::cout << "\nSaved sweep JSON: " << out_json << "\n";
    return 0;
}
