#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "libsmctrl.h"

using namespace std;
using WallClock = std::chrono::steady_clock;

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        cerr << "CUDA error: " << cudaGetErrorString(_e)                   \
             << " (" << (int)_e << ") at " << __FILE__ << ":" << __LINE__  \
             << endl;                                                     \
        exit(1);                                                          \
    }                                                                     \
} while(0)

// ----------------------- device helper: read SM id --------------------------
__device__ __forceinline__ unsigned _getsmid() {
    unsigned smid;
#if defined(__CUDA_ARCH__)
    asm("mov.u32 %0, %smid;" : "=r"(smid));
#else
    smid = 0;
#endif
    return smid;
}

// One atomic per block -> low overhead but gives "which SMs executed blocks"
__global__ void smid(unsigned int* sm_counts, int sm_counts_len) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned sm = _getsmid();
        if ((int)sm < sm_counts_len) atomicAdd(&sm_counts[sm], 1u);
    }
}

// Clear histogram
__global__ void clearCounts(unsigned int* sm_counts, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) sm_counts[i] = 0u;
}

static vector<int> list_active_sms(const vector<unsigned int>& h) {
    vector<int> out;
    for (int i = 0; i < (int)h.size(); ++i)
        if (h[i] != 0u) out.push_back(i);
    return out;
}

// Build "disable mask" enabling low-index TPCs [0..tpc_active-1].
// bit=1 disabled, bit=0 enabled.
static uint64_t build_disable_mask_from_active_tpcs(int tpc_active) {
    if (tpc_active <= 0) return ~0ULL;           // disable all
    if (tpc_active >= 64) return 0ULL;           // enable all 64
    uint64_t enable_low = (1ULL << tpc_active) - 1ULL;
    return ~enable_low;
}

 #define LIBSMCTRL_SET_STREAM_MASK(stream, mask) libsmctrl_set_stream_mask((stream),(mask))


// Run kernel, return active SM list
static vector<int> run_and_get_active_sms(cudaStream_t stream,
                                         unsigned int* d_sm_counts,
                                         vector<unsigned int>& h_sm_counts,
                                         int num_sms,
                                         dim3 grid, dim3 block)
{
    int t = 256;
    int b = (num_sms + t - 1) / t;

    clearCounts<<<b, t, 0, stream>>>(d_sm_counts, num_sms);
    smid<<<grid, block, 0, stream>>>(d_sm_counts, num_sms);

    CUDA_CHECK(cudaMemcpyAsync(h_sm_counts.data(), d_sm_counts,
                               (size_t)num_sms * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return list_active_sms(h_sm_counts);
}

static pair<double,double> mean_std(const vector<double>& v) {
    if (v.empty()) return {0.0, 0.0};
    double mean = accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= (double)v.size();
    return {mean, sqrt(max(0.0, var))};
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = num_sms / 2;  // your assumption: 2 SM per TPC

    cout << "GPU: " << prop.name << "\n";
    cout << "SMs reported: " << num_sms << "\n";
    cout << "Assuming 2 SM/TPC => MAX_TPC=" << MAX_TPC << "\n\n";

    // One stream for the entire experiment
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Histogram buffer
    unsigned int* d_sm_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sm_counts, (size_t)num_sms * sizeof(unsigned int)));
    vector<unsigned int> h_sm_counts((size_t)num_sms, 0u);

    // Use MANY blocks so scheduler has reason to use all enabled SMs.
    const int blocks_total = max(4096, num_sms * 256);
    dim3 block(32, 1, 1);
    dim3 grid(blocks_total, 1, 1);

    auto print_result = [&](const char* tag, uint64_t mask, const vector<int>& active) {
        cout << tag
             << " disable_mask=0x" << hex << setw(16) << setfill('0') << mask
             << dec << setfill(' ')
             << " | unique_sms=" << active.size() << "\n";
        cout << "  SM IDs: ";
        for (size_t i = 0; i < active.size(); ++i)
            cout << active[i] << (i + 1 < active.size() ? "," : "");
        cout << "\n";
    };

    // Choose two settings to demonstrate runtime adjustment
    int tpc_small = max(1, MAX_TPC / 4);
    int tpc_large = max(1, (MAX_TPC * 3) / 4);

    uint64_t mask_small = build_disable_mask_from_active_tpcs(tpc_small);
    uint64_t mask_large = build_disable_mask_from_active_tpcs(tpc_large);

    // -------------------- Phase 1: set small mask, measure -------------------
    cout << "=== Phase 1: Apply SMALL stream mask and measure active SMs ===\n";
    {
        auto t0 = WallClock::now();
        LIBSMCTRL_SET_STREAM_MASK(stream, mask_small);
        auto t1 = WallClock::now();

        auto active = run_and_get_active_sms(stream, d_sm_counts, h_sm_counts, num_sms, grid, block);
        print_result("SMALL", mask_small, active);

        double us = chrono::duration<double, micro>(t1 - t0).count();
        cout << "  mask_set_cpu_time(us)=" << fixed << setprecision(2) << us << "\n\n";
    }

    // -------------------- Phase 2: change mask at runtime --------------------
    cout << "=== Phase 2: Change to LARGE stream mask at runtime ===\n";
    {
        auto t0 = WallClock::now();
        LIBSMCTRL_SET_STREAM_MASK(stream, mask_large);
        auto t1 = WallClock::now();

        auto active = run_and_get_active_sms(stream, d_sm_counts, h_sm_counts, num_sms, grid, block);
        print_result("LARGE", mask_large, active);

        double us = chrono::duration<double, micro>(t1 - t0).count();
        cout << "  mask_set_cpu_time(us)=" << fixed << setprecision(2) << us << "\n\n";
    }

    // -------------------- Phase 3: flip back and forth -----------------------
    cout << "=== Phase 3: Flip masks repeatedly (same stream) ===\n";
    const int FLIPS = 10;
    const int REPS_PER_FLIP = 10; // <-- what you asked: run each flip 10 times and average

    // store per-flip averages
    vector<double> flip_set_only_us;  flip_set_only_us.reserve(FLIPS);
    vector<double> flip_run_sync_us;  flip_run_sync_us.reserve(FLIPS);

    for (int i = 0; i < FLIPS; ++i) {
        uint64_t m = (i & 1) ? mask_small : mask_large;

        vector<double> rep_set_only, rep_run_sync;
        rep_set_only.reserve(REPS_PER_FLIP);
        rep_run_sync.reserve(REPS_PER_FLIP);

        vector<int> last_active;

        for (int r = 0; r < REPS_PER_FLIP; ++r) {
            // time ONLY the mask-setting call
            auto s0 = WallClock::now();
            LIBSMCTRL_SET_STREAM_MASK(stream, m);
            auto s1 = WallClock::now();
            rep_set_only.push_back(chrono::duration<double, micro>(s1 - s0).count());

            // NOTE: since kernels aren't preemptible, mask affects *next* launch only.
            auto t0 = WallClock::now();
            last_active = run_and_get_active_sms(stream, d_sm_counts, h_sm_counts, num_sms, grid, block);
            auto t1 = WallClock::now();
            rep_run_sync.push_back(chrono::duration<double, micro>(t1 - t0).count());
        }

        double avg_set = accumulate(rep_set_only.begin(), rep_set_only.end(), 0.0) / rep_set_only.size();
        double avg_run = accumulate(rep_run_sync.begin(), rep_run_sync.end(), 0.0) / rep_run_sync.size();

        flip_set_only_us.push_back(avg_set);
        flip_run_sync_us.push_back(avg_run);

        cout << "flip " << setw(2) << i
             << " | disable_mask=0x" << hex << setw(16) << setfill('0') << m
             << dec << setfill(' ')
             << " | unique_sms=" << setw(2) << last_active.size()
             << " | set_only_avg(us)=" << fixed << setprecision(2) << avg_set
             << " | (run+sync)_avg(us)=" << fixed << setprecision(2) << avg_run
             << "\n";
    }

    // Summary statistics
    {
        auto s_set = mean_std(flip_set_only_us);
        auto s_run = mean_std(flip_run_sync_us);

        cout << "\nSet-mask-only overhead summary over " << FLIPS
             << " flips (each averaged over " << REPS_PER_FLIP << " runs):\n";
        cout << "  mean(us)=" << fixed << setprecision(2) << s_set.first
             << "  std(us)=" << fixed << setprecision(2) << s_set.second << "\n";

        cout << "\n(run+sync) summary over " << FLIPS
             << " flips (each averaged over " << REPS_PER_FLIP << " runs):\n";
        cout << "  mean(us)=" << fixed << setprecision(2) << s_run.first
             << "  std(us)=" << fixed << setprecision(2) << s_run.second << "\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_sm_counts));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}