#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdint>
#include <fstream>
#include <cmath>

#include "libsmctrl.h"  // TPC mask control (1 = disabled, 0 = enabled)
//#include "helper_cuda.h"

using namespace std;

#define BLOCK_SIZE 512

// ==========================================================
// CUDA Kernel — Histogram
// ==========================================================
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
                                 unsigned int num_elements, unsigned int num_bins)
{
    // Repeat a few times to simulate heavier GPU load
    for (int iter = 0; iter < 100; ++iter) {
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int stride = blockDim.x * gridDim.x;
        for (unsigned int i = tid; i < num_elements; i += stride) {
            unsigned int val = input[i];
            if (val < num_bins)
                atomicAdd(&bins[val], 1);
        }
    }
}

// ==========================================================
// Helper — Build disable mask for libsmctrl
// ==========================================================
static inline uint64_t build_mask(int active)
{
    if (active <= 0) return 0xFFFFFFFFFFFFFFFEULL;  // all disabled
    if (active >= 64) return 0ULL;                  // all enabled
    uint64_t enable_mask = (1ULL << active) - 1ULL; // enable lower bits
    return ~enable_mask;                            // disable others
}

// ==========================================================
// Main Benchmark
// ==========================================================
int main()
{
    const unsigned int NUM_ELEMS = 1 << 20;  // 1M elements
    const unsigned int NUM_BINS  = 4096;
    const int RUNS = 10;
    cudaDeviceProp prop;
    int device = 0; // Assuming device 0
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    int MAX_TPC = num_sms / 2;

    cout << "[CUDA Benchmark] Histogram (" << NUM_ELEMS << " elements, "
         << NUM_BINS << " bins)\n";
    cout << "Using libsmctrl TPC mask control (0–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    // Host memory
    unsigned int *h_in = nullptr, *h_bins = nullptr;
    cudaMallocHost(&h_in, NUM_ELEMS * sizeof(unsigned int));
    cudaMallocHost(&h_bins, NUM_BINS * sizeof(unsigned int));

    // Fill input data
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist(0U, NUM_BINS - 1U);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = dist(gen);

    // Device memory
    unsigned int *d_in = nullptr, *d_bins = nullptr;
    cudaMalloc(&d_in, NUM_ELEMS * sizeof(unsigned int));
    cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int));

    // Copy input once
    cudaMemcpy(d_in, h_in, NUM_ELEMS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Grid setup
    unsigned int grid_size = static_cast<unsigned int>(
        ceil(static_cast<float>(NUM_ELEMS) / BLOCK_SIZE));
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    // Results
    vector<pair<int, float>> results;

    for (int tpc_active = 0; tpc_active <= MAX_TPC; ++tpc_active) {
        uint64_t mask = build_mask(tpc_active);
        libsmctrl_set_global_mask(mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "] Mask = 0x" << hex << mask << dec << endl;

        vector<float> times;
        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int));
            cudaEventRecord(start);

            histogram_kernel<<<grid, block>>>(d_in, d_bins, NUM_ELEMS, NUM_BINS);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            times.push_back(ms);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float avg = 0.0f;
        for (auto t : times) avg += t;
        avg /= times.size();

        cout << "  → Average GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    // ======================================================
    // Save JSON Output
    // ======================================================
    ofstream json("logs/tpc_results_hist");
    if (!json) {
        cerr << "⚠Could not open logs/tpc_results_hist.json, writing fallback file.\n";
        json.open("tpc_results_hist_fallback");
    }
    json << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    cout << "\n[CUDA Benchmark] Results saved to logs/tpc_results_hist.json\n";

    // Cleanup
    cudaFreeHost(h_in);
    cudaFreeHost(h_bins);
    cudaFree(d_in);
    cudaFree(d_bins);

    cout << "[CUDA Benchmark] Done.\n";
    return 0;
}
