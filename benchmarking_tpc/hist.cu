#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdint>
#include <fstream>
#include <cmath>

#include "tpc_control.h"
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
// Main Benchmark
// ==========================================================
int main()
{
    const unsigned int NUM_ELEMS = 1 << 20;  // 1M elements
    const unsigned int NUM_BINS  = 4096;
    const int RUNS = 10;
    cudaDeviceProp prop;
    int device = 0; // Assuming device 0
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    cout << "[CUDA Benchmark] Histogram (" << NUM_ELEMS << " elements, "
         << NUM_BINS << " bins)\n";
    cout << "Using libsmctrl TPC mask control (1–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    // Host memory
    unsigned int *h_in = nullptr, *h_bins = nullptr;
    benchmark_cuda_check(cudaMallocHost(&h_in, NUM_ELEMS * sizeof(unsigned int)));
    benchmark_cuda_check(cudaMallocHost(&h_bins, NUM_BINS * sizeof(unsigned int)));

    // Fill input data
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist(0U, NUM_BINS - 1U);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = dist(gen);

    // Device memory
    unsigned int *d_in = nullptr, *d_bins = nullptr;
    benchmark_cuda_check(cudaMalloc(&d_in, NUM_ELEMS * sizeof(unsigned int)));
    benchmark_cuda_check(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));

    // Copy input once
    benchmark_cuda_check(cudaMemcpy(d_in, h_in, NUM_ELEMS * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Grid setup
    unsigned int grid_size = static_cast<unsigned int>(
        ceil(static_cast<float>(NUM_ELEMS) / BLOCK_SIZE));
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    // Results
    vector<pair<int, float>> results;

    cudaStream_t stream;
    benchmark_cuda_check(cudaStreamCreate(&stream));

    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active) {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "]" << std::endl;

        vector<float> times;
        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            benchmark_cuda_check(cudaEventCreate(&start));
            benchmark_cuda_check(cudaEventCreate(&stop));

            benchmark_cuda_check(cudaMemsetAsync(d_bins, 0, NUM_BINS * sizeof(unsigned int), stream));
            benchmark_cuda_check(cudaEventRecord(start, stream));

            histogram_kernel<<<grid, block, 0, stream>>>(d_in, d_bins, NUM_ELEMS, NUM_BINS);

            benchmark_cuda_check(cudaGetLastError());

            benchmark_cuda_check(cudaEventRecord(stop, stream));
            benchmark_cuda_check(cudaEventSynchronize(stop));

            float ms = 0.0f;
            benchmark_cuda_check(cudaEventElapsedTime(&ms, start, stop));
            times.push_back(ms);

            benchmark_cuda_check(cudaEventDestroy(start));
            benchmark_cuda_check(cudaEventDestroy(stop));
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
    ofstream json(benchmark_log_path("tpc_results_hist"));
    if (!json) {
        cerr << "⚠Could not open logs/tpc_results_hist.json, writing fallback file.\n";
        json.open("tpc_results_hist_fallback");
    }
    json << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    std::cout << "[DONE] " << benchmark_log_path("tpc_results_hist") << std::endl;

    // Cleanup
    benchmark_cuda_check(cudaFreeHost(h_in));
    benchmark_cuda_check(cudaFreeHost(h_bins));
    benchmark_cuda_check(cudaFree(d_in));
    benchmark_cuda_check(cudaFree(d_bins));

    cout << "[CUDA Benchmark] Done.\n";
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
