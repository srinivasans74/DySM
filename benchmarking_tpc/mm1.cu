#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include "tpc_control.h"

using namespace std;

// ==========================================================
// CUDA Kernel — Basic Matrix Multiplication
// ==========================================================
__global__ void matMulKernel(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

// ==========================================================
// Benchmark Runner — zero-copy mapped memory with SM mask sweep
// ==========================================================
int main()
{
    const int N = 1024;                // Matrix size (N x N)
    const int RUNS = 10;               // Runs per mask
    const int BLOCK_SIZE = 32;        // Threads per block
    int device = 0; // Assuming device 0
    cudaDeviceProp prop;
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    cout << "[CUDA Benchmark] Matrix Multiply (" << N << "x" << N << ")\n";
    cout << "Running with libsmctrl TPC masks (1–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    size_t size = N * N * sizeof(float);

    // Enable mapping of host memory for zero-copy access
    benchmark_cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Allocate pinned, mapped host memory
    float *h_A, *h_B, *h_C;
    benchmark_cuda_check(cudaHostAlloc(&h_A, size, cudaHostAllocMapped));
    benchmark_cuda_check(cudaHostAlloc(&h_B, size, cudaHostAllocMapped));
    benchmark_cuda_check(cudaHostAlloc(&h_C, size, cudaHostAllocMapped));

    // Obtain device pointers that map to the same host memory
    float *d_A, *d_B, *d_C;
    benchmark_cuda_check(cudaHostGetDevicePointer((void**)&d_A, h_A, 0));
    benchmark_cuda_check(cudaHostGetDevicePointer((void**)&d_B, h_B, 0));
    benchmark_cuda_check(cudaHostGetDevicePointer((void**)&d_C, h_C, 0));

    // Initialize host data
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // Configure kernel launch
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Store results for JSON output
    vector<pair<int, float>> results;

    // ======================================================
    // Sweep through masks
    // ======================================================
    cudaStream_t stream;
    benchmark_cuda_check(cudaStreamCreate(&stream));

    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active)
    {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "]" << std::endl;

        vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run)
        {
            cudaEvent_t start, stop;
            benchmark_cuda_check(cudaEventCreate(&start));
            benchmark_cuda_check(cudaEventCreate(&stop));

            benchmark_cuda_check(cudaEventRecord(start, stream));

            // Launch kernel
            matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);

            benchmark_cuda_check(cudaGetLastError());

            benchmark_cuda_check(cudaEventRecord(stop, stream));
            benchmark_cuda_check(cudaEventSynchronize(stop));

            float ms = 0.0f;
            benchmark_cuda_check(cudaEventElapsedTime(&ms, start, stop));
            timings.push_back(ms);

            benchmark_cuda_check(cudaEventDestroy(start));
            benchmark_cuda_check(cudaEventDestroy(stop));
        }

        // Average over runs
        float avg = 0.0f;
        for (float t : timings) avg += t;
        avg /= timings.size();

        cout << "  → Average GPU time (" << RUNS << " runs) = "
             << avg << " ms\n";

        // Save result
        results.push_back({tpc_active, avg});
    }

    // ======================================================
    // Save JSON output
    // ======================================================
    ofstream jsonFile(benchmark_log_path("tpc_results_mm"));
    jsonFile << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i != results.size() - 1) jsonFile << ",";
        jsonFile << "\n";
    }

    jsonFile << "  ]\n}\n";
    jsonFile.close();

    std::cout << "[DONE] " << benchmark_log_path("tpc_results_mm") << std::endl;

    // ======================================================
    // Cleanup
    // ======================================================
    benchmark_cuda_check(cudaFreeHost(h_A));
    benchmark_cuda_check(cudaFreeHost(h_B));
    benchmark_cuda_check(cudaFreeHost(h_C));

    cout << "[CUDA Benchmark] All masks completed.\n";
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
