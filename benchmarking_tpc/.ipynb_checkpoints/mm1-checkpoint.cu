#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include "libsmctrl.h"  // Header for TPC mask control

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
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    int MAX_TPC = num_sms / 2;

    cout << "[CUDA Benchmark] Matrix Multiply (" << N << "x" << N << ")\n";
    cout << "Running with libsmctrl TPC masks (0–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    size_t size = N * N * sizeof(float);

    // Enable mapping of host memory for zero-copy access
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // Allocate pinned, mapped host memory
    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, size, cudaHostAllocMapped);
    cudaHostAlloc(&h_B, size, cudaHostAllocMapped);
    cudaHostAlloc(&h_C, size, cudaHostAllocMapped);

    // Obtain device pointers that map to the same host memory
    float *d_A, *d_B, *d_C;
    cudaHostGetDevicePointer((void**)&d_A, h_A, 0);
    cudaHostGetDevicePointer((void**)&d_B, h_B, 0);
    cudaHostGetDevicePointer((void**)&d_C, h_C, 0);

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
    for (int tpc_active = 0; tpc_active <= MAX_TPC; ++tpc_active)
    {
        // Build disable mask: 1 = disabled, 0 = enabled
        uint64_t mask = 0xFFFFFFFFFFFFFFFEULL;  // all disabled by default
        if (tpc_active > 0 && tpc_active < 64)
            mask = ~((1ULL << tpc_active) - 1ULL);

        // Apply global TPC disable mask via libsmctrl
        libsmctrl_set_global_mask(mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "] Mask = 0x" << hex << mask << dec << '\n';

        vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            // Launch kernel
            matMulKernel<<<grid, block,0>>>(d_A, d_B, d_C, N);
            cudaDeviceSynchronize();

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            timings.push_back(ms);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
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
    ofstream jsonFile("logs/tpc_results_mm");
    jsonFile << "{\n  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i != results.size() - 1) jsonFile << ",";
        jsonFile << "\n";
    }

    jsonFile << "  ]\n}\n";
    jsonFile.close();

    cout << "\n[CUDA Benchmark] Results saved to 'tpc_results.json'.\n";

    // ======================================================
    // Cleanup
    // ======================================================
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cout << "[CUDA Benchmark] All masks completed.\n";
    return 0;
}
