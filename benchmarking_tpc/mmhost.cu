#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include "libsmctrl.h"

using namespace std;

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

struct PerfResult {
    int tpc;
    float h2d_ms;
    float ek_ms;
    float d2h_ms;
};

int main()
{
    const int N = 1024;
    const int RUNS = 10;
    const int BLOCK_SIZE = 32;
    int device = 0;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    int MAX_TPC = num_sms / 2;

    cout << "[CUDA Benchmark] Matrix Multiply (" << N << "x" << N << ")\n";
    size_t size = N * N * sizeof(float);

    // Standard Host Allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Device Allocation (Required to measure explicit H2D/D2H)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; h_B[i] = 1.0f;
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    vector<PerfResult> results;

    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active)
    {
        uint64_t mask = (tpc_active < 64) ? ~((1ULL << tpc_active) - 1ULL) : 0;
        libsmctrl_set_global_mask(mask);

        cout << "\r[Processing TPC: " << tpc_active << "/" << MAX_TPC << "]" << flush;

        float total_h2d = 0, total_ek = 0, total_d2h = 0;

        for (int run = 0; run < RUNS; ++run)
        {
            cudaEvent_t startH2D, stopH2D, startEK, stopEK, startD2H, stopD2H;
            cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
            cudaEventCreate(&startEK);  cudaEventCreate(&stopEK);
            cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

            // 1. Measure H2D
            cudaEventRecord(startH2D);
            cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
            cudaEventRecord(stopH2D);

            // 2. Measure EK (Kernel)
            cudaEventRecord(startEK);
            matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stopEK);

            // 3. Measure D2H
            cudaEventRecord(startD2H);
            cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
            cudaEventRecord(stopD2H);

            cudaEventSynchronize(stopD2H);

            float ms_h2d, ms_ek, ms_d2h;
            cudaEventElapsedTime(&ms_h2d, startH2D, stopH2D);
            cudaEventElapsedTime(&ms_ek, startEK, stopEK);
            cudaEventElapsedTime(&ms_d2h, startD2H, stopD2H);

            total_h2d += ms_h2d; total_ek += ms_ek; total_d2h += ms_d2h;

            cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
            cudaEventDestroy(startEK);  cudaEventDestroy(stopEK);
            cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);
        }

        results.push_back({tpc_active, total_h2d/RUNS, total_ek/RUNS, total_d2h/RUNS});
    }

    // JSON Export
    ofstream jsonFile("logs/tpc_detailed_results.json");
    jsonFile << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].tpc 
                 << ", \"H2D_ms\": " << results[i].h2d_ms 
                 << ", \"EK_ms\": " << results[i].ek_ms 
                 << ", \"D2H_ms\": " << results[i].d2h_ms << "}";
        if (i != results.size() - 1) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";
    
    cout << "\nBenchmark complete. Data saved to logs/tpc_detailed_results.json\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
