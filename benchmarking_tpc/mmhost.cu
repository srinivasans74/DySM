#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <fstream>
#include "tpc_control.h"

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
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    cout << "[CUDA Benchmark] Matrix Multiply (" << N << "x" << N << ")\n";
    size_t size = N * N * sizeof(float);

    // Standard Host Allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Device Allocation (Required to measure explicit H2D/D2H)
    float *d_A, *d_B, *d_C;
    benchmark_cuda_check(cudaMalloc(&d_A, size));
    benchmark_cuda_check(cudaMalloc(&d_B, size));
    benchmark_cuda_check(cudaMalloc(&d_C, size));

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; h_B[i] = 1.0f;
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    vector<PerfResult> results;

    cudaStream_t stream;
    benchmark_cuda_check(cudaStreamCreate(&stream));

    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active)
    {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        cout << "\r[Processing TPC: " << tpc_active << "/" << MAX_TPC << "]" << flush;

        float total_h2d = 0, total_ek = 0, total_d2h = 0;

        for (int run = 0; run < RUNS; ++run)
        {
            cudaEvent_t startH2D, stopH2D, startEK, stopEK, startD2H, stopD2H;
            benchmark_cuda_check(cudaEventCreate(&startH2D)); benchmark_cuda_check(cudaEventCreate(&stopH2D));
            benchmark_cuda_check(cudaEventCreate(&startEK));  benchmark_cuda_check(cudaEventCreate(&stopEK));
            benchmark_cuda_check(cudaEventCreate(&startD2H)); benchmark_cuda_check(cudaEventCreate(&stopD2H));

            // 1. Measure H2D
            benchmark_cuda_check(cudaEventRecord(startH2D, stream));
            benchmark_cuda_check(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
            benchmark_cuda_check(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));
            benchmark_cuda_check(cudaEventRecord(stopH2D, stream));

            // 2. Measure EK (Kernel)
            benchmark_cuda_check(cudaEventRecord(startEK, stream));
            matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);
            benchmark_cuda_check(cudaGetLastError());
            benchmark_cuda_check(cudaEventRecord(stopEK, stream));

            // 3. Measure D2H
            benchmark_cuda_check(cudaEventRecord(startD2H, stream));
            benchmark_cuda_check(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
            benchmark_cuda_check(cudaEventRecord(stopD2H, stream));

            benchmark_cuda_check(cudaEventSynchronize(stopD2H));

            float ms_h2d, ms_ek, ms_d2h;
            benchmark_cuda_check(cudaEventElapsedTime(&ms_h2d, startH2D, stopH2D));
            benchmark_cuda_check(cudaEventElapsedTime(&ms_ek, startEK, stopEK));
            benchmark_cuda_check(cudaEventElapsedTime(&ms_d2h, startD2H, stopD2H));

            total_h2d += ms_h2d; total_ek += ms_ek; total_d2h += ms_d2h;

            benchmark_cuda_check(cudaEventDestroy(startH2D)); benchmark_cuda_check(cudaEventDestroy(stopH2D));
            benchmark_cuda_check(cudaEventDestroy(startEK));  benchmark_cuda_check(cudaEventDestroy(stopEK));
            benchmark_cuda_check(cudaEventDestroy(startD2H)); benchmark_cuda_check(cudaEventDestroy(stopD2H));
        }

        results.push_back({tpc_active, total_h2d/RUNS, total_ek/RUNS, total_d2h/RUNS});
    }

    // JSON Export
    ofstream jsonFile(benchmark_log_path("tpc_detailed_results.json"));
    jsonFile << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].tpc
                 << ", \"H2D_ms\": " << results[i].h2d_ms
                 << ", \"EK_ms\": " << results[i].ek_ms
                 << ", \"D2H_ms\": " << results[i].d2h_ms << "}";
        if (i != results.size() - 1) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";

    cout << "[DONE] " << benchmark_log_path("tpc_detailed_results.json") << std::endl;

    benchmark_cuda_check(cudaFree(d_A)); benchmark_cuda_check(cudaFree(d_B)); benchmark_cuda_check(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
