// mttkrp_smctrl_sweep_projection_style.cu
// Benchmark sparse MTTKRP (mode-0) with TPC mask sweep (all detected TPCs) — projection-style

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include "tpc_control.h"
#include <sys/stat.h>
#include <sys/types.h>
// -------------------------------
// Configuration (tune as needed)
// -------------------------------
#define I_DIM  32768     // mode-0 size (rows of output M)
#define J_DIM  4096      // mode-1 size (rows of B)
#define K_DIM  4096      // mode-2 size (rows of C)
#define RANK   16        // CP rank (columns of factors / output)
#define NNZ    4000000   // number of nonzeros in COO tensor
#define RUNS   10
#define THREADS 256

// -------------------------------
// CUDA error checking
// -------------------------------
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char *const func, const char *const file, int const line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code="
                  << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorName(result) << ") "
                  << "\"" << func << "\"" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// -------------------------------
// Simple PRNG helper
// -------------------------------
static inline float frand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// -------------------------------
// Sparse MTTKRP (mode-0) kernel
// Each thread handles one NNZ entry and accumulates into M[i,*] with atomics
// -------------------------------
__global__ void mttkrp_mode0_coo(
    const int* __restrict__ ti,
    const int* __restrict__ tj,
    const int* __restrict__ tk,
    const float* __restrict__ tv,
    const float* __restrict__ B,   // [J_DIM * RANK]
    const float* __restrict__ C,   // [K_DIM * RANK]
    float* __restrict__ M,         // [I_DIM * RANK]
    int nnz, int rank)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= nnz) return;

    const int i = ti[n];
    const int j = tj[n];
    const int k = tk[n];
    const float x = tv[n];

    const float* Bj = B + (size_t)j * rank;
    const float* Ck = C + (size_t)k * rank;
    float* Mi = M + (size_t)i * rank;

    // Rank loop (small)
    #pragma unroll
    for (int r = 0; r < RANK; r++) {
        float contrib = x * Bj[r] * Ck[r];
        atomicAdd(&Mi[r], contrib);
    }
}

// -------------------------------
// Main
// -------------------------------
int main()
{
    // Query GPU SM/TPC info
    int num_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    const int num_tpcs = benchmark_tpc_count();

    int sms_per_tpc = (num_tpcs > 0) ? (num_sms / (int)num_tpcs) : 0;
    std::cout << "GPU: " << num_sms << " SMs, " << num_tpcs << " TPCs ("
              << sms_per_tpc << " SMs/TPC)\n";

    const int sweep_max = num_tpcs;
    std::cout << "[MTTKRP Benchmark] Sweep active TPC = 1.." << sweep_max << "\n";
    std::cout << std::fixed << std::setprecision(3);

    // -------------------------------
    // Host data init
    // -------------------------------
    std::vector<int>   h_i(NNZ), h_j(NNZ), h_k(NNZ);
    std::vector<float> h_v(NNZ);

    // Random COO indices (uniform) + values
    // NOTE: This is synthetic; replace with your real tensor if needed.
    srand(1);
    for (int n = 0; n < NNZ; n++) {
        h_i[n] = rand() % I_DIM;
        h_j[n] = rand() % J_DIM;
        h_k[n] = rand() % K_DIM;
        h_v[n] = frand();
    }

    std::vector<float> h_B((size_t)J_DIM * RANK);
    std::vector<float> h_C((size_t)K_DIM * RANK);
    for (auto &x : h_B) x = frand();
    for (auto &x : h_C) x = frand();

    // -------------------------------
    // Device allocations
    // -------------------------------
    int *d_i=nullptr, *d_j=nullptr, *d_k=nullptr;
    float *d_v=nullptr, *d_B=nullptr, *d_C=nullptr, *d_M=nullptr;

    checkCudaErrors(cudaMalloc(&d_i, (size_t)NNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_j, (size_t)NNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_k, (size_t)NNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_v, (size_t)NNZ * sizeof(float)));

    checkCudaErrors(cudaMalloc(&d_B, (size_t)J_DIM * RANK * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, (size_t)K_DIM * RANK * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_M, (size_t)I_DIM * RANK * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_i, h_i.data(), (size_t)NNZ * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_j, h_j.data(), (size_t)NNZ * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_k, h_k.data(), (size_t)NNZ * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, h_v.data(), (size_t)NNZ * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), (size_t)J_DIM * RANK * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C.data(), (size_t)K_DIM * RANK * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    dim3 block(THREADS);
    dim3 grid((NNZ + THREADS - 1) / THREADS);

    std::vector<std::pair<int, float>> results;
    results.reserve(sweep_max);

    // -------------------------------
    // Sweep active TPC count
    // -------------------------------
    for (int tpc_active = 1; tpc_active <= sweep_max; ++tpc_active) {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        std::cout << "\n[Active TPCs: " << std::setw(2) << tpc_active
                  << "]" << std::endl;

        std::vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run) {
            // Clear output outside timed region (so timing is kernel-only)
            checkCudaErrors(cudaMemsetAsync(d_M, 0, (size_t)I_DIM * RANK * sizeof(float), stream));
            checkCudaErrors(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));

            checkCudaErrors(cudaEventRecord(start, stream));

            mttkrp_mode0_coo<<<grid, block, 0, stream>>>(
                d_i, d_j, d_k, d_v, d_B, d_C, d_M, NNZ, RANK
            );

            benchmark_cuda_check(cudaGetLastError());

            checkCudaErrors(cudaEventRecord(stop, stream));
            checkCudaErrors(cudaEventSynchronize(stop));

            float ms = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
            timings.push_back(ms);

            checkCudaErrors(cudaEventDestroy(start));
            checkCudaErrors(cudaEventDestroy(stop));
        }

        float avg = 0.0f;
        for (float t : timings) avg += t;
        avg /= (float)timings.size();

        std::cout << "  Average GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    // -------------------------------
    // Write results to JSON
    // -------------------------------
    const std::string outpath = benchmark_log_path("tpc_results_mttkrp");
    std::ofstream jsonFile(outpath);
    jsonFile << "{\n  \"total_tpcs\": " << num_tpcs << ",\n"
             << "  \"kernel\": \"mttkrp_mode0_coo\",\n"
             << "  \"I\": " << I_DIM << ", \"J\": " << J_DIM << ", \"K\": " << K_DIM << ",\n"
             << "  \"R\": " << RANK << ", \"NNZ\": " << NNZ << ", \"RUNS\": " << RUNS << ",\n"
             << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";
    jsonFile.close();

    std::cout << "\n[MTTKRP Benchmark] Results saved to " << outpath << "\n";

    // Cleanup
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_i));
    checkCudaErrors(cudaFree(d_j));
    checkCudaErrors(cudaFree(d_k));
    checkCudaErrors(cudaFree(d_v));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_M));

    return 0;
}
