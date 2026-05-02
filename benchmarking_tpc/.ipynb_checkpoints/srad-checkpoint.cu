// srad_smctrl_sweep_projection_style.cu
// Benchmark SRAD kernels with TPC mask sweep (1..41) — projection-style

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include "libsmctrl.h"

// -------------------------------
// Constants and configuration
// -------------------------------
#define ROWS (4096 * 4)
#define COLS (4096 * 4)
#define SIZE (ROWS * COLS)
#define BLOCK_SIZE 8
#define RUNS 10

const float lambda = 2.5f;

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
        exit(EXIT_FAILURE);
    }
}

// -------------------------------
// Kernel 1: compute conduction coefficients
// -------------------------------
__global__ void srad1(float* E, float* W, float* N, float* S,
                      float* J, float* C, int cols, int rows, float q0sqr)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = (rows * BLOCK_SIZE * by) + (BLOCK_SIZE * bx) + (rows * ty) + tx;

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileC[BLOCK_SIZE][BLOCK_SIZE];

    tile[ty][tx] = J[idx];
    __syncthreads();

    float jc = tile[ty][tx];
    float north = (ty == 0) ? jc : tile[ty - 1][tx] - jc;
    float south = (ty == BLOCK_SIZE - 1) ? jc : tile[ty + 1][tx] - jc;
    float west  = (tx == 0) ? jc : tile[ty][tx - 1] - jc;
    float east  = (tx == BLOCK_SIZE - 1) ? jc : tile[ty][tx + 1] - jc;

    float g2 = (north * north + south * south + west * west + east * east) / (jc * jc);
    float l  = (north + south + west + east) / jc;
    float num = 0.5f * g2 - (1.0f / 16.0f) * (l * l);
    float den = 1.0f + 0.25f * l;
    float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    float c = 1.0f / (1.0f + den);
    tileC[ty][tx] = fminf(fmaxf(c, 0.0f), 1.0f);
    __syncthreads();

    C[idx] = tileC[ty][tx];
    E[idx] = east;
    W[idx] = west;
    N[idx] = north;
    S[idx] = south;
}

// -------------------------------
// Kernel 2: update J using diffusion
// -------------------------------
__global__ void srad2(float* E, float* W, float* N, float* S,
                      float* J, float* C, int cols, int rows, float lambda)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = (rows * BLOCK_SIZE * by) + (BLOCK_SIZE * bx) + (rows * ty) + tx;

    __shared__ float tileJ[BLOCK_SIZE][BLOCK_SIZE];
    tileJ[ty][tx] = J[idx];
    __syncthreads();

    float cc = C[idx];
    float d = cc * (N[idx] + S[idx] + W[idx] + E[idx]);
    tileJ[ty][tx] += 0.25f * lambda * d;
    __syncthreads();

    J[idx] = tileJ[ty][tx];
}

// -------------------------------
// Build TPC disable mask
// -------------------------------
static inline uint64_t build_mask_from_active_tpc(int active_tpc) {
    if (active_tpc <= 0)  return ~0ULL;
    if (active_tpc >= 64) return 0ULL;
    const uint64_t allowed = (1ULL << active_tpc) - 1ULL;
    return ~allowed;
}

// -------------------------------
// Main
// -------------------------------
int main()
{
    int num_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop;
    uint32_t num_tpcs = 0;
    if (libsmctrl_get_tpc_info_cuda(&num_tpcs, 0) != 0 || num_tpcs == 0) {
        std::cerr << "libsmctrl_get_tpc_info_cuda failed.\n";
        return 1;
    }

    int sms_per_tpc = num_sms / num_tpcs;
    std::cout << "GPU: " << num_sms << " SMs, " << num_tpcs << " TPCs ("
              << sms_per_tpc << " SMs/TPC)\n";
    std::cout << "[SRAD Benchmark] Sweep TPC active = 1..41\n";
    std::cout << std::fixed << std::setprecision(3);

    // Host data
    std::vector<float> h_J(SIZE);
    for (size_t i = 0; i < SIZE; ++i)
        h_J[i] = expf(static_cast<float>(rand()) / RAND_MAX);

    // Device memory
    float *d_J, *d_C, *d_E, *d_W, *d_N, *d_S;
    checkCudaErrors(cudaMalloc(&d_J, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_E, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_W, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_N, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_S, SIZE * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_J, h_J.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(COLS / BLOCK_SIZE, ROWS / BLOCK_SIZE);

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    std::vector<std::pair<int, float>> results;
    results.reserve(41);

    // Sweep 1–41 active TPCs
    for (int tpc_active = 1; tpc_active <= 4; ++tpc_active) {
        uint64_t mask = build_mask_from_active_tpc(tpc_active);
        libsmctrl_set_global_mask(mask);

        std::cout << "\n[Active TPCs: " << std::setw(2) << tpc_active
                  << "] Mask = 0x" << std::hex << mask << std::dec << "\n";

        std::vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));

            checkCudaErrors(cudaEventRecord(start, stream));

            srad1<<<grid, block, 0, stream>>>(d_E, d_W, d_N, d_S, d_J, d_C, COLS, ROWS, 1.0f);
            srad2<<<grid, block, 0, stream>>>(d_E, d_W, d_N, d_S, d_J, d_C, COLS, ROWS, lambda);

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
        avg /= timings.size();

        std::cout << "  Average GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    // Write results to JSON
    std::ofstream jsonFile("logs/tpc_results_srad");
    jsonFile << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";
    jsonFile.close();

    std::cout << "\n[SRAD Benchmark] Results saved to logs/tpc_results_srad.json\n";

    // Cleanup
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_J));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_E));
    checkCudaErrors(cudaFree(d_W));
    checkCudaErrors(cudaFree(d_N));
    checkCudaErrors(cudaFree(d_S));

    return 0;
}
