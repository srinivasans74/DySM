#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <cmath>

#include "tpc_control.h"
#include "common/include/support.h"

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

// ==========================================================
// CUDA Kernel — Compute-heavy point "projection"
// ==========================================================
__global__ void projectPoints(float *d_input, float *d_output, int num_points, float f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int loopIters = 5000;           // new macro

    if (idx < num_points) {
        float x = d_input[3 * idx];
        float y = d_input[3 * idx + 1];
        float z = d_input[3 * idx + 2];

        #pragma unroll 1
        for (int i = 0; i < loopIters; i++) {
            x = __sinf(x) + __cosf(y);
            y = __cosf(y) + __sinf(x);
            z = __tanf(z) + __cosf(x);
        }

        float scale = f / (f + z);
        d_output[2 * idx]     = x * scale;
        d_output[2 * idx + 1] = y * scale;
    }
}

// ----------------------------------------------------------
// CPU helper: simple prime generator
// ----------------------------------------------------------
static inline bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; 1LL * i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

static std::vector<int> computePrimes(int limit) {
    std::vector<int> primes;
    for (int i = 2; i <= limit; i++) if (isPrime(i)) primes.push_back(i);
    return primes;
}

int main(int argc, char* argv[])
{
    const int num_points   = 16834*4;
    const float focal_len  = 2048.0f;
    const int RUNS         = 100;
    cudaDeviceProp prop;
    int device = 0; // Assuming device 0
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    const int threadsPerBlock = 128;

    std::cout << "[CUDA Benchmark] projectPoints with cudaHostAllocMapped + TPC masks\n";
    std::cout << std::fixed << std::setprecision(3);

    // ---------------- Enable mapped memory -----------------
    checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

    // ---------------- Host allocations (mapped) -------------
    float *h_input = nullptr, *h_output = nullptr;
    float *d_input = nullptr, *d_output = nullptr;

    checkCudaErrors(cudaHostAlloc(&h_input,  sizeof(float) * 3 * num_points, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&h_output, sizeof(float) * 2 * num_points, cudaHostAllocMapped));

    // Get device pointers that map to same host memory
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_input,  h_input,  0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&d_output, h_output, 0));

    // ---------------- Initialize input ----------------------
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
    for (int i = 0; i < 3 * num_points; ++i)
        h_input[i] = dis(gen);

    // Optional CPU compute (like before)
    auto primes = computePrimes(num_points);
    (void)primes;

    // ---------------- Stream -------------------------------
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    const int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<std::pair<int, float>> results;
    results.reserve(MAX_TPC + 1);

    // ---------------- Sweep over TPC masks ------------------
    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active)
    {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        std::cout << "\n[Active TPCs: " << std::setw(2) << tpc_active
                  << "]" << std::endl;

        std::vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run)
        {
            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));

            checkCudaErrors(cudaEventRecord(start, stream));
            projectPoints<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, num_points, focal_len);
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
        avg /= timings.size();

        std::cout << "Avg GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    // ---------------- Save JSON -----------------------------
    std::ofstream jsonFile(benchmark_log_path("tpc_results_projection"));
    jsonFile << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";
    jsonFile.close();

    std::cout << "[DONE] " << benchmark_log_path("tpc_results_projection") << std::endl;

    // ---------------- Cleanup -------------------------------
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFreeHost(h_input));
    checkCudaErrors(cudaFreeHost(h_output));

    std::cout << "taskFinish\n";
    return 0;
}
