/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Standalone TPC sweep for blackscholes.
// Kernel, input size and complete-job dispatch copied from application/blackscholes/blackscholes.cu.
// Original kernel provenance is documented in that application source.
// Keeps the existing benchmark JSON format for fit_coefficients.py.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "tpc_control.h"

namespace
{
namespace cg = cooperative_groups;
__device__ inline float cndGPU(float d)
{
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float cnd =
        RSQRT2PI * __expf(-0.5f * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
////////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(float &CallResult, float &PutResult,
                                           float S, // Stock price
                                           float X, // Option strike
                                           float T, // Option years
                                           float R, // Riskless rate
                                           float V  // Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    // Calculate Call and Put simultaneously
    expRT = __expf(-R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128) __global__
    void BlackScholesGPU(float2 *__restrict d_CallResult, float2 *__restrict d_PutResult,
                         float2 *__restrict d_StockPrice, float2 *__restrict d_OptionStrike,
                         float2 *__restrict d_OptionYears, float Riskfree, float Volatility,
                         int optN)
{
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculating 2 options per thread to increase ILP (instruction level
    // parallelism)
    if (opt < (optN / 2)) {
        float callResult1, callResult2;
        float putResult1, putResult2;
        BlackScholesBodyGPU(callResult1, putResult1, d_StockPrice[opt].x, d_OptionStrike[opt].x,
                            d_OptionYears[opt].x, Riskfree, Volatility);
        BlackScholesBodyGPU(callResult2, putResult2, d_StockPrice[opt].y, d_OptionStrike[opt].y,
                            d_OptionYears[opt].y, Riskfree, Volatility);
        d_CallResult[opt] = make_float2(callResult1, callResult2);
        d_PutResult[opt] = make_float2(putResult1, putResult2);
    }
}

} // namespace

int main()
{
    // Match the current application workload, including all repeats per job.
    const int task_id = 7;
    const int RUNS = 100;
    // Work per release; repetitions/time steps are included in the measured GPU time.
    constexpr int NUM_OPTIONS = 2 * 1024 * 1024;
    constexpr int REPEATS = 20;

    // Pinned stock/strike/expiry inputs; device inputs and separate call/put outputs.
    float *h_stock = nullptr, *h_strike = nullptr, *h_years = nullptr;
    float *d_stock = nullptr, *d_strike = nullptr, *d_years = nullptr;
    float *d_call = nullptr, *d_put = nullptr;

    benchmark_cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop;
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, 0));
    const int total_tpcs = benchmark_tpc_count();
    std::cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount
              << " SMs, " << total_tpcs << " TPCs)\n";
    cudaStream_t stream;
    cudaEvent_t st, sp;
    benchmark_cuda_check(cudaStreamCreate(&stream));
    benchmark_cuda_check(cudaEventCreate(&st));
    benchmark_cuda_check(cudaEventCreate(&sp));

    const size_t bytes = NUM_OPTIONS * sizeof(float);
    // Host (pinned) + Device buffers
    benchmark_cuda_check(cudaHostAlloc(&h_stock, bytes, cudaHostAllocDefault));
    benchmark_cuda_check(cudaHostAlloc(&h_strike, bytes, cudaHostAllocDefault));
    benchmark_cuda_check(cudaHostAlloc(&h_years, bytes, cudaHostAllocDefault));
    // Reproducible input matching this workload's 6.9 task ID.
    std::mt19937 gen(5489u + task_id);
    std::uniform_real_distribution<float> stocks(5.0f, 30.0f), strikes(1.0f, 100.0f),
        years(0.25f, 10.0f);
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        h_stock[i] = stocks(gen);
        h_strike[i] = strikes(gen);
        h_years[i] = years(gen);
    }
    benchmark_cuda_check(cudaMalloc(&d_stock, bytes));
    benchmark_cuda_check(cudaMalloc(&d_strike, bytes));
    benchmark_cuda_check(cudaMalloc(&d_years, bytes));
    benchmark_cuda_check(cudaMalloc(&d_call, bytes));
    benchmark_cuda_check(cudaMalloc(&d_put, bytes));
    benchmark_cuda_check(cudaMemcpyAsync(d_stock, h_stock, bytes, cudaMemcpyHostToDevice, stream));
    benchmark_cuda_check(cudaMemcpyAsync(d_strike, h_strike, bytes, cudaMemcpyHostToDevice, stream));
    benchmark_cuda_check(cudaMemcpyAsync(d_years, h_years, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  options=" << NUM_OPTIONS << " repeats=" << REPEATS << "\n";


    // Complete application job; transfers and setup are outside the timed region.
    const auto launch_job = [&]() {
        // Price all options REPEATS times; every pass reads the same input.
        // float2 packs two options per thread; risk-free rate = 0.02, volatility = 0.30.
        for (int i = 0; i < REPEATS; ++i)
            BlackScholesGPU<<<(NUM_OPTIONS / 2 + 127) / 128, 128, 0, stream>>>(
                reinterpret_cast<float2 *>(d_call), reinterpret_cast<float2 *>(d_put),
                reinterpret_cast<float2 *>(d_stock), reinterpret_cast<float2 *>(d_strike),
                reinterpret_cast<float2 *>(d_years), 0.02f, 0.30f, NUM_OPTIONS);
    };
    benchmark_cuda_check(cudaStreamSynchronize(stream));

    std::vector<std::pair<int, double>> results;
    results.reserve(total_tpcs);
    for (int tpcs = 1; tpcs <= total_tpcs; ++tpcs) {
        libsmctrl_set_stream_mask_ext(stream, benchmark_tpc_mask(tpcs));
        // Warm up each new mask before collecting the same 100-job mean.
        launch_job();
        benchmark_cuda_check(cudaGetLastError());
        benchmark_cuda_check(cudaStreamSynchronize(stream));

        double total_ms = 0.0;
        for (int run = 0; run < RUNS; ++run) {
            benchmark_cuda_check(cudaEventRecord(st, stream));
            launch_job();
            benchmark_cuda_check(cudaGetLastError());
            benchmark_cuda_check(cudaEventRecord(sp, stream));
            benchmark_cuda_check(cudaEventSynchronize(sp));
            float ms = 0.0f;
            benchmark_cuda_check(cudaEventElapsedTime(&ms, st, sp));
            total_ms += ms;
        }
        const double mean_ms = total_ms / RUNS;
        results.emplace_back(tpcs, mean_ms);
        std::cout << "[Active TPCs: " << tpcs << "] Avg GPU job time = "
                  << std::setprecision(9) << mean_ms << " ms" << std::endl;
    }

    const std::string path = benchmark_log_path("tpc_results_blackscholes");
    std::ofstream json(path);
    if (!json) {
        std::cerr << "Cannot open result file: " << path << "\n";
        return EXIT_FAILURE;
    }
    json << std::setprecision(9)
         << "{\n  \"total_tpcs\": " << total_tpcs << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}"
             << (i + 1 == results.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    json.close();
    if (!json) {
        std::cerr << "Failed to write result file: " << path << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "[DONE] " << path << "\n";

    // Wait for queued GPU work before freeing its buffers and stream.
    benchmark_cuda_check(cudaStreamSynchronize(stream));
    benchmark_cuda_check(cudaEventDestroy(st));
    benchmark_cuda_check(cudaEventDestroy(sp));
    benchmark_cuda_check(cudaFree(d_stock));
    benchmark_cuda_check(cudaFree(d_strike));
    benchmark_cuda_check(cudaFree(d_years));
    benchmark_cuda_check(cudaFree(d_call));
    benchmark_cuda_check(cudaFree(d_put));
    benchmark_cuda_check(cudaFreeHost(h_stock));
    benchmark_cuda_check(cudaFreeHost(h_strike));
    benchmark_cuda_check(cudaFreeHost(h_years));
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
