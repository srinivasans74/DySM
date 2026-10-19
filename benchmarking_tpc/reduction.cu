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

// Standalone TPC sweep for reduction.
// Kernel, input size and complete-job dispatch copied from application/reduction/reduction.cu.
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
template <class T> struct SharedMemory {
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
template <class T> __global__ void reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

} // namespace

int main()
{
    // Match the current application workload, including all repeats per job.
    const int task_id = 8;
    const int RUNS = 100;
    // Work per release; repetitions/time steps are included in the measured GPU time.
    constexpr unsigned int NUM_ELEMS = 2 * 1024 * 1024;
    constexpr unsigned int THREADS = 256;
    constexpr int REPEATS = 12;

    // Pinned input, device input, and two alternating arrays of block sums.
    float *h_in = nullptr, *d_in = nullptr, *d_partial_a = nullptr, *d_partial_b = nullptr;

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

    const size_t bytes = NUM_ELEMS * sizeof(float);
    // Host (pinned) + Device buffers
    benchmark_cuda_check(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    // Reproducible input matching this workload's 6.9 task ID.
    std::mt19937 gen(5489u + task_id);
    std::uniform_int_distribution<int> dist(0, 3);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = static_cast<float>(dist(gen));
    benchmark_cuda_check(cudaMalloc(&d_in, bytes));
    const size_t partial_bytes = ((NUM_ELEMS + THREADS - 1) / THREADS) * sizeof(float);
    benchmark_cuda_check(cudaMalloc(&d_partial_a, partial_bytes));
    benchmark_cuda_check(cudaMalloc(&d_partial_b, partial_bytes));
    benchmark_cuda_check(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  elements=" << NUM_ELEMS << " repeats=" << REPEATS << "\n";


    // Complete application job; transfers and setup are outside the timed region.
    const auto launch_job = [&]() {
        // Repeat a complete array-to-scalar reduction, using alternating scratch buffers.
        for (int repeat = 0; repeat < REPEATS; ++repeat) {
            unsigned int count = NUM_ELEMS;
            float *input = d_in;
            float *output = d_partial_a;
            // Each pass produces one sum per block; keep reducing until one value remains.
            do {
                const unsigned int blocks = (count + THREADS - 1) / THREADS;
                reduce0<float>
                    <<<blocks, THREADS, THREADS * sizeof(float), stream>>>(input, output, count);
                count = blocks;
                input = output;
                output = (output == d_partial_a) ? d_partial_b : d_partial_a;
            } while (count > 1);
        }
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

    const std::string path = benchmark_log_path("tpc_results_reduction");
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
    benchmark_cuda_check(cudaFree(d_in));
    benchmark_cuda_check(cudaFree(d_partial_a));
    benchmark_cuda_check(cudaFree(d_partial_b));
    benchmark_cuda_check(cudaFreeHost(h_in));
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
