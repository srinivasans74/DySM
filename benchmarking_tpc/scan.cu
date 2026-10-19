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

// Standalone TPC sweep for scan.
// Kernel, input size and complete-job dispatch copied from application/scan/scan.cu.
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
using uint = unsigned int;
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation
// and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size,
                                      cg::thread_block cta)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size,
                                      cg::thread_block cta)
{
    return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size,
                                       cg::thread_block cta)
{
    // Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    // Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size,
                                       cg::thread_block cta)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    uint4 idata4 = d_Src[pos];

    // Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

    // Write back
    d_Dst[pos] = odata4;
}

// Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src, uint N,
                                     uint arrayLength)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    // Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load top elements
    // Convert results of bottom-level scan back to inclusive
    uint idata = 0;

    if (pos < N)
        idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
                d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    // Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength, cta);

    // Avoid out-of-bound access
    if (pos < N) {
        d_Buf[pos] = odata;
    }
}

// Final step of large-array scan: combine basic inclusive scan with exclusive
// scan of top elements of input arrays
__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        buf = d_Buffer[blockIdx.x];
    }

    cg::sync(cta);

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

} // namespace

int main()
{
    // Match the current application workload, including all repeats per job.
    const int task_id = 9;
    const int RUNS = 100;
    // Work per release; repetitions/time steps are included in the measured GPU time.
    // Each array has 256 blocks of 1024 values; prefixes restart at each array.
    constexpr unsigned int ARRAY_LENGTH = 256 * 1024;
    constexpr unsigned int BATCH_SIZE = 8;
    constexpr unsigned int NUM_ELEMS = ARRAY_LENGTH * BATCH_SIZE;
    constexpr int REPEATS = 12;

    // Pinned input, device input/output, and one offset per bottom-level block.
    unsigned int *h_in = nullptr, *d_in = nullptr, *d_out = nullptr, *d_block_sums = nullptr;

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

    const size_t bytes = NUM_ELEMS * sizeof(unsigned int);
    // Host (pinned) + Device buffers
    benchmark_cuda_check(cudaHostAlloc(&h_in, bytes, cudaHostAllocDefault));
    // Reproducible input matching this workload's 6.9 task ID.
    std::mt19937 gen(5489u + task_id);
    std::uniform_int_distribution<unsigned int> dist(0, 15);
    for (unsigned int i = 0; i < NUM_ELEMS; ++i)
        h_in[i] = dist(gen);
    benchmark_cuda_check(cudaMalloc(&d_in, bytes));
    benchmark_cuda_check(cudaMalloc(&d_out, bytes));
    benchmark_cuda_check(
        cudaMalloc(&d_block_sums, (NUM_ELEMS / (4 * THREADBLOCK_SIZE)) * sizeof(unsigned int)));
    benchmark_cuda_check(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    std::cout << "  arrays=" << BATCH_SIZE << " array_length=" << ARRAY_LENGTH
              << " repeats=" << REPEATS << "\n";


    // Complete application job; transfers and setup are outside the timed region.
    const auto launch_job = [&]() {
        // Scan each block, scan block totals within each array, then add block offsets.
        const unsigned int blocks = NUM_ELEMS / (4 * THREADBLOCK_SIZE);
        for (int repeat = 0; repeat < REPEATS; ++repeat) {
            // 1. Exclusive scan within each 1024-value block.
            scanExclusiveShared<<<blocks, THREADBLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<uint4 *>(d_out), reinterpret_cast<uint4 *>(d_in),
                4 * THREADBLOCK_SIZE);
            // 2. Exclusive scan of block totals, separately for each input array.
            scanExclusiveShared2<<<(blocks + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE, THREADBLOCK_SIZE,
                                   0, stream>>>(d_block_sums, d_out, d_in, blocks,
                                                ARRAY_LENGTH / (4 * THREADBLOCK_SIZE));
            // 3. Add the preceding-block sum to every element of each block.
            uniformUpdate<<<blocks, THREADBLOCK_SIZE, 0, stream>>>(reinterpret_cast<uint4 *>(d_out),
                                                                   d_block_sums);
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

    const std::string path = benchmark_log_path("tpc_results_scan");
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
    benchmark_cuda_check(cudaFree(d_out));
    benchmark_cuda_check(cudaFree(d_block_sums));
    benchmark_cuda_check(cudaFreeHost(h_in));
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
