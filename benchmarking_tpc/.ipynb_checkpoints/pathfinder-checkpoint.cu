// dp_smctrl_sweep_projection_style.cu
// Dynamic Programming 1D (pathfinder) benchmark with TPC mask sweep (1..41)

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include "libsmctrl.h"

#define BLOCK_SIZE 256
#define HALO 1
#ifndef RUNS
#define RUNS 10
#endif

// -----------------------------
// CUDA error check
// -----------------------------
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char *const func, const char *const file, int const line)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " code=" << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorName(result) << ") "
                  << "\"" << func << "\"" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// -----------------------------
// Utility macros
// -----------------------------
#define MINI(a,b) (((a) < (b)) ? (a) : (b))
#define MAXI(a,b) (((a) > (b)) ? (a) : (b))
#define IN_RANGE(x, lo, hi) ((x) >= (lo) && (x) <= (hi))

// -----------------------------
// Kernel: 1D DP per row segment (Rodinia-style)
// -----------------------------
__global__ void dynproc_kernel(int iteration,
                               const int* __restrict__ gpuWall,
                               const int* __restrict__ gpuSrc,
                               int* __restrict__ gpuDst,
                               int cols, int rows,
                               int startStep, int border)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int buf [BLOCK_SIZE];

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
    const int blkX    = small_block_cols * bx - border;
    const int blkXmax = blkX + BLOCK_SIZE - 1;
    const int xidx    = blkX + tx;

    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > cols - 1)
                        ? (BLOCK_SIZE - 1 - (blkXmax - cols + 1))
                        : (BLOCK_SIZE - 1);
    const int W = MAXI(tx - 1, validXmin);
    const int E = MINI(tx + 1, validXmax);
    const bool isValid = (tx >= validXmin && tx <= validXmax);

    if (xidx >= 0 && xidx < cols)
        prev[tx] = gpuSrc[xidx];
    __syncthreads();

    bool computed = false;

    for (int i = 0; i < iteration; ++i) {
        computed = false;
        if (isValid && tx >= (i + 1) && tx <= (BLOCK_SIZE - i - 2)) {
            int left  = prev[W];
            int up    = prev[tx];
            int right = prev[E];
            int shortest = MINI(MINI(left, up), right);
            int wallRow = startStep + i;
            int wallIdx = wallRow * cols + xidx;
            buf[tx] = shortest + gpuWall[wallIdx];
            computed = true;
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) prev[tx] = buf[tx];
        __syncthreads();
    }
    if (computed && xidx >= 0 && xidx < cols)
        gpuDst[xidx] = buf[tx];
}

// -----------------------------
// Mask builder (bit=1 disabled, bit=0 enabled)
// -----------------------------
static inline uint64_t build_mask_from_active_tpc(int active_tpc) {
    if (active_tpc <= 0)  return ~0ULL;  // disable all (no TPC)
    if (active_tpc >= 64) return 0ULL;   // enable all
    const uint64_t allowed = (1ULL << active_tpc) - 1ULL;
    return ~allowed;
}

// -----------------------------
// Run full DP computation
// -----------------------------
static int run_dp_pass(const int* d_wall,
                       int* d_result[2],
                       int rows, int cols,
                       int pyramid_height,
                       int blockCols, int borderCols,
                       cudaStream_t stream)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);
    int src = 1, dst = 0;

    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src; src = dst; dst = temp;
        int iters = MINI(pyramid_height, (rows - 1) - t);
        dynproc_kernel<<<dimGrid, dimBlock, 0, stream>>>(
            iters, d_wall, d_result[src], d_result[dst],
            cols, rows, t, borderCols);
    }
    return dst;
}

// -----------------------------
// Main
// -----------------------------
int main()
{
    const int cols = 4096;
    const int rows = 4096;
    const int pyramid_height = 16;  // smaller than rows to keep iterations valid
    cudaDeviceProp prop;
    if (cols <= 0 || rows <= 1 || pyramid_height <= 0) {
        std::cerr << "Invalid parameters: cols>0, rows>1, pyramid_height>0 required.\n";
        return 1;
    }

    const int borderCols    = pyramid_height * HALO;
    const int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
    const int blockCols     = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    const int size = rows * cols;
    std::vector<int> h_data(size);
    for (int i = 0; i < size; ++i) h_data[i] = std::rand() % 10;

    int *d_wall = nullptr, *d_result[2] = {nullptr, nullptr};
    checkCudaErrors(cudaMalloc((void**)&d_result[0], cols * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_result[1], cols * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_wall,     (size - cols) * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_result[0], h_data.data(), cols * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_wall, h_data.data() + cols, (size - cols) * sizeof(int), cudaMemcpyHostToDevice));

    int num_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    uint32_t num_tpcs = 0;
    if (libsmctrl_get_tpc_info_cuda(&num_tpcs, 0) != 0 || num_tpcs == 0) {
        std::cerr << "libsmctrl_get_tpc_info_cuda failed.\n";
        return 1;
    }

    std::cout << "GPU: " << num_sms << " SMs, " << num_tpcs << " TPCs\n";
    std::cout << "[DP Benchmark] Sweep TPC active = 1..41\n";
    std::cout << std::fixed << std::setprecision(3);

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    std::vector<std::pair<int, float>> results;
    results.reserve(41);

    // Sweep 1–41 TPCs
    for (int tpc_active = 1; tpc_active <= num_tpcs; ++tpc_active) {
        const uint64_t mask = build_mask_from_active_tpc(tpc_active);
        libsmctrl_set_global_mask(mask);

        std::cout << "\n[Active TPCs: " << std::setw(2) << tpc_active
                  << "] Mask = 0x" << std::hex << mask << std::dec << "\n";

        std::vector<float> timings;
        timings.reserve(RUNS);

        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));

            checkCudaErrors(cudaMemcpyAsync(d_result[0], h_data.data(),
                                            cols * sizeof(int), cudaMemcpyHostToDevice, stream));

            checkCudaErrors(cudaEventRecord(start, stream));
            (void)run_dp_pass(d_wall, d_result, rows, cols, pyramid_height, blockCols, borderCols, stream);
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

    // Save JSON
    std::ofstream jsonFile("logs/tpc_results_pathfinder");
    jsonFile << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        jsonFile << "    {\"TPC\": " << results[i].first
                 << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ]\n}\n";
    jsonFile.close();

    std::cout << "\n[DP Benchmark] Results saved to logs/tpc_results_pathfinder.json\n";

    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_wall));
    checkCudaErrors(cudaFree(d_result[0]));
    checkCudaErrors(cudaFree(d_result[1]));
    return 0;
}
