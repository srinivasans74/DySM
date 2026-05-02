#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>

#include "libsmctrl.h"  // for TPC mask control
#include "common/include/support.h"

using namespace std;

// -------------------------------------------------------------
// Config
// -------------------------------------------------------------
#define NX 480
#define NY 480
#define NZ 128
#define BSIZE 16
#define XTILE 32
#define HEAVY 4
#define REPEAT_COUNT 10

// -------------------------------------------------------------
// Error checking
// -------------------------------------------------------------
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        cerr << "[CUDA] " << func << " failed at " << file << ":" << line
             << " (" << cudaGetErrorName(result) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// -------------------------------------------------------------
// Mask builder
// -------------------------------------------------------------
static inline uint64_t build_mask(int active) {
    if (active >= 64) return 0ULL;
    return ~((1ULL << active) - 1ULL);
}

// -------------------------------------------------------------
// Kernel
// -------------------------------------------------------------
__global__ void stencil3d(const float *psi, float *npsi,
                          const float *sigmaX, const float *sigmaY, const float *sigmaZ,
                          int nx, int ny, int nz) {
    __shared__ float tile[4][BSIZE][BSIZE];

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int ox = XTILE * blockIdx.x;
    int oy = (BSIZE - 2) * blockIdx.y;
    int oz = (BSIZE - 2) * blockIdx.z;

    int maxX = nx - 1 - ox;
    int maxY = ny - 1 - oy;
    int maxZ = nz - 1 - oz;
    int nLast_x = (XTILE + 1 < maxX ? XTILE + 1 : maxX);
    int nLast_y = (BSIZE - 1 < maxY ? BSIZE - 1 : maxY);
    int nLast_z = (BSIZE - 1 < maxZ ? BSIZE - 1 : maxZ);
    if (tj > nLast_y || tk > nLast_z) return;

    auto idx = [=] __device__ (int x, int y, int z) {
        return (size_t)x * ny * nz + (size_t)y * nz + (size_t)z;
    };

    tile[1][tj][tk] = psi[idx(ox + 0, oy + tj, oz + tk)];
    tile[2][tj][tk] = psi[idx(ox + 1, oy + tj, oz + tk)];
    __syncthreads();

    int pi = 0, ci = 1, ni = 2, ti;
    for (int rep = 0; rep < HEAVY; rep++) {
        for (int i = 1; i < nLast_x; ++i) {
            tile[ni][tj][tk] = psi[idx(ox + i + 1, oy + tj, oz + tk)];
            __syncthreads();

            float c = tile[ci][tj][tk];
            float l = tile[pi][tj][tk > 0 ? tk - 1 : 0];
            float r = tile[pi][tj][tk < nLast_z ? tk + 1 : nLast_z];
            float sum = c + l + r
                      + sigmaX[idx(ox + i, oy + tj, oz + tk)]
                      + sigmaY[idx(ox + i, oy + tj, oz + tk)]
                      + sigmaZ[idx(ox + i, oy + tj, oz + tk)];

            npsi[idx(ox + i, oy + tj, oz + tk)] = sum;

            __syncthreads();
            ti = pi; pi = ci; ci = ni; ni = ti;
        }
    }
}

// -------------------------------------------------------------
// Main benchmark
// -------------------------------------------------------------
int main() {
    cout << "[CUDA Benchmark] 3D Stencil with libsmctrl TPC mask sweep\n";
    cout << fixed << setprecision(3);

    // --- Device setup ---
    int num_sms = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    uint32_t num_tpcs = 0;
    libsmctrl_get_tpc_info_cuda(&num_tpcs, 0);
    cout << "Detected " << num_sms << " SMs across " << num_tpcs << " TPCs\n";

    // --- Allocate memory ---
    size_t vol = (size_t)NX * NY * NZ;
    size_t bytes = vol * sizeof(float);

    vector<float> h_psi(vol, 1.0f), h_sigX(vol, 0.1f), h_sigY(vol, 0.2f), h_sigZ(vol, 0.3f);
    float *d_psi, *d_npsi, *d_sigX, *d_sigY, *d_sigZ;
    checkCudaErrors(cudaMalloc(&d_psi, bytes));
    checkCudaErrors(cudaMalloc(&d_npsi, bytes));
    checkCudaErrors(cudaMalloc(&d_sigX, bytes));
    checkCudaErrors(cudaMalloc(&d_sigY, bytes));
    checkCudaErrors(cudaMalloc(&d_sigZ, bytes));
    checkCudaErrors(cudaMemcpy(d_psi, h_psi.data(), bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sigX, h_sigX.data(), bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sigY, h_sigY.data(), bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sigZ, h_sigZ.data(), bytes, cudaMemcpyHostToDevice));

    // --- Kernel geometry ---
    int bdimx = (NX - 2) / XTILE + ((NX - 2) % XTILE != 0);
    int bdimy = (NY - 2) / (BSIZE - 2) + ((NY - 2) % (BSIZE - 2) != 0);
    int bdimz = (NZ - 2) / (BSIZE - 2) + ((NZ - 2) % (BSIZE - 2) != 0);
    dim3 gridDim(bdimx, bdimy, bdimz);
    dim3 blockDim(BSIZE, BSIZE);

    // --- Sweep TPC masks ---
    vector<pair<int, float>> results;

    for (int tpc_active = 1; tpc_active <= num_tpcs; ++tpc_active) {
        uint64_t mask = build_mask(tpc_active);
        libsmctrl_set_global_mask(mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "] Mask = 0x" << hex << mask << dec << "\n";

        vector<float> timings;
        for (int run = 0; run < REPEAT_COUNT; ++run) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            stencil3d<<<gridDim, blockDim>>>(d_psi, d_npsi, d_sigX, d_sigY, d_sigZ, NX, NY, NZ);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            timings.push_back(ms);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float avg = 0.0f;
        for (auto t : timings) avg += t;
        avg /= timings.size();

        cout << "  Average GPU time (" << REPEAT_COUNT << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    // --- Save results ---
    ofstream json("logs/tpc_results_stencil");
    json << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    cout << "\n[CUDA Benchmark] Results saved to logs/tpc_results_stencil.json\n";

    // --- Cleanup ---
    cudaFree(d_psi);
    cudaFree(d_npsi);
    cudaFree(d_sigX);
    cudaFree(d_sigY);
    cudaFree(d_sigZ);
    cout << "[CUDA Benchmark] Done.\n";
    return 0;
}
