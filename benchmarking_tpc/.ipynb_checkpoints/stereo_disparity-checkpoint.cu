#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cstdint>
#include <chrono>
#include "libsmctrl.h"  // for TPC mask control

using namespace std;

#define BLOCK_X 32
#define BLOCK_Y 8
#define RAD 8
#define STEPS 3
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

// ============================================================
// TPC Mask Builder
// ============================================================
static inline uint64_t build_mask(int active) {
    if (active <= 0) return 0xFFFFFFFFFFFFFFFEULL; // all disabled
    if (active >= 64) return 0ULL;                 // all enabled
    return ~((1ULL << active) - 1ULL);             // lower bits enabled
}

// ============================================================
// Utility: integer ceil division
// ============================================================
inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// ============================================================
// Device Helper: vector absdiff
// ============================================================
__device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C = 0)
{
    unsigned int result;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(result) : "r"(A), "r"(B), "r"(C));
    return result;
}

// ============================================================
// Stereo Disparity CUDA Kernel
// ============================================================
__global__ void stereoDisparityKernel(
    unsigned int *g_img0,
    unsigned int *g_img1,
    unsigned int *g_odata,
    int w, int h, int minDisp, int maxDisp,
    cudaTextureObject_t tex2Dleft,
    cudaTextureObject_t tex2Dright)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;

    const unsigned int sidx = threadIdx.x + RAD;
    const unsigned int sidy = threadIdx.y + RAD;

    __shared__ unsigned int diff[BLOCK_Y + 2*RAD][BLOCK_X + 2*RAD];

    unsigned int imLeftA[STEPS], imLeftB[STEPS];

    for (int i = 0; i < STEPS; i++) {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2D<unsigned int>(tex2Dleft, (float)(tidx - RAD), (float)(tidy + offset));
        imLeftB[i] = tex2D<unsigned int>(tex2Dleft, (float)(tidx - RAD + BLOCK_X), (float)(tidy + offset));
    }

    unsigned int bestCost = 0xFFFFFFFF;
    unsigned int bestDisparity = 0;

    for (int d = minDisp; d <= maxDisp; d++) {
        #pragma unroll
        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;
            unsigned int imLeft = imLeftA[i];
            unsigned int imRight = tex2D<unsigned int>(tex2Dright, (float)(tidx - RAD + d), (float)(tidy + offset));
            unsigned int cost = __usad4(imLeft, imRight);
            diff[sidy + offset][sidx - RAD] = cost;
        }

        #pragma unroll
        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;
            if (threadIdx.x < 2*RAD) {
                unsigned int imLeft = imLeftB[i];
                unsigned int imRight = tex2D<unsigned int>(tex2Dright, (float)(tidx - RAD + BLOCK_X + d), (float)(tidy + offset));
                unsigned int cost = __usad4(imLeft, imRight);
                diff[sidy + offset][sidx - RAD + BLOCK_X] = cost;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < STEPS; j++) {
            int offset = -RAD + j * RAD;
            unsigned int cost = 0;
            #pragma unroll
            for (int i = -RAD; i <= RAD; i++) cost += diff[sidy + offset][sidx + i];
            __syncthreads();
            diff[sidy + offset][sidx] = cost;
            __syncthreads();
        }

        unsigned int cost = 0;
        #pragma unroll
        for (int i = -RAD; i <= RAD; i++) cost += diff[sidy + i][sidx];

        if (cost < bestCost) { bestCost = cost; bestDisparity = d + 8; }
        __syncthreads();
    }

    if (tidy < h && tidx < w)
        g_odata[tidy * w + tidx] = bestDisparity;
}

// ============================================================
// Main Benchmark
// ============================================================
int main() {
    const int W = 960, H = 960;
    const int MIN_DISP = -16;
    const int MAX_DISP = 0;
    cudaDeviceProp prop;
    const int RUNS = 3;
    int device = 0; // Assuming device 0
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    int MAX_TPC = num_sms / 2;

    cout << "[CUDA Benchmark] Stereo Disparity (" << W << "x" << H << ")\n";
    cout << "Using libsmctrl TPC mask control (0–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    const size_t size = W * H * sizeof(unsigned int);

    // Allocate host memory
    vector<unsigned int> h_img0(W * H), h_img1(W * H), h_out(W * H);

    // Random test images
    mt19937 rng{random_device{}()};
    uniform_int_distribution<unsigned int> dist(0, 255);
    for (auto &p : h_img0) p = dist(rng);
    for (auto &p : h_img1) p = dist(rng);

    // Allocate device memory
    unsigned int *d_img0, *d_img1, *d_out;
    cudaMalloc(&d_img0, size);
    cudaMalloc(&d_img1, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_img0, h_img0.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img1, h_img1.data(), size, cudaMemcpyHostToDevice);

    // Create texture objects
    cudaChannelFormatDesc ca_desc = cudaCreateChannelDesc<unsigned int>();

    cudaResourceDesc texResLeft = {};
    texResLeft.resType = cudaResourceTypePitch2D;
    texResLeft.res.pitch2D.devPtr = d_img0;
    texResLeft.res.pitch2D.desc = ca_desc;
    texResLeft.res.pitch2D.width = W;
    texResLeft.res.pitch2D.height = H;
    texResLeft.res.pitch2D.pitchInBytes = W * sizeof(unsigned int);

    cudaTextureDesc texDescr = {};
    texDescr.normalizedCoords = 0;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex2Dleft, tex2Dright;
    checkCudaErrors(cudaCreateTextureObject(&tex2Dleft, &texResLeft, &texDescr, nullptr));

    cudaResourceDesc texResRight = texResLeft;
    texResRight.res.pitch2D.devPtr = d_img1;
    checkCudaErrors(cudaCreateTextureObject(&tex2Dright, &texResRight, &texDescr, nullptr));

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks(iDivUp(W, threads.x), iDivUp(H, threads.y));

    vector<pair<int, float>> results;

    for (int tpc_active = 0; tpc_active <= MAX_TPC; ++tpc_active) {
        uint64_t mask = build_mask(tpc_active);
        libsmctrl_set_global_mask(mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "] Mask = 0x" << hex << mask << dec << "\n";

        vector<float> timings;
        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            stereoDisparityKernel<<<blocks, threads>>>(
                d_img0, d_img1, d_out, W, H, MIN_DISP, MAX_DISP,
                tex2Dleft, tex2Dright
            );

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

        cout << "  Average GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

    // Save disparity output
    ofstream outFile("logs/output_stereodisparity.txt");
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            outFile << h_out[y * W + x] << " ";
        outFile << "\n";
    }
    outFile.close();

    // Save JSON benchmark results
    ofstream json("logs/tpc_results_stereodisparity");
    json << "{\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    cout << "\n[CUDA Benchmark] Results saved to logs/tpc_results_stereodisparity\n";

    cudaDestroyTextureObject(tex2Dleft);
    cudaDestroyTextureObject(tex2Dright);
    cudaFree(d_img0);
    cudaFree(d_img1);
    cudaFree(d_out);

    cout << "[CUDA Benchmark] Done.\n";
    return 0;
}

