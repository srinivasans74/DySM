#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cstdint>
#include <chrono>
#include "tpc_control.h"

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
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    cout << "[CUDA Benchmark] Stereo Disparity (" << W << "x" << H << ")\n";
    cout << "Using libsmctrl TPC mask control (1–" << MAX_TPC << ")\n";
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
    benchmark_cuda_check(cudaMalloc(&d_img0, size));
    benchmark_cuda_check(cudaMalloc(&d_img1, size));
    benchmark_cuda_check(cudaMalloc(&d_out, size));
    benchmark_cuda_check(cudaMemcpy(d_img0, h_img0.data(), size, cudaMemcpyHostToDevice));
    benchmark_cuda_check(cudaMemcpy(d_img1, h_img1.data(), size, cudaMemcpyHostToDevice));

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

    cudaStream_t stream;
    benchmark_cuda_check(cudaStreamCreate(&stream));

    for (int tpc_active = 1; tpc_active <= MAX_TPC; ++tpc_active) {
        const uint128_t mask = benchmark_tpc_mask(tpc_active);
        libsmctrl_set_stream_mask_ext(stream, mask);

        cout << "\n[Active TPCs: " << setw(2) << tpc_active
             << "]" << std::endl;

        vector<float> timings;
        for (int run = 0; run < RUNS; ++run) {
            cudaEvent_t start, stop;
            benchmark_cuda_check(cudaEventCreate(&start));
            benchmark_cuda_check(cudaEventCreate(&stop));

            benchmark_cuda_check(cudaEventRecord(start, stream));

            stereoDisparityKernel<<<blocks, threads, 0, stream>>>(
                d_img0, d_img1, d_out, W, H, MIN_DISP, MAX_DISP,
                tex2Dleft, tex2Dright
            );

            benchmark_cuda_check(cudaGetLastError());

            benchmark_cuda_check(cudaEventRecord(stop, stream));
            benchmark_cuda_check(cudaEventSynchronize(stop));

            float ms = 0.0f;
            benchmark_cuda_check(cudaEventElapsedTime(&ms, start, stop));
            timings.push_back(ms);

            benchmark_cuda_check(cudaEventDestroy(start));
            benchmark_cuda_check(cudaEventDestroy(stop));
        }

        float avg = 0.0f;
        for (auto t : timings) avg += t;
        avg /= timings.size();

        cout << "  Average GPU time (" << RUNS << " runs) = " << avg << " ms\n";
        results.emplace_back(tpc_active, avg);
    }

    benchmark_cuda_check(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    // Save disparity output
    ofstream outFile(benchmark_log_path("output_stereodisparity.txt"));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            outFile << h_out[y * W + x] << " ";
        outFile << "\n";
    }
    outFile.close();

    // Save JSON benchmark results
    ofstream json(benchmark_log_path("tpc_results_stereodisparity"));
    json << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    std::cout << "[DONE] " << benchmark_log_path("tpc_results_stereodisparity") << std::endl;

    benchmark_cuda_check(cudaDestroyTextureObject(tex2Dleft));
    benchmark_cuda_check(cudaDestroyTextureObject(tex2Dright));
    benchmark_cuda_check(cudaFree(d_img0));
    benchmark_cuda_check(cudaFree(d_img1));
    benchmark_cuda_check(cudaFree(d_out));

    cout << "[CUDA Benchmark] Done.\n";
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}

