// stereo.cu — MATCHED to your “working” structure, BUT:
// A) Uses steady_clock + sleep_until (no clock_nanosleep / MONOTONIC_RAW helpers)
// B) Keeps termination condition: while (gpu_running.load(...))
// C) Uses explicit cudaMemcpyAsync + cudaStreamSynchronize (NOT mapped host memory)
// D) No hostalloc *mapped pointer* usage (no cudaHostGetDevicePointer, no cudaDeviceMapHost flag needed)
// E) Keeps your texture path + kernel as-is
//
// Metrics (same style as your matrixMul example):
// - ET: kernel-only time via CUDA events
// - RT: (finish - actual_release) taken immediately after sleep_until returns

#include <cuda_runtime.h>
#include <iostream>
#include <atomic>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <random>

#include "../../libsmctrl/libsmctrl.h"
#include "../../Scheduler/Parameters.h"

extern std::atomic<bool> gpu_running;

using steady_clock = std::chrono::steady_clock;

// ============================================================
// Utility
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call);  \
    if (_e != cudaSuccess) { \
        std::cerr << "[CUDA] " << #call << " failed: " \
                  << cudaGetErrorString(_e) << " (" << (int)_e << ")\n"; \
        std::terminate(); \
    } \
} while (0)

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

// ============================================================
// CUDA Kernel — Stereo Disparity (UNCHANGED)
// ============================================================
#define BLOCK_X 8
#define BLOCK_Y 8
#define RAD 8
#define STEPS 3

__device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C = 0)
{
    unsigned int result;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;" : "=r"(result) : "r"(A), "r"(B), "r"(C));
    return result;
}

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
// StereoDisparity class — matched to matrixMul structure
// ============================================================
class StereoDisparity {
public:
    StereoDisparity(int id, parameters* p_)
        : task_id(id), p(p_),
          W(480), H(480),
          MIN_DISP(-16), MAX_DISP(0),
          h_img0(nullptr), h_img1(nullptr), h_out(nullptr),
          d_img0(nullptr), d_img1(nullptr), d_out(nullptr),
          tex2Dleft(0), tex2Dright(0),
          stream(nullptr),
          ev_start(nullptr), ev_stop(nullptr)
    {}

    void taskInit();
    void runContinuous();   // runs until gpu_running == false
    void taskFinish();

private:
    void kernelOnceMeasure(const steady_clock::time_point& actual_release_tp);
    void prepare_images();

    int task_id;
    parameters* p;

    const int W, H;
    const int MIN_DISP, MAX_DISP;

    unsigned int *h_img0, *h_img1, *h_out;
    unsigned int *d_img0, *d_img1, *d_out;

    cudaTextureObject_t tex2Dleft, tex2Dright;

    cudaStream_t stream;
    cudaEvent_t  ev_start, ev_stop;
    dim3 grid, block;
};

// ============================================================
// Input preparation
// ============================================================
void StereoDisparity::prepare_images()
{
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<unsigned int> dist(0, 255);
    const size_t size = (size_t)W * (size_t)H;
    for (size_t i = 0; i < size; ++i) {
        h_img0[i] = dist(rng);
        h_img1[i] = dist(rng);
        h_out[i]  = 0;
    }
}

// ============================================================
// Init
// ============================================================
void StereoDisparity::taskInit()
{
    std::cout << "[StereoDisparity] Init Task " << task_id << " (" << W << "x" << H << ")\n";

    CUDA_CHECK(cudaStreamCreate(&stream));
    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));

    const size_t elems = (size_t)W * (size_t)H;
    const size_t bytes = elems * sizeof(unsigned int);

    // Host buffers: pinned (good for memcpyAsync), NOT mapped
    CUDA_CHECK(cudaHostAlloc((void**)&h_img0, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_img1, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_out,  bytes, cudaHostAllocDefault));

    prepare_images();

    // Device buffers
    CUDA_CHECK(cudaMalloc((void**)&d_img0, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_img1, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out,  bytes));

    // Initial H2D
    CUDA_CHECK(cudaMemcpyAsync(d_img0, h_img0, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_img1, h_img1, bytes, cudaMemcpyHostToDevice, stream));

    // Texture setup (pitch2D) from DEVICE pointers (no host mapping)
    cudaChannelFormatDesc ca_desc = cudaCreateChannelDesc<unsigned int>();

    cudaResourceDesc texResLeft{};
    texResLeft.resType = cudaResourceTypePitch2D;
    texResLeft.res.pitch2D.devPtr = d_img0;
    texResLeft.res.pitch2D.desc = ca_desc;
    texResLeft.res.pitch2D.width = W;
    texResLeft.res.pitch2D.height = H;
    texResLeft.res.pitch2D.pitchInBytes = (size_t)W * sizeof(unsigned int);

    cudaTextureDesc texDescr{};
    texDescr.normalizedCoords = 0;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    CUDA_CHECK(cudaCreateTextureObject(&tex2Dleft, &texResLeft, &texDescr, nullptr));

    cudaResourceDesc texResRight = texResLeft;
    texResRight.res.pitch2D.devPtr = d_img1;
    CUDA_CHECK(cudaCreateTextureObject(&tex2Dright, &texResRight, &texDescr, nullptr));

    block = dim3(BLOCK_X, BLOCK_Y);
    grid  = dim3(iDivUp(W, block.x), iDivUp(H, block.y));

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm-up: (already H2D'd) + kernel + sync
    stereoDisparityKernel<<<grid, block, 0, stream>>>(
        d_img0, d_img1, d_out, W, H, MIN_DISP, MAX_DISP, tex2Dleft, tex2Dright);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "[StereoDisparity] Task " << task_id << " initialized.\n";
}

// ============================================================
// One job (ET + RT) — “working code style”
//   RT is measured from ACTUAL release (after sleep_until returns).
// ============================================================
void StereoDisparity::kernelOnceMeasure(const steady_clock::time_point& actual_release_tp)
{
    // (Optional) if images change each period, you can update host then H2D here.
    // For now, keep like your matrixMul example: kernel-only each period.
    // If you need per-period H2D, uncomment:
    // const size_t bytes = (size_t)W * (size_t)H * sizeof(unsigned int);
    // CUDA_CHECK(cudaMemcpyAsync(d_img0, h_img0, bytes, cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_img1, h_img1, bytes, cudaMemcpyHostToDevice, stream));

    auto finish_tp = steady_clock::now();
    float rt_ms = std::chrono::duration<float, std::milli>(finish_tp - actual_release_tp).count();

    libsmctrl_set_stream_mask_ext(stream, p->getMask(task_id));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    stereoDisparityKernel<<<grid, block, 0, stream>>>(
        d_img0, d_img1, d_out, W, H, MIN_DISP, MAX_DISP, tex2Dleft, tex2Dright);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float et_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&et_ms, ev_start, ev_stop));

    std::lock_guard<std::mutex> lk(p->task_mutexes[task_id]);
    p->executiontimes[task_id] = et_ms;
    p->responsetimes[task_id]  = et_ms + rt_ms;
}

// ============================================================
// Periodic loop — MATCHED structure:
// - steady_clock
// - sleep_until(next_release)
// - while(gpu_running.load())
// - update next_release using current p->period[task_id] (seconds) each iteration
// ============================================================
void StereoDisparity::runContinuous()
{
    auto next_release = steady_clock::now();

    while (gpu_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_until(next_release);
        auto actual_release = steady_clock::now();

        kernelOnceMeasure(actual_release);

        double ps = p->period[task_id];
        long long period_us = (long long)llround(ps * 1e6);
        next_release += std::chrono::microseconds(period_us);
    }
}

// ============================================================
// Cleanup
// ============================================================
void StereoDisparity::taskFinish()
{
    std::cout << "[StereoDisparity] Cleaning Task " << task_id << "\n";

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    if (tex2Dleft)  CUDA_CHECK(cudaDestroyTextureObject(tex2Dleft));
    if (tex2Dright) CUDA_CHECK(cudaDestroyTextureObject(tex2Dright));

    if (d_img0) CUDA_CHECK(cudaFree(d_img0));
    if (d_img1) CUDA_CHECK(cudaFree(d_img1));
    if (d_out)  CUDA_CHECK(cudaFree(d_out));

    if (h_img0) CUDA_CHECK(cudaFreeHost(h_img0));
    if (h_img1) CUDA_CHECK(cudaFreeHost(h_img1));
    if (h_out)  CUDA_CHECK(cudaFreeHost(h_out));

    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================
// Public API (C linkage)
// ============================================================
extern "C" {

void* stereo_create(int task_id, parameters* p) {
    auto* s = new StereoDisparity(task_id, p);
    s->taskInit();
    return s;
}

void stereo_run(void* handle) {
    static_cast<StereoDisparity*>(handle)->runContinuous();
}

void stereo_destroy(void* handle) {
    auto* s = static_cast<StereoDisparity*>(handle);
    s->taskFinish();
    delete s;
}

} // extern "C"