#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include "libsmctrl.h"

#define SAFE(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define RANGE_MASK(lo, hi) (((1ULL << ((hi) - (lo) + 1)) - 1ULL) << (lo))

// -----------------------------------------------------------------------------
// Kernel that records which SMs execute blocks
// -----------------------------------------------------------------------------
__global__ void probe_kernel(uint8_t *smid_log) {
    if (threadIdx.x == 0 && blockIdx.x < 256) {
        unsigned smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        smid_log[blockIdx.x] = (uint8_t)smid;
    }
}

// -----------------------------------------------------------------------------
// Utility to test a given TPC range and measure execution time
// -----------------------------------------------------------------------------
void test_tpc_mask(cudaStream_t stream,
                   int tpc_lo, int tpc_hi,
                   int num_tpcs, int sms_per_tpc, int num_sms,
                   uint8_t *smid_d, uint8_t *smid_h)
{
    printf("\n=== Testing TPC[%d - %d] (SM[%d - %d]) ===\n",
           tpc_lo, tpc_hi,
           tpc_lo * sms_per_tpc,
           (tpc_hi + 1) * sms_per_tpc - 1);

    if (tpc_lo < 0 || tpc_hi >= num_tpcs) {
        printf(" Invalid TPC range (available TPCs: 0 - %d)\n", num_tpcs - 1);
        return;
    }

    uint64_t allowed = RANGE_MASK(tpc_lo, tpc_hi);
    uint64_t mask = ~allowed;   // 1 = disable, 0 = enable
    printf("Mask = 0x%016llx\n", (unsigned long long)mask);

    libsmctrl_set_stream_mask(stream, mask);

    auto start = std::chrono::high_resolution_clock::now();
    probe_kernel<<<256, 32, 0, stream>>>(smid_d);
    SAFE(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    SAFE(cudaMemcpy(smid_h, smid_d, 256, cudaMemcpyDeviceToHost));

    int seen[128] = {0}, count = 0;
    for (int i = 0; i < 256; i++)
        seen[smid_h[i]] = 1;
    for (int i = 0; i < num_sms; i++)
        if (seen[i]) count++;

    printf("Active SMs: %d\nIDs: ", count);
    for (int i = 0; i < num_sms; i++)
        if (seen[i]) printf("%d ", i);
    printf("\n");

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Execution time: %.3f ms\n", ms);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    int dev = 0;
    SAFE(cudaSetDevice(dev));

    // ------------------------------------------------------------
    // Query GPU topology
    // ------------------------------------------------------------
    uint32_t num_tpcs = 0;
    if (libsmctrl_get_tpc_info(&num_tpcs, dev) != 0) {
        fprintf(stderr, "libsmctrl_get_tpc_info() failed.\n");
        return -1;
    }

    int num_sms = 0;
    SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));

    if (num_tpcs == 0 || num_sms == 0) {
        fprintf(stderr, "Failed to query GPU topology properly.\n");
        return -1;
    }

    int sms_per_tpc = num_sms / num_tpcs;
    printf("Detected %u TPCs (%d SMs total, %d SMs/TPC)\n\n",
           num_tpcs, num_sms, sms_per_tpc);

    // ------------------------------------------------------------
    // Allocate resources
    // ------------------------------------------------------------
    uint8_t *smid_d, *smid_h;
    SAFE(cudaMalloc(&smid_d, 256));
    smid_h = (uint8_t*)malloc(256);
    cudaStream_t stream;
    SAFE(cudaStreamCreate(&stream));

    // ------------------------------------------------------------
    // Test different partitions
    // ------------------------------------------------------------
    test_tpc_mask(stream, 0, 0, num_tpcs, sms_per_tpc, num_sms, smid_d, smid_h);   // only TPC0
    test_tpc_mask(stream, 0, 1, num_tpcs, sms_per_tpc, num_sms, smid_d, smid_h);   // 2 TPCs
    test_tpc_mask(stream, 0, 20, num_tpcs, sms_per_tpc, num_sms, smid_d, smid_h);  // ~half GPU
    test_tpc_mask(stream, 0, num_tpcs - 1, num_tpcs, sms_per_tpc, num_sms, smid_d, smid_h); // all TPCs

    // ------------------------------------------------------------
    // Disable all TPCs
    // ------------------------------------------------------------
    printf("\n=== Testing mask = ~0ULL (disables all TPCs) ===\n");
    libsmctrl_set_stream_mask(stream, ~0ULL);

    probe_kernel<<<256, 32, 0, stream>>>(smid_d);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        printf("Expected hang or failure (no TPCs enabled): %s\n",
               cudaGetErrorString(err));
    else
        printf("Unexpected success — some SMs still active?\n");

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    SAFE(cudaStreamDestroy(stream));
    SAFE(cudaFree(smid_d));
    free(smid_h);

    return 0;
}