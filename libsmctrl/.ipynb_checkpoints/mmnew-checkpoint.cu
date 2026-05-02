#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"

// Helper macro for error checking
#define SAFE(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel to perform dummy matrix multiply and record SM IDs
__global__ void mm_kernel(const float *A, const float *B, float *C,
                          int N, uint8_t *smid_arr) {
    // Record SM ID only once per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        smid_arr[idx] = (uint8_t)smid;
    }

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Make a range mask (enable bits [lo, hi])
#define RANGE_MASK(lo, hi) (((1ULL << ((hi) - (lo) + 1)) - 1ULL) << (lo))

// Count unique SM IDs in array
int count_unique(uint8_t *arr, int len) {
    int count = 0;
    uint8_t seen[256] = {0};
    for (int i = 0; i < len; i++)
        seen[arr[i]] = 1;
    for (int i = 0; i < 256; i++)
        if (seen[i]) count++;
    return count;
}

int main() {
    const int N = 256;
    const size_t bytes = N * N * sizeof(float);

    // Host and device memory
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    SAFE(cudaMalloc(&dA, bytes));
    SAFE(cudaMalloc(&dB, bytes));
    SAFE(cudaMalloc(&dC, bytes));
    SAFE(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    int grid_x = (N + 15) / 16, grid_y = (N + 15) / 16;
    dim3 grid(grid_x, grid_y);
    dim3 block(16, 16);
    int num_blocks = grid_x * grid_y;

    // Allocate SMID arrays
    uint8_t *smidA_d, *smidA_h, *smidB_d, *smidB_h;
    SAFE(cudaMalloc(&smidA_d, num_blocks));
    SAFE(cudaMalloc(&smidB_d, num_blocks));
    smidA_h = (uint8_t*)malloc(num_blocks);
    smidB_h = (uint8_t*)malloc(num_blocks);

    // Create two streams
    cudaStream_t streamA, streamB;
    SAFE(cudaStreamCreate(&streamA));
    SAFE(cudaStreamCreate(&streamB));

    // Partition 1 → TPCs 0–11
    uint64_t allowedA = RANGE_MASK(0, 11);
    uint64_t maskA = ~allowedA;
    libsmctrl_set_stream_mask(streamA, maskA);

    // Partition 2 → TPCs 12–23
    uint64_t allowedB = RANGE_MASK(12, 23);
    uint64_t maskB = ~allowedB;
    libsmctrl_set_stream_mask(streamB, maskB);

    printf("Partition A: TPCs 0–11 | mask = 0x%016llx\n", maskA);
    printf("Partition B: TPCs 12–23 | mask = 0x%016llx\n\n", maskB);

    // CUDA timing events
    cudaEvent_t startA, stopA, startB, stopB;
    SAFE(cudaEventCreate(&startA));
    SAFE(cudaEventCreate(&stopA));
    SAFE(cudaEventCreate(&startB));
    SAFE(cudaEventCreate(&stopB));

    // --- Launch on Stream A ---
    SAFE(cudaEventRecord(startA, streamA));
    mm_kernel<<<grid, block, 0, streamA>>>(dA, dB, dC, N, smidA_d);
    SAFE(cudaEventRecord(stopA, streamA));

    // --- Launch on Stream B ---
    SAFE(cudaEventRecord(startB, streamB));
    mm_kernel<<<grid, block, 0, streamB>>>(dA, dB, dC, N, smidB_d);
    SAFE(cudaEventRecord(stopB, streamB));

    // Wait
    SAFE(cudaEventSynchronize(stopA));
    SAFE(cudaEventSynchronize(stopB));

    // Copy back SMID logs
    SAFE(cudaMemcpy(smidA_h, smidA_d, num_blocks, cudaMemcpyDeviceToHost));
    SAFE(cudaMemcpy(smidB_h, smidB_d, num_blocks, cudaMemcpyDeviceToHost));

    // Measure times
    float timeA, timeB;
    SAFE(cudaEventElapsedTime(&timeA, startA, stopA));
    SAFE(cudaEventElapsedTime(&timeB, startB, stopB));

    printf("\nExecution times:\n");
    printf("Stream A: %.3f ms\n", timeA);
    printf("Stream B: %.3f ms\n", timeB);

    // Count unique SMs used
    int uniqA = count_unique(smidA_h, num_blocks);
    int uniqB = count_unique(smidB_h, num_blocks);

    printf("\nUnique SMs used:\n");
    printf("Stream A: %d\n", uniqA);
    printf("Stream B: %d\n\n", uniqB);

    // Print SM IDs observed
    printf("Stream A SM IDs: ");
    for (int i = 0; i < num_blocks; i++)
        printf("%d ", smidA_h[i]);
    printf("\n\nStream B SM IDs: ");
    for (int i = 0; i < num_blocks; i++)
        printf("%d ", smidB_h[i]);
    printf("\n");

    // Cleanup
    cudaEventDestroy(startA);
    cudaEventDestroy(stopA);
    cudaEventDestroy(startB);
    cudaEventDestroy(stopB);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(smidA_d); cudaFree(smidB_d);
    free(smidA_h); free(smidB_h);
    free(hA); free(hB); free(hC);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);

    return 0;
}