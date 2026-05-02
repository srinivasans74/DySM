// Copyright 2023 Joshua Bakita
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"
//#include "testbench.h"
#include "libsmctrl_test_mask_shared.h"

#define SAFE(x) x

// Matrix size for multiplication
#define N 256

// A simple matrix multiplication kernel that also records SM IDs.
// Each thread computes one element of C.
// Additionally, thread (0,0) in each block uses inline assembly to obtain the SM ID
// and writes it into smid_arr at an index corresponding to the block index.
__global__ void mm_kernel(const float* A, const float* B, float* C, int n, uint8_t* smid_arr) {
    // Record SM ID from one thread per block.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        // Use 2D block indexing flattened to 1D.
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        smid_arr[idx] = (uint8_t)smid;
    }
    
    // Compute matrix multiplication.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

#define NUM_BLOCKS_X ((N + 15) / 16)
#define NUM_BLOCKS_Y ((N + 15) / 16)
#define NUM_BLOCKS (NUM_BLOCKS_X * NUM_BLOCKS_Y)

// Sorting comparator for qsort (ascending order)
static int sort_asc(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}

// Count unique SM IDs (this function mutates the array via qsort)
static int count_unique(uint8_t* arr, int len) {
    qsort(arr, len, sizeof(uint8_t), sort_asc);
    int num_uniq = 1;
    for (int i = 0; i < len - 1; i++) {
        if (arr[i] != arr[i+1])
            num_uniq++;
    }
    return num_uniq;
}

int main(void) {
    int res;
    uint8_t *smids_d, *smids_h;
    int uniq;
    uint32_t num_tpcs;
    int num_sms, sms_per_tpc;

    // Create a CUDA stream.
    cudaStream_t stream;
    SAFE(cudaStreamCreate(&stream));

    // Query total number of SMs.
    SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    // Get total number of TPCs using libsmctrl API.
    if ((res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0)) != 0)
        error(1, res, "Unable to get TPC configuration");
    
    // For your GPU, each TPC contains 2 SMs.
    sms_per_tpc = num_sms / num_tpcs;
    printf("Total TPCs on GPU: %u\n", num_tpcs);
    printf("Total SMs on GPU: %d\n", num_sms);
    printf("SMs per TPC: %d\n\n", sms_per_tpc);

    // Allocate memory for SM IDs recorded by the mm kernel.
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS * sizeof(uint8_t)));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS * sizeof(uint8_t));
    if (!smids_h)
        error(1, errno, "Unable to allocate host memory for SM IDs");

    // Allocate host and device memory for matrices.
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C)
        error(1, errno, "Unable to allocate host matrices");
    // Initialize matrices (fill A and B with 1.0f).
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    float *d_A, *d_B, *d_C;
    SAFE(cudaMalloc(&d_A, bytes));
    SAFE(cudaMalloc(&d_B, bytes));
    SAFE(cudaMalloc(&d_C, bytes));
    SAFE(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define an array of desired SM counts.
    // For a GPU with 82 SMs (41 TPCs, 2 SMs/TPC), valid desired SM counts must be even.
    // Enabling the first k TPCs yields 2*k SMs.
    //int desired_sms_arr[] = {2, 4, 8, 16, 32, 64, 82};
    int desired_sms_arr[]={2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
    int num_tests = sizeof(desired_sms_arr) / sizeof(desired_sms_arr[0]);

    // Define 2D grid and block dimensions for mm kernel.
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Loop over each desired SM count.
    for (int test = 0; test < num_tests; test++) {
        int desired_sms = desired_sms_arr[test];
        // Number of TPCs to enable.
        int n_tpc_enable = desired_sms / 2;
        // Compute allowed mask: lower n_tpc_enable bits = 1, others = 0.
        uint64_t allowed = (1ULL << n_tpc_enable) - 1;
        // Disable mask = bitwise complement.
        uint64_t mask = ~allowed;
        printf("=== Testing for desired SMs: %d (enabling %d TPCs) ===\n", desired_sms, n_tpc_enable);
        printf("Using mask: 0x%016llx\n", mask);

        // Apply the global mask.
        libsmctrl_set_global_mask(mask);

        // Allocate an additional array for SM IDs recorded by the mm kernel.
        // (Re-use smids_d and smids_h already allocated.)

        // Warmup: launch mm kernel once to prime GPU.
        mm_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N, smids_d);
        cudaDeviceSynchronize();

        // Create CUDA events for timing.
        cudaEvent_t start, stop;
        SAFE(cudaEventCreate(&start));
        SAFE(cudaEventCreate(&stop));

        // Start timing.
        SAFE(cudaEventRecord(start, stream));
        mm_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N, smids_d);
        SAFE(cudaEventRecord(stop, stream));
        SAFE(cudaEventSynchronize(stop));
        float elapsed_ms;
        SAFE(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Copy SM IDs back to host.
        SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        // Count unique SM IDs.
        uniq = count_unique(smids_h, NUM_BLOCKS);
        printf("Unique SMs used: %d\n", uniq);

        // Print all recorded SM IDs.
        printf("SM IDs recorded: ");
      //  for (int i = 0; i < NUM_BLOCKS; i++) {
        //    printf("%d ", smids_h[i]);
        //}
        printf("\n");

        // Verify that all SM IDs are within the expected range: 0 to desired_sms - 1.
        bool valid = true;
        for (int i = 0; i < NUM_BLOCKS; i++) {
            if (smids_h[i] < 0 || smids_h[i] >= desired_sms) {
                printf("Test failure: Found SM ID %d outside expected range (0 to %d).\n", smids_h[i], desired_sms - 1);
                valid = false;
                break;
            }
        }
        if (valid)
            printf("Test passed: All blocks executed on SMs in range 0 to %d.\n", desired_sms - 1);
        else
            printf("Test failed for desired SMs = %d.\n", desired_sms);

        printf("Kernel execution time: %f ms\n\n", elapsed_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup.
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(smids_h);
    cudaFree(smids_d);
    cudaStreamDestroy(stream);
    return 0;
}
