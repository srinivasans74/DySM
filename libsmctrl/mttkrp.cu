// Copyright 2023
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

// Matrix size
#define N 256

// Each thread computes one element of C = A * B^T
// Records SM ID for each block using inline PTX
__global__ void mttrkp_kernel(const float* A, const float* B, float* C, int n, uint8_t* smid_arr) {
    // Record SM ID from one thread per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        smid_arr[idx] = (uint8_t)smid;
    }

    // Compute C[row, col] = sum over k (A[row,k] * B[col,k])  -> B is transposed
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[col * n + k]; // Transpose B access
        }
        C[row * n + col] = sum;
    }
}

#define NUM_BLOCKS_X ((N + 15) / 16)
#define NUM_BLOCKS_Y ((N + 15) / 16)
#define NUM_BLOCKS (NUM_BLOCKS_X * NUM_BLOCKS_Y)

// Sorting comparator for qsort
static int sort_asc(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}

// Count unique SM IDs
static int count_unique(uint8_t* arr, int len) {
    qsort(arr, len, sizeof(uint8_t), sort_asc);
    int num_uniq = 1;
    for (int i = 0; i < len - 1; i++) {
        if (arr[i] != arr[i + 1])
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

    // Create CUDA stream
    cudaStream_t stream;
    SAFE(cudaStreamCreate(&stream));

    // Query GPU SM info
    SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    if ((res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0)) != 0)
        error(1, res, "Unable to get TPC configuration");

    sms_per_tpc = num_sms / num_tpcs;
    printf("Total TPCs on GPU: %u\n", num_tpcs);
    printf("Total SMs on GPU: %d\n", num_sms);
    printf("SMs per TPC: %d\n\n", sms_per_tpc);

    // Allocate SM ID array
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS * sizeof(uint8_t)));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS * sizeof(uint8_t));
    if (!smids_h)
        error(1, errno, "Unable to allocate host memory for SM IDs");

    // Allocate matrices
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C)
        error(1, errno, "Unable to allocate host matrices");

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

    // Desired SM configurations
    int desired_sms_arr[] = {2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
    int num_tests = sizeof(desired_sms_arr) / sizeof(desired_sms_arr[0]);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    for (int test = 0; test < num_tests; test++) {
        int desired_sms = desired_sms_arr[test];
        int n_tpc_enable = desired_sms / 2;
        uint64_t allowed = (1ULL << n_tpc_enable) - 1;
        uint64_t mask = ~allowed;
        printf("=== Testing for desired SMs: %d (enabling %d TPCs) ===\n", desired_sms, n_tpc_enable);
        printf("Using mask: 0x%016llx\n", mask);

        libsmctrl_set_global_mask(mask);

        // Warmup
        mttrkp_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N, smids_d);
        cudaDeviceSynchronize();

        // Timing
        cudaEvent_t start, stop;
        SAFE(cudaEventCreate(&start));
        SAFE(cudaEventCreate(&stop));
        SAFE(cudaEventRecord(start, stream));
        mttrkp_kernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N, smids_d);
        SAFE(cudaEventRecord(stop, stream));
        SAFE(cudaEventSynchronize(stop));
        float elapsed_ms;
        SAFE(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Copy SM IDs
        SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        uniq = count_unique(smids_h, NUM_BLOCKS);
        printf("Unique SMs used: %d\n", uniq);

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

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(smids_h);
    cudaFree(smids_d);
    cudaStreamDestroy(stream);
    return 0;
}