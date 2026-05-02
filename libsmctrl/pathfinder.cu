#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <errno.h>
#include <error.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"

#define BLOCK_SIZE 256
#define HALO 1
#define CHECK_CUDA(call) {                                \
    cudaError_t err = call;                               \
    if (err != cudaSuccess) {                             \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n",    \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
}

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// Kernel: Dynamic Programming (Path Calculation)
__global__ void dynproc_kernel(
    int iteration, int *gpuWall, int *gpuSrc, int *gpuResults,
    int cols, int rows, int startStep, int border, uint8_t *smid_arr)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Record SM ID for validation
    if (tx == 0) {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = blockIdx.x;
        smid_arr[idx] = (uint8_t)smid;
    }

    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;
    int xidx = blkX + tx;

    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

    int W = max(tx - 1, validXmin);
    int E = min(tx + 1, validXmax);
    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (xidx >= 0 && xidx < cols) {
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads();

    bool computed;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if ((tx >= i + 1 && tx <= BLOCK_SIZE - i - 2) && isValid) {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols * (startStep + i) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed)
            prev[tx] = result[tx];
        __syncthreads();
    }

    if (computed) {
        gpuResults[xidx] = result[tx];
    }
}

// Count unique SM IDs (sort + count)
static int sort_asc(const void *a, const void *b) {
    return *(uint8_t *)a - *(uint8_t *)b;
}
static int count_unique(uint8_t *arr, int len) {
    qsort(arr, len, sizeof(uint8_t), sort_asc);
    int num_uniq = 1;
    for (int i = 0; i < len - 1; i++) {
        if (arr[i] != arr[i + 1])
            num_uniq++;
    }
    return num_uniq;
}

int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols,
              uint8_t *smid_arr)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);
    int src = 1, dst = 0;

    for (int t = 0; t < rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            MIN(pyramid_height, rows - t - 1),
            gpuWall, gpuResult[src], gpuResult[dst],
            cols, rows, t, borderCols, smid_arr);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    return dst;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s cols rows pyramid_height\n", argv[0]);
        return -1;
    }

    int cols = atoi(argv[1]);
    int rows = atoi(argv[2]);
    int pyramid_height = atoi(argv[3]);

    int borderCols = pyramid_height * HALO;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
    int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

    int size = rows * cols;
    int *data = (int *)malloc(sizeof(int) * size);
    int *result = (int *)malloc(sizeof(int) * cols);

    for (int i = 0; i < size; i++)
        data[i] = rand() % 10;

    int *gpuWall, *gpuResult[2];
    CHECK_CUDA(cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols));
    CHECK_CUDA(cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols));
    CHECK_CUDA(cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols)));

    CHECK_CUDA(cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice));

    // Allocate SM ID tracker
    uint8_t *smids_d, *smids_h;
    CHECK_CUDA(cudaMalloc(&smids_d, blockCols * sizeof(uint8_t)));
    smids_h = (uint8_t *)malloc(blockCols * sizeof(uint8_t));

    // Query GPU info
    int res;
    uint32_t num_tpcs;
    int num_sms, sms_per_tpc;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    if ((res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0)) != 0)
        error(1, res, "Unable to get TPC configuration");
    sms_per_tpc = num_sms / num_tpcs;
    printf("Total TPCs: %u | Total SMs: %d | SMs/TPC: %d\n", num_tpcs, num_sms, sms_per_tpc);

    int desired_sms_arr[] = {2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
    int num_tests = sizeof(desired_sms_arr) / sizeof(desired_sms_arr[0]);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    for (int test = 0; test < num_tests; test++) {
        int desired_sms = desired_sms_arr[test];
        int n_tpc_enable = desired_sms / 2;
        uint64_t allowed = (1ULL << n_tpc_enable) - 1;
        uint64_t mask = ~allowed;

        printf("=== SM Config: %d SMs (%d TPCs) ===\n", desired_sms, n_tpc_enable);
        printf("Mask: 0x%016llx\n", mask);

        libsmctrl_set_global_mask(mask);

        // Warmup
        calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols, smids_d);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols, smids_d);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        cudaMemcpy(smids_h, smids_d, blockCols * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        int uniq = count_unique(smids_h, blockCols);
        printf("Unique SMs used: %d | Exec Time: %.4f ms\n\n", uniq, elapsed_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    free(data);
    free(result);
    free(smids_h);
    cudaFree(smids_d);
    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);
    cudaStreamDestroy(stream);
    return 0;
}