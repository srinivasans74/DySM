// srad_smctrl_sweep.cu
// Benchmark SRAD kernels under varying SM masks using libsmctrl

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"

#define SAFE(x) x

// Problem dimensions
#define ROWS (4096*2)
#define COLS (4096*2)
#define SIZE (ROWS*COLS)

// SRAD parameters
#define BLOCK_SIZE 16
#define repeatCount 1  // kernel launches per measurement
const float lambda = 2.5f;

// Derived constants
#define NUM_BLOCKS_X (COLS/BLOCK_SIZE)
#define NUM_BLOCKS_Y (ROWS/BLOCK_SIZE)
#define NUM_BLOCKS   (NUM_BLOCKS_X * NUM_BLOCKS_Y)

// Utility to count unique SM IDs
static int sort_asc(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}
static int count_unique(uint8_t* arr, int len) {
    qsort(arr, len, sizeof(uint8_t), sort_asc);
    int uniq = 1;
    for (int i = 1; i < len; ++i)
        if (arr[i] != arr[i-1]) ++uniq;
    return uniq;
}

// Kernel 1: compute conduction coefficients and stamps SM ID
__global__ void srad1(
    float* E, float* W, float* N, float* S,
    float* J, float* C, int cols, int rows,
    float q0sqr, uint8_t* smid_arr)
{
    // Stamp SM ID once per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int sm;
        asm("mov.u32 %0, %%smid;" : "=r"(sm));
        int idx = blockIdx.y * gridDim.x + blockIdx.x;
        smid_arr[idx] = (uint8_t)sm;
    }

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = (rows*BLOCK_SIZE*by) + (BLOCK_SIZE*bx) + (rows*ty) + tx;

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileC[BLOCK_SIZE][BLOCK_SIZE];

    tile[ty][tx] = J[idx];
    __syncthreads();

    float jc = tile[ty][tx];
    float north = (ty==0)               ? jc : tile[ty-1][tx] - jc;
    float south = (ty==BLOCK_SIZE-1)    ? jc : tile[ty+1][tx] - jc;
    float west  = (tx==0)               ? jc : tile[ty][tx-1] - jc;
    float east  = (tx==BLOCK_SIZE-1)    ? jc : tile[ty][tx+1] - jc;

    float g2 = (north*north + south*south + west*west + east*east)/(jc*jc);
    float l  = (north + south + west + east)/jc;
    float num= 0.5f*g2 - (1.0f/16.0f)*(l*l);
    float den= 1.0f + 0.25f*l;
    float qsqr = num / (den*den);
    den = (qsqr - q0sqr)/(q0sqr*(1.0f+q0sqr));
    float c = 1.0f/(1.0f+den);
    tileC[ty][tx] = fminf(fmaxf(c, 0.0f), 1.0f);
    __syncthreads();

    C[idx] = tileC[ty][tx];
    E[idx] = east;  W[idx] = west;
    N[idx] = north; S[idx] = south;
}

// Kernel 2: update J using diffusion
__global__ void srad2(
    float* E, float* W, float* N, float* S,
    float* J, float* C, int cols, int rows,
    float lambda)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int idx = (rows*BLOCK_SIZE*by) + (BLOCK_SIZE*bx) + (rows*ty) + tx;

    __shared__ float tileJ[BLOCK_SIZE][BLOCK_SIZE];
    tileJ[ty][tx] = J[idx];
    __syncthreads();

    float cc = C[idx];
    float d  = cc*(N[idx] + S[idx] + W[idx] + E[idx]);
    tileJ[ty][tx] += 0.25f * lambda * d;
    __syncthreads();

    J[idx] = tileJ[ty][tx];
}

int main(){
    // Query GPU SM/TPC info
    int num_sms; uint32_t num_tpcs;
    SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    if(libsmctrl_get_tpc_info_cuda(&num_tpcs,0)!=0){
        fprintf(stderr,"Unable to get TPC info\n"); return 1;
    }
    int sms_per_tpc = num_sms / num_tpcs;
    printf("Total TPCs: %u, Total SMs: %d, SMs/TPC: %d\n\n",
           num_tpcs, num_sms, sms_per_tpc);

    // Allocate host data
    float *h_I = (float*)malloc(SIZE*sizeof(float));
    float *h_J = (float*)malloc(SIZE*sizeof(float));
    srand(7);
    for(int i=0;i<SIZE;i++){
        h_I[i] = rand()/(float)RAND_MAX;
        h_J[i] = expf(h_I[i]);
    }

    // Device buffers
    float *d_J, *d_C, *d_E, *d_W, *d_N, *d_S;
    SAFE(cudaMalloc(&d_J, SIZE*sizeof(float)));
    SAFE(cudaMalloc(&d_C, SIZE*sizeof(float)));
    SAFE(cudaMalloc(&d_E, SIZE*sizeof(float)));
    SAFE(cudaMalloc(&d_W, SIZE*sizeof(float)));
    SAFE(cudaMalloc(&d_N, SIZE*sizeof(float)));
    SAFE(cudaMalloc(&d_S, SIZE*sizeof(float)));
    SAFE(cudaMemcpy(d_J, h_J, SIZE*sizeof(float), cudaMemcpyHostToDevice));

    free(h_I); free(h_J);

    // SM‐ID buffers
    uint8_t *smid_d, *smid_h;
    SAFE(cudaMalloc(&smid_d, NUM_BLOCKS*sizeof(uint8_t)));
    smid_h = (uint8_t*)malloc(NUM_BLOCKS);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid(NUM_BLOCKS_X, NUM_BLOCKS_Y);

    // Desired SM counts
    int desired_sms_arr[] = {2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
    int num_tests = sizeof(desired_sms_arr)/sizeof(desired_sms_arr[0]);

    for(int t=0; t<num_tests; ++t){
        int desired_sms = desired_sms_arr[t];
        int tpcs = desired_sms / sms_per_tpc;
        uint64_t allowed = (1ULL<<tpcs)-1;
        uint64_t mask = ~allowed;
        printf("=== Testing for desired SMs: %d (enabling %d TPCs) ===\n",
               desired_sms, tpcs);
        printf("Using mask: 0x%016llx\n", mask);
        libsmctrl_set_global_mask(mask);

        // Warmup
        srad1<<<grid,block>>>(d_E,d_W,d_N,d_S,
                              d_J,d_C,COLS,ROWS,
                              /*q0sqr=*/1.0f, smid_d);
        SAFE(cudaDeviceSynchronize());
        srad2<<<grid,block>>>(d_E,d_W,d_N,d_S,
                              d_J,d_C,COLS,ROWS,
                              lambda);
        SAFE(cudaDeviceSynchronize());

        // Timing
        cudaEvent_t start, stop;
        SAFE(cudaEventCreate(&start));
        SAFE(cudaEventCreate(&stop));
        SAFE(cudaEventRecord(start,0));

        for(int r=0;r<repeatCount;r++){
            srad1<<<grid,block>>>(d_E,d_W,d_N,d_S,
                                  d_J,d_C,COLS,ROWS,
                                  1.0f, smid_d);
            srad2<<<grid,block>>>(d_E,d_W,d_N,d_S,
                                  d_J,d_C,COLS,ROWS,
                                  lambda);
        }

        SAFE(cudaEventRecord(stop,0));
        SAFE(cudaEventSynchronize(stop));
        float elapsed_ms;
        SAFE(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // SM usage
        SAFE(cudaMemcpy(smid_h, smid_d, NUM_BLOCKS*sizeof(uint8_t),
                        cudaMemcpyDeviceToHost));
        int uniq = count_unique(smid_h, NUM_BLOCKS);
        printf("Unique SMs used: %d\n", uniq);
        printf("Total kernel execution time: %f ms\n\n", elapsed_ms);

        SAFE(cudaEventDestroy(start));
        SAFE(cudaEventDestroy(stop));
    }

    // Cleanup
    free(smid_h);
    SAFE(cudaFree(smid_d));
    SAFE(cudaFree(d_J));
    SAFE(cudaFree(d_C));
    SAFE(cudaFree(d_E));
    SAFE(cudaFree(d_W));
    SAFE(cudaFree(d_N));
    SAFE(cudaFree(d_S));
    return 0;
}
