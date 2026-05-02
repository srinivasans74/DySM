// stencil_smctrl_sweep.cu
// Sweep through specified SM counts, apply mask, run 3D stencil, and measure execution time.

#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"

#define SAFE(x) x

// Stencil problem dimensions
#define NX 512
#define NY 512
#define NZ 512
#define vol (NX*NY*NZ)

// Number of kernel launches per measurement
#define repeatCount 10

// Tile sizes
#define BSIZE 16
#define XTILE 20

// 3D‐stencil kernel stamping SM ID per block
__global__ void stencil3d(
    const float* __restrict__ psi,
          float* __restrict__ npsi,
    const float* __restrict__ sigmaX,
    const float* __restrict__ sigmaY,
    const float* __restrict__ sigmaZ,
    int nx, int ny, int nz,
    uint8_t* smid_arr)
{
    // Record SM ID from one thread per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = blockIdx.x
                + blockIdx.y * gridDim.x
                + blockIdx.z * gridDim.x * gridDim.y;
        smid_arr[idx] = (uint8_t)smid;
    }

    __shared__ float tile[4][BSIZE][BSIZE];
    int tj = threadIdx.y, tk = threadIdx.x;

    #define IDX(x,y,z)    ((x)*ny*nz + (y)*nz + (z))
    #define PSI(x,y,z)    psi[IDX(x,y,z)]
    #define NPSI(x,y,z)   npsi[IDX(x,y,z)]
    #define SIGX(x,y,z,d) sigmaX[(d)*vol + IDX(x,y,z)]
    #define SIGY(x,y,z,d) sigmaY[(d)*vol + IDX(x,y,z)]
    #define SIGZ(x,y,z,d) sigmaZ[(d)*vol + IDX(x,y,z)]

    int ox = XTILE * blockIdx.x;
    int oy = (BSIZE - 2) * blockIdx.y;
    int oz = (BSIZE - 2) * blockIdx.z;

    int maxX = nx - 1 - ox;
    int maxY = ny - 1 - oy;
    int maxZ = nz - 1 - oz;
    int nLast_x = (XTILE+1 < maxX ? XTILE+1 : maxX);
    int nLast_y = (BSIZE-1 < maxY ? BSIZE-1 : maxY);
    int nLast_z = (BSIZE-1 < maxZ ? BSIZE-1 : maxZ);
    if (tj > nLast_y || tk > nLast_z) return;

    tile[1][tj][tk] = PSI(ox+0, oy+tj, oz+tk);
    tile[2][tj][tk] = PSI(ox+1, oy+tj, oz+tk);
    __syncthreads();

    int pi=0, ci=1, ni=2, ti;
    for (int i = 1; i < nLast_x; ++i) {
        tile[ni][tj][tk] = PSI(ox+i+1, oy+tj, oz+tk);
        __syncthreads();

        float c = tile[ci][tj][tk];
        float l = tile[pi][tj][ tk>0      ? tk-1      : 0     ];
        float r = tile[pi][tj][ tk<nLast_z? tk+1 : nLast_z ];
        float sum = c + l + r
                  + SIGX(ox+i,oy+tj,oz+tk,0)
                  + SIGY(ox+i,oy+tj,oz+tk,1)
                  + SIGZ(ox+i,oy+tj,oz+tk,2);
        NPSI(ox+i,oy+tj,oz+tk) = sum;

        __syncthreads();
        ti=pi; pi=ci; ci=ni; ni=ti;
    }
}

// Comparator for qsort
static int sort_asc(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}

// Count unique SM IDs
static int count_unique(uint8_t* arr, int len) {
    qsort(arr, len, sizeof(uint8_t), sort_asc);
    int uniq = 1;
    for (int i = 1; i < len; ++i) {
        if (arr[i] != arr[i-1]) ++uniq;
    }
    return uniq;
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

    // Allocate SM ID buffers
    int bdimx = (NX-2)/XTILE + ((NX-2)%XTILE!=0);
    int bdimy = (NY-2)/(BSIZE-2) + ((NY-2)%(BSIZE-2)!=0);
    int bdimz = (NZ-2)/(BSIZE-2) + ((NZ-2)%(BSIZE-2)!=0);
    int totalBlocks = bdimx * bdimy * bdimz;
    SAFE(cudaMalloc(&smids_d, totalBlocks * sizeof(uint8_t)));
    smids_h = (uint8_t*)malloc(totalBlocks * sizeof(uint8_t));
    if (!smids_h) error(1, errno, "Unable to allocate SM ID buffer");

    // Allocate and initialize 3D arrays on host
    size_t volBytes = vol * sizeof(float);
    float *h_psi   = (float*)malloc(volBytes);
    float *h_sigX  = (float*)malloc(volBytes);
    float *h_sigY  = (float*)malloc(volBytes);
    float *h_sigZ  = (float*)malloc(volBytes);
    for (int i = 0; i < vol; ++i) {
        h_psi[i]  = 1.0f;
        h_sigX[i] = 0.1f;
        h_sigY[i] = 0.2f;
        h_sigZ[i] = 0.3f;
    }
    float *d_psi, *d_npsi, *d_sigX, *d_sigY, *d_sigZ;
    SAFE(cudaMalloc(&d_psi,  volBytes));
    SAFE(cudaMalloc(&d_npsi, volBytes));
    SAFE(cudaMalloc(&d_sigX, volBytes));
    SAFE(cudaMalloc(&d_sigY, volBytes));
    SAFE(cudaMalloc(&d_sigZ, volBytes));
    SAFE(cudaMemcpy(d_psi,  h_psi,  volBytes, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_sigX, h_sigX, volBytes, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_sigY, h_sigY, volBytes, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_sigZ, h_sigZ, volBytes, cudaMemcpyHostToDevice));

    free(h_psi);
    free(h_sigX);
    free(h_sigY);
    free(h_sigZ);

    // Configure kernel launch
    dim3 blockDim(BSIZE, BSIZE, 1);
    dim3 gridDim(bdimx, bdimy, bdimz);

    // Desired SM configurations
    int desired_sms_arr[] = {2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
    int num_tests = sizeof(desired_sms_arr) / sizeof(desired_sms_arr[0]);

    for (int test = 0; test < num_tests; test++) {
        int desired_sms = desired_sms_arr[test];
        int n_tpc_enable = desired_sms / 2;
        uint64_t allowed = (1ULL << n_tpc_enable) - 1;
        uint64_t mask = ~allowed;

        printf("=== Testing for desired SMs: %d (enabling %d TPCs) ===\n",
               desired_sms, n_tpc_enable);
        printf("Using mask: 0x%016llx\n", mask);

        libsmctrl_set_global_mask(mask);

        // Warmup
        stencil3d<<<gridDim, blockDim, 0, stream>>>(d_psi, d_npsi,
            d_sigX, d_sigY, d_sigZ,
            NX, NY, NZ,
            smids_d);
        cudaDeviceSynchronize();

        // Timing
        cudaEvent_t start, stop;
        SAFE(cudaEventCreate(&start));
        SAFE(cudaEventCreate(&stop));
        SAFE(cudaEventRecord(start, stream));

        stencil3d<<<gridDim, blockDim, 0, stream>>>(d_psi, d_npsi,
            d_sigX, d_sigY, d_sigZ,
            NX, NY, NZ,
            smids_d);

        SAFE(cudaEventRecord(stop, stream));
        SAFE(cudaEventSynchronize(stop));
        float elapsed_ms;
        SAFE(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Copy SM IDs back and count unique
        SAFE(cudaMemcpy(smids_h, smids_d,
            totalBlocks * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));

        uniq = count_unique(smids_h, totalBlocks);
        printf("Unique SMs used: %d\n", uniq);

        bool valid = true;
        for (int i = 0; i < totalBlocks; i++) {
            if (smids_h[i] >= desired_sms) {
                printf("Test failure: Found SM ID %d outside expected range (0 to %d).\n",
                       smids_h[i], desired_sms - 1);
                valid = false;
                break;
            }
        }
        if (valid)
            printf("Test passed: All blocks executed on SMs in range 0 to %d.\n",
                   desired_sms - 1);
        else
            printf("Test failed for desired SMs = %d.\n", desired_sms);

        printf("Kernel execution time: %f ms\n\n", elapsed_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Cleanup
    free(smids_h);
    SAFE(cudaFree(smids_d));
    SAFE(cudaFree(d_psi));
    SAFE(cudaFree(d_npsi));
    SAFE(cudaFree(d_sigX));
    SAFE(cudaFree(d_sigY));
    SAFE(cudaFree(d_sigZ));
    return 0;
}