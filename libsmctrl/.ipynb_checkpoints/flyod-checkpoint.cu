// gpu_benchmarks2_sm_sweep_stats.cu
// Benchmarks with SM‐mask sweep and statistics:
//   1) Black-Scholes
//   2) N-Body simulation
//   3) Blelloch exclusive scan (with mean/stddev over 10 runs)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <error.h>
#include <errno.h>
#include "libsmctrl.h"
#include "libsmctrl_test_mask_shared.h"

// Provide sqrt(1/2) if missing
#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F 0.707106781186547524f
#endif

// Easy error checking for CUDA calls
#define CHECK_CUDA(call) do {                                     \
  cudaError_t err = call;                                         \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(err));         \
    exit(1);                                                       \
  }                                                                \
} while(0)

// -----------------------------------------------------------------------------
// 1) Black-Scholes
// -----------------------------------------------------------------------------
struct BSData { float S, X, T, R, V; };

__device__ float norm_cdf(float x) {
  return 0.5f * erfcf(-x * M_SQRT1_2_F);
}

__global__ void blackScholes(
    const BSData* d_in,
    float*        d_call,
    float*        d_put,
    int           n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;
  BSData b = d_in[i];
  float sqrtT = sqrtf(b.T);
  float d1 = (logf(b.S/b.X) + (b.R + 0.5f*b.V*b.V)*b.T) / (b.V*sqrtT);
  float d2 = d1 - b.V*sqrtT;
  float cdf1 = norm_cdf(d1), cdf2 = norm_cdf(d2);
  float eRT = expf(-b.R*b.T);
  d_call[i] = b.S*cdf1 - b.X*eRT*cdf2;
  d_put[i]  = d_call[i] - b.S + b.X*eRT;
}

// -----------------------------------------------------------------------------
// 2) N-Body (naive O(N²))
// -----------------------------------------------------------------------------
struct Body { float3 pos; float3 vel; };

__global__ void nbody(
    Body* bodies,
    int    n,
    float  dt)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 p = bodies[i].pos;
  float3 v = bodies[i].vel;
  float3 acc = make_float3(0,0,0);

  for(int j = 0; j < n; ++j){
    float3 q = bodies[j].pos;
    float3 r = { q.x - p.x, q.y - p.y, q.z - p.z };
    float dist2 = r.x*r.x + r.y*r.y + r.z*r.z + 1e-9f;
    float invDist3 = rsqrtf(dist2*dist2*dist2);
    acc.x += r.x * invDist3;
    acc.y += r.y * invDist3;
    acc.z += r.z * invDist3;
  }

  v.x += acc.x * dt;  v.y += acc.y * dt;  v.z += acc.z * dt;
  p.x += v.x   * dt;  p.y += v.y   * dt;  p.z += v.z   * dt;

  bodies[i].vel = v;
  bodies[i].pos = p;
}

// -----------------------------------------------------------------------------
// 3) Blelloch scan (exclusive prefix sum)
// -----------------------------------------------------------------------------
__global__ void blelloch_up(float* data, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int offset = 1;
  s[2*tid]   = data[2*tid];
  s[2*tid+1] = data[2*tid+1];
  for(int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1) - 1;
      int bi = offset*(2*tid+2) - 1;
      s[bi] += s[ai];
    }
    offset <<= 1;
  }
  if (tid == 0) s[n - 1] = 0;
}

__global__ void blelloch_down(float* data, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int offset = n;
  for(int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1) - 1;
      int bi = offset*(2*tid+2) - 1;
      float t = s[ai];
      s[ai]  = s[bi];
      s[bi] += t;
    }
  }
  __syncthreads();
  data[2*tid]   = s[2*tid];
  data[2*tid+1] = s[2*tid+1];
}

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
int main(){
  // Query GPU topology
  int numSMS;
  CHECK_CUDA(cudaDeviceGetAttribute(&numSMS, cudaDevAttrMultiProcessorCount, 0));
  uint32_t numTPCs;
  if (libsmctrl_get_tpc_info_cuda(&numTPCs, 0) != 0)
    error(1, errno, "Unable to get TPC configuration");
  int smsPerTPC = numSMS / numTPCs;
  printf("Total TPCs: %u, Total SMs: %d, SMs/TPC: %d\n\n", numTPCs, numSMS, smsPerTPC);

  // Masks to sweep
  int desired_sms[] = {2,4,5,8,10,15,16,20,25,30,32,35,40,45,50,55,60,64,65,70,75,80,82};
  int nTests = sizeof(desired_sms) / sizeof(desired_sms[0]);

  // CUDA events
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // ------------------- 1) Black-Scholes -------------------
  {
    printf("=== Black-Scholes Benchmark ===\n");
    const int N = 1<<20;
    size_t bytes = N * sizeof(BSData);

    // Allocate host
    BSData *h_in   = (BSData*)malloc(bytes);
    float  *h_call = (float*)malloc(N*sizeof(float));
    float  *h_put  = (float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++){
      h_in[i] = {100.0f,100.0f,1.0f,0.05f,0.2f};
    }

    // Allocate device
    BSData *d_in; float *d_call,*d_put;
    CHECK_CUDA(cudaMalloc(&d_in,   bytes));
    CHECK_CUDA(cudaMalloc(&d_call, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_put,  N*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));

    dim3 blkA(256), grdA((N+255)/256);

    // Sweep masks
    for(int t=0;t<nTests;t++){
      int want = desired_sms[t];
      int tpcs = (want + smsPerTPC - 1) / smsPerTPC;
      uint64_t allow = (1ULL<<tpcs) - 1;
      uint64_t mask  = ~allow;
      printf(" BS: want %2d SM → enable %2d SM (%2d TPCs), mask=0x%016llx … ",
             want, tpcs*smsPerTPC, tpcs, mask);
      libsmctrl_set_global_mask(mask);

      // Warmup
      blackScholes<<<grdA,blkA>>>(d_in,d_call,d_put,N);
      CHECK_CUDA(cudaDeviceSynchronize());

      // Timed
      CHECK_CUDA(cudaEventRecord(start));
      blackScholes<<<grdA,blkA>>>(d_in,d_call,d_put,N);
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
      printf("%.3f ms\n", ms);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_call));
    CHECK_CUDA(cudaFree(d_put));
    free(h_in); free(h_call); free(h_put);
  }

  // ------------------- 2) N-Body -------------------
  {
    printf("=== N-Body Benchmark ===\n");
    const int N = 1<<16;
    size_t bytes = N * sizeof(Body);

    // Host init
    Body *h_b = (Body*)malloc(bytes);
    for(int i=0;i<N;i++){
      h_b[i].pos = make_float3(rand()/float(RAND_MAX),
                               rand()/float(RAND_MAX),
                               rand()/float(RAND_MAX));
      h_b[i].vel = make_float3(0,0,0);
    }

    // Device
    Body* d_b;
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

    dim3 blkB(128), grdB((N+127)/128);
    float dt = 0.01f;

    // Sweep masks
    for(int t=0;t<nTests;t++){
      int want = desired_sms[t];
      int tpcs = (want + smsPerTPC - 1) / smsPerTPC;
      uint64_t allow = (1ULL<<tpcs) - 1;
      uint64_t mask  = ~allow;
      printf(" NB: want %2d SM → enable %2d SM (%2d TPCs), mask=0x%016llx … ",
             want, tpcs*smsPerTPC, tpcs, mask);
      libsmctrl_set_global_mask(mask);

      // Warmup
      nbody<<<grdB,blkB>>>(d_b,N,dt);
      CHECK_CUDA(cudaDeviceSynchronize());

      // Timed
      CHECK_CUDA(cudaEventRecord(start));
      nbody<<<grdB,blkB>>>(d_b,N,dt);
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
      printf("%.3f ms\n", ms);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_b));
    free(h_b);
  }

  // ------------------- 3) Blelloch Scan -------------------
  {
    printf("=== Blelloch Scan Benchmark (10 runs each) ===\n");
    const int SZ = 1<<20;
    size_t bytes = SZ * sizeof(float);

    float *h = (float*)malloc(bytes);
    for(int i=0;i<SZ;i++) h[i] = 1.0f;

    float* d;
    CHECK_CUDA(cudaMalloc(&d, bytes));
    CHECK_CUDA(cudaMemcpy(d,h,bytes,cudaMemcpyHostToDevice));

    dim3 blkC(SZ/2), grdC(1);
    size_t smem = SZ * sizeof(float);

    // Sweep masks
    for(int t=0;t<nTests;t++){
      int want = desired_sms[t];
      int tpcs = (want + smsPerTPC - 1) / smsPerTPC;
      uint64_t allow = (1ULL<<tpcs) - 1;
      uint64_t mask  = ~allow;
      printf(" SC: want %2d SM → enable %2d SM (%2d TPCs), mask=0x%016llx … ",
             want, tpcs*smsPerTPC, tpcs, mask);
      libsmctrl_set_global_mask(mask);

      // Warmup
      blelloch_up  <<<grdC,blkC,smem>>>(d,SZ);
      blelloch_down<<<grdC,blkC,smem>>>(d,SZ);
      CHECK_CUDA(cudaDeviceSynchronize());

      // 10 runs for stats
      double sum=0, sumsq=0;
      for(int r=0;r<10;r++){
        CHECK_CUDA(cudaEventRecord(start));
        blelloch_up  <<<grdC,blkC,smem>>>(d,SZ);
        blelloch_down<<<grdC,blkC,smem>>>(d,SZ);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        sum   += ms; 
        sumsq += ms*ms;
      }
      double mean = sum/10.0;
      double var  = sumsq/10.0 - mean*mean;
      double sd   = sqrt(var>0?var:0);
      printf("mean=%.3f ms  stddev=%.3f ms\n", mean, sd);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d));
    free(h);
  }

  // Cleanup events
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return 0;
}