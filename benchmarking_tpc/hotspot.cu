#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include "helper_cuda.h"
#include "tpc_control.h"

using namespace std;

// ============================================================
// Parameters
// ============================================================
#define BLOCK_SIZE 16
#define MAX_PD        (3.0e6)
#define PRECISION     0.001
#define SPEC_HEAT_SI  1.75e6
#define K_SI          100
#define FACTOR_CHIP   0.5
#define EXPAND_RATE   2

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define EXPAND_RATE 2

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

// ============================================================
// Utility helpers
// ============================================================
void fatal(const char* s) {
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void readinput(float *vect, int grid_rows, int grid_cols, const char *file) {
    FILE *fp = fopen(file, "r");
    if (!fp) fatal("could not open input file");
    int i, j;
    char str[256];
    float val;
    for (i = 0; i < grid_rows; ++i) {
        for (j = 0; j < grid_cols; ++j) {
            if (!fgets(str, 256, fp)) fatal("not enough lines in file");
            if (sscanf(str, "%f", &val) != 1) fatal("invalid file format");
            vect[i * grid_cols + j] = val;
        }
    }
    fclose(fp);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, const char *file) {
    FILE *fp = fopen(file, "w");
    if (!fp) fatal("could not open output file for writing");
    for (int i = 0; i < grid_rows; ++i)
        for (int j = 0; j < grid_cols; ++j)
            fprintf(fp, "%d\t%g\n", i * grid_cols + j, vect[i * grid_cols + j]);
    fclose(fp);
}

// ============================================================
// CUDA Kernel
// ============================================================
__global__ void calculate_temp(
    int iteration, float *power, float *temp_src, float *temp_dst,
    int grid_cols, int grid_rows, int border_cols, int border_rows,
    float Cap, float Rx, float Ry, float Rz, float step)
{
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

    float step_div_Cap = step / Cap;
    float Rx_1 = 1 / Rx, Ry_1 = 1 / Ry, Rz_1 = 1 / Rz;

    float amb_temp = 80.0;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int small_block_rows = BLOCK_SIZE - iteration * 2;
    int small_block_cols = BLOCK_SIZE - iteration * 2;

    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int yidx = blkY + ty;
    int xidx = blkX + tx;
    int index = grid_cols * yidx + xidx;

    if (yidx >= 0 && yidx < grid_rows && xidx >= 0 && xidx < grid_cols) {
        temp_on_cuda[ty][tx] = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }
    __syncthreads();

    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

    for (int i = 0; i < iteration; i++) {
        bool computed = false;
        if (tx >= i + 1 && tx <= BLOCK_SIZE - i - 2 &&
            ty >= i + 1 && ty <= BLOCK_SIZE - i - 2 &&
            tx >= validXmin && tx <= validXmax &&
            ty >= validYmin && ty <= validYmax)
        {
            computed = true;
            float N = temp_on_cuda[ty - 1][tx];
            float S = temp_on_cuda[ty + 1][tx];
            float E = temp_on_cuda[ty][tx + 1];
            float W = temp_on_cuda[ty][tx - 1];

            temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (
                power_on_cuda[ty][tx] +
                (S + N - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 +
                (E + W - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 +
                (amb_temp - temp_on_cuda[ty][tx]) * Rz_1
            );
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) temp_on_cuda[ty][tx] = temp_t[ty][tx];
        __syncthreads();
    }

    if (yidx < grid_rows && xidx < grid_cols) {
        temp_dst[index] = temp_t[ty][tx];
    }
}

// ============================================================
// Main Benchmark
// ============================================================
int main() {
    const int GRID_ROWS = 512;
    const int GRID_COLS = 512;
    const int TOTAL_ITER = 60;
    const int PYRAMID_HEIGHT = 2;
    const int RUNS = 3;
    cudaDeviceProp prop;
    int device = 0; // Assuming device 0
    benchmark_cuda_check(cudaGetDeviceProperties(&prop, device));
    int num_sms = prop.multiProcessorCount;
    const int MAX_TPC = benchmark_tpc_count(device);
    std::cout << "GPU: " << prop.name << " (" << num_sms << " SMs, "
              << MAX_TPC << " TPCs)\n";

    const char* TEMP_FILE = "data/hotspot/temp_512";
    const char* POWER_FILE = "data/hotspot/power_512";
    const std::string OUT_FILE = benchmark_log_path("output_hotspot.out");

    cout << "[CUDA Benchmark] Hotspot " << GRID_ROWS << "x" << GRID_COLS
         << ", " << TOTAL_ITER << " iterations\n";
    cout << "Using libsmctrl TPC mask control (1–" << MAX_TPC << ")\n";
    cout << fixed << setprecision(3);

    const int size = GRID_ROWS * GRID_COLS;
    vector<float> temp_h(size);
    vector<float> power_h(size);

    // Read input data
    readinput(temp_h.data(), GRID_ROWS, GRID_COLS, TEMP_FILE);
    readinput(power_h.data(), GRID_ROWS, GRID_COLS, POWER_FILE);

    // Device memory
    float *temp_d[2], *power_d;
    benchmark_cuda_check(cudaMalloc(&temp_d[0], size * sizeof(float)));
    benchmark_cuda_check(cudaMalloc(&temp_d[1], size * sizeof(float)));
    benchmark_cuda_check(cudaMalloc(&power_d, size * sizeof(float)));

    benchmark_cuda_check(cudaMemcpy(power_d, power_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    benchmark_cuda_check(cudaMemcpy(temp_d[0], temp_h.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // Compute constants
    float grid_height = chip_height / GRID_ROWS;
    float grid_width = chip_width / GRID_COLS;
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);
    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;

    int borderCols = (PYRAMID_HEIGHT) * EXPAND_RATE / 2;
    int borderRows = (PYRAMID_HEIGHT) * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (PYRAMID_HEIGHT) * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (PYRAMID_HEIGHT) * EXPAND_RATE;
    int blockCols = GRID_COLS / smallBlockCol + ((GRID_COLS % smallBlockCol == 0) ? 0 : 1);
    int blockRows = GRID_ROWS / smallBlockRow + ((GRID_ROWS % smallBlockRow == 0) ? 0 : 1);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);

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

            int src = 0, dst = 1;
            for (int t = 0; t < TOTAL_ITER; t += PYRAMID_HEIGHT) {
                int temp = src; src = dst; dst = temp;
                calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(
                    min(PYRAMID_HEIGHT, TOTAL_ITER - t),
                    power_d, temp_d[src], temp_d[dst],
                    GRID_COLS, GRID_ROWS, borderCols, borderRows,
                    Cap, Rx, Ry, Rz, step
                );
            }

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

    // Copy back final temp and write to file
    vector<float> out_h(size);
    benchmark_cuda_check(cudaMemcpy(out_h.data(), temp_d[0], size * sizeof(float), cudaMemcpyDeviceToHost));
    writeoutput(out_h.data(), GRID_ROWS, GRID_COLS, OUT_FILE.c_str());

    // Save JSON
    ofstream json(benchmark_log_path("tpc_results_hotspot"));
    json << "{\n  \"total_tpcs\": " << MAX_TPC << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        json << "    {\"TPC\": " << results[i].first
             << ", \"ExecTime_ms\": " << results[i].second << "}";
        if (i + 1 != results.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n}\n";
    json.close();

    std::cout << "[DONE] " << benchmark_log_path("tpc_results_hotspot") << std::endl;

    benchmark_cuda_check(cudaFree(power_d));
    benchmark_cuda_check(cudaFree(temp_d[0]));
    benchmark_cuda_check(cudaFree(temp_d[1]));
    cout << "[CUDA Benchmark] Done.\n";
    benchmark_cuda_check(cudaStreamDestroy(stream));
    return 0;
}
