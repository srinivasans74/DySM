#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// Simple error checking macro
void check(cudaError_t err, const char* const func, const char* const file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << std::endl;
        std::cerr << func << " failed with error " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void printDeviceProperties(cudaDeviceProp prop) {
    printf("  Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.0f MBytes\n", (float)prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("  Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Grid Size (X, Y, Z): %d, %d, %d\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("------------------------------------------\n");
}

int main() {
    int deviceCount = 0;
    
    // 1. Get the number of CUDA-capable devices
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    printf("Total CUDA Devices Found: %d\n", deviceCount);
    printf("==========================================\n");

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    // 2. Loop through all devices and get their properties
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        
        // Get properties for the current device (i)
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));

        printf("GPU Device #%d:\n", i);
        printDeviceProperties(prop);
    }

    return 0;
}
