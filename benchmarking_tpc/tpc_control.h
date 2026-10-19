#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "libsmctrl.h"

inline void benchmark_cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA benchmark failed: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline int benchmark_tpc_count(int device = 0) {
    uint32_t count = 0;
    if (libsmctrl_get_tpc_info_cuda(&count, device) != 0 || count == 0 || count > 128) {
        std::fprintf(stderr, "Cannot obtain a TPC count supported by the 128-bit mask.\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(count);
}

// Enable the first active TPCs; all remaining mask bits stay disabled.
inline uint128_t benchmark_tpc_mask(int active) {
    if (active < 1 || active > 128) {
        std::fprintf(stderr, "Active TPC count must be between 1 and 128.\n");
        std::exit(EXIT_FAILURE);
    }
    return active == 128 ? uint128_t(0) : ~((uint128_t(1) << active) - 1);
}

inline std::string benchmark_log_path(const char* name) {
    const char* directory = std::getenv("BENCHMARK_LOG_DIR");
    return std::string(directory ? directory : "logs") + "/" + name;
}
