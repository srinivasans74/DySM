// cuda_runner.cpp
// Adds a SEPARATE template + helper for "mminterupt" that:
//  - creates the handle
//  - stores it so controller can call interrupt later
//  - runs (blocking)
//  - clears handle and destroys on exit
//
// Everything else stays the same.

#include "cuda_runner.h"
#include "Parameters.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

#include "../application/matrixmul/matrixMul.h"
#include "../application/hist/hist.h"
#include "../application/hotspot/hotspot.h"
#include "../application/projection/projection.h"
#include "../application/stencil/stencil.h"
#include "../application/stereodisparity/stereodisparity.h"
//#include "../application/srad/srad.h"
#include "../application/matrixmulinter/matrixMul.h"   
#include "../application/mttkrp/mttkrp.h"   

extern std::atomic<bool> gpu_running;

// -------------------- Generic runner (no interrupt) --------------------
template<typename CreateFn, typename RunFn, typename DestroyFn>
static void run_task_blocking(const char* name,
                              int task_id,
                              parameters* p,
                              CreateFn create_fn,
                              RunFn run_fn,
                              DestroyFn destroy_fn)
{
    std::cout << "[GPU] Launching persistent " << name
              << " workload (Task " << task_id << ")...\n";

    void* handle = create_fn(task_id, p);
    if (!handle) {
        std::cerr << "[GPU] Failed to create " << name << " handle!\n";
        return;
    }

    try {
        run_fn(handle);   // blocks until gpu_running == false
    } catch (...) {
        std::cerr << "[GPU] Exception in " << name << " run loop (Task "
                  << task_id << ").\n";
    }

    destroy_fn(handle);
    std::cout << "[GPU] " << name << " workload stopped for task "
              << task_id << ".\n";
}

// -------------------- Interrupt-capable runner (stores handle) --------------------
// We keep handles only for mminterupt tasks.
static std::mutex g_mm_int_mu;
static std::vector<void*> g_mm_int_handles;  // index by task_id

static void store_mm_int_handle(int task_id, void* h)
{
    std::lock_guard<std::mutex> lk(g_mm_int_mu);
    if ((int)g_mm_int_handles.size() <= task_id)
        g_mm_int_handles.resize(task_id + 1, nullptr);
    g_mm_int_handles[task_id] = h;
}

static void* load_mm_int_handle(int task_id)
{
    std::lock_guard<std::mutex> lk(g_mm_int_mu);
    if (task_id < 0 || task_id >= (int)g_mm_int_handles.size()) return nullptr;
    return g_mm_int_handles[task_id];
}

// Call this from your controller thread when you want to interrupt that task:
void mminterupt_interrupt_task(int task_id)
{
    void* h = load_mm_int_handle(task_id);
    if (!h) {
        std::cout << "[CTRL] mminterupt task " << task_id << " not running (no handle)\n";
        return;
    }
    matrixMul_interrupt(h);
    std::cout << "[CTRL] Interrupt sent to mminterupt task " << task_id << "\n";
}

template<typename CreateFn, typename RunFn, typename DestroyFn, typename InterruptFn>
static void run_task_blocking_interruptible(const char* name,
                                           int task_id,
                                           parameters* p,
                                           CreateFn create_fn,
                                           RunFn run_fn,
                                           DestroyFn destroy_fn,
                                           InterruptFn /*interrupt_fn*/)
{
    // NOTE: interrupt_fn is not used here; it will be called externally using the stored handle.
    // We keep it in the signature so your call site can pass matrixMul_interrupt.

    std::cout << "[GPU] Launching persistent " << name
              << " workload (Task " << task_id << ")...\n";

    void* handle = create_fn(task_id, p);
    if (!handle) {
        std::cerr << "[GPU] Failed to create " << name << " handle!\n";
        return;
    }

    // Store handle so controller can call mminterupt_interrupt_task(task_id)
    store_mm_int_handle(task_id, handle);

    try {
        run_fn(handle);   // blocks until gpu_running == false (or internal stop)
    } catch (...) {
        std::cerr << "[GPU] Exception in " << name << " run loop (Task "
                  << task_id << ").\n";
    }

    // Clear handle BEFORE destroy so controller won't call after free
    store_mm_int_handle(task_id, nullptr);

    destroy_fn(handle);
    std::cout << "[GPU] " << name << " workload stopped for task "
              << task_id << ".\n";
}

// -------------------- Dispatcher --------------------
void runCudaKernel(const std::string& workload, int task_id, parameters* p)
{
    if (workload == "mm") {
        run_task_blocking("MatrixMul", task_id, p,
            matrixMul_create, matrixMul_run, matrixMul_destroy);
    }
    // else if (workload == "vadd") {
    //     run_task_blocking("VectorAdd", task_id, p,
    //         vectorAdd_create, vectorAdd_run, vectorAdd_destroy);
    // }
    else if (workload == "projection") {
        run_task_blocking("Projection", task_id, p,
            projection_create, projection_run, projection_destroy);
    }
    else if (workload == "hist") {
        run_task_blocking("Histogram", task_id, p,
            hist_create, hist_run, hist_destroy);
    }
    else if (workload == "hotspot") {
        run_task_blocking("Hotspot", task_id, p,
            hotspot_create, hotspot_run, hotspot_destroy);
    }
    // else if (workload == "srad") {
    //     run_task_blocking("SRAD", task_id, p,
    //         srad_create, srad_run, srad_destroy);
    // }
    else if (workload == "stencil") {
        run_task_blocking("Stencil3D", task_id, p,
            stencil_create, stencil_run, stencil_destroy);
    }
    else if (workload == "stereo") {
        run_task_blocking("StereoDisparity", task_id, p,
            stereo_create, stereo_run, stereo_destroy);
    }
     else if (workload == "mttkrp") {
        run_task_blocking("mttkrp", task_id, p,
           mttkrp_create, mttkrp_run, mttkrp_destroy);
    }
    
    else if (workload == "mminterupt") {
        run_task_blocking_interruptible("mminterupt", task_id, p,
            matrixMul_create, matrixMul_run, matrixMul_destroy, matrixMul_interrupt);
    }
    else {
        std::cerr << "[GPU] Unknown workload: " << workload << "\n";
    }
}