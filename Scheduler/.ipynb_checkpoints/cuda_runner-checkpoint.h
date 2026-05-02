#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include "Parameters.h"

extern std::atomic<bool> gpu_running;
extern void runCudaKernel(const std::string& workload, int task_id, parameters* p);
extern void mminterupt_interrupt_task(int task_id);

