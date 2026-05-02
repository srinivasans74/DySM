#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle-based CUDA workload control
void* mttkrp_create(int task_id, parameters* p);
void  mttkrp_run(void* handle);
void  mttkrp_destroy(void* handle);

#ifdef __cplusplus
}
#endif