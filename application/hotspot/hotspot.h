#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a Histogram task (returns opaque handle)
void* hotspot_create(int task_id, parameters* p);

// Runs the continuous periodic loop
void  hotspot_run(void* handle);

// Stops and destroys task, frees resources
void  hotspot_destroy(void* handle);

#ifdef __cplusplus
}
#endif