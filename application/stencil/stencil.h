#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a Histogram task (returns opaque handle)
void* stencil_create(int task_id, parameters* p);

// Runs the continuous periodic loop
void  stencil_run(void* handle);

// Stops and destroys task, frees resources
void  stencil_destroy(void* handle);

#ifdef __cplusplus
}
#endif