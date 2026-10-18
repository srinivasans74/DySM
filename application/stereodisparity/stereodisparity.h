#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a Histogram task (returns opaque handle)
void* stereo_create(int task_id, parameters* p);

// Runs the continuous periodic loop
void  stereo_run(void* handle);

// Stops and destroys task, frees resources
void  stereo_destroy(void* handle);

#ifdef __cplusplus
}
#endif