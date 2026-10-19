#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a Reduction task (returns opaque handle)
void *reduction_create(int task_id, parameters *p);

// Runs the continuous periodic loop
void reduction_run(void *handle);

// Destroys task and frees resources after run() returns
void reduction_destroy(void *handle);

#ifdef __cplusplus
}
#endif
