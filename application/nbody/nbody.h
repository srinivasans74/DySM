#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a NBody task (returns opaque handle)
void *nbody_create(int task_id, parameters *p);

// Runs the continuous periodic loop
void nbody_run(void *handle);

// Destroys task and frees resources after run() returns
void nbody_destroy(void *handle);

#ifdef __cplusplus
}
#endif
