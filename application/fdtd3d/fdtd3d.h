#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a FDTD3D task (returns opaque handle)
void *fdtd3d_create(int task_id, parameters *p);

// Runs the continuous periodic loop
void fdtd3d_run(void *handle);

// Destroys task and frees resources after run() returns
void fdtd3d_destroy(void *handle);

#ifdef __cplusplus
}
#endif
