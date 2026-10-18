#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif
// Creates and initializes a Projection task (returns opaque handle)
void* projection_create(int task_id, parameters* p);

// Runs the continuous periodic loop
void  projection_run(void* handle);

// Stops and destroys task, frees resources
void  projection_destroy(void* handle);

} // extern "C"


#ifdef __cplusplus
#endif