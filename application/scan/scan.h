#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a Scan task (returns opaque handle)
void *scan_create(int task_id, parameters *p);

// Runs the continuous periodic loop
void scan_run(void *handle);

// Destroys task and frees resources after run() returns
void scan_destroy(void *handle);

#ifdef __cplusplus
}
#endif
