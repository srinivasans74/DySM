#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and initializes a ConvolutionSeparable task (returns opaque handle)
void *convolution_create(int task_id, parameters *p);

// Runs the continuous periodic loop
void convolution_run(void *handle);

// Destroys task and frees resources after run() returns
void convolution_destroy(void *handle);

#ifdef __cplusplus
}
#endif
