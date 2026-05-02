#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void* stereo_create(int task_id, parameters* p);
void  stereo_run(void* handle);
void  stereo_destroy(void* handle);

#ifdef __cplusplus
}
#endif