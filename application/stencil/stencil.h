#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void* stencil_create(int task_id, parameters* p);
void  stencil_run(void* handle);
void  stencil_destroy(void* handle);

#ifdef __cplusplus
}
#endif