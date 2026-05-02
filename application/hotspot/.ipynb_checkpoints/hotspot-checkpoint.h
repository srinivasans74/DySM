#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void* hotspot_create(int task_id, parameters* p);
void  hotspot_run(void* handle);
void  hotspot_destroy(void* handle);

#ifdef __cplusplus
}
#endif