#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif
void* hist_create(int task_id, parameters* p);
void  hist_run(void* handle);
void  hist_destroy(void* handle);

#ifdef __cplusplus
}
#endif