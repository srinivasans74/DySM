#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif
    
void* projection_create(int task_id, parameters* p);
void  projection_run(void* handle);
void  projection_destroy(void* handle);

} // extern "C"


#ifdef __cplusplus
#endif