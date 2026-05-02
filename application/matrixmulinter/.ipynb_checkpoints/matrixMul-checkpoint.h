#pragma once
#include "../../Scheduler/Parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

void* matrixMul_create(int task_id, parameters* p);
void  matrixMul_run(void* handle);
void  matrixMul_destroy(void* handle);
void  matrixMul_interrupt(void* handle);

#ifdef __cplusplus
}
#endif