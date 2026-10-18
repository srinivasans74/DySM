#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <atomic>
#include <vector>
#include "Parameters.h"

extern std::atomic<bool> running;
extern std::atomic<bool> outerlooprunning;

extern float inner_loop_duration;
extern float outer_loop_duration;
// ===== Per-task runtime measurement container =====
struct Task {
    std::vector<float> response_time_measurement;
    std::vector<float> execution_time_measurement;
    std::vector<float> task_rate_measurement;
    std::vector<float> rtr_measurement;
//    std::vector<float> preemption_measurement;

};

// ===== Functions =====
void initialize(parameters* p);
//void FC_GPU(parameters* p);
void smscaling(parameters* p);
void monitor(parameters* p);
//void innerloop(parameters* p);
// void outerloop(parameters* p);
void adhoc(parameters* p);
void stgm(parameters* p);
void dysm_mpc(parameters* p);
// ===== Helper functions =====
float calculateDeadlineMiss(const std::vector<float>& responseTimes, float period);
float calculateAverage(const std::vector<float>& values);

#endif // CONTROLLER_H
