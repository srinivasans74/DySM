#ifndef GAIN_MATRIX_H_
#define GAIN_MATRIX_H_

constexpr int NUM_TASKS = 2;
constexpr int NUM_INPUTS = 2;

constexpr float gain[NUM_TASKS][NUM_TASKS] = {
    {10.000000000f, 0.986481549f},
    {2.590131454f, 6.576543661f}
};

// Model source CSV: /home/yuanyang/ECRTS_original_20261017/ECRTS_25/experiment/traces/trace_mm_stencil.csv
// Model operating total SMs: 80
// Same plant used for gain above: delta_RRT = model_B * delta_SM.
constexpr double model_B[NUM_TASKS][NUM_TASKS] = {
    {-0.0001184902682798527, 1.7773540241977904e-05},
    {4.6666666666666665e-05, -0.00031111111111111113}
};

#endif
