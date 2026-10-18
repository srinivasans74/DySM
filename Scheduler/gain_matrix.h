#ifndef GAIN_MATRIX_H_
#define GAIN_MATRIX_H_

constexpr int NUM_TASKS = 3;
constexpr int NUM_INPUTS = 3;

constexpr float gain[NUM_TASKS][NUM_TASKS] = {
    {10.000000000f, 2.282762294f, 1.482558140f},
    {5.660573477f, 10.000000000f, 1.482558140f},
    {5.660573477f, 2.282762294f, 10.000000000f}
};

// Model source CSV: /home/yuanyang/ECRTS_original_20261017/ECRTS_25/experiment/6.3/3traces/trace_projection_hist_stencil.csv
// Model operating total SMs: 170
// Same plant used for gain above: delta_RRT = model_B * delta_SM.
constexpr double model_B[NUM_TASKS][NUM_TASKS] = {
    {-2.900036319749966e-05, 2.1750272398124746e-06, 2.1750272398124746e-06},
    {5.3934224948261574e-06, -7.1912299931015434e-05, 5.3934224948261574e-06},
    {8.3044982698961941e-06, 8.3044982698961941e-06, -0.00011072664359861593}
};

#endif
