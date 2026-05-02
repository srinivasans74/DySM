#ifndef GAIN_MATRIX_H_
#define GAIN_MATRIX_H_

constexpr int NUM_TASKS = 2;
constexpr int NUM_INPUTS = 2;

constexpr float gain[NUM_TASKS][NUM_TASKS] = {
    {10.000000000f, 6.958485934f},
    {2.470503078f, 10.000000000f}
};

#endif
