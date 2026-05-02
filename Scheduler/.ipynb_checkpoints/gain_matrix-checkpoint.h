
#ifndef GAIN_MATRIX_H_
#define GAIN_MATRIX_H_

constexpr int NUM_TASKS = 2;
constexpr int NUM_INPUTS = 2;

constexpr float gain[NUM_TASKS][NUM_TASKS] = {
    {6.439778024f, 4.534613333f},
    {1.609944506f, 10.000000000f}
};

#endif
