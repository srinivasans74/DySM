#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <future>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "libsmctrl.h"
using namespace std::chrono;


typedef double datatype;

inline std::string mask_hex(uint128_t mask)
{
    std::ostringstream out;
    out << std::hex << static_cast<uint64_t>(mask >> 64)
        << std::setw(16) << std::setfill('0') << static_cast<uint64_t>(mask);
    return out.str();
}

class parameters {
public:
    int  tasks;
    bool rateadaption;

    std::vector<std::string> workloads;
    std::vector<datatype> setpoints;
    std::vector<datatype> taskrates;
    std::vector<bool> tunable;
    int maxsms;
    int minsms;
    datatype* sms;
    datatype* rtr;
    datatype* error;
    datatype* outererror;
    datatype* delta;
    datatype* range;
    datatype* outerdelta;
    datatype* period;
    datatype* responsetimes;
    datatype* preemption;
    datatype* executiontimes;
    std::mutex* task_mutexes;  // one per task
    uint128_t* masks;
    char*path;
    int finsolution;
    steady_clock::time_point start_time;
    std::string setpointsel[3];
    std::vector<datatype>wallclocktime;

    
    std::vector<std::vector<datatype>> output_responsetime;
    std::vector<std::vector<datatype>> output_executiontime;
    std::vector<std::vector<datatype>> output_deadlinemiss;
    std::vector<std::vector<datatype>> output_rtr_measured;
    std::vector<std::vector<datatype>> output_current_taskrate;
    std::vector<std::vector<datatype>> output_preemptiontime;
    std::vector<std::vector<datatype>> output_setpoints;

    
    std::vector<bool>controller_hold;
    std::vector<std::vector<datatype>> outer_output_responsetime;
    std::vector<std::vector<datatype>> outer_output_executiontime;
    std::vector<std::vector<datatype>> outer_output_deadlinemiss;
    std::vector<std::vector<datatype>> outer_output_rtr_measured;
    std::vector<std::vector<datatype>> outer_output_current_taskrate;
    std::vector<std::vector<datatype>> outer_output_sms;

    std::vector<std::vector<datatype>> inter_responsetime;
    std::vector<std::vector<datatype>> inter_executiontime;
    std::vector<std::vector<datatype>> inter_deadlinemiss;
    std::vector<std::vector<datatype>> inter_rtr_measured;
    std::vector<std::vector<datatype>> inter_current_taskrate;
    std::vector<std::vector<datatype>> inter_preemptiontime;


    parameters(int n, bool adapt = false);
    ~parameters();

    void setTaskConfig(int idx, const std::string& wl, double rate, double sp, bool tun);
    void show() const;
    inline int getTasks() const { return tasks; }
    inline uint128_t getMask(int task_id) {
        std::lock_guard<std::mutex> lock(task_mutexes[task_id]);
        return masks[task_id];
    }
    void dumpWallClock(const std::string& output_dir) const;
    void dumpLogsToJSON(const std::string& output_dir) const;
    void dumpinterLogs(const std::string& output_dir) const;

        // Output logs (per-task vectors)
 
};

extern bool stgminit(parameters* p,
                     const std::vector<std::string>& workloads,
                     const std::vector<double>& periods_ms,
                     int Np = 1,
                     double exec_scale = 0.25);
#endif // PARAMETERS_H
