#include "Parameters.h"
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
parameters::parameters(int n, bool adapt)
{
    if (n <= 0)
        throw std::invalid_argument("Number of tasks must be > 0");

    tasks = n;
    rateadaption = adapt;

    // --- Allocate raw arrays ---
    rtr = new datatype[tasks];
    sms = new datatype[tasks];
    period = new datatype[tasks];
    error = new datatype[tasks];
    outererror = new datatype[tasks];
    outerdelta=new datatype[tasks];
    delta = new datatype[tasks];
    range=new datatype[tasks];
    responsetimes = new datatype[tasks];
    executiontimes = new datatype[tasks];
    task_mutexes = new std::mutex[tasks];
    masks = new uint64_t[tasks];
    start_time = steady_clock::now();
    preemption = new datatype[tasks];
    controller_hold.assign(tasks, false);
    for (int i = 0; i < tasks; ++i) {
        rtr[i] = 0.0;
        sms[i] = 0;
        period[i] = 0.0;
        responsetimes[i] = 0.0;
        preemption[i]=0;
        executiontimes[i] = 0.0;
        error[i] = 0.0;
        outererror[i] = 0.0;
        delta[i] = 0.0;
        outerdelta[i] = 0.0;
        range[i] = 0.0;
    }

    // --- Resize configuration vectors ---
    workloads.resize(tasks, "undefined");
    taskrates.resize(tasks, 0.0);
    setpoints.resize(tasks, 0.0);
    tunable.resize(tasks, false);
    // --- Allocate per-task log vectors ---
    output_responsetime.resize(tasks);
    output_executiontime.resize(tasks);
    output_deadlinemiss.resize(tasks);
    output_rtr_measured.resize(tasks);
    output_current_taskrate.resize(tasks);
    output_preemptiontime.resize(tasks);


    
    output_setpoints.resize(tasks);
    //output_preemptiontime.resize(tasks);
        
    outer_output_responsetime.resize(tasks);
    outer_output_executiontime.resize(tasks);
    outer_output_deadlinemiss.resize(tasks);
    outer_output_rtr_measured.resize(tasks);
    outer_output_current_taskrate.resize(tasks);
    outer_output_sms.resize(tasks);

    cudaDeviceProp prop;
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        device_id = 0; // fallback to device 0
    }
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        std::cerr << "[parameters] Warning: Unable to query GPU properties. Setting default SM count = 80.\n";
        maxsms = 80; // default for e.g., V100
    } else {
        maxsms = prop.multiProcessorCount;
        std::cout << "[parameters] Detected GPU: " << prop.name
                  << " (" << maxsms << " SMs)\n";
    }
    minsms = 1;

    
    inter_responsetime.resize(tasks);
    inter_executiontime.resize(tasks);
    inter_rtr_measured.resize(tasks);
    inter_deadlinemiss.resize(tasks);
    inter_current_taskrate.resize(tasks);
    inter_preemptiontime.resize(tasks);
    setpointsel[0] = "FC-GPU";
    setpointsel[1] = "SM-Masking";
    setpointsel[2] = "Proposed";

    std::cout << "[parameters] Initialized with " << tasks
              << " tasks. RateAdaption = " << (rateadaption ? "true" : "false") << "\n";
}

parameters::~parameters()
{
    delete[] rtr;
    delete[] sms;
    delete[] period;
    delete[] error;
    delete[] outererror;
    delete[] outerdelta;
    delete[] responsetimes;
    delete[] executiontimes;
    delete[] task_mutexes;
    delete[] masks;
    delete[] delta;
    delete[] range;
    delete[] preemption;
    //delete[] preemptiontime;
    std::cout << "[parameters] Memory released.\n";
}

void parameters::setTaskConfig(int idx, const std::string& wl, double rate, double sp, bool tun)
{
    if (idx < 0 || idx >= tasks) {
        std::cerr << "[parameters] Invalid task index " << idx << "\n";
        return;
    }

    std::lock_guard<std::mutex> lock(task_mutexes[idx]);
    workloads[idx] = wl;
    taskrates[idx] = rate;
    setpoints[idx] = sp;
    tunable[idx] = tun;
}

void parameters::show() const
{
    std::cout << "\n=== [System Configuration Summary] ===\n";
    std::cout << "Total Tasks     : " << tasks << "\n"
              << "Rate Adaption   : " << (rateadaption ? "true" : "false") << "\n\n";

    std::cout << std::left << std::setw(5) << "ID"
              << std::setw(12) << "Workload"
              << std::setw(10) << "Rate"
              << std::setw(10) << "Setpoint"
              << std::setw(10) << "Tunable"
              << std::setw(10) << "Delta" << "\n";

    std::cout << "-----------------------------------------------\n";

    for (int i = 0; i < tasks; ++i) {
        std::cout << std::left << std::setw(5) << (i + 1)
                  << std::setw(12) << workloads[i]
                  << std::setw(10) << taskrates[i]
                  << std::setw(10) << setpoints[i]
                  << std::setw(10) << (tunable[i] ? "Yes" : "No")
                  << std::setw(10) << range[i]<< "\n" ;
    }

    std::cout << "-----------------------------------------------\n";
}


void parameters::dumpWallClock(const std::string& output_dir) const
{
    namespace fs = std::filesystem;
    fs::create_directories(output_dir);

    std::string filename = output_dir + "/wallclocktime";
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[parameters] Failed to open " << filename << "\n";
        return;
    }

    file << std::fixed << std::setprecision(4);
    file << "{\n  \"wallclocktime\": [";

    for (size_t i = 0; i < wallclocktime.size(); ++i) {
        file << wallclocktime[i];
        if (i + 1 < wallclocktime.size())
            file << ", ";
    }

    file << "]\n}\n";
    file.close();

    std::cout << "[parameters] Dumped wallclocktime → " << filename << "\n";
}

void parameters::dumpLogsToJSON(const std::string& output_dir) const
{
    namespace fs = std::filesystem;
    fs::create_directories(output_dir);  // ensure logs/ exists

    for (int i = 0; i < tasks; ++i) {
        std::string filename = output_dir + "/task_" + std::to_string(i);
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[parameters] Failed to open " << filename << " for writing.\n";
            continue;
        }

        file << std::fixed << std::setprecision(4);
        file << "{\n";

        auto writeArray = [&](const std::string& key, const std::vector<datatype>& vec, bool isLast) {
            file << "  \"" << key << "\": [";
            for (size_t j = 0; j < vec.size(); ++j) {
                file << vec[j];
                if (j + 1 < vec.size()) file << ", ";
            }
            file << "]";
            if (!isLast) file << ",\n";
        };

        writeArray("responsetime", output_responsetime[i], false);
        writeArray("executiontime", output_executiontime[i], false);
        writeArray("deadlinemiss", output_deadlinemiss[i], false);
        writeArray("rtr_measured", output_rtr_measured[i], false);
        writeArray("interresponsetime", inter_responsetime[i], false);
        writeArray("interexecutiontime", inter_executiontime[i], false);
        writeArray("intercurrent_taskrate", inter_current_taskrate[i], false);        
        writeArray("current_taskrate", output_current_taskrate[i], false);
        writeArray("setpoint", output_setpoints[i], false);
        writeArray("outerresponsetime", outer_output_responsetime[i], false);
        writeArray("outerexecutiontime", outer_output_executiontime[i], false);
        writeArray("outerdeadlinemiss", outer_output_deadlinemiss[i], false);
        writeArray("outerrtr_measured", outer_output_rtr_measured[i], false);
        writeArray("outersms", outer_output_sms[i], false);
        writeArray("outercurrent_taskrate", outer_output_current_taskrate[i], true);


        file << "\n}\n";
        file.close();

        std::cout << "[parameters] Dumped metrics for task " << i
                  << "at" << filename << "\n";
    }

}


void parameters::dumpinterLogs(const std::string& output_dir) const
{
    namespace fs = std::filesystem;

    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    for (int i = 0; i < tasks; ++i) {

        // ---------- Response Time ----------
        {
            std::string filename = output_dir + "/inter_responsetime_task" + std::to_string(i) + ".txt";
            std::ofstream file(filename);
            if (file.is_open()) {
                for (const auto& val : inter_responsetime[i]) {
                    file << val << "\n";
                }
                file.close();
            } else {
                std::cerr << "[WARN] Could not open " << filename << " for writing.\n";
            }
        }

        // ---------- Execution Time ----------
        {
            std::string filename = output_dir + "/inter_executiontime_task" + std::to_string(i) + ".txt";
            std::ofstream file(filename);
            if (file.is_open()) {
                for (const auto& val : inter_executiontime[i]) {
                    file << val << "\n";
                }
                file.close();
            } else {
                std::cerr << "[WARN] Could not open " << filename << " for writing.\n";
            }
        }

        // ---------- Task Rate ----------
        {
            std::string filename = output_dir + "/inter_current_taskrate_task" + std::to_string(i) + ".txt";
            std::ofstream file(filename);
            if (file.is_open()) {
                for (const auto& val : inter_current_taskrate[i]) {
                    file << val << "\n";
                }
                file.close();
            } else {
                std::cerr << "[WARN] Could not open " << filename << " for writing.\n";
            }
        }
    }

    std::cout << "[INFO] Intermediate logs written to " << output_dir << std::endl;
}