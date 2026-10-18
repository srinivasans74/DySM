#include "Parameters.h"
#include "controller.h"
#include "gain_matrix.h"
#include "cuda_runner.h"  
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem> // Requires C++17
// Structure for one CSV row
struct TaskInfo {
    std::string workload;
    double task_rate;
    double setpoint;
    bool tunable;
    double range;  

};
float inner_loop_duration;
float outer_loop_duration;
// Load tasks from CSV
std::atomic<bool> gpu_running{true};

#include <cstdint>
#include <algorithm>



// Must match your platform mapping
static inline int sm_to_tpc(int sm)
{
    // Jetson / Ampere: 2 SMs per TPC
    return (sm + 1) / 2;   // integer ceil(sm/2)
}

uint64_t assignmask(int sm_req, int total_sms)
{
    const int TOTAL_SMS  = total_sms;
    const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

    if (TOTAL_SMS <= 0 || TOTAL_TPCS <= 0)
        return 0ULL;
    sm_req = std::clamp(sm_req, 1, TOTAL_SMS);
    int tpc_req = sm_to_tpc(sm_req);
    tpc_req = std::clamp(tpc_req, 1, TOTAL_TPCS);
    uint64_t enabled = (tpc_req >= 64)
                         ? ~0ULL
                         : ((1ULL << tpc_req) - 1ULL);

    uint64_t disable = ~enabled; // Disable all unassigned TPCs.

    return disable;
}

std::vector<TaskInfo> loadTasksFromCSV(const std::string& filename) {
    std::vector<TaskInfo> tasks;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV file: " << filename << std::endl;
        return tasks;
    }

    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }

        std::stringstream ss(line);
        std::string sl_no, workload, rate, setpoint, tunable, range_str;
        std::getline(ss, sl_no, ',');
        std::getline(ss, workload, ',');
        std::getline(ss, rate, ',');
        std::getline(ss, setpoint, ',');
        std::getline(ss, tunable, ',');
        std::getline(ss, range_str, ',');

        if (workload.empty()) continue;

        TaskInfo t;
        t.workload = workload;
        t.task_rate = std::stod(rate);
        t.task_rate=t.task_rate/1000;
        t.setpoint = std::stod(setpoint);
        t.tunable = (tunable.find("tunable") != std::string::npos);

        int range_val = 0;
        if (!range_str.empty()) {
            try {
                range_val = std::stoi(range_str);  // directly parse integer
            } catch (...) {
                range_val = 0;
            }
        }
       

        t.range = range_val;
        tasks.push_back(t);
    }

    file.close();
    return tasks;
}

// ===== Main =====
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <task_csv_path> <duration_seconds> <output_dir> <solution_id>\n";
        return 1;
    }

    std::string csv_path = argv[1];
    int duration = std::atoi(argv[2]);
    std::string filepath = argv[3];
    int solution = std::atoi(argv[4]);

    // ============================================================
    // === Select Scheduling / Control Method =====================
    // ============================================================
    std::cout << "\n===== SELECTED CONTROL MODE =====\n";
    if (solution == 3) {
        std::cout << "Proposed";
    } 
    else if (solution == 5) {
        std::cout << "STGM";
    } 
    else if (solution == 4) {
        std::cout << "Adhoc\n";
    } 
     else if (solution == 6) {
        std::cout << "Equal\n";
    }
    else if (solution == 7) {
        std::cout << "MPC\n";
    } 
    
    else {
        std::cerr << "Unknown solution ID: " << solution << " (must be 3, 4, 5, 6, or 7)\n";
        return 1;
    }
    std::cout << "=================================\n\n";

    // ============================================================
    // === Load Tasks =============================================
    // ============================================================
    auto task_list = loadTasksFromCSV(csv_path);
    if (task_list.empty()) {
        std::cerr << "No tasks loaded from " << csv_path << std::endl;
        return 1;
    }

    int num_tasks = static_cast<int>(task_list.size());
    if (solution == 7 && num_tasks != NUM_TASKS) {
        std::cerr << "[MPC] Regenerate gain_matrix.h for this task CSV.\n";
        return 1;
    }
    std::cout << "Loaded " << num_tasks << " tasks from " << csv_path << "\n";

    parameters* p = new parameters(num_tasks, true);
    inner_loop_duration = 1.00;
    for (int i = 0; i < num_tasks; ++i) {
        const auto& t = task_list[i];
        p->setTaskConfig(i, t.workload, t.task_rate, t.setpoint, t.tunable);
        p->period[i] = t.task_rate;
        p->range[i]  = t.range;
    }

    std::cout << "\nParameters initialized and ready for controller.\n";
    p->show();
    initialize(p);
    p->finsolution=solution;
    
    // ============================================================
    // === Launch CUDA workloads ==================================
    // ============================================================
    std::vector<std::thread> gpu_threads;
    gpu_running.store(true);    
    std::cout<<"solution = "<<solution<<std::endl;


    for (int i = 0; i < num_tasks; ++i) {
    p->sms[i]   = p->maxsms / num_tasks;
    p->masks[i] = assignmask(p->sms[i], p->maxsms);
        }
    
     for (int i = 0; i < num_tasks; ++i) {
        const std::string workload = task_list[i].workload;
        gpu_threads.emplace_back([&, i, workload]() {
            runCudaKernel(workload, i, p);
        });
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    running = true;
    outerlooprunning = true;
   // std::this_thread::sleep_for(std::chrono::seconds(3));
    // ============================================================
    // === Launch Control Threads Based on Solution ===============
    // ============================================================
    std::thread ctrl_thread;
    std::thread outer_thread;
    std::thread monitor_thread(monitor, p);
    auto start_time = std::chrono::steady_clock::now();

    // if (solution == 1) {
    //     // FC-GPU baseline: only inner control loop
    //     std::cout<<"FC-GPU"<<std::endl;
    //     ctrl_thread = std::thread(FC_GPU, p);
    // }
//     if (solution == 2) {
//         // Proposed hybrid: both loops active
//         ctrl_thread = std::thread(innerloop, p);
//         outer_thread = std::thread(outerloop, p);
//         std::cout<<"Two-tier"<<std::endl;

//     }
    if (solution == 3) {
        outer_loop_duration = inner_loop_duration;
        outer_thread = std::thread(smscaling, p);
        std::cout<<"Proposed"<<std::endl;

    }
     else if (solution == 4) {
        outer_loop_duration = inner_loop_duration;
        outer_thread = std::thread(adhoc, p);
        std::cout<<"Adhoc"<<std::endl;

    }
    
    else if (solution == 5)
    {
        std::cout << "STGM" << std::endl;
        outer_loop_duration = inner_loop_duration;
        std::vector<std::string> workloads;
        std::vector<double> periods;
        std::vector<bool> tunables;

        workloads.reserve(num_tasks);
        periods.reserve(num_tasks);
        tunables.reserve(num_tasks);

        for (int i = 0; i < num_tasks; ++i) {
            workloads.push_back(task_list[i].workload);
            periods.push_back(task_list[i].task_rate*1000);   // Pi in seconds
            tunables.push_back(task_list[i].tunable);
        }

        bool ok = stgminit(p, workloads, periods, /*Np=*/1, /*exec_scale=*/1.25);
        if (!ok) {
            std::cerr << "[MAIN] STGM: UNSCHEDULABLE (keeping current sms/masks)\n";
        }

        outer_thread = std::thread(stgm, p);
        std::cout << "STGM" << std::endl;
    }
    else if (solution == 6)
    {
    
    
    std::cout<<"HPCA 2012 \t" <<std::endl;
    outer_loop_duration = inner_loop_duration;
    for (int i = 0; i < num_tasks; ++i) 
        {
        p->sms[i]   = p->maxsms / num_tasks;
        p->masks[i] = assignmask(p->sms[i], p->maxsms);
        }
        outer_thread = std::thread(stgm, p);

    }
    
    else if (solution == 7) {
        outer_loop_duration = inner_loop_duration;
        std::cout << "MPC\n";
        outer_thread = std::thread(dysm_mpc, p);
    }
    
    // ============================================================
    // === Run for Given Duration ================================
    // ============================================================
    bool interrupt_fired = false;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - start_time).count();  // seconds (fractional)

        if (elapsed >= duration) {
            std::cout << "[MAIN] Time limit reached (" << elapsed << "s). Exiting...\n";
            break;
        }
    
        
        
        if (!interrupt_fired && elapsed >= (duration / 2.0)) 
        {
            interrupt_fired = true;
            std::cout << "[MAIN] Half time reached (" << elapsed << "s). Interrupting tunable tasks...\n";

            for (int i = 0; i < num_tasks; ++i) {
                if (!task_list[i].tunable) continue;

                if (task_list[i].workload == "mminterupt") {
                    mminterupt_interrupt_task(i);
                } else {
                    std::cout << "[MAIN] Task " << i << " is tunable but workload="
                              << task_list[i].workload << " has no interrupt hook.\n";
                }
            }
        }
        
     

       
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
    }
        
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    

    // ============================================================
    // === Graceful Shutdown ======================================
    // ============================================================
    running = false;
    outerlooprunning = false;
    gpu_running.store(false);

    if (ctrl_thread.joinable()) ctrl_thread.join();
    if (outer_thread.joinable()) outer_thread.join();
    if (monitor_thread.joinable()) monitor_thread.join();

    // for (auto& t : gpu_threads)
    //     if (t.joinable()) t.join();


    for (auto& t : gpu_threads) {
        if (t.joinable()) {
            t.detach();  // GPU threads clean up internally
        }
    }


    std::cout << "[MAIN] All threads stopped. Experiment complete.\n";
    p->dumpLogsToJSON(filepath);
    p->dumpinterLogs(filepath);
    p->dumpWallClock(filepath);
    return 0;
}


