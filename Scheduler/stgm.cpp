#include "Parameters.h"
#include <unordered_map>
#include <map>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>   // setw, left

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Declare (defined elsewhere, e.g., scheduler.cpp)
uint128_t assignmask(int sm_req, int total_sms);

// workload -> (TPC -> ExecTime_ms)
using TPCMap = std::map<int, double>;
static std::unordered_map<std::string, TPCMap> workload_profile_db;

// JSON paths (keys MUST match your workload names passed in)
static const std::unordered_map<std::string, std::string> workload_json_path = {
    {"mm",         "../benchmarking_tpc/logs/tpc_results_mm"},
    {"projection", "../benchmarking_tpc/logs/tpc_results_projection_mapped"},
    {"hist",       "../benchmarking_tpc/logs/tpc_results_hist"},
    {"hotspot",    "../benchmarking_tpc/logs/tpc_results_hotspot"},
    {"stencil",    "../benchmarking_tpc/logs/tpc_results_stencil"},
    {"stereo",     "../benchmarking_tpc/logs/tpc_results_stereodisparity"},
    {"mminterupt",  "../benchmarking_tpc/logs/tpc_results_mm"}
};

// Jetson/Ampere mapping: 2 SM per TPC
static inline int sm_to_tpc(int sm) { return (sm + 1) / 2; }

// -------------------- NO-OVERLAP MASK HELPERS --------------------
// Mask convention: bit=1 means "disable this TPC" (same as your assignmask()).
// We enable ONLY the slice [tpc_start, tpc_start+tpc_count-1] and disable everything else.
static uint128_t mask_disable_all_except_range(int tpc_start, int tpc_count, int total_tpcs)
{
    if (total_tpcs <= 0) return ~uint128_t(0);

    const uint128_t all_bits =
        (total_tpcs >= 128) ? ~uint128_t(0) : ((uint128_t(1) << total_tpcs) - 1ULL);

    if (tpc_count <= 0) {
        // enable none, disable all valid
        return ~uint128_t(0);
    }

    if (tpc_start < 0) tpc_start = 0;
    if (tpc_start >= total_tpcs) return ~uint128_t(0);

    int end = tpc_start + tpc_count;   // exclusive
    if (end > total_tpcs) end = total_tpcs;
    tpc_count = end - tpc_start;
    if (tpc_count <= 0) return ~uint128_t(0);

    uint128_t enable =
        (tpc_count >= 128) ? ~uint128_t(0) : (((uint128_t(1) << tpc_count) - 1ULL) << tpc_start);
    enable &= all_bits;

    uint128_t disable = ~enable;
    return disable;
}

// Allocate disjoint contiguous slices based on requested TPC counts.
// If total requested > total_tpcs, it shrinks everyone (largest-first) until it fits.
static std::vector<std::pair<int,int>> allocate_disjoint_tpcs(std::vector<int> req_tpcs,
                                                              int total_tpcs)
{
    const int n = (int)req_tpcs.size();
    std::vector<std::pair<int,int>> slices(n, {0,0});
    if (n == 0 || total_tpcs <= 0) return slices;

    for (int& x : req_tpcs) x = std::max(0, x);

    int sum = 0;
    for (int x : req_tpcs) sum += x;

    if (sum == 0) return slices;

    if (sum > total_tpcs) {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return req_tpcs[a] > req_tpcs[b]; });

        while (sum > total_tpcs) {
            bool changed = false;
            for (int k = 0; k < n && sum > total_tpcs; ++k) {
                int i = idx[k];
                if (req_tpcs[i] > 0) { req_tpcs[i]--; sum--; changed = true; }
            }
            if (!changed) break;
        }
    }

    int cursor = 0;
    for (int i = 0; i < n; ++i) {
        int cnt = req_tpcs[i];
        if (cnt <= 0) { slices[i] = {cursor, 0}; continue; }

        if (cursor + cnt > total_tpcs) cnt = std::max(0, total_tpcs - cursor);
        slices[i] = {cursor, cnt};
        cursor += cnt;
        if (cursor >= total_tpcs) break;
    }
    return slices;
}

// ------------------------------------------------------------
// Load one workload profile from its JSON file.
// Expects JSON shape:
// { "results": [ {"TPC": 1, "ExecTime_ms": 81.0}, ... ] }
// ------------------------------------------------------------
static bool load_profile_for_workload(const std::string& workload)
{
    if (workload_profile_db.count(workload)) return true;

    auto pit = workload_json_path.find(workload);
    if (pit == workload_json_path.end()) {
        std::cerr << "[STGM] No JSON path for workload='" << workload << "'\n";
        return false;
    }

    const auto profile_path =
        std::filesystem::read_symlink("/proc/self/exe").parent_path() / pit->second;
    std::ifstream f(profile_path);
    if (!f.is_open()) {
        std::cerr << "[STGM] Cannot open JSON file: " << profile_path << "\n";
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "[STGM] JSON parse error in " << pit->second << " : " << e.what() << "\n";
        return false;
    }

    if (!j.contains("results") || !j["results"].is_array()) {
        std::cerr << "[STGM] Bad JSON format (missing results array): " << pit->second << "\n";
        return false;
    }

    TPCMap profile;
    for (const auto& e : j["results"]) {
        if (!e.contains("TPC") || !e.contains("ExecTime_ms")) continue;

        int tpc = 0;
        double exec_ms = 0.0;
        try {
            tpc     = e["TPC"].get<int>();
            exec_ms = e["ExecTime_ms"].get<double>();
        } catch (...) {
            continue;
        }
        profile[tpc] = exec_ms;
    }

    if (profile.empty()) {
        std::cerr << "[STGM] Empty profile for workload='" << workload << "'\n";
        return false;
    }

    std::cout << "[STGM] Loaded profile for '" << workload << "' (" << profile.size()
              << " points) from " << profile_path << "\n";

    workload_profile_db.emplace(workload, std::move(profile));
    return true;
}

// Gi(N): execution time (ms) for N SMs
static double Gi_ms_for_sms(const std::string& workload, int sm_count)
{
    const auto& profile = workload_profile_db.at(workload);

    sm_count = std::max(1, sm_count);
    int tpc_req = sm_to_tpc(sm_count);

    auto it = profile.lower_bound(tpc_req);
    if (it != profile.end()) return it->second;

    return profile.rbegin()->second; // clamp
}

// Utilization Ui = Ei/Pi   (ALL in ms)
// Ei = Gi_ms(Ni) * exec_scale
static double util_i_ms(const std::string& workload, int Ni, double Pi_ms, double exec_scale)
{
    double gi_ms = Gi_ms_for_sms(workload, Ni);
    double Ei_ms = gi_ms * exec_scale;
    return Ei_ms / Pi_ms;
}

// WFD feasibility on Np identical cores capacity 1 each.
static bool schedulable_wfd(const std::vector<double>& U, int Np)
{
    if (Np <= 0) return false;

    std::vector<double> sorted = U;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());

    std::vector<double> load(Np, 0.0);

    for (double ui : sorted) {
        if (!std::isfinite(ui) || ui > 1.0) return false;

        int best = 0;
        for (int c = 1; c < Np; ++c)
            if (load[c] < load[best]) best = c;

        if (load[best] + ui <= 1.0 + 1e-12) load[best] += ui;
        else return false;
    }
    return true;
}

// Score = (Gi(N) - Gi(N+1))/Pi, scaled
static double marginal_gain_score_ms(const std::string& workload, int Ni, double Pi_ms, double exec_scale)
{
    double gN  = Gi_ms_for_sms(workload, Ni);
    double gNp = Gi_ms_for_sms(workload, Ni + 1);
    double delta_ms = (gN - gNp) * exec_scale;
    return delta_ms / Pi_ms;
}

static void print_state(const std::vector<std::string>& workloads,
                        const std::vector<double>& periods_ms,
                        const std::vector<int>& Ni,
                        int Np, int NSM,
                        double exec_scale)
{
    std::cout << "\n[STGM] ---- STATE ---- Np=" << Np << " NSM=" << NSM
              << " exec_scale=" << exec_scale << "\n";
    for (size_t i = 0; i < workloads.size(); ++i) {
        double gi = Gi_ms_for_sms(workloads[i], Ni[(int)i]);
        double ei = gi * exec_scale;
        double ui = (periods_ms[i] > 0) ? (ei / periods_ms[i]) : 1e9;
        std::cout << "[STGM] task " << i
                  << " wl=" << workloads[i]
                  << " Pi(ms)=" << periods_ms[i]
                  << " Ni=" << Ni[i]
                  << " Gi(ms)=" << gi
                  << " Ei(ms)=" << ei
                  << " Ui=" << ui
                  << "\n";
    }
    std::cout << "[STGM] --------------\n\n";
}

// Your requested initial table print (setpoint column header kept; value omitted)
static void print_task_table(parameters* /*p*/,
                             const std::vector<std::string>& workloads,
                             const std::vector<double>& periods_ms,
                             const std::vector<int>& sms_init)
{
    const int tasks = (int)workloads.size();

    std::cout << "\n[STGM] -----------------------------------------------\n";
    std::cout << std::left
              << std::setw(5)  << "#"
              << std::setw(12) << "workload"
              << std::setw(12) << "period_ms"
              << std::setw(12) << "setpoint"
              << std::setw(8)  << "sms"
              << "\n";
    std::cout << "[STGM] -----------------------------------------------\n";

    for (int i = 0; i < tasks; ++i) {
        std::cout << std::left
                  << std::setw(5)  << (i + 1)
                  << std::setw(12) << workloads[i]
                  << std::setw(12) << periods_ms[i]
                  << std::setw(12) << "-"          // you didn’t have setpoints available here
                  << std::setw(8)  << sms_init[i]
                  << "\n";
    }

    std::cout << "[STGM] -----------------------------------------------\n\n";
}

// ============================================================
// PUBLIC API
// ============================================================
bool stgminit(parameters* p,
              const std::vector<std::string>& workloads,
              const std::vector<double>& periods_ms,
              int Np,
              double exec_scale)
{
    if (!p) return false;

    const int n = (int)workloads.size();
    if (n == 0 || (int)periods_ms.size() != n) {
        std::cerr << "[STGM] bad input sizes\n";
        return false;
    }

    const int NSM = p->maxsms;
    if (NSM <= 0) {
        std::cerr << "[STGM] p->maxsms <= 0\n";
        return false;
    }

    // Load profiles for ALL tasks
    for (int i = 0; i < n; ++i) {
        if (!load_profile_for_workload(workloads[i])) {
            std::cerr << "[STGM] failed to load profile for " << workloads[i] << "\n";
            return false;
        }
    }

    // Ni init = 1 for ALL tasks (kept same as your logic)
    std::vector<int> Ni(n, 1);

    // Print your requested table before beginning
    print_task_table(p, workloads, periods_ms, Ni);

    // Debug print initial state
    print_state(workloads, periods_ms, Ni, Np, NSM, exec_scale);

    while (true) {
        // Build U for ALL tasks
        std::vector<double> U;
        U.reserve(n);

        for (int i = 0; i < n; ++i) {
            if (periods_ms[i] <= 0) {
                std::cerr << "[STGM] period <= 0 at task " << i << "\n";
                return false;
            }
            U.push_back(util_i_ms(workloads[i], Ni[i], periods_ms[i], exec_scale));
        }

        // Check schedulability
        if (schedulable_wfd(U, Np)) {

            // ---- COMMIT WITH NO OVERLAP ----
            // Convert total SM -> total TPC
            const int TOTAL_TPCS = sm_to_tpc(NSM);

            // Convert each Ni (SM) -> requested TPCs
            std::vector<int> req_tpcs(n, 0);
            for (int i = 0; i < n; ++i) {
                p->sms[i] = Ni[i];
                req_tpcs[i] = sm_to_tpc(p->sms[i]);
            }

            // Allocate disjoint slices
            auto slices = allocate_disjoint_tpcs(req_tpcs, TOTAL_TPCS);

            // Assign per-task masks enabling only its slice
            for (int i = 0; i < n; ++i) {
                int start = slices[i].first;
                int count = slices[i].second;

                std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
                p->masks[i] = mask_disable_all_except_range(start, count, TOTAL_TPCS);

                std::cout << "[STGM] FINAL task " << i
                          << " wl=" << workloads[i]
                          << " sms=" << p->sms[i]
                          << " tpcs=" << req_tpcs[i]
                          << " slice=[" << start << "," << (start + count - 1) << "]"
                          << " mask=0x" << std::hex << mask_hex(p->masks[i]) << std::dec
                          << "\n";
            }

            std::cout << "[STGM] SUCCESS\n";
            return true;
        }

        // Not schedulable -> increase Ni for best gain task
        int best_i = -1;
        double best_score = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < n; ++i) {
            if (Ni[i] >= NSM) continue;

            double score = marginal_gain_score_ms(workloads[i], Ni[i], periods_ms[i], exec_scale);
            if (score > best_score) {
                best_score = score;
                best_i = i;
            }
        }

        if (best_i < 0) {
            std::cerr << "[STGM] UNSCHEDULABLE: cannot increase Ni further\n";
            print_state(workloads, periods_ms, Ni, Np, NSM, exec_scale);
            return false;
        }

        Ni[best_i] += 1;
    }
}
