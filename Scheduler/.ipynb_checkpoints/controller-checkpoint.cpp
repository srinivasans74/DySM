#include "controller.h"
#include <atomic>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>
#include <numeric>   // std::accumulate
#include <cmath>
#include <iomanip>
#include "gain_matrix.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

static std::thread g_act_thread;
static std::atomic<bool> g_stop_act{false};
static std::mutex g_sms_masks_mutex;
using namespace std;
extern float inner_loop_duration;
extern float outer_loop_duration;
using namespace std::chrono;

std::atomic<bool> running{false};
std::atomic<bool> outerlooprunning{false};
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[ ";
    for (const auto& x : v) os << x << " ";
    os << "]";
    return os;
}

// ================================
// Global task tracking
// ================================
static std::vector<Task> innerloop_tasklist;
static std::vector<Task> outerloop_tasklist;
static int totaltasks = 0;
// ================================
// Initialization
// ================================
void initialize(parameters* p)
{
    totaltasks = p->getTasks();
    if (totaltasks <= 0) {
        std::cerr << "[initialize] No tasks found in parameters!\n";
        return;
    }

    innerloop_tasklist.resize(totaltasks);
    outerloop_tasklist.resize(totaltasks);

    std::cout << "[initialize] Allocated tracking structures for "
              << totaltasks << " tasks.\n";
}

// ================================
// Utility Functions
// ================================
float calculateDeadlineMiss(const std::vector<float>& responseTimes, float period)
{
    if (responseTimes.empty() || period <= 0.0f) return 0.0f;
    int missed = 0;
    for (float rt : responseTimes)
        if (rt/period > 1) missed++;
    return (100.0f * missed / static_cast<float>(responseTimes.size()));
}

float calculateAverage(const std::vector<float>& values)
{
    if (values.empty()) return 0.0f;
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / static_cast<float>(values.size());
}

#define RANGE_MASK(lo, hi) (((1ULL << ((hi) - (lo) + 1)) - 1ULL) << (lo))



static constexpr int SMS_PER_TPC = 2;

// Convert requested SM count into TPC count (ceil)
static inline int sm_to_tpc(int sms)
{
    if (sms <= 0) return 1;
    return (sms + SMS_PER_TPC - 1) / SMS_PER_TPC;   // ceil(sms / 2)
}






void assign_overlap(parameters* p)
{
    const int N          = p->tasks;
    const int TOTAL_SMS  = p->maxsms;
    const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

    if (N <= 0 || TOTAL_SMS <= 0 || TOTAL_TPCS <= 0) return;

    const uint64_t all_bits =
        (TOTAL_TPCS >= 64) ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL);

    auto popcount64 = [](uint64_t x) -> int {
#if defined(__GNUG__) || defined(__clang__)
        return __builtin_popcountll(x);
#else
        int c = 0; while (x) { x &= (x - 1); ++c; } return c;
#endif
    };

    auto build_ring_mask = [&](int start, int len) -> uint64_t {
        uint64_t m = 0ULL;
        if (len <= 0) return 0ULL;
        len   = std::min(len, TOTAL_TPCS);
        start = ((start % TOTAL_TPCS) + TOTAL_TPCS) % TOTAL_TPCS;
        for (int k = 0; k < len; ++k) {
            int idx = (start + k) % TOTAL_TPCS;
            m |= (1ULL << idx);
        }
        return m;
    };

    auto find_free_run = [&](uint64_t free_mask, int len) -> int {
        if (len <= 0) return -1;
        len = std::min(len, TOTAL_TPCS);
        if (popcount64(free_mask) < len) return -1;

        for (int start = 0; start < TOTAL_TPCS; ++start) {
            bool ok = true;
            for (int k = 0; k < len; ++k) {
                int idx = (start + k) % TOTAL_TPCS;
                if (((free_mask >> idx) & 1ULL) == 0ULL) { ok = false; break; }
            }
            if (ok) return start;
        }
        return -1;
    };

    auto pick_any_free = [&](uint64_t free_mask, int need) -> uint64_t {
        uint64_t sel = 0ULL;
        for (int i = 0; i < TOTAL_TPCS && need > 0; ++i) {
            if ((free_mask >> i) & 1ULL) {
                sel |= (1ULL << i);
                --need;
            }
        }
        return sel;
    };

    // ----------------------------
    // 1) Quantize sms -> tpc_req
    // ----------------------------
    struct Req {
        int idx;
        int tpc_req;
        double priority; // higher => more important
    };

    std::vector<Req> reqs;
    reqs.reserve(N);

    for (int i = 0; i < N; ++i) {
        int sm = std::clamp((int)p->sms[i], p->minsms, TOTAL_SMS);

        // quantize SM to TPC granularity (even)
        sm = (sm / SMS_PER_TPC) * SMS_PER_TPC;
        if (sm < SMS_PER_TPC) sm = SMS_PER_TPC;

        int tpc = sm / SMS_PER_TPC;
        tpc = std::clamp(tpc, 1, TOTAL_TPCS);

        p->sms[i] = tpc * SMS_PER_TPC;

        // Priority idea:
        // If outererror = setpoint - rtr:
        //   outererror > 0 means rtr below setpoint (needs help) -> higher priority
        //   outererror <= 0 means ok or over -> lower priority
        double pr = 0.0;
        if (p->outererror) pr = (double)p->outererror[i];

        // Map to priority weight (tune this):
        // clamp negative to 0 (don’t reward tasks that are already above setpoint)
        pr = std::max(0.0, pr);

        reqs.push_back({i, tpc, pr});
    }

    // ----------------------------
    // Policy knobs
    // ----------------------------
    const int MAX_BORROW_PER_TASK = TOTAL_TPCS; // or smaller like 2/4 to limit overlap
    const int MAX_OVERLAP_PER_TPC = N;          

    // ----------------------------
    // 2) Sort by (priority, size)
    // ----------------------------
    std::sort(reqs.begin(), reqs.end(),
              [&](const Req& a, const Req& b) {
                  if (a.priority != b.priority) return a.priority > b.priority;
                  return a.tpc_req > b.tpc_req;
              });

    // enabled mask for each task
    std::vector<uint64_t> enabled_masks(N, 0ULL);

    // per-TPC sharing count
    std::vector<int> tpc_share(TOTAL_TPCS, 0);

    // free TPCs = those with share==0
    auto compute_free_mask = [&]() -> uint64_t {
        uint64_t m = 0ULL;
        for (int t = 0; t < TOTAL_TPCS; ++t) {
            if (tpc_share[t] == 0) m |= (1ULL << t);
        }
        return m & all_bits;
    };

    // helper: add a TPC bit to a task, updating share count
    auto add_tpc = [&](int task_idx, int tpc_idx) {
        enabled_masks[task_idx] |= (1ULL << tpc_idx);
        tpc_share[tpc_idx] += 1;
    };

    // helper: can we add this TPC (respect overlap cap)?
    auto can_use_tpc = [&](int tpc_idx) -> bool {
        return tpc_share[tpc_idx] < MAX_OVERLAP_PER_TPC;
    };

    // ----------------------------
    // 3) PASS A: isolate using FREE TPCs only
    // ----------------------------
    for (const auto& r : reqs) {
        int i    = r.idx;
        int need = r.tpc_req;

        uint64_t free_mask = compute_free_mask();

        // try a contiguous run first
        int start = find_free_run(free_mask, need);
        uint64_t enabled = 0ULL;

        if (start >= 0) {
            enabled = build_ring_mask(start, need) & free_mask;
        } else {
            enabled = pick_any_free(free_mask, need);
        }

        // commit
        for (int t = 0; t < TOTAL_TPCS; ++t) {
            if ((enabled >> t) & 1ULL) add_tpc(i, t);
        }
    }

    // ----------------------------
    // 4) PASS B: overlap/borrow for tasks still short
    //    Borrow from "less critical" tasks implicitly by using already-used TPCs
    // ----------------------------
    for (const auto& r : reqs) {
        int i      = r.idx;
        int need   = r.tpc_req;
        int got    = popcount64(enabled_masks[i]);
        int shortf = need - got;
        if (shortf <= 0) continue;

        int borrowed = 0;

        // Build a candidate ordering of TPCs:
        // Prefer TPCs that are currently shared by tasks with low priority.
        // We approximate this by preferring lower-share first OR based on existing enabled distribution.
        // Simple heuristic: try near a "base" position for stability.
        int base = (i * TOTAL_TPCS) / std::max(1, N);

        for (int k = 0; k < 2 * TOTAL_TPCS && shortf > 0 && borrowed < MAX_BORROW_PER_TASK; ++k) {
            int t = (base + k) % TOTAL_TPCS;

            // skip if already enabled for this task
            if ((enabled_masks[i] >> t) & 1ULL) continue;

            // enforce overlap limit
            if (!can_use_tpc(t)) continue;

            // Add overlap TPC
            add_tpc(i, t);
            --shortf;
            ++borrowed;
        }

        // If still short, do a fallback scan
        for (int t = 0; t < TOTAL_TPCS && shortf > 0 && borrowed < MAX_BORROW_PER_TASK; ++t) {
            if ((enabled_masks[i] >> t) & 1ULL) continue;
            if (!can_use_tpc(t)) continue;
            add_tpc(i, t);
            --shortf;
            ++borrowed;
        }
    }

    // ----------------------------
    // 5) Write disable masks
    // ----------------------------
    for (int i = 0; i < N; ++i) {
        uint64_t enabled = enabled_masks[i] & all_bits;
        p->masks[i] = (~enabled) & all_bits;

        // Keep p->sms consistent with actual enabled count (optional but recommended)
        int tpcs_enabled = popcount64(enabled);
        p->sms[i] = tpcs_enabled * SMS_PER_TPC;
    }
}



void actuation_outerloop(parameters* p)
{
    const int N = p->tasks;
    static std::vector<double> sm_acc;
    if ((int)sm_acc.size() != N) {
        sm_acc.assign(N, 0.0);
        for (int i = 0; i < N; ++i)
            sm_acc[i] = (double)p->sms[i];
    }

    for (int i = 0; i < N; ++i)
    {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

        sm_acc[i] += (double)p->outerdelta[i];
        int sm_rounded = (int)std::llround(sm_acc[i]);
        sm_rounded = std::clamp(sm_rounded, p->minsms, p->maxsms);
        int sm_quant = (sm_rounded / SMS_PER_TPC) * SMS_PER_TPC;
        if (sm_quant < SMS_PER_TPC) sm_quant = SMS_PER_TPC;

        p->sms[i] = sm_quant;
        sm_acc[i] = (double)p->sms[i];
    }

    assign_overlap(p);
}


void actuation_outerloop_for_period(parameters* p,
                                    double outer_loop_duration_sec,
                                    int slice_ms = 10)
{
    const int N = p->tasks;
    if (N <= 0) return;

    // persistent delta-sigma accumulator per task (fractional part)
    static std::vector<double> acc_tpc;
    if ((int)acc_tpc.size() != N) acc_tpc.assign(N, 0.0);

    std::vector<double> x_tpc(N, 1.0);

    for (int i = 0; i < N; ++i) {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
        double cur_tpc = std::max(1.0, (double)p->sms[i] / (double)SMS_PER_TPC);

        // controller output outerdelta is in SM units in your code => convert to TPC units
        double delta_tpc = (double)p->outerdelta[i] / (double)SMS_PER_TPC;

        x_tpc[i] = cur_tpc + delta_tpc;

        // clamp to actuator limits in TPC units
        double min_tpc = std::max(1.0, (double)p->minsms / (double)SMS_PER_TPC);
        double max_tpc = std::max(min_tpc, (double)p->maxsms / (double)SMS_PER_TPC);
        x_tpc[i] = std::clamp(x_tpc[i], min_tpc, max_tpc);
    }

    using clock = std::chrono::steady_clock;
    const auto tick = std::chrono::milliseconds(std::max(1, slice_ms));
    const auto end_time = clock::now() + std::chrono::duration<double>(outer_loop_duration_sec);

    while (!g_stop_act.load(std::memory_order_relaxed) && clock::now() < end_time)
    {
        // lock around masks rebuild so tasks don't see inconsistent masks
        std::lock_guard<std::mutex> txn(g_sms_masks_mutex);

        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

            // two nearest discrete levels in TPC domain
            int L = (int)std::floor(x_tpc[i]);   // lower TPC
            int U = L + 1;                       // upper TPC

            double min_tpc = std::max(1.0, (double)p->minsms / (double)SMS_PER_TPC);
            double max_tpc = std::max(min_tpc, (double)p->maxsms / (double)SMS_PER_TPC);

            L = (int)std::clamp((double)L, min_tpc, max_tpc);
            U = (int)std::clamp((double)U, min_tpc, max_tpc);

            if (U <= L) {
                // exact level, nothing to modulate
                p->sms[i] = L * SMS_PER_TPC;
                continue;
            }

            // first-order delta-sigma
            // desired duty cycle of U
            double f = (x_tpc[i] - (double)L) / (double)(U - L);  // in [0,1]
            f = std::clamp(f, 0.0, 1.0);

            acc_tpc[i] += f;

            int chosen_tpc;
            if (acc_tpc[i] >= 1.0) {
                chosen_tpc = U;
                acc_tpc[i] -= 1.0;
            } else {
                chosen_tpc = L;
            }

            // apply as SMs (must be even)
            p->sms[i] = chosen_tpc * SMS_PER_TPC;
        }

        // rebuild masks for instantaneous SM selection
        assign_overlap(p);

        if (clock::now() + tick >= end_time) break;
        std::this_thread::sleep_for(tick);
    }
}

void start_new_actuation_period(parameters* p, double outer_loop_duration_sec)
{
    // 1) terminate old thread if exists
    if (g_act_thread.joinable()) {
        g_stop_act.store(true, std::memory_order_relaxed);
        g_act_thread.join();
        auto now = steady_clock::now();
        auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();
        std::cout<<"Old actuation stopped at = \t"<<elapsed_time;
    }

    // 2) start new one
    g_stop_act.store(false, std::memory_order_relaxed);
    g_act_thread = std::thread([=]() {
        actuation_outerloop_for_period(p, outer_loop_duration_sec);
    });
}
void stop_actuation_thread()
{
    if (g_act_thread.joinable()) {
        g_stop_act.store(true, std::memory_order_relaxed);
        g_act_thread.join();
    }
}

/*
void smscaling(parameters* p)
{
    std::cout << "[outerloop] Activated.\n";

    int control_period = 0;
    const int N = p->getTasks();

    while (outerlooprunning) 
    {
        auto now = steady_clock::now();
        auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();
        p->wallclocktime.push_back(elapsed_time);

        std::cout << "[outerloop] Control period  " << control_period
                  << "\tTime = " << elapsed_time << "\n";

        // ====================================================
        // 1) Measure, compute RTR, error, and log
        // ====================================================
        for (int i = 0; i < N; ++i) {
            const float avg_exec = calculateAverage(
                outerloop_tasklist[i].execution_time_measurement);
            const float avg_resp = calculateAverage(
                outerloop_tasklist[i].response_time_measurement);
            float avg_rate = calculateAverage(
                innerloop_tasklist[i].task_rate_measurement);

            float avg_period_ms = avg_rate * 1000.0f;

            const float miss = calculateDeadlineMiss(
                outerloop_tasklist[i].response_time_measurement,
                avg_period_ms
            );
            
         

            const float rtr_meas =
                (avg_period_ms > 1e-9f) ? (avg_resp / avg_period_ms) : 0.0f;

            // Log outputs
            p->outer_output_executiontime[i].push_back(avg_exec);
            p->outer_output_responsetime[i].push_back(avg_resp);
            p->outer_output_deadlinemiss[i].push_back(miss);
            p->outer_output_rtr_measured[i].push_back(rtr_meas);
            p->outer_output_current_taskrate[i].push_back(avg_period_ms);
            p->outer_output_sms[i].push_back(p->sms[i]);

            // Fill intermediate outputs (if needed elsewhere)


            if (p->finsolution == 3) {
                p->output_setpoints[i].push_back(p->setpoints[i]);
            }

            // Error = setpoint - measured RTR
            const float target_rtr = static_cast<float>(p->setpoints[i]);
            const float err = target_rtr - rtr_meas;
            p->outererror[i] = err;
            if (err > -0.02f && err < 0.02f) {
                  p->outererror[i] = 0.0;
                }


            // std::cout << "Before Clearance Task " << i << "] "
            //           << "Exec=" << outerloop_tasklist[i].execution_time_measurement.size() << ", "
            //           << "Resp=" << outerloop_tasklist[i].response_time_measurement.size() << ", "
            //           << "RTR="  << outerloop_tasklist[i].rtr_measurement.size() << ", "
            //           << "Rate=" << outerloop_tasklist[i].task_rate_measurement.size()
            //           << std::endl;

            // Clear measurement windows for next period
            outerloop_tasklist[i].execution_time_measurement.clear();
            outerloop_tasklist[i].response_time_measurement.clear();
            outerloop_tasklist[i].rtr_measurement.clear();
            outerloop_tasklist[i].task_rate_measurement.clear();
        }

        // ====================================================
        // 2) Controller: Δd = -K * error
        //    U_i = - Σ_j K[i][j] * error[j]
        // ====================================================
        std::vector<float> U(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            float control = 0.0f;
            for (int j = 0; j < N; ++j) {
                control += gain[i][j] * p->outererror[j];
            }
            U[i] = -control;
            p->outerdelta[i] = U[i];  // Δd_i(k)
        }
        
      
        // ====================================================
        // 3) Debug: show RT, error, Δd, current SMs & masks
        // ====================================================
        for (int i = 0; i < N; ++i) {
            const auto& rr = p->outer_output_responsetime[i];
            const auto& pp = p->outer_output_current_taskrate[i];
            const auto& rt = p->outer_output_rtr_measured[i];
            const auto& et = p->outer_output_executiontime[i];

            float last_rt   = rr.empty() ? 0.0f : rr.back();
            float last_et   = et.empty() ? 0.0f : et.back();
            float last_per  = pp.empty() ? 0.0f : pp.back();
            float last_rtr  = rt.empty() ? 0.0f : rt.back();

            std::cout << "Task " << (i + 1)
                      << " | RT="     << last_rt
                      << " | ET="     << last_et
                      << " | Period=" << last_per
                      << " | RTR="    << last_rtr
                      << " | Error="  << p->outererror[i]
                      << " | Δd="     << p->outerdelta[i]
                      << " | Current SMs=" << p->sms[i]
                      << " | News SMs=" << p->sms[i]+p->outerdelta[i]
                      << " | Current Mask=0x"
                      << std::hex << std::setw(16) << std::setfill('0')
                      << static_cast<unsigned long long>(p->masks[i])
                      << std::dec << std::setfill(' ') << "\n";

            // --- Sanity check: decode disable mask → enabled TPCs → SMs ---
           // {
//                 const int TOTAL_SMS  = p->maxsms;
//                 const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

//                 const uint64_t all_bits =
//                     (TOTAL_TPCS >= 64) ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL);

//                 uint64_t disable_mask = p->masks[i];
//                 uint64_t enabled_mask = (~disable_mask) & all_bits;

//                 int tpcs_enabled = __builtin_popcountll(enabled_mask);
//                 int sms_from_mask = tpcs_enabled * SMS_PER_TPC;

                // std::cout << "    [mask-check] enabled_mask=0x"
                //           << std::hex << std::setw(16) << std::setfill('0')
                //           << static_cast<unsigned long long>(enabled_mask)
                //           << std::dec << std::setfill(' ')
                //           << " | tpcs_enabled=" << tpcs_enabled
                //           << " | sms_from_mask=" << sms_from_mask
                //           << " | reported_sms=" << p->sms[i]
                //           << "\n";
            //}
        }

        // ====================================================
        // 4) Actuation: update SMs (inverse model) + rebuild masks
        // ====================================================
        //actuation_outerloop(p);
        
        start_new_actuation_period(p, outer_loop_duration);
        control_period++;
        std::this_thread::sleep_for(std::chrono::duration<float>(outer_loop_duration));
    }
    stop_actuation_thread();
    std::cout << "[outerloop] Stopped after " << control_period
              << " periods.\n";
}
*/
void smscaling(parameters* p)
{
    std::cout << "[DYSM] Activated.\n";

    int control_period = 0;
    const int N = p->getTasks();

    constexpr float DEADBAND = 0.03f;

    while (outerlooprunning)
    {
        auto now = steady_clock::now();
        auto elapsed_time =
            duration_cast<seconds>(now - p->start_time).count();

        p->wallclocktime.push_back(elapsed_time);

        std::cout << "\n[outerloop] Control period "
                  << control_period
                  << " | Time = " << elapsed_time << "s\n";

        // ====================================================
        // 1) Measure, compute RRT, error, and log
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            const float avg_exec = calculateAverage(
                outerloop_tasklist[i].execution_time_measurement);

            const float avg_resp = calculateAverage(
                outerloop_tasklist[i].response_time_measurement);

            const float avg_rate = calculateAverage(
                innerloop_tasklist[i].task_rate_measurement);

            const float avg_period_ms = avg_rate * 1000.0f;

            const float miss = calculateDeadlineMiss(
                outerloop_tasklist[i].response_time_measurement,
                avg_period_ms);

            const float rtr_meas =
                (avg_period_ms > 1e-9f) ? (avg_resp / avg_period_ms) : 0.0f;

            p->outer_output_executiontime[i].push_back(avg_exec);
            p->outer_output_responsetime[i].push_back(avg_resp);
            p->outer_output_deadlinemiss[i].push_back(miss);
            p->outer_output_rtr_measured[i].push_back(rtr_meas);
            p->outer_output_current_taskrate[i].push_back(avg_period_ms);
            p->outer_output_sms[i].push_back(p->sms[i]);

            if (p->finsolution == 3) {
                p->output_setpoints[i].push_back(p->setpoints[i]);
            }

            const float target_rtr = static_cast<float>(p->setpoints[i]);
            float err = target_rtr - rtr_meas;

            if (err > -DEADBAND && err < DEADBAND) {
                err = 0.0f;
            }

            p->outererror[i] = err;

            outerloop_tasklist[i].execution_time_measurement.clear();
            outerloop_tasklist[i].response_time_measurement.clear();
            outerloop_tasklist[i].rtr_measurement.clear();
            outerloop_tasklist[i].task_rate_measurement.clear();
        }

        // ====================================================
        // 2) Full N x N MIMO controller
        //
        // gain dimension:
        //   gain[NUM_TASKS][NUM_TASKS]
        //
        // outererror = setpoint - measured_RRT
        // outerdelta = -K * outererror
        // ====================================================
        std::vector<float> raw_delta(N, 0.0f);
        std::vector<float> projected_delta(N, 0.0f);

        for (int i = 0; i < N; ++i)
        {
            float control = 0.0f;

            for (int j = 0; j < N; ++j)
            {
                control += gain[i][j] * p->outererror[j];
            }

            raw_delta[i] = -control;
            projected_delta[i] = raw_delta[i];
        }

        // ====================================================
        // 3) Per-task bounds:
        //    minsms <= sms_i + delta_i <= maxsms
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            float min_delta = static_cast<float>(p->minsms - p->sms[i]);
            float max_delta = static_cast<float>(p->maxsms - p->sms[i]);

            projected_delta[i] =
                std::clamp(projected_delta[i], min_delta, max_delta);
        }

        // ====================================================
        // 4) Inequality budget:
        //    sum requested SMs <= maxsms
        // ====================================================
        float total_current_sms = 0.0f;
        float total_raw_requested_sms = 0.0f;
        float total_projected_before_budget = 0.0f;

        for (int i = 0; i < N; ++i)
        {
            total_current_sms += static_cast<float>(p->sms[i]);
            total_raw_requested_sms +=
                static_cast<float>(p->sms[i]) + raw_delta[i];

            total_projected_before_budget +=
                static_cast<float>(p->sms[i]) + projected_delta[i];
        }

        float budget_excess_before =
            std::max(0.0f,
                     total_projected_before_budget -
                     static_cast<float>(p->maxsms));

        if (total_projected_before_budget > static_cast<float>(p->maxsms))
        {
            float excess =
                total_projected_before_budget - static_cast<float>(p->maxsms);

            while (excess > 1e-6f)
            {
                int victim = -1;
                float best_error = -1e30f;

                for (int i = 0; i < N; ++i)
                {
                    float requested =
                        static_cast<float>(p->sms[i]) + projected_delta[i];

                    if (requested <= static_cast<float>(p->minsms))
                        continue;

                    // Reduce task that is most below setpoint first.
                    // outererror > 0 means RRT < setpoint, task is fast.
                    if (p->outererror[i] > best_error)
                    {
                        best_error = p->outererror[i];
                        victim = i;
                    }
                }

                if (victim < 0)
                    break;

                float requested =
                    static_cast<float>(p->sms[victim]) + projected_delta[victim];

                float can_reduce =
                    requested - static_cast<float>(p->minsms);

                if (can_reduce <= 0.0f)
                    break;

                float reduce = std::min(excess, can_reduce);

                projected_delta[victim] -= reduce;
                excess -= reduce;
            }
        }

        for (int i = 0; i < N; ++i)
        {
            p->outerdelta[i] = projected_delta[i];
        }

        float total_final_requested_sms = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            total_final_requested_sms +=
                static_cast<float>(p->sms[i]) + p->outerdelta[i];
        }

        // ====================================================
        // 5) Debug print
        // ====================================================
        std::cout << "[DYSM-debug]"
                  << " total_current_sms=" << total_current_sms
                  << " total_raw_requested_sms=" << total_raw_requested_sms
                  << " total_projected_before_budget="
                  << total_projected_before_budget
                  << " total_final_requested_sms="
                  << total_final_requested_sms
                  << " maxsms=" << p->maxsms
                  << " budget_excess_before=" << budget_excess_before
                  << " budget_excess_after="
                  << std::max(0.0f,
                              total_final_requested_sms -
                              static_cast<float>(p->maxsms))
                  << "\n";

        for (int i = 0; i < N; ++i)
        {
            const auto& rr = p->outer_output_responsetime[i];
            const auto& pp = p->outer_output_current_taskrate[i];
            const auto& rt = p->outer_output_rtr_measured[i];
            const auto& et = p->outer_output_executiontime[i];

            const float last_rt  = rr.empty() ? 0.0f : rr.back();
            const float last_et  = et.empty() ? 0.0f : et.back();
            const float last_per = pp.empty() ? 0.0f : pp.back();
            const float last_rtr = rt.empty() ? 0.0f : rt.back();

            std::cout << "Task " << (i + 1)
                      << " | RT=" << last_rt
                      << " | ET=" << last_et
                      << " | Period=" << last_per
                      << " | RTR=" << last_rtr
                      << " | Setpoint=" << p->setpoints[i]
                      << " | Error=" << p->outererror[i]
                      << " | RawDeltaSM=" << raw_delta[i]
                      << " | ProjectedDeltaSM=" << projected_delta[i]
                      << " | DeltaSM=" << p->outerdelta[i]
                      << " | CurrentSMs=" << p->sms[i]
                      << " | RequestedSMs=" << p->sms[i] + p->outerdelta[i]
                      << " | CurrentMask=0x"
                      << std::hex << std::setw(16) << std::setfill('0')
                      << static_cast<unsigned long long>(p->masks[i])
                      << std::dec << std::setfill(' ') << "\n";
        }

        // ====================================================
        // 6) Actuation
        // ====================================================
        start_new_actuation_period(p, outer_loop_duration);

        ++control_period;

        std::this_thread::sleep_for(
            std::chrono::duration<float>(outer_loop_duration));
    }

    stop_actuation_thread();

    std::cout << "[outerloop] Stopped after "
              << control_period << " periods.\n";
}

void adhoc(parameters* p)
{
    std::cout << "[outerloop] Activated (Ad-hoc SM Controller).\n";

    int control_period = 0;
    const int N = p->getTasks();

    constexpr int SM_STEP = 5;  
    constexpr float EPS   = 0.0f; 

    while (outerlooprunning)
    {
        using namespace std::chrono;
        auto now = steady_clock::now();
        auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();
        p->wallclocktime.push_back(elapsed_time);

        std::cout << "\n[outerloop] Control period " << control_period
                  << " | Time = " << elapsed_time << "s\n";

        // ====================================================
        // 1) Measurement + logging
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            float avg_exec = calculateAverage(
                outerloop_tasklist[i].execution_time_measurement);
            float avg_resp = calculateAverage(
                outerloop_tasklist[i].response_time_measurement);
            float avg_rate = calculateAverage(
                innerloop_tasklist[i].task_rate_measurement);

            float avg_period_ms = avg_rate * 1000.0f;

            float miss = calculateDeadlineMiss(
                outerloop_tasklist[i].response_time_measurement,
                avg_period_ms);

            float rtr_meas =
                (avg_period_ms > 1e-9f) ? (avg_resp / avg_period_ms) : 0.0f;

            // ---- Logs ----
            p->outer_output_executiontime[i].push_back(avg_exec);
            p->outer_output_responsetime[i].push_back(avg_resp);
            p->outer_output_deadlinemiss[i].push_back(miss);
            p->outer_output_rtr_measured[i].push_back(rtr_meas);
            p->outer_output_current_taskrate[i].push_back(avg_period_ms);
            p->outer_output_sms[i].push_back(p->sms[i]);

            if (p->finsolution == 2)
                p->output_setpoints[i].push_back(p->setpoints[i]);

            // Error stored only for debugging
            p->outererror[i] =
                static_cast<float>(p->setpoints[i]) - rtr_meas;

            // ---- Clear measurement windows ----
            outerloop_tasklist[i].execution_time_measurement.clear();
            outerloop_tasklist[i].response_time_measurement.clear();
            outerloop_tasklist[i].rtr_measurement.clear();
            outerloop_tasklist[i].task_rate_measurement.clear();
        }

        // ====================================================
        // 2) DEBUG PRINT (before actuation)
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            float rtr =
                p->outer_output_rtr_measured[i].empty()
                ? 0.0f
                : p->outer_output_rtr_measured[i].back();

            std::cout << "Task " << i
                      << " | RTR=" << rtr
                      << " | Setpoint=" << p->setpoints[i]
                      << " | SM_current=" << p->sms[i]
                      << " | Error=" << p->outererror[i]
                      << "\n";
        }

        // ====================================================
        // 3) AD-HOC ACTUATION
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

            int sm_old = p->sms[i];
            int sm_new = sm_old;

            float rtr =
                p->outer_output_rtr_measured[i].empty()
                ? 0.0f
                : p->outer_output_rtr_measured[i].back();

            float setp = static_cast<float>(p->setpoints[i]);

            if (rtr < setp - EPS) {
                sm_new = sm_old - SM_STEP;   // decrease SMs
            }
            else if (rtr > setp + EPS) {
                sm_new = sm_old + SM_STEP;   // increase SMs
            }

            sm_new = std::clamp(sm_new, p->minsms, p->maxsms);

            std::cout << "[adhoc-actuation] Task " << i
                      << " | RTR=" << rtr
                      << " | Setpoint=" << setp
                      << " | SM_old=" << sm_old
                      << " | SM_new=" << sm_new
                      << "\n";

            p->sms[i] = sm_new;
        }

        // ====================================================
        // 4) Rebuild SM masks (single shot)
        // ====================================================
        assign_overlap(p);

        // ====================================================
        // 5) Sleep
        // ====================================================
        control_period++;
        std::this_thread::sleep_for(
            std::chrono::duration<float>(outer_loop_duration));
    }

    std::cout << "[outerloop] Stopped after "
              << control_period << " periods.\n";
}


void stgm(parameters* p)
{
    std::cout << "[outerloop] Activated (MONITOR-ONLY).\n";

    int control_period = 0;
    const int N = p->getTasks();

    while (outerlooprunning)
    {
        using namespace std::chrono;
        auto now = steady_clock::now();
        auto elapsed_time =
            duration_cast<seconds>(now - p->start_time).count();

        // Log wall-clock time once per control period
        p->wallclocktime.push_back(elapsed_time);

        std::cout << "\n[outerloop] Control period "
                  << control_period
                  << " | Time = " << elapsed_time << "s\n";

        // ====================================================
        // 1) MEASURE + LOG (NO CONTROL)
        // ====================================================
        for (int i = 0; i < N; ++i)
        {
            float avg_resp = calculateAverage(
                outerloop_tasklist[i].response_time_measurement);

            float avg_rate = calculateAverage(
                innerloop_tasklist[i].task_rate_measurement);

            float avg_period_ms = avg_rate * 1000.0f;

            float rtr_meas =
                (avg_period_ms > 1e-9f)
                ? (avg_resp / avg_period_ms)
                : 0.0f;

            // ---- LOGS ONLY ----
            float miss = calculateDeadlineMiss(
                outerloop_tasklist[i].response_time_measurement,
                avg_period_ms);
            
            p->outer_output_responsetime[i].push_back(avg_resp);
            p->outer_output_rtr_measured[i].push_back(rtr_meas);
            p->outer_output_current_taskrate[i].push_back(avg_period_ms);
            p->outer_output_sms[i].push_back(p->sms[i]);
            p->outer_output_deadlinemiss[i].push_back(miss);

            // ---- PRINT ----
            std::cout << "Task " << i
                      << " | Period(ms)=" << avg_period_ms
                      << " | RTR=" << rtr_meas
                      << " | SMs=" << p->sms[i]
                      << "\n";

            // ---- CLEAR WINDOWS ----
            outerloop_tasklist[i].response_time_measurement.clear();
            outerloop_tasklist[i].execution_time_measurement.clear();
            outerloop_tasklist[i].rtr_measurement.clear();
            outerloop_tasklist[i].task_rate_measurement.clear();
        }

        // ====================================================
        // 2) NO ACTUATION Since open-loop
        // ====================================================


        // ====================================================
        // 3) Sleep to next control period
        // ====================================================
        control_period++;
        std::this_thread::sleep_for(
            std::chrono::duration<float>(outer_loop_duration));
    }

    std::cout << "[outerloop] Stopped after "
              << control_period << " periods.\n";
}




void monitor(parameters* p)
{
    std::cout << "[monitor] Started" << std::endl;

    auto push_if_changed = [&](auto& v, double val)
    {
        constexpr double EPS = 1e-6;
        if (v.empty() || std::fabs(v.back() - val) > EPS)
            v.push_back(val);
    };

    while (running) {
        for (int i = 0; i < totaltasks; ++i)
        {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

            double resp = p->responsetimes[i];
            double exec = p->executiontimes[i];
            double per  = p->period[i];

            // Inner loop task list
            push_if_changed(innerloop_tasklist[i].response_time_measurement, resp);
            push_if_changed(innerloop_tasklist[i].execution_time_measurement, exec);
            push_if_changed(innerloop_tasklist[i].task_rate_measurement, per);

            // Outer loop task list
            push_if_changed(outerloop_tasklist[i].response_time_measurement, resp);
            push_if_changed(outerloop_tasklist[i].execution_time_measurement, exec);
            push_if_changed(outerloop_tasklist[i].task_rate_measurement, per);

            // Intermediate storage in parameters
            push_if_changed(p->inter_responsetime[i], resp);
            push_if_changed(p->inter_executiontime[i], exec);
            push_if_changed(p->inter_current_taskrate[i], per);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

