#include "controller.h"
#include <limits>
#include <cstdio>
#include <cstdlib>
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




// void assign_overlap(parameters* p)
// {
//     const int N         = p->tasks;
//     const int TOTAL_SMS = p->maxsms;
//     const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

//     if (N <= 0 || TOTAL_SMS <= 0 || TOTAL_TPCS <= 0)
//         return;

//     // All valid TPC bits
//     const uint64_t all_bits =
//         (TOTAL_TPCS >= 64) ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL);

//     // Build a contiguous mask of tpc_req TPCs, starting at start_tpc
//     auto build_tpc_mask = [&](int start_tpc, int tpc_req) -> uint64_t
//     {
//         uint64_t m = 0ULL;
//         tpc_req = std::clamp(tpc_req, 1, TOTAL_TPCS);

//         for (int k = 0; k < tpc_req; ++k)
//         {
//             int idx = (start_tpc + k) % TOTAL_TPCS;
//             m |= (1ULL << idx);
//         }
//         return m;
//     };

//     for (int i = 0; i < N; i++)
//     {
//         int sm_req = std::clamp((int)p->sms[i], p->minsms, TOTAL_SMS);
//         int tpc_req = sm_to_tpc(sm_req);

//         // VALID EVEN SPREAD ACROSS THE TPC RING
//         int start = (i * TOTAL_TPCS) / N;   // safe for all valid configs

//         uint64_t enabled = build_tpc_mask(start, tpc_req);
//         uint64_t disable = (~enabled) & all_bits;

//         p->masks[i] = disable;
//         p->sms[i]   = sm_req;

//         std::cout << "[assign_overlap] Task " << i
//                   << " | reqSM=" << sm_req
//                   << " | TPC=" << tpc_req
//                   << " | startTPC=" << start
//                   << " | enabled=0x" << std::hex << enabled
//                   << " | disable=0x" << disable << std::dec
//                   << " | TOTAL_TPCS=" << TOTAL_TPCS
//                   << "\n";
//     }
//     std::cout<<endl;
// }

void assign_overlap(parameters* p)
{
    const int N          = p->tasks;
    const int TOTAL_SMS  = p->maxsms;
    const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

    if (N <= 0 || TOTAL_SMS <= 0 || TOTAL_TPCS <= 0) return;

    // One scheduler process owns one parameters object for the experiment.
    static parameters* initialized = nullptr;
    const bool initial = (initialized != p);

    const uint128_t all_bits =
        (TOTAL_TPCS >= 128)
            ? ~uint128_t(0)
            : ((uint128_t(1) << TOTAL_TPCS) - 1);

    auto popcount128 = [](uint128_t x) -> int {
#if defined(__GNUG__) || defined(__clang__)
        return __builtin_popcountll(static_cast<uint64_t>(x))
             + __builtin_popcountll(static_cast<uint64_t>(x >> 64));
#else
        int c = 0;
        while (x) {
            x &= (x - 1);
            ++c;
        }
        return c;
#endif
    };

    auto build_ring_mask = [&](int start, int len) -> uint128_t {
        uint128_t m = 0ULL;
        if (len <= 0) return 0ULL;

        len   = std::min(len, TOTAL_TPCS);
        start = ((start % TOTAL_TPCS) + TOTAL_TPCS) % TOTAL_TPCS;

        for (int k = 0; k < len; ++k) {
            int idx = (start + k) % TOTAL_TPCS;
            m |= (uint128_t(1) << idx);
        }
        return m;
    };

    auto find_free_run = [&](uint128_t free_mask, int len) -> int {
        if (len <= 0) return -1;
        len = std::min(len, TOTAL_TPCS);
        if (popcount128(free_mask) < len) return -1;

        for (int start = 0; start < TOTAL_TPCS; ++start) {
            bool ok = true;
            for (int k = 0; k < len; ++k) {
                int idx = (start + k) % TOTAL_TPCS;
                if (((free_mask >> idx) & 1ULL) == 0ULL) {
                    ok = false;
                    break;
                }
            }
            if (ok) return start;
        }
        return -1;
    };

    auto pick_any_free = [&](uint128_t free_mask, int need) -> uint128_t {
        uint128_t sel = 0ULL;
        for (int i = 0; i < TOTAL_TPCS && need > 0; ++i) {
            if ((free_mask >> i) & 1ULL) {
                sel |= (uint128_t(1) << i);
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
        double priority;
    };

    std::vector<Req> reqs;
    reqs.reserve(N);

    for (int i = 0; i < N; ++i) {
        int sm = std::clamp((int)p->sms[i], p->minsms, TOTAL_SMS);

        // Quantize to whole TPCs: one TPC contains two SMs.
        sm = (sm / SMS_PER_TPC) * SMS_PER_TPC;
        if (sm < SMS_PER_TPC) sm = SMS_PER_TPC;

        int tpc = sm / SMS_PER_TPC;
        tpc = std::clamp(tpc, 1, TOTAL_TPCS);

        p->sms[i] = tpc * SMS_PER_TPC;

        // Preserve the original priority rule for initial placement.
        double pr = 0.0;
        if (p->outererror) pr = (double)p->outererror[i];
        pr = std::max(0.0, pr);

        reqs.push_back({i, tpc, pr});
    }

    // Preserve the original sharing limits.
    const int MAX_BORROW_PER_TASK = TOTAL_TPCS;
    const int MAX_OVERLAP_PER_TPC = N;

    // ----------------------------
    // 2) Sort only for initial placement.
    //    Later requests remain in task order.
    // ----------------------------
    if (initial) {
        std::sort(reqs.begin(), reqs.end(),
                  [&](const Req& a, const Req& b) {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.tpc_req > b.tpc_req;
                  });
    }

    std::vector<uint128_t> enabled_masks(N, 0ULL);
    std::vector<int> tpc_share(TOTAL_TPCS, 0);

    auto compute_free_mask = [&]() -> uint128_t {
        uint128_t m = 0ULL;
        for (int t = 0; t < TOTAL_TPCS; ++t) {
            if (tpc_share[t] == 0)
                m |= (uint128_t(1) << t);
        }
        return m & all_bits;
    };

    auto add_tpc = [&](int task_idx, int tpc_idx) {
        enabled_masks[task_idx] |= (uint128_t(1) << tpc_idx);
        ++tpc_share[tpc_idx];
    };

    auto can_use_tpc = [&](int tpc_idx) -> bool {
        return tpc_share[tpc_idx] < MAX_OVERLAP_PER_TPC;
    };

    if (initial) {
        // ----------------------------
        // 3) Initial placement: allocate free TPCs first.
        // ----------------------------
        for (const auto& r : reqs) {
            int i    = r.idx;
            int need = r.tpc_req;

            uint128_t free_mask = compute_free_mask();
            int start = find_free_run(free_mask, need);
            uint128_t enabled = 0ULL;

            if (start >= 0) {
                enabled = build_ring_mask(start, need) & free_mask;
            } else {
                enabled = pick_any_free(free_mask, need);
            }

            for (int t = 0; t < TOTAL_TPCS; ++t) {
                if ((enabled >> t) & 1ULL)
                    add_tpc(i, t);
            }
        }

        // ----------------------------
        // 4) Initial placement: share if free TPCs were insufficient.
        // ----------------------------
        for (const auto& r : reqs) {
            int i      = r.idx;
            int need   = r.tpc_req;
            int got    = popcount128(enabled_masks[i]);
            int shortf = need - got;
            if (shortf <= 0) continue;

            int borrowed = 0;

            // Preserve the original ring scan from each task's base.
            int base = (i * TOTAL_TPCS) / std::max(1, N);

            for (int k = 0;
                 k < 2 * TOTAL_TPCS &&
                 shortf > 0 &&
                 borrowed < MAX_BORROW_PER_TASK;
                 ++k) {
                int t = (base + k) % TOTAL_TPCS;

                if ((enabled_masks[i] >> t) & 1ULL) continue;
                if (!can_use_tpc(t)) continue;

                add_tpc(i, t);
                --shortf;
                ++borrowed;
            }

            for (int t = 0;
                 t < TOTAL_TPCS &&
                 shortf > 0 &&
                 borrowed < MAX_BORROW_PER_TASK;
                 ++t) {
                if ((enabled_masks[i] >> t) & 1ULL) continue;
                if (!can_use_tpc(t)) continue;

                add_tpc(i, t);
                --shortf;
                ++borrowed;
            }
        }
    } else {
        // ----------------------------
        // 3) Keep old positions and release excess TPCs for ALL tasks.
        // ----------------------------
        for (const auto& r : reqs) {
            const int i = r.idx;
            enabled_masks[i] = (~p->masks[i]) & all_bits;

            int got = popcount128(enabled_masks[i]);

            // Shrink from the highest TPC index; retain all other positions.
            for (int t = TOTAL_TPCS - 1;
                 t >= 0 && got > r.tpc_req;
                 --t) {
                const uint128_t bit = uint128_t(1) << t;
                if (enabled_masks[i] & bit) {
                    enabled_masks[i] &= ~bit;
                    --got;
                }
            }

            // Count sharing after all retained positions are identified.
            for (int t = 0; t < TOTAL_TPCS; ++t) {
                if ((enabled_masks[i] >> t) & 1ULL)
                    ++tpc_share[t];
            }
        }

        // ----------------------------
        // 4) Grow into free TPCs, then least-shared TPCs.
        //    Existing positions are retained.
        // ----------------------------
        for (const auto& r : reqs) {
            const int i = r.idx;
            int shortf = r.tpc_req - popcount128(enabled_masks[i]);

            while (shortf > 0) {
                int best = -1;

                for (int t = 0; t < TOTAL_TPCS; ++t) {
                    if ((enabled_masks[i] >> t) & 1ULL) continue;
                    if (!can_use_tpc(t)) continue;

                    // Free TPCs have sharing count zero and are chosen first.
                    // Equal counts are resolved by the lowest TPC index.
                    if (best < 0 || tpc_share[t] < tpc_share[best])
                        best = t;
                }

                if (best < 0) break;

                add_tpc(i, best);
                --shortf;
            }
        }
    }

    // ----------------------------
    // 5) Write disable masks and actual SM counts.
    // ----------------------------
    for (int i = 0; i < N; ++i) {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

        uint128_t enabled = enabled_masks[i] & all_bits;
        p->masks[i] = ~enabled;

        int tpcs_enabled = popcount128(enabled);
        p->sms[i] = tpcs_enabled * SMS_PER_TPC;
    }

    initialized = p;
}

/*
void assign_overlap(parameters* p)
{
    const int N          = p->tasks;
    const int TOTAL_SMS  = p->maxsms;
    const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

    if (N <= 0 || TOTAL_SMS <= 0 || TOTAL_TPCS <= 0) return;

    const uint128_t all_bits =
        (TOTAL_TPCS >= 128) ? ~uint128_t(0) : ((uint128_t(1) << TOTAL_TPCS) - 1);

    auto popcount128 = [](uint128_t x) -> int {
#if defined(__GNUG__) || defined(__clang__)
        return __builtin_popcountll(static_cast<uint64_t>(x))
             + __builtin_popcountll(static_cast<uint64_t>(x >> 64));
#else
        int c = 0; while (x) { x &= (x - 1); ++c; } return c;
#endif
    };

    auto build_ring_mask = [&](int start, int len) -> uint128_t {
        uint128_t m = 0ULL;
        if (len <= 0) return 0ULL;
        len   = std::min(len, TOTAL_TPCS);
        start = ((start % TOTAL_TPCS) + TOTAL_TPCS) % TOTAL_TPCS;
        for (int k = 0; k < len; ++k) {
            int idx = (start + k) % TOTAL_TPCS;
            m |= (uint128_t(1) << idx);
        }
        return m;
    };

    auto find_free_run = [&](uint128_t free_mask, int len) -> int {
        if (len <= 0) return -1;
        len = std::min(len, TOTAL_TPCS);
        if (popcount128(free_mask) < len) return -1;

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

    auto pick_any_free = [&](uint128_t free_mask, int need) -> uint128_t {
        uint128_t sel = 0ULL;
        for (int i = 0; i < TOTAL_TPCS && need > 0; ++i) {
            if ((free_mask >> i) & 1ULL) {
                sel |= (uint128_t(1) << i);
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
    const int MAX_OVERLAP_PER_TPC = N;          // or 2 if you want at most 2 tasks per TPC

    // ----------------------------
    // 2) Sort by (priority, size)
    // ----------------------------
    std::sort(reqs.begin(), reqs.end(),
              [&](const Req& a, const Req& b) {
                  if (a.priority != b.priority) return a.priority > b.priority;
                  return a.tpc_req > b.tpc_req;
              });

    // enabled mask for each task
    std::vector<uint128_t> enabled_masks(N, 0ULL);

    // per-TPC sharing count
    std::vector<int> tpc_share(TOTAL_TPCS, 0);

    // free TPCs = those with share==0
    auto compute_free_mask = [&]() -> uint128_t {
        uint128_t m = 0ULL;
        for (int t = 0; t < TOTAL_TPCS; ++t) {
            if (tpc_share[t] == 0) m |= (uint128_t(1) << t);
        }
        return m & all_bits;
    };

    // helper: add a TPC bit to a task, updating share count
    auto add_tpc = [&](int task_idx, int tpc_idx) {
        enabled_masks[task_idx] |= (uint128_t(1) << tpc_idx);
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

        uint128_t free_mask = compute_free_mask();

        // try a contiguous run first
        int start = find_free_run(free_mask, need);
        uint128_t enabled = 0ULL;

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
        int got    = popcount128(enabled_masks[i]);
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
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
        uint128_t enabled = enabled_masks[i] & all_bits;
        p->masks[i] = ~enabled; // Disable TPCs outside the effective budget too.

        // Keep p->sms consistent with actual enabled count (optional but recommended)
        int tpcs_enabled = popcount128(enabled);
        p->sms[i] = tpcs_enabled * SMS_PER_TPC;
    }
}
*/


/*
void assign_overlap(parameters* p)
{
    const int N          = p->tasks;
    const int TOTAL_SMS  = p->maxsms;
    const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

    const bool valid =
        (N > 0) && (TOTAL_SMS > 0) && (TOTAL_TPCS > 0);

    const uint64_t all_bits =
        (!valid) ? 0ULL :
        (TOTAL_TPCS >= 64 ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL));

    auto popcount64 = [](uint64_t x) -> int {
#if defined(__GNUG__) || defined(__clang__)
        return __builtin_popcountll(x);
#else
        int c = 0; while (x) { x &= (x - 1); ++c; } return c;
#endif
    };

    auto build_ring_mask = [&](int start, int len) -> uint64_t {
        uint64_t m = 0ULL;
        if (len <= 0 || !valid) return 0ULL;
        len   = std::min(len, TOTAL_TPCS);
        start = ((start % TOTAL_TPCS) + TOTAL_TPCS) % TOTAL_TPCS;
        for (int k = 0; k < len; ++k) {
            int idx = (start + k) % TOTAL_TPCS;
            m |= (1ULL << idx);
        }
        return m;
    };

    auto find_free_run = [&](uint64_t free_mask, int len) -> int {
        if (!valid || len <= 0) return -1;
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
        if (!valid) return sel;
        for (int i = 0; i < TOTAL_TPCS && need > 0; ++i) {
            if ((free_mask >> i) & 1ULL) {
                sel |= (1ULL << i);
                --need;
            }
        }
        return sel;
    };

    struct Req { int idx, tpc_req; };
    std::vector<Req> reqs;
    reqs.reserve(N);

    // --- quantize SMs DOWN to TPC granularity ---
    for (int i = 0; i < N; ++i) {
        int sm = std::clamp((int)p->sms[i], p->minsms, TOTAL_SMS);
        sm = (sm / SMS_PER_TPC) * SMS_PER_TPC;
        if (sm < SMS_PER_TPC) sm = SMS_PER_TPC;

        int tpc = sm / SMS_PER_TPC;
        tpc = std::clamp(tpc, 1, TOTAL_TPCS);

        p->sms[i] = tpc * SMS_PER_TPC;
        reqs.push_back({i, tpc});
    }

    // largest-first
    std::sort(reqs.begin(), reqs.end(),
              [](const Req& a, const Req& b) {
                  return a.tpc_req > b.tpc_req;
              });

    std::vector<uint64_t> enabled_masks(N, 0ULL);

    uint64_t used      = 0ULL;
    uint64_t free_mask = all_bits;

    // --- allocation (overlap allowed) ---
    for (const auto& r : reqs) {
        int i    = r.idx;
        int need = r.tpc_req;

        uint64_t enabled = 0ULL;

        int start = find_free_run(free_mask, need);
        if (start >= 0) {
            enabled = build_ring_mask(start, need);
        } else {
            enabled = pick_any_free(free_mask, need);
        }

        int got = popcount64(enabled);

        if (got < need) {
            int remain = need - got;
            int base   = (i * TOTAL_TPCS) / std::max(1, N);
            for (int k = 0; remain > 0 && k < 2 * TOTAL_TPCS; ++k) {
                int idx = (base + k) % TOTAL_TPCS;
                uint64_t bit = (1ULL << idx);
                if ((enabled & bit) == 0ULL) {
                    enabled |= bit;
                    --remain;
                }
            }
        }

        enabled &= all_bits;
        enabled_masks[i] = enabled;

        uint64_t newly_reserved = enabled & free_mask;
        used |= newly_reserved;
        free_mask = (~used) & all_bits;
    }

    // --- write disable masks ---
    for (int i = 0; i < N; ++i) {
        uint64_t enabled = enabled_masks[i] & all_bits;
        p->masks[i] = (~enabled) & all_bits;
    }
}
*/
// void actuation_outerloop(parameters* p)
// {
//     const int N = p->tasks;

//     // fixed-point accumulator per task (Q10)
//     static std::vector<int64_t> acc_q10;
//     if ((int)acc_q10.size() != N) acc_q10.assign(N, 0);

//     const int64_t SCALE = 1024; // Q10

//     for (int i = 0; i < N; ++i)
//     {
//         std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
//         int64_t delta_q10 = (int64_t) llround((double)p->outerdelta[i] * (double)SCALE);

//         acc_q10[i] += delta_q10;
//         int step = (int)(acc_q10[i] / SCALE);      // can be negative
//         acc_q10[i] -= (int64_t)step * SCALE;       // keep remainder

//         int sm = (int)p->sms[i] + step;
//         sm = std::clamp(sm, p->minsms, p->maxsms);
//         sm = (sm / SMS_PER_TPC) * SMS_PER_TPC;
//         if (sm < SMS_PER_TPC) sm = SMS_PER_TPC;

//         p->sms[i] = sm;
//     }

//     assign_overlap(p);
// }



void actuation_outerloop(parameters* p)
{
    const int N = p->tasks;

    // Fractional SM accumulator per task
    static std::vector<double> sm_acc;
    if ((int)sm_acc.size() != N) {
        sm_acc.assign(N, 0.0);
        for (int i = 0; i < N; ++i)
            sm_acc[i] = (double)p->sms[i];
    }

    for (int i = 0; i < N; ++i)
    {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

        // 1) additive control update
        sm_acc[i] += (double)p->outerdelta[i];

        // 2) round to nearest integer
        int sm_rounded = (int)std::llround(sm_acc[i]);

        // 3) clamp to allowed range
        sm_rounded = std::clamp(sm_rounded, p->minsms, p->maxsms);

        // 4) quantize to TPC granularity
        int sm_quant = (sm_rounded / SMS_PER_TPC) * SMS_PER_TPC;
        if (sm_quant < SMS_PER_TPC) sm_quant = SMS_PER_TPC;

        p->sms[i] = sm_quant;

        // 5) keep accumulator consistent with what was actually applied
        sm_acc[i] = (double)p->sms[i];
    }

    assign_overlap(p);
}


/*
void actuation_outerloop_for_period(parameters* p, double outer_loop_duration_sec)
{
    const int N = p->tasks;

    // delta-sigma accumulator per task (persistent)
    static std::vector<double> acc;
    if ((int)acc.size() != N) acc.assign(N, 0.0);

    // Snapshot fractional targets ONCE per control period
    std::vector<double> target(N);
    for (int i = 0; i < N; ++i) {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
        target[i] = (double)p->sms[i] + (double)p->outerdelta[i];  // e.g. 15.3
    }

    const int quantum = std::lcm(SMS_PER_TPC, 2);   // discrete step (e.g. 2)
    const auto tick = std::chrono::milliseconds(1);

    using clock = std::chrono::steady_clock;
    const auto end_time =
        clock::now() + std::chrono::duration<double>(outer_loop_duration_sec);

    while (!g_stop_act.load(std::memory_order_relaxed) &&
           clock::now() < end_time)
    {
        std::lock_guard<std::mutex> txn(g_sms_masks_mutex);

        for (int i = 0; i < N; ++i)
        {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

            double x = std::clamp(target[i],
                                  (double)p->minsms,
                                  (double)p->maxsms);

            // Two nearest discrete levels
            int L = (int)std::floor(x / quantum) * quantum;
            int U = L + quantum;

            L = std::clamp(L, p->minsms, p->maxsms);
            U = std::clamp(U, p->minsms, p->maxsms);

            if (U <= L) {
                p->sms[i] = L;
                continue;
            }

            // --- PURE delta-sigma ---
            double f = (x - L) / (double)(U - L);   // in [0,1]
            acc[i] += f;

            if (acc[i] >= 1.0) {
                p->sms[i] = U;
                acc[i] -= 1.0;
            } else {
                p->sms[i] = L;
            }
        }

        // rebuild masks for instantaneous sms
        assign_overlap(p);

        if (clock::now() + tick >= end_time) break;
        std::this_thread::sleep_for(tick);
    }
}
*/

// Delta-sigma over a control period, in TPC domain (1 TPC = 2 SMs).
// p->outerdelta[i] is interpreted as "delta in SM units" (can be fractional).
// If you want outerdelta in TPC units instead, set x_tpc = (p->sms[i]/2) + p->outerdelta[i].
void actuation_outerloop_for_period(parameters* p,
                                    double outer_loop_duration_sec,
                                    int slice_ms = 10)
{
    const int N = p->tasks;
    if (N <= 0) return;

    // persistent delta-sigma accumulator per task (fractional part)
    static std::vector<double> acc_tpc;
    if ((int)acc_tpc.size() != N) acc_tpc.assign(N, 0.0);

    // snapshot target ONCE per control period (so controller output is held constant this period)
    std::vector<double> x_tpc(N, 1.0);

    for (int i = 0; i < N; ++i) {
        std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

        // current applied SMs -> TPCs
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

void dysm_mpc(parameters* p)
{
    const int N = p->getTasks();
    const double control_seconds = outer_loop_duration;
    int control_period = 0;
    // Enabled only by the 6.9 runner; ordinary experiments do not record timing.
    const char* timing_env = std::getenv("DYSM_MEASURE_DECISION");
    const bool measure_decision = timing_env && std::string(timing_env) == "1";
    // H=1, frozen model from the SAME gain.py invocation as P's gain.
    // model_B is RRT/SM, from the same model used to generate P's gain.
    std::vector<int> command_sms;

    while (outerlooprunning) {
        // Freeze the previous actuator before reading the SM baseline. This
        // same baseline is used to convert the chosen target into a delta.
        stop_actuation_thread();
        const auto cycle_start = steady_clock::now();
        const auto elapsed = duration_cast<seconds>(cycle_start - p->start_time).count();
        std::vector<double> current_sms(N), measured_rrt(N), target_rrt(N);
        std::vector<float> exec_ms(N), response_ms(N), period_ms(N), miss_percent(N);
        bool have_measurements = true;

        // 1) Snapshot the monitor's measurement windows under the same locks.
        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
            const auto& samples = outerloop_tasklist[i];
            exec_ms[i] = calculateAverage(samples.execution_time_measurement);
            response_ms[i] = calculateAverage(samples.response_time_measurement);
            period_ms[i] = 1000.0f * calculateAverage(samples.task_rate_measurement);
            current_sms[i] = p->sms[i];
            target_rrt[i] = p->setpoints[i];
            if (samples.response_time_measurement.empty() ||
                !std::isfinite(period_ms[i]) || period_ms[i] <= 0.0f ||
                !std::isfinite(response_ms[i]) || response_ms[i] <= 0.0f) {
                have_measurements = false;
                continue;
            }
            measured_rrt[i] = response_ms[i] / period_ms[i];
            miss_percent[i] = calculateDeadlineMiss(samples.response_time_measurement,
                                                    period_ms[i]);
        }
        if (!have_measurements) {
            // Hold the previous MEAN command, not the last physical pulse.
            if (!command_sms.empty()) {
                for (int i = 0; i < N; ++i) {
                    std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
                    p->outerdelta[i] = command_sms[i] - current_sms[i];
                }
                start_new_actuation_period(p, control_seconds);
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(control_seconds));
            continue;
        }

        // Time the complete decision, excluding monitoring, logging and actuation.
        const auto decision_start = measure_decision
            ? steady_clock::now() : steady_clock::time_point{};

        // 2) Previous MEAN command -> next MEAN command. Never use the last
        // lower/upper hardware pulse as the model's previous mean input.
        if (command_sms.empty()) {
            command_sms.resize(N);
            for (int i = 0; i < N; ++i)
                command_sms[i] = static_cast<int>(std::lround(current_sms[i]));
        }
        std::vector<std::vector<int>> candidates;
        const auto consider = [&](const std::vector<int>& v) {
            if (std::accumulate(v.begin(), v.end(), 0) > p->maxsms) return;
            for (int i = 0; i < N; ++i) {
                if (v[i] < 2 || v[i] > p->maxsms) return;
                const int low_sms = (v[i] / 2) * 2;
                if (low_sms < command_sms[i] &&
                    !(1.06 * measured_rrt[i] * command_sms[i] / low_sms < 1.0))
                    return;
            }
            if (std::find(candidates.begin(), candidates.end(), v) == candidates.end())
                candidates.push_back(v);
        };
        // Same finite action set: hold, +1 SM, -1 SM, or exchange one SM.
        consider(command_sms);
        for (int step : {1}) {
            for (int i = 0; i < N; ++i) {
                auto v = command_sms;
                v[i] += step;
                consider(v);
            }
            for (int i = 0; i < N; ++i) {
                auto v = command_sms;
                v[i] -= step;
                consider(v);
            }
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (i == j) continue;
                    auto v = command_sms;
                    v[i] += step;
                    v[j] -= step;
                    consider(v);
                }
            }
        }

        if (candidates.empty()) {
            auto safe_hold = command_sms;
            for (int i = 0; i < N; ++i) {
                const int low_sms = (safe_hold[i] / 2) * 2;
                if (low_sms < command_sms[i] &&
                    !(1.06 * measured_rrt[i] * command_sms[i] / low_sms < 1.0))
                    safe_hold[i] += safe_hold[i] % 2;
            }
            consider(safe_hold);
        }
        if (candidates.empty()) candidates.push_back(command_sms);

        // Preserve the existing MPC's priority for the most overdue task.
        int urgent = -1;
        double worst = 1.0;
        for (int i = 0; i < N; ++i)
            if (measured_rrt[i] > worst) { worst = measured_rrt[i]; urgent = i; }
        if (urgent >= 0) {
            std::vector<std::vector<int>> progress;
            for (const auto& v : candidates)
                if (v[urgent] > command_sms[urgent]) progress.push_back(v);
            if (!progress.empty()) candidates.swap(progress);
        }

        double best_cost = std::numeric_limits<double>::infinity();
        auto best_sms = command_sms;
        for (const auto& v : candidates) {
            std::vector<double> predicted(N);
            double tracking = 0.0, deadline = 0.0, move_tpc = 0.0;
            for (int i = 0; i < N; ++i) {
                double delta_rrt = 0.0;
                for (int j = 0; j < N; ++j)
                    delta_rrt += model_B[i][j] * (v[j] - command_sms[j]);
                predicted[i] = std::clamp(measured_rrt[i] + delta_rrt, 0.05, 2.50);
                const double target = std::isfinite(target_rrt[i]) && target_rrt[i] > 0.05
                    ? target_rrt[i] : 0.8;
                const double error = std::log(std::max(predicted[i], 1e-6) / target);
                const double excess = std::max(predicted[i] - 1.0, 0.0);
                tracking += error * error;
                deadline += excess * excess;
                move_tpc += std::abs(v[i] - command_sms[i]) / 2.0;
            }
            // Original H=1 weights, including the terminal tracking term.
            const double cost = 8.0 * tracking + 120.0 * deadline + 0.004 * move_tpc
                              + 0.90 * 2.0 * (8.0 * tracking);
            if (cost + 1e-12 < best_cost) {
                best_cost = cost;
                best_sms = v;
            }
        }
        command_sms = best_sms;
        const auto& target_sms = command_sms;
        if (measure_decision) {
            const double decision_us = duration<double, std::micro>(
                steady_clock::now() - decision_start).count();
            std::printf("[CONTROL-TIMING] solution=7 tasks=%d period=%d decision_us=%.6f\n",
                        N, control_period, decision_us);
        }

        // 3) Preserve the existing log fields. The measurements belong to the
        // PREVIOUS allocation, not to the new target selected above.
        p->wallclocktime.push_back(elapsed);
        std::cout << "[MPC] Control period " << control_period
                  << "\tTime = " << elapsed << "\n";
        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
            p->outererror[i] = target_rrt[i] - measured_rrt[i];
            p->outerdelta[i] = target_sms[i] - current_sms[i]; // delta in SMs
            p->outer_output_executiontime[i].push_back(exec_ms[i]);
            p->outer_output_responsetime[i].push_back(response_ms[i]);
            p->outer_output_deadlinemiss[i].push_back(miss_percent[i]);
            p->outer_output_rtr_measured[i].push_back(measured_rrt[i]);
            p->outer_output_current_taskrate[i].push_back(period_ms[i]);
            p->outer_output_sms[i].push_back(current_sms[i]);
            p->output_setpoints[i].push_back(target_rrt[i]);

            std::cout << "Task " << (i + 1)
                      << " | RT=" << response_ms[i] << " | ET=" << exec_ms[i]
                      << " | Period=" << period_ms[i] << " | RTR=" << measured_rrt[i]
                      << " | Error=" << p->outererror[i]
                      << " | Δd=" << p->outerdelta[i]
                      << " | Current SMs=" << current_sms[i]
                      << " | News SMs=" << target_sms[i]
                      << " | Current Mask=0x" << std::hex
                      << mask_hex(p->masks[i]) << std::dec << "\n";

            auto& samples = outerloop_tasklist[i];
            samples.execution_time_measurement.clear();
            samples.response_time_measurement.clear();
            samples.rtr_measurement.clear();
            samples.task_rate_measurement.clear();
        }

        // 4) Reuse the ZIP delta-sigma / overlap actuator without changing it.
        start_new_actuation_period(p, control_seconds);
        ++control_period;
        std::this_thread::sleep_until(cycle_start +
                                     std::chrono::duration<double>(control_seconds));
    }
    stop_actuation_thread();
}

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
                      << mask_hex(p->getMask(i))
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

/*

void smscaling(parameters* p)
{
    std::cout << "[outerloop] Activated.\n";

    int control_period = 0;
    const int N = p->getTasks();
    if (N <= 0) return;

    const double DELTA_SM_LIM = 8.0;

    while (outerlooprunning)
    {
        using namespace std::chrono;
        auto now = steady_clock::now();
        auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();
        p->wallclocktime.push_back(elapsed_time);

        std::cout << "[outerloop] Control period  " << control_period
                  << "\tTime = " << elapsed_time << "\n";

        for (int i = 0; i < N; ++i) {
            const float avg_exec = calculateAverage(outerloop_tasklist[i].execution_time_measurement);
            const float avg_resp = calculateAverage(outerloop_tasklist[i].response_time_measurement);
            float avg_rate = calculateAverage(innerloop_tasklist[i].task_rate_measurement);

            float avg_period_ms = avg_rate * 1000.0f;

            const float miss = calculateDeadlineMiss(
                outerloop_tasklist[i].response_time_measurement,
                avg_period_ms
            );

            const float rtr_meas =
                (avg_period_ms > 1e-9f) ? (avg_resp / avg_period_ms) : 0.0f;

            p->outer_output_executiontime[i].push_back(avg_exec);
            p->outer_output_responsetime[i].push_back(avg_resp);
            p->outer_output_deadlinemiss[i].push_back(miss);
            p->outer_output_rtr_measured[i].push_back(rtr_meas);
            p->outer_output_current_taskrate[i].push_back(avg_period_ms);
            p->outer_output_sms[i].push_back(p->sms[i]);

            if (p->finsolution == 3)
                p->output_setpoints[i].push_back(p->setpoints[i]);

            const float target_rtr = (float)p->setpoints[i];
            const float err = target_rtr - rtr_meas;
            p->outererror[i] = err;

            outerloop_tasklist[i].execution_time_measurement.clear();
            outerloop_tasklist[i].response_time_measurement.clear();
            outerloop_tasklist[i].rtr_measurement.clear();
            outerloop_tasklist[i].task_rate_measurement.clear();
        }

        for (int i = 0; i < N; ++i) {
            double dsm = 0.0;
            for (int j = 0; j < N; ++j)
                dsm += (double)gain[i][j] * (double)p->outererror[j];

            p->outerdelta[i] = (float)dsm;
        }

        for (int i = 0; i < N; ++i) {
            const auto& rr = p->outer_output_responsetime[i];
            const auto& pp = p->outer_output_current_taskrate[i];
            const auto& rt = p->outer_output_rtr_measured[i];
            const auto& et = p->outer_output_executiontime[i];

            float last_rt  = rr.empty() ? 0.0f : rr.back();
            float last_et  = et.empty() ? 0.0f : et.back();
            float last_per = pp.empty() ? 0.0f : pp.back();
            float last_rtr = rt.empty() ? 0.0f : rt.back();

            std::cout << "Task " << (i + 1)
                      << " | RT="     << last_rt
                      << " | ET="     << last_et
                      << " | Period=" << last_per
                      << " | RTR="    << last_rtr
                      << " | Error="  << p->outererror[i]
                      << " | ΔSM="    << p->outerdelta[i]
                      << " | Current SMs=" << p->sms[i]
                      <<"|New SMs= "<<std::ceil(p->sms[i]+p->outerdelta[i])
                      << " | Current Mask=0x"
                      << std::hex << std::setw(16) << std::setfill('0')
                      << (unsigned long long)p->masks[i]
                      << std::dec << std::setfill(' ')
                      << "\n";
        }

        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

            double sm_old = (double)p->sms[i];
            double sm_new = sm_old + (double)p->outerdelta[i];

            sm_new = std::clamp(sm_new, (double)p->minsms, (double)p->maxsms);

            int sm_int = (int)std::llround(sm_new);
            if (sm_int & 1) sm_int -= 1;
            if (sm_int < 2) sm_int = 2;
            sm_int = std::clamp(sm_int, p->minsms, p->maxsms);

            p->sms[i] = sm_int;
        }

        assign_overlap(p);

        control_period++;
        std::this_thread::sleep_for(std::chrono::duration<float>(outer_loop_duration));
    }

    std::cout << "[outerloop] Stopped after " << control_period << " periods.\n";
}
*/



// ============================================================
// actuation_outerloop_mimo
// d_i(k) = d_i(k-1) + Δd_i(k)
// s_i(k) = 1 / d_i(k)
// ============================================================

// ============================================================
// actuation_outerloop_mimo
// d_i(k) = d_i(k-1) + Δd_i(k)
// s_i(k) = 1 / d_i(k)
// ============================================================

// void actuation_outerloop_mimo(parameters* p)
// {
//     const int N = p->getTasks();
//     if (N <= 0) return;

//     constexpr int MAX_SM_STEP = 2;   // critical for stability

//     auto round_even = [](int x) {
//         return (x & 1) ? (x + 1) : x;
//     };

//     const int SM_MIN = round_even(p->minsms);
//     const int SM_MAX = round_even(p->maxsms);

//     // ---------- Snapshot previous SMs ----------
//     std::vector<int> prev_sms(N);
//     for (int i = 0; i < N; ++i)
//         prev_sms[i] = round_even(p->sms[i]);

//     // ---------- MIMO update ----------
//     for (int i = 0; i < N; ++i)
//     {
//         std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

//         int sm_old = std::clamp(prev_sms[i], SM_MIN, SM_MAX);

//         // correct inverse-SM
//         double d_old =sm_old;

//         // controller output is Δd
//         double d_cmd = d_old + p->outerdelta[i];
//         int sm_new = (int)std::lround(d_cmd);
//         sm_new = round_even(sm_new);
//         sm_new = std::clamp(sm_new, SM_MIN, SM_MAX);

//         // rate limit (SM domain)
//        //f (sm_new > sm_old + MAX_SM_STEP)
//        //   sm_new = sm_old + MAX_SM_STEP;
//        //f (sm_new < sm_old - MAX_SM_STEP)
//         //  sm_new = sm_old - MAX_SM_STEP;

//         sm_new = round_even(sm_new);

//         std::cout << "[outer-actuation] Task " << i
//                   << " | SM_prev=" << sm_old
//                   << " | Δd=" << p->outerdelta[i]
//                   << " | d_old=" << d_old
//                   << " | d_cmd=" << d_cmd
//                   << " | SM_new=" << sm_new
//                   << "\n";

//         p->sms[i] = sm_new;
//     }

//     // ---------- Single mask rebuild ----------
//     assign_overlap(p);
// }


// void assign_overlap(parameters* p)
// {
//     const int N         = p->tasks;
//     const int TOTAL_SMS = p->maxsms;
//     const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

//     if (N <= 0 || TOTAL_SMS <= 0 || TOTAL_TPCS <= 0)
//         return;

//     // All valid TPC bits
//     const uint64_t all_bits =
//         (TOTAL_TPCS >= 64) ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL);

//     // Build a contiguous mask of tpc_req TPCs, starting at start_tpc
//     auto build_tpc_mask = [&](int start_tpc, int tpc_req) -> uint64_t
//     {
//         uint64_t m = 0ULL;
//         tpc_req = std::clamp(tpc_req, 1, TOTAL_TPCS);

//         for (int k = 0; k < tpc_req; ++k)
//         {
//             int idx = (start_tpc + k) % TOTAL_TPCS;
//             m |= (1ULL << idx);
//         }
//         return m;
//     };

//     for (int i = 0; i < N; i++)
//     {
//         int sm_req = std::clamp((int)p->sms[i], p->minsms, TOTAL_SMS);
//         int tpc_req = sm_to_tpc(sm_req);

//         // VALID EVEN SPREAD ACROSS THE TPC RING
//         int start = (i * TOTAL_TPCS) / N;   // safe for all valid configs

//         uint64_t enabled = build_tpc_mask(start, tpc_req);
//         uint64_t disable = (~enabled) & all_bits;

//         p->masks[i] = disable;
//         p->sms[i]   = sm_req;

//         std::cout << "[assign_overlap] Task " << i
//                   << " | reqSM=" << sm_req
//                   << " | TPC=" << tpc_req
//                   << " | startTPC=" << start
//                   << " | enabled=0x" << std::hex << enabled
//                   << " | disable=0x" << disable << std::dec
//                   << " | TOTAL_TPCS=" << TOTAL_TPCS
//                   << "\n";
//     }
//     std::cout<<endl;
// }


// void outerloop(parameters* p)
// {
//     std::cout << "\n[smscaling] Activated (MIMO, inverse-SM)\n";

//     const int N = p->getTasks();
//     int control_period = 0;

//     constexpr float RTR_DEV_TH = 0.05f;   // 5%
//     constexpr int   PERSIST_N  = 1;
// //
//  // static std::vector<int> bad_cnt;
//    //f ((int)bad_cnt.size() != N)
//      // bad_cnt.assign(N, 0);

//     while (outerlooprunning)
//     {
//         std::cout << "\n========== SMSCALING Period "
//                   << control_period << " ==========\n";

//         bool trigger = false;

//         // -----------------------------------
//         // 1) Measure RTR + persistence
//         // -----------------------------------
//         for (int i = 0; i < N; ++i)
//         {
//             float avg_resp = calculateAverage(
//                 outerloop_tasklist[i].response_time_measurement);
//             float avg_rate = calculateAverage(
//                 outerloop_tasklist[i].task_rate_measurement);

//             float period_ms = avg_rate * 1000.0f;
//             float rtr_meas =
//                 (period_ms > 1e-6f) ? avg_resp / period_ms : 0.0f;

//             float rtr_sp = p->setpoints[i];
//             float err = rtr_sp - rtr_meas;
//             float dev = (rtr_sp > 1e-6f) ? std::fabs(err) / rtr_sp : 0.0f;

//             p->outererror[i] = err;
//             p->outer_output_sms[i].push_back(p->sms[i]);

//             std::cout << "[smscaling] Task " << i
//                       << " | RTR=" << rtr_meas
//                       << " | SP=" << rtr_sp
//                       << " | err=" << err
//                       << " | dev=" << dev * 100.0f << "%"
//                       // " | bad_cnt=" << bad_cnt[i]
//                       << "\n";

//             // (dev > RTR_DEV_TH) bad_cnt[i]++;
//             //se bad_cnt[i] = 0;

//             // (bad_cnt[i] >= PERSIST_N)
//             //
//               //trigger = true;
//                //->controller_hold[i] = true;
//            //

//             // clear only OUTER buffers
//             outerloop_tasklist[i].execution_time_measurement.clear();
//             outerloop_tasklist[i].response_time_measurement.clear();
//             outerloop_tasklist[i].task_rate_measurement.clear();
//         }

//         // -----------------------------------
//         // 2) MIMO control law
//         // Δd = -K * e
//         // -----------------------------------
//         for (int i = 0; i < N; ++i)
//         {
//             float acc = 0.0f;
//             for (int j = 0; j < N; ++j)
//                 acc += outerloopgain[i][j] * p->outererror[j];

//             p->outerdelta[i] = -acc;
//         }

//         // -----------------------------------
//         // 3) Actuation
//         // -----------------------------------
//        /// (trigger)
//        //
//             std::cout << "\n[smscaling] >>> APPLYING SM SCALING (MIMO) <<<\n";
//             actuation_outerloop_mimo(p);
//           //std::fill(bad_cnt.begin(), bad_cnt.end(), 0);
//         //
//        //lse
//        //
//          // std::cout << "[smscaling] No SM scaling this period\n";
//         //

//         control_period++;
//         std::this_thread::sleep_for(
//             std::chrono::duration<float>(outer_loop_duration));
//     }
// }




// // ============================================================
// // actuation_outerloop(parameters*) — SM update + mask rebuild
// // ============================================================

// void actuation_outerloop(parameters* p)
// {
//     const int N = p->tasks;
//     if (N <= 0) return;

//     std::vector<int> new_sms(N);

//     // Compute new SM allocation per task
//     for (int i = 0; i < N; i++)
//     {
//         std::lock_guard<std::mutex> lock(p->task_mutexes[i]);
//         double sms = std::max(1.0, (double)p->sms[i]);  // avoid div by 0
//         double d    = 1/sms;                        // d(k) = 1 / SM(k)
//         double delta_d =p->outerdelta[i];              // Δd(k)
//         double d_new = d + delta_d;                     // d(k+1) = d(k) + Δd
//         int s_int = (int)1/std::round(d_new);
//         s_int = std::clamp(s_int, p->minsms, p->maxsms);
//         std::cout << "[outer-actuation] Task " << i
//                   << " | Δd="  << delta_d
//                   << " | d_old=" << d
//                   << " | d_new=" << d_new
//                   << " | SM_old=" << p->sms[i]
//                   << " | SM_new=" << s_int
//                   << "\n";
    
//         new_sms[i] = s_int;
//         p->outer_output_sms[i].push_back(p->sms[i]);

//     }
//     // Commit SM allocations
//     for (int i = 0; i < N; i++)
//         p->sms[i] = new_sms[i];

//     // Convert SM allocation → TPC disable mask
//     assign_overlap(p);
// }

/*

void smscaling(parameters* p)
{
    std::cout << "[outerloop] Activated.\n";

    int control_period = 0;
    const int N = p->getTasks();

    while (outerlooprunning) 
    {
        using namespace std::chrono;
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

            std::cout << "Before Clearance Task " << i << "] "
                      << "Exec=" << outerloop_tasklist[i].execution_time_measurement.size() << ", "
                      << "Resp=" << outerloop_tasklist[i].response_time_measurement.size() << ", "
                      << "RTR="  << outerloop_tasklist[i].rtr_measurement.size() << ", "
                      << "Rate=" << outerloop_tasklist[i].task_rate_measurement.size()
                      << std::endl;

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
            U[i] = control;
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
                      << " | Current Mask=0x"
                      << std::hex << std::setw(16) << std::setfill('0')
                      << static_cast<unsigned long long>(p->masks[i])
                      << std::dec << std::setfill(' ') << "\n";

            // --- Sanity check: decode disable mask → enabled TPCs → SMs ---
            {
                const int TOTAL_SMS  = p->maxsms;
                const int TOTAL_TPCS = sm_to_tpc(TOTAL_SMS);

                const uint64_t all_bits =
                    (TOTAL_TPCS >= 64) ? ~0ULL : ((1ULL << TOTAL_TPCS) - 1ULL);

                uint64_t disable_mask = p->masks[i];
                uint64_t enabled_mask = (~disable_mask) & all_bits;

                int tpcs_enabled = __builtin_popcountll(enabled_mask);
                int sms_from_mask = tpcs_enabled * SMS_PER_TPC;

                std::cout << "    [mask-check] enabled_mask=0x"
                          << std::hex << std::setw(16) << std::setfill('0')
                          << static_cast<unsigned long long>(enabled_mask)
                          << std::dec << std::setfill(' ')
                          << " | tpcs_enabled=" << tpcs_enabled
                          << " | sms_from_mask=" << sms_from_mask
                          << " | reported_sms=" << p->sms[i]
                          << "\n";
            }
        }

        // ====================================================
        // 4) Actuation: update SMs (inverse model) + rebuild masks
        // ====================================================
        actuation_outerloop(p);

        // ====================================================
        // 5) Sleep until next control period
        // ====================================================
        control_period++;
        std::this_thread::sleep_for(
            std::chrono::duration<float>(outer_loop_duration));
    }

    std::cout << "[outerloop] Stopped after " << control_period
              << " periods.\n";
}
*/



void adhoc(parameters* p)
{
    std::cout << "[outerloop] Activated (Ad-hoc SM Controller).\n";

    int control_period = 0;
    const int N = p->getTasks();

    constexpr int SM_STEP = 5;   // safe step size
    constexpr float EPS   = 0.0f; // set to 0.02f if you want deadband

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
        // 2) NO ACTUATION
        // ====================================================
        // (intentionally empty)

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



// void innerloop(parameters* p)
// {
//     std::cout << "\n[FC_GPU] Activated\n";
//     int control_period = 0;

//     static int hold_cnt = 0;
//     constexpr int HOLD_PERIODS = 4;

//     while (running)
//     {
//         std::cout << "\n========== FC_GPU Period " << control_period << " ==========\n";
//         auto now = steady_clock::now();
//         auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();

//         for (int i = 0; i < totaltasks; ++i)
//         {
//             float avg_resp   = calculateAverage(innerloop_tasklist[i].response_time_measurement);
//             float avg_period = calculateAverage(innerloop_tasklist[i].task_rate_measurement) * 1000.0f;

//             float rtr_meas = (avg_period > 1e-6f) ? (avg_resp / avg_period) : 0.0f;
//             float rtr_sp   = p->setpoints[i];
//             float err      = rtr_sp - rtr_meas;

//             p->error[i] = err;
            
//             const float avg_exec = calculateAverage(innerloop_tasklist[i].execution_time_measurement);
//             const float miss = calculateDeadlineMiss(innerloop_tasklist[i].response_time_measurement,avg_period);
            
//             p->output_executiontime[i].push_back(avg_exec);
//             p->output_responsetime[i].push_back(avg_resp);
//             p->output_deadlinemiss[i].push_back(miss);
//             p->output_rtr_measured[i].push_back(rtr_meas);
//             p->output_current_taskrate[i].push_back(avg_period);
//             p->output_setpoints[i].push_back(p->setpoints[i]);
//             p->wallclocktime.push_back(elapsed_time);
//             std::cout << "[FC_GPU] Task " << i
//                       << " | RTR_meas=" << rtr_meas
//                       << " | RTR_sp=" << rtr_sp
//                       << " | err=" << err
//                       << " | hold=" << (p->controller_hold[i] ? "YES" : "NO")
//                       << "\n";

//             innerloop_tasklist[i].execution_time_measurement.clear();
//             innerloop_tasklist[i].response_time_measurement.clear();
//             innerloop_tasklist[i].task_rate_measurement.clear();
//             p->inter_responsetime[i].push_back(10000);
//             p->inter_executiontime[i].push_back(10000);
//         }

//         // Control law
//         for (int i = 0; i < totaltasks; ++i)
//         {
//             float acc = 0.0f;
//             for (int j = 0; j < totaltasks; ++j)
//                 acc += innerloopgain[i][j] * p->error[j];

//             p->delta[i] = -acc;
//         }

//         // Period update
//         for (int i = 0; i < totaltasks; ++i)
//         {
//             std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

//             if (p->controller_hold[i]) {
//                //td::cout << "[FC_GPU] Task " << i << " PERIOD FROZEN\n";
//                 continue;
//             }

//             double base = p->taskrates[i];
//             double minp = base * (1.0 - p->range[i] / 100.0);
//             double maxp = base * (1.0 + p->range[i] / 100.0);

//             double old_p = p->period[i];
//             double new_p = std::clamp(old_p + p->delta[i], minp, maxp);

//             p->period[i] = new_p;

//             //d::cout << "[FC_GPU] Task " << i << " | period " << old_p << " -> " << new_p << "\n";
//         }

//         // Release hold
//         hold_cnt++;
//         if (hold_cnt >= HOLD_PERIODS)
//         {
//             std::fill(p->controller_hold.begin(),
//                       p->controller_hold.end(), false);
//             hold_cnt = 0;
//             std::cout << "[FC_GPU] HOLD RELEASED\n";
//         }

//         control_period++;
//         std::this_thread::sleep_for(std::chrono::duration<float>(inner_loop_duration));
//     }
// }






// void FC_GPU(parameters* p)
// {
//     std::cout << "[innerloop] Activated.\n";
//     int control_period = 0;
//     std::cout << "Total number of tasks = " << totaltasks << std::endl;

//     while (running)
//     {
//         auto now = steady_clock::now();
//         auto elapsed_time = duration_cast<seconds>(now - p->start_time).count();
//         p->wallclocktime.push_back(elapsed_time);

//         std::cout << "[innerloop] Control Period " << control_period
//                   << "\tTime = " << elapsed_time << "\n";

//         // ============================================================
//         // Step 1. Collect runtime measurements and compute errors
//         // ============================================================
//         for (int i = 0; i < totaltasks; ++i)
//         {
//             float avg_exec = calculateAverage(innerloop_tasklist[i].execution_time_measurement);
//             float avg_resp = calculateAverage(innerloop_tasklist[i].response_time_measurement);
//             float avg_period = calculateAverage(innerloop_tasklist[i].task_rate_measurement);
//             avg_period *= 1000.0f;   // ms for logging

//             float miss = calculateDeadlineMiss(
//                 innerloop_tasklist[i].response_time_measurement,
//                 avg_period
//             );

//             float rtr_measured =
//                 (avg_period > 0.0f ? avg_resp / avg_period : 0.0f);

//             float err = p->setpoints[i] - rtr_measured;
//             p->error[i] = err;

//             // Save measurements
//             p->output_executiontime[i].push_back(avg_exec);
//             p->output_responsetime[i].push_back(avg_resp);
//             p->output_deadlinemiss[i].push_back(miss);
//             p->output_rtr_measured[i].push_back(rtr_measured);
//             p->output_current_taskrate[i].push_back(avg_period);
//             p->output_setpoints[i].push_back(p->setpoints[i]);

//             // Debug before clearing
//             std::cout << "Before Clearance Task " << i << "] "
//                       << "Exec=" << innerloop_tasklist[i].execution_time_measurement.size()
//                       << ", Resp=" << innerloop_tasklist[i].response_time_measurement.size()
//                       << ", RTR=" << innerloop_tasklist[i].rtr_measurement.size()
//                       << ", Rate=" << innerloop_tasklist[i].task_rate_measurement.size()
//                       << std::endl;

//             innerloop_tasklist[i].execution_time_measurement.clear();
//             innerloop_tasklist[i].response_time_measurement.clear();
//             innerloop_tasklist[i].rtr_measurement.clear();
//             innerloop_tasklist[i].task_rate_measurement.clear();

//             std::cout << "Clearance Task " << i << "] Exec=0, Resp=0, RTR=0, Rate=0\n";

         
//         }

//         // ============================================================
//         // Step 2. Compute coupled control terms
//         // ============================================================
//         std::vector<float> s(totaltasks, 0.0f);

//         for (int i = 0; i < totaltasks; ++i)
//         {
//             float acc = 0.0f;

//             for (int j = 0; j < totaltasks; ++j)
//                 acc +=innerloopgain[i][j] * p->error[j];

//             s[i] = -acc;
//             p->delta[i] = s[i];    // store control signal Δ_i
//         }

//         // ============================================================
//         // Step 3. Update periods directly (range-bounded)
//         // ============================================================
//         for (int i = 0; i < totaltasks; ++i)
//         {
//             double old_period    = 0.0;
//             double base_period   = p->taskrates[i];   // original nominal period
//             double range_pct     = p->range[i];       // allowed ±range%
//             double min_period    = base_period * (1.0 - range_pct / 100.0);
//             double max_period    = base_period * (1.0 + range_pct / 100.0);
//             if (min_period < 0.001)    // prevent zero/negative
//                 min_period = 0.001;
//             double updated_raw   = 0.0;
//             double updated       = 0.0;

//             {
//                 // Protect period access against concurrent GPU threads
//                 std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

//                 old_period = p->period[i];

//                 if (p->tunable[i])
//                 {
//                     updated_raw = old_period + p->delta[i];
//                     updated     = std::clamp(updated_raw, min_period, max_period);
                    
//                     bool saturated = (updated != updated_raw);
                    
//                     if (saturated) {
//                         if (updated == min_period)
//                             std::cout << "Period saturated to MIN (" << min_period << ")\n";
//                         else if (updated == max_period)
//                             std::cout << "Period saturated to MAX (" << max_period << ")\n";
//                     }
                    
//                     p->period[i] = updated;
//                 }
//                 else
//                 {
//                     // Non-tunable: no change
//                     updated_raw = old_period;
//                     updated     = old_period;
//                 }
//             }

//             // ===== DEBUG PRINT =====
//             std::cout << "[Period-Update] Task " << i
//                       << " | Old=" << old_period
//                       << " | Δ="   << p->delta[i]
//                       << " | RawNew=" << updated_raw
//                       << " | ClampedNew=" << updated
//                       << " | Min=" << min_period
//                       << " | Max=" << max_period
//                       << (p->tunable[i] ? " | tunable" : " | NON-TUNABLE")
//                       << std::endl;
//         }

//         // ============================================================
//         // Step 4. (Optional) Print summary with updated periods
//         // ============================================================
//         for (int i = 0; i < totaltasks; ++i)
//         {
//             std::cout << "Task " << (i + 1)
//                       << " | RT=" << p->output_responsetime[i].back()
//                       << " | ET=" << p->output_executiontime[i].back()
//                       << " | Period=" << p->period[i]
//                       << " | RTR=" << p->output_rtr_measured[i].back()
//                       << " | Error=" << p->error[i]
//                       << " | Δ(s_i)=" << p->delta[i]
//                       << std::endl;
//         }
//         std::cout << "==========================================\n";

//         control_period++;
//         std::this_thread::sleep_for(std::chrono::duration<float>(inner_loop_duration));
//     }

//     std::cout << "[innerloop] Stopped after " << control_period << " periods.\n";
// }



// ================================
// Monitor Thread
// ================================
// void monitor(parameters* p)
// {
//     std::cout << "[monitor] Started.\n";
//     totaltasks = p->getTasks();
//     auto push_if_changed = [&](auto& v, auto val)
//     {
//         constexpr double EPS = 1e-6;
//         if (v.empty() || std::fabs(v.back() - val) > EPS)
//             v.push_back(val);
//     };

//     while (running) 
//     {

//         for (int i = 0; i < totaltasks; ++i)
//         {
//             std::lock_guard<std::mutex> lock(p->task_mutexes[i]);

//             double resp   = p->responsetimes[i];
//             double exec   = p->executiontimes[i];
//             double period = p->period[i];

//             // ---- Shared logs ----
//             push_if_changed(p->inter_responsetime[i], resp);
//             push_if_changed(p->inter_executiontime[i], exec);
//             push_if_changed(p->inter_current_taskrate[i], period);

//             // ---- Per-loop logs ----
//             for (Task* tlist : { &innerloop_tasklist[i], &outerloop_tasklist[i] })
//             {
//                 push_if_changed(tlist->response_time_measurement, resp);
//                 push_if_changed(tlist->execution_time_measurement, exec);
//                 push_if_changed(tlist->task_rate_measurement, period);

//                 // Compute RTR correctly: resp (ms) / period(ms)
//                 double rtr = (period > 1e-9 ? resp / (period * 1000.0) : 0.0);

//                 push_if_changed(tlist->rtr_measurement, rtr);
//             }
//         }

//         // Sampling frequency (~200 Hz)
//         std::this_thread::sleep_for(std::chrono::milliseconds(5));
      
//     }

//     std::cout << "[monitor] Stopped.\n";
// }


