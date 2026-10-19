// Thin ABI around build-extracted DySM code. No CUDA dependency at link time.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <vector>
using uint128_t = unsigned __int128;
using partition_mask_t = uint128_t;
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include <sstream>
#include <cstring>
#include <dlfcn.h>

using namespace std::chrono;


static thread_local std::string error;
template<class F> int guarded(F action) {
    try { action(); error.clear(); return 0; }
    catch(const std::exception& e) { error=e.what(); return -1; }
    catch(...) { error="Native controller failure"; return -1; }
}
struct parameters {
    int tasks,maxsms,minsms=1;
    std::unique_ptr<double[]> sms,outerdelta,outererror;
    std::unique_ptr<partition_mask_t[]> masks;
    std::unique_ptr<std::mutex[]> task_mutexes;
    steady_clock::time_point start_time=steady_clock::now();
    parameters(int n,int budget):tasks(n),maxsms(budget),sms(new double[n]{}),
        outerdelta(new double[n]{}),outererror(new double[n]{}),
        masks(new partition_mask_t[n]{}),task_mutexes(new std::mutex[n]) {
        for(int i=0;i<n;++i) sms[i]=2*(budget/2/n + (i >= n-budget/2%n));
    }
};
struct Pulse {
    uint64_t ns;
    std::vector<uint64_t> stream_ns;
    std::vector<int> sms;
    std::vector<partition_mask_t> masks;
};
struct Context {
    int n, budget;
    std::vector<std::vector<double>> model;
    parameters state;
    std::unique_ptr<std::mutex[]> submit_locks;
    std::vector<void*> streams;
    void* library=nullptr;
    void (*set_mask)(void*,partition_mask_t)=nullptr;
    std::vector<Pulse> pulses;
    Context(int count,int sms):n(count),budget(sms),model(count,std::vector<double>(count)),state(count,sms),
        submit_locks(new std::mutex[count]),streams(count,nullptr) {}
    ~Context() { if(library) dlclose(library); }
};
static Context* live=nullptr;
static bool created=false;
static std::mutex lifecycle;
static std::thread g_act_thread;
static std::atomic<bool> g_stop_act{false};
static std::mutex g_sms_masks_mutex;
static constexpr int SMS_PER_TPC=2;
static int sm_to_tpc(int sms) { return sms<=0 ? 1 : (sms+1)/2; }

static std::string mask_hex(uint128_t mask) {
    std::ostringstream out;
    out << std::hex << static_cast<uint64_t>(mask >> 64)
        << std::setw(16) << std::setfill('0') << static_cast<uint64_t>(mask);
    return out.str();
}
#include "frozen_allocator.inc"
static void assign_overlap(parameters* p) {
    baseline_assign_overlap(p);
    auto& c=*live;
    Pulse pulse{0,{}, {}, {}};
    for(int i=0;i<c.n;++i) {
        std::lock_guard<std::mutex> lock(c.submit_locks[i]);
        c.set_mask(c.streams[i],p->masks[i]);
        pulse.stream_ns.push_back(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
        pulse.sms.push_back(static_cast<int>(p->sms[i]));
        pulse.masks.push_back(p->masks[i]);
    }
    pulse.ns=duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    c.pulses.push_back(std::move(pulse));
}
#include "frozen_actuator.inc"

static Context& context(void* p) {
    if(!p || p!=live) throw std::invalid_argument("Controller is closed");
    return *static_cast<Context*>(p);
}
static void validate_command(Context& c,const int* sms) {
    if(!sms) throw std::invalid_argument("Missing SM command");
    long total=0;
    for(int i=0;i<c.n;++i) {
        if(sms[i]<2 || sms[i]>c.budget) throw std::invalid_argument("Invalid SM command");
        total+=sms[i];
    }
    if(total>c.budget) throw std::invalid_argument("SM command exceeds budget");
}
extern "C" {
const char* dysm_last_error() { return error.c_str(); }
int dysm_create(int n,int budget,const double* b,void** out) {
    return guarded([&] {
        std::lock_guard<std::mutex> lock(lifecycle);
        if(!out || !b || n<1 || budget>256 || budget%2 || n>budget/2)
            throw std::invalid_argument("Invalid model or budget");
        if(created) throw std::invalid_argument("Use one controller run per process");
        *out=nullptr;
        auto c=std::make_unique<Context>(n,budget);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
            if(!std::isfinite(b[i*n+j])) throw std::invalid_argument("Nonfinite model");
            c->model[i][j]=b[i*n+j]; // RRT per SM, same units as current MPC.
        }
        live=c.release(); *out=live; created=true;
    });
}
int dysm_step(void* ctx,const int* mean,const double* y,const double* target,
              int* next,double* prediction) {
    return guarded([&] {
        auto& c=context(ctx); validate_command(c,mean);
        if(!y || !target || !next || !prediction) throw std::invalid_argument("Missing vector");
        const int N = c.n;
        auto* p = &c.state;
        const auto& model_B = c.model;
        std::vector<int> command_sms(mean,mean+N);
        std::vector<double> current_sms(mean,mean+N), measured_rrt(y,y+N), target_rrt(target,target+N);
        for(int i=0;i<N;++i)
            if(!std::isfinite(y[i]) || y[i]<=0 || !std::isfinite(target[i]) || target[i]<=0 || target[i]>=1)
                throw std::invalid_argument("Invalid RRT or reference");
        // Generated verbatim from the current Scheduler/controller.cpp decision block.
        #include "frozen_decision.inc"
        std::copy(command_sms.begin(),command_sms.end(),next);
        for(int i=0;i<N;++i) {
            double change=0;
            for(int j=0;j<N;++j) change+=model_B[i][j]*(next[j]-mean[j]);
            prediction[i]=std::clamp(y[i]+change,0.05,2.50);
        }
    });
}
int dysm_bind_streams(void* ctx,const uintptr_t* handles,const char* library) {
    return guarded([&] {
        auto& c=context(ctx);
        if(!handles || !library || c.library) throw std::invalid_argument("Invalid or duplicate stream binding");
        for(int i=0;i<c.n;++i) {
            if(!handles[i]) throw std::invalid_argument("Null CUDA stream");
            c.streams[i]=reinterpret_cast<void*>(handles[i]);
        }
        c.library=dlopen(library,RTLD_NOW|RTLD_LOCAL);
        if(!c.library) throw std::invalid_argument(dlerror());
        c.set_mask=reinterpret_cast<decltype(c.set_mask)>(dlsym(c.library,"libsmctrl_set_stream_mask_ext"));
        if(!c.set_mask) throw std::invalid_argument("Missing 128-bit libsmctrl setter");
        assign_overlap(&c.state);
    });
}
int dysm_publish(void* ctx,const int* mean,const double* errors,double seconds) {
    return guarded([&] {
        auto& c=context(ctx); validate_command(c,mean);
        if(!c.set_mask || !errors || !std::isfinite(seconds) || seconds<=0)
            throw std::invalid_argument("Invalid actuation or unbound streams");
        for(int i=0;i<c.n;++i) if(!std::isfinite(errors[i])) throw std::invalid_argument("Invalid error");
        stop_actuation_thread();
        for(int i=0;i<c.n;++i) {
            c.state.outerdelta[i]=mean[i]-c.state.sms[i];
            c.state.outererror[i]=errors[i];
        }
        start_new_actuation_period(&c.state,seconds);
    });
}
int dysm_begin_submit(void* ctx,int task) {
    return guarded([&] {
        auto& c=context(ctx);
        if(task<0 || task>=c.n || !c.set_mask) throw std::invalid_argument("Invalid submission");
        c.submit_locks[task].lock();
    });
}
int dysm_end_submit(void* ctx,int task) {
    return guarded([&] {
        auto& c=context(ctx);
        if(task<0 || task>=c.n) throw std::invalid_argument("Invalid task");
        c.submit_locks[task].unlock();
    });
}
int dysm_stop(void* ctx) {
    return guarded([&] { context(ctx); stop_actuation_thread(); });
}
int dysm_copy_actuation_log(void* ctx,char* out,size_t capacity,size_t* required) {
    return guarded([&] {
        auto& c=context(ctx);
        if(!required || g_act_thread.joinable()) throw std::invalid_argument("Stop actuator before reading log");
        std::ostringstream stream; stream<<'[';
        for(size_t k=0;k<c.pulses.size();++k) {
            const auto& p=c.pulses[k];
            if(k) stream<<',';
            stream<<"{\"time_ns\":"<<p.ns<<",\"stream_time_ns\":[";
            for(int i=0;i<c.n;++i) { if(i) stream<<','; stream<<p.stream_ns[i]; }
            stream<<"],\"physical_sms\":[";
            for(int i=0;i<c.n;++i) { if(i) stream<<','; stream<<p.sms[i]; }
            stream<<"],\"masks_hex\":[";
            for(int i=0;i<c.n;++i) { if(i) stream<<','; stream<<"\"0x"<<mask_hex(p.masks[i])<<'"'; }
            stream<<"]}";
        }
        stream<<']'; auto text=stream.str(); *required=text.size()+1;
        if(out) {
            if(capacity<*required) throw std::invalid_argument("Log buffer too small");
            std::memcpy(out,text.c_str(),*required);
        }
    });
}
void dysm_destroy(void* ctx) {
    if(ctx && ctx==live) { stop_actuation_thread(); delete live; live=nullptr; }
}
}
