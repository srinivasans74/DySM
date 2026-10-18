
static inline void sleep_until_abs_hybrid_raw(const timespec& target)
{
    const long long guard_ns = 50'000; // 50 µs
    while (true) {
        timespec now = ts_now_raw();
        long long diff = ts_diff_ns(target, now);
        if (diff <= 0)
            break;
        if (diff > guard_ns) {
            // Sleep until slightly before target
            timespec pre = ts_add(target, -guard_ns);
            clock_nanosleep(CLOCK_MONOTONIC_RAW, TIMER_ABSTIME, &pre, nullptr);
        } else {
            // Busy-spin for the last few microseconds
#if defined(__x86_64__) || defined(__i386__)
            // Original x86 pause
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            // FIX: Use ARM's equivalent to PAUSE/YIELD
            __asm__ __volatile__ ("yield" ::: "memory");
#else
            // Fallback for other/unknown architectures
            std::this_thread::yield();
#endif
        }
    }
}
