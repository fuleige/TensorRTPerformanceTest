
#pragma once
#include <chrono>

namespace fulei{
    
    class TimeRecoder
    {
    private:
        std::chrono::steady_clock::time_point start_time;
    public:
        TimeRecoder() {
            reset();
        }

        uint32_t cost_ms() {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
        }

        uint32_t cost_us() {
            return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
        }

        void reset() {
            start_time = std::chrono::steady_clock::now();
        }
    };

} 
