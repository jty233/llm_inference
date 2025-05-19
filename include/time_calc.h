#pragma once
#include <chrono>
#include <iomanip>
#include <ios>
#include <iostream>
#include <queue>
#include <string>

namespace time_calc {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    struct time_info {
        std::string name;
        time_point start_time;
    };
    inline std::queue<time_info> time_queue;
}

inline void startTimeCalc(const std::string& name) {
    time_calc::time_queue.push(time_calc::time_info{name, std::chrono::system_clock::now()});
}

inline void coutTimeCalc() {
    std::cout << std::fixed << std::setprecision(3);
    auto& info = time_calc::time_queue.back();
    std::cout << info.name << " use time: " << 
        std::chrono::duration<double>(std::chrono::system_clock::now() - info.start_time).count() << 's' << std::endl;
    time_calc::time_queue.pop();
}

inline double getTimeCalcSec() {
    std::cout << std::fixed << std::setprecision(3);
    auto info = time_calc::time_queue.back();
    time_calc::time_queue.pop();
    return std::chrono::duration<double>(std::chrono::system_clock::now() - info.start_time).count();
}

struct TimeCalcGuard {
    TimeCalcGuard(const std::string& name, int precision = 3) : name(name), precision(precision), start_time(std::chrono::system_clock::now()) {}
    ~TimeCalcGuard() {
        std::cout << std::fixed << std::setprecision(precision);
        // std::cout << name << " use time: " << 
        //     std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count() << 's' << std::endl;
    }

    time_calc::time_point start_time;
    std::string name;
    int precision;
};