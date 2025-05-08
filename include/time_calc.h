#pragma once
#include <chrono>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

namespace time_calc {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    inline time_point start_time = std::chrono::system_clock::now();
    inline std::string name;
}

inline void startTimeCalc(const std::string& name) {
    time_calc::name = name;
    time_calc::start_time = std::chrono::system_clock::now();
}

inline void finishTimeCalc() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << time_calc::name << " use time: " << 
        std::chrono::duration<double>(std::chrono::system_clock::now() - time_calc::start_time).count() << 's' << std::endl;
}

struct TimeCalcGuard {
    TimeCalcGuard(const std::string& name, int precision = 3) : name(name), precision(precision), start_time(std::chrono::system_clock::now()) {}
    ~TimeCalcGuard() {
        std::cout << std::fixed << std::setprecision(precision);
        std::cout << name << " use time: " << 
            std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count() << 's' << std::endl;
    }

    time_calc::time_point start_time;
    std::string name;
    int precision;
};