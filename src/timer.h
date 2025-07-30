#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <map>

class PerformanceTimer {
public:
    PerformanceTimer();
    
    void start(const std::string& phaseName);
    
    void next(const std::string& phaseName);
    
    void finish();
    
    long long getPhaseDuration(const std::string& phaseName) const;
    
    long long getTotalDuration() const;
    
    // Multi-run support methods
    void startRun(const std::string& phaseName);
    void nextRun(const std::string& phaseName);
    void finishRun();
    void finishAllRuns();
    
    long long getAveragePhaseDuration(const std::string& phaseName) const;
    int getRunCount() const;

private:
    struct Phase {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        long long duration_us;
    };
    
    std::vector<Phase> phases;
    std::chrono::high_resolution_clock::time_point total_start;
    std::chrono::high_resolution_clock::time_point total_end;
    bool is_running;
    
    std::map<std::string, std::vector<long long>> run_phase_durations;
    std::vector<Phase> current_run_phases;
    std::chrono::high_resolution_clock::time_point current_run_start;
    bool is_multi_run_mode;
    int run_count;
    
    void endCurrentPhase();
    void endCurrentRunPhase();
    void printResults() const;
    void printMultiRunResults() const;
};
