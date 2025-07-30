#include "timer.h"
#include <iomanip>

PerformanceTimer::PerformanceTimer() : is_running(false), is_multi_run_mode(false), run_count(0) {
}

void PerformanceTimer::start(const std::string& phaseName) {
    phases.clear();
    total_start = std::chrono::high_resolution_clock::now();
    is_running = true;
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = total_start;
    phases.push_back(phase);
}

void PerformanceTimer::next(const std::string& phaseName) {
    if (!is_running) {
        std::cerr << "Timer not started! Call start() first." << std::endl;
        return;
    }
    
    endCurrentPhase();
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = std::chrono::high_resolution_clock::now();
    phases.push_back(phase);
}

void PerformanceTimer::finish() {
    if (!is_running) {
        std::cerr << "Timer not started! Call start() first." << std::endl;
        return;
    }
    
    endCurrentPhase();
    total_end = std::chrono::high_resolution_clock::now();
    is_running = false;
    
    printResults();
}

void PerformanceTimer::endCurrentPhase() {
    if (!phases.empty() && phases.back().end_time == std::chrono::high_resolution_clock::time_point{}) {
        phases.back().end_time = std::chrono::high_resolution_clock::now();
        phases.back().duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            phases.back().end_time - phases.back().start_time).count();
    }
}

long long PerformanceTimer::getPhaseDuration(const std::string& phaseName) const {
    for (const auto& phase : phases) {
        if (phase.name == phaseName) {
            return phase.duration_us;
        }
    }
    return -1; // Phase not found
}

long long PerformanceTimer::getTotalDuration() const {
    if (total_start == std::chrono::high_resolution_clock::time_point{} || 
        total_end == std::chrono::high_resolution_clock::time_point{}) {
        return -1;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
}

void PerformanceTimer::printResults() const {
    if (is_multi_run_mode) {
        printMultiRunResults();
        return;
    }
    
    std::cout << "\n=== Detailed Performance Summary ===" << std::endl;
    
    for (const auto& phase : phases) {
        std::cout << std::left << std::setw(30) << (phase.name + ":") 
                  << std::right << std::setw(20) << phase.duration_us << " microseconds ("
                  << std::fixed << std::setprecision(2) << (double)phase.duration_us / 1000.0 << " ms)" << std::endl;
    }
    
    long long total_us = getTotalDuration();
    if (total_us > 0) {
        std::cout << std::left << std::setw(25) << "Total Execution Time:" 
                  << std::right << std::setw(10) << total_us << " microseconds ("
                  << std::fixed << std::setprecision(2) << (double)total_us / 1000.0 << " ms)" << std::endl;
    }
}

void PerformanceTimer::startRun(const std::string& phaseName) {
    if (!is_multi_run_mode) {
        is_multi_run_mode = true;
        total_start = std::chrono::high_resolution_clock::now();
    }
    
    current_run_phases.clear();
    current_run_start = std::chrono::high_resolution_clock::now();
    is_running = true;
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = current_run_start;
    current_run_phases.push_back(phase);
}

void PerformanceTimer::nextRun(const std::string& phaseName) {
    if (!is_running) {
        std::cerr << "Timer not started! Call startRun() first." << std::endl;
        return;
    }
    
    endCurrentRunPhase();
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = std::chrono::high_resolution_clock::now();
    current_run_phases.push_back(phase);
}

void PerformanceTimer::finishRun() {
    if (!is_running) {
        std::cerr << "Timer not started! Call startRun() first." << std::endl;
        return;
    }
    
    endCurrentRunPhase();
    is_running = false;
    run_count++;
    
    for (const auto& phase : current_run_phases) {
        run_phase_durations[phase.name].push_back(phase.duration_us);
    }
}

void PerformanceTimer::finishAllRuns() {
    total_end = std::chrono::high_resolution_clock::now();
    printMultiRunResults();
}

void PerformanceTimer::endCurrentRunPhase() {
    if (!current_run_phases.empty() && current_run_phases.back().end_time == std::chrono::high_resolution_clock::time_point{}) {
        current_run_phases.back().end_time = std::chrono::high_resolution_clock::now();
        current_run_phases.back().duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            current_run_phases.back().end_time - current_run_phases.back().start_time).count();
    }
}

long long PerformanceTimer::getAveragePhaseDuration(const std::string& phaseName) const {
    auto it = run_phase_durations.find(phaseName);
    if (it == run_phase_durations.end() || it->second.empty()) {
        return -1; // Phase not found
    }
    
    long long sum = 0;
    for (long long duration : it->second) {
        sum += duration;
    }
    return sum / it->second.size();
}

int PerformanceTimer::getRunCount() const {
    return run_count;
}

void PerformanceTimer::printMultiRunResults() const {
    std::cout << "\n=== Multi-Run Performance Summary (" << run_count << " runs) ===" << std::endl;
    
    for (const auto& phase_data : run_phase_durations) {
        const std::string& phase_name = phase_data.first;
        const std::vector<long long>& durations = phase_data.second;
        
        if (!durations.empty()) {
            long long sum = 0;
            long long min_duration = durations[0];
            long long max_duration = durations[0];
            
            for (long long duration : durations) {
                sum += duration;
                min_duration = std::min(min_duration, duration);
                max_duration = std::max(max_duration, duration);
            }
            
            long long avg_duration = sum / durations.size();
            
            std::cout << std::left << std::setw(30) << (phase_name + " (avg):");
            std::cout << std::right << std::setw(15) << avg_duration << " μs ("
                      << std::fixed << std::setprecision(2) << (double)avg_duration / 1000.0 << " ms)" << std::endl;
            
            std::cout << std::left << std::setw(30) << "  [min/max]:";
            std::cout << std::right << std::setw(15) << min_duration << " / " << max_duration << " μs" << std::endl;
        }
    }
    
    long long total_us = getTotalDuration();
    if (total_us > 0) {
        std::cout << std::left << std::setw(25) << "Total Execution Time:" 
                  << std::right << std::setw(10) << total_us << " microseconds ("
                  << std::fixed << std::setprecision(2) << (double)total_us / 1000.0 << " ms)" << std::endl;
    }
}
