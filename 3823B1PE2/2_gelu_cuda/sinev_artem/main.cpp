#include "gelu_cuda.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

int main() {
    const std::size_t size = 134217728; // 2^27
    std::vector<float> input(size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : input) v = dist(rng);

    // Warming-up
    GeluCUDA(input);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        GeluCUDA(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << "Time: " << time << " s\n";
    return 0;
}