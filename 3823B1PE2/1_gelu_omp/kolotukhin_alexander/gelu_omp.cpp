#include "gelu_omp.h"

#include <omp.h>

#include <cmath>

constexpr float PI = 3.1415926535f;
constexpr float LOG2E = 1.4426950408889634f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t input_size = input.size();
    std::vector<float> output(input_size);
    const float sr = std::sqrt(2.0f / PI);

    #pragma omp parallel for simd schedule(static) \
        default(none) shared(sr, input, output, input_size)
    for (std::size_t i = 0; i < input_size; i++) {
        const float val = input[i];
        const float arg  = sr * val * (1.0f + 0.044715f * val * val);
        output[i] = val / (1.0f + exp2f(-2.0f * LOG2E * arg));
    }

    return output;
}
