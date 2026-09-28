#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = input.size();
    std::vector<float> output(n);

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x*x*x) ))

    // Using tanh(a) = (e^(2a) − 1) / (e^(2a) + 1):
    // GELU(x) = 0.5 * x * (1 + (e^(2t) − 1) / (e^(2t) + 1) )
    // GELU(x) = x * e^(2t) / (e^(2t) + 1)
    // u = 2 * t
    // GELU(x) = x * e^(u) / (e^(u) + 1) | :e^(u)
    // GELU(x) = x / (1 + e^(-u))

    // u = 2*sqrt(2/π)*(x + 0.044715*x*x*x)
    // GELU(x) = x / (1 + exp(−u))

    const float kPi = 3.14159265358979323846f;
    const float kValue = sqrt(2 / kPi);
    const float kAlpha = 0.044715f;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const float x = input[i];
        const float u = 2.0f * kValue * (x + kAlpha * x * x * x);
        output[i] = x / (1.0f + std::exp(-u));
    }

    return output;
}