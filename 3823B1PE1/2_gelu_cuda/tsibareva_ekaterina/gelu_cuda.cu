#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Using the do {...} while (0) construct for a one‑time execution
#define CHECK_ERROR(X)                            \
    do {                                          \
        cudaError_t err = (X);                    \
        if (err != cudaSuccess) {                 \
            printf("Error: %s\n",                 \
                   cudaGetErrorString(err));      \
            exit(0);                              \
        }                                         \
    } while (0)

__global__ void gelu_kernel(const float* in, float* out, int N,
                            float kValue) {
    const float kAlpha = 0.044715f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float x = in[i];
        const float u = 2.0f * kValue * (x + kAlpha * x * x * x);
        out[i] = x / (1.0f + expf(-u));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int N = (int)input.size();
    std::vector<float> output(N);
    if (N == 0) return output;

    const float kPi = 3.14159265358979323846f;
    const float kValue = std::sqrt(2.0f / kPi);

    const size_t size = N * sizeof(float);

    float* d_in;
    float* d_out;
    CHECK_ERROR(cudaMalloc(&d_in, size));
    CHECK_ERROR(cudaMalloc(&d_out, size));

    CHECK_ERROR(cudaMemcpy(d_in, input.data(), size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, kValue);
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(output.data(), d_out, size, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_in));
    CHECK_ERROR(cudaFree(d_out));

    return output;
}