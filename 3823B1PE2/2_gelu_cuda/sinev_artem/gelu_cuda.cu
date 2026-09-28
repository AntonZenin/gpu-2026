#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>

constexpr float kSqrt2OverPi = 0.7978845608028654f;

__device__ __forceinline__ float fast_tanh(float z) {
    float e = expf(2.0f * z);
    return (e - 1.0f) / (e + 1.0f);
}

__global__ void GetResult(size_t len, float* __restrict__ in) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = in[i];
        in[i] = 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int len = input.size();
    std::vector<float> output(len);

    const int bytes = len * sizeof(float);

    static float* d_in = nullptr;
    if(d_in == nullptr){
        cudaMalloc(&d_in, bytes);
    }


    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = ((len + blockSize - 1) / blockSize);
    GetResult<<<gridSize, blockSize>>>(len, d_in);

    cudaMemcpy(output.data(), d_in, bytes, cudaMemcpyDeviceToHost);

    // cudaFree(d_in);

    return output;
}