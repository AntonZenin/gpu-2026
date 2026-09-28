#include "gelu_cuda.h"
#include <mutex>
#include <iostream>

__device__ inline float fast_tanh(float z)
{
    return 1.0f - 2.0f / (expf(2.0f * z) + 1.0f);
}  
constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2.0 / M_PI)

__global__ void GetResult(size_t len, float* in, float* output){

    int i = blockIdx.x * blockDim.x +threadIdx.x;

    if (i < len)
    {
        float x = in[i];
        output[i] = 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {

    const size_t len = input.size();
    std::vector<float> output(len);

    const size_t bytes = len * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = (int)((len + blockSize - 1) / blockSize);
    GetResult<<<gridSize, blockSize>>>(len, d_in, d_out);

    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);


    return output;
} 