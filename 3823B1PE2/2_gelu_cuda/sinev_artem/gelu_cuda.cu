#include "gelu_cuda.h"
#include <mutex>

__device__ inline float fast_tanh(float z)
{
    return 1.0f - 2.0f / (expf(2.0f * z) + 1.0f);
}  
constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2.0 / M_PI)

__global__ void GetResult(size_t len, float* gpu_buffer){

    int i = blockIdx.x * blockDim.x +threadIdx.x;

    if (i < len)
    {
        float x = gpu_buffer[i];
        gpu_buffer[i] = 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {

    const size_t len = input.size();
    static float *gpu_buffer = nullptr;
    static std::once_flag flag;
    
    std::call_once(flag, [len](){
        cudaMalloc(&gpu_buffer, len*sizeof(float));
    });
        
    int aCountThreads = 256;
    int aCountBlocks = (len + aCountThreads - 1) / aCountThreads;

    cudaMemcpy(gpu_buffer, input.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    GetResult<<<aCountBlocks, aCountThreads>>>(len, gpu_buffer);

    std::vector<float> output(len);

    cudaMemcpy(output.data(), gpu_buffer, len * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaFree(gpu_buffer);


    return output;
}