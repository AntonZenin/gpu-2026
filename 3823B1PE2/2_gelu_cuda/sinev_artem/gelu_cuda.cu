#include "gelu_cuda.h"
#include <mutex>

__device__ inline float fast_tanh(float z)
{
    return 1.0f - 2.0f / (expf(2.0f * z) + 1.0f);
}  
constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2.0 / M_PI)

__global__ void GetResult(size_t len, float* host_buffer){

    int i = blockIdx.x * blockDim.x +threadIdx.x;

    if (i < len)
    {
        float x = host_buffer[i];
        host_buffer[i] = 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {

    const size_t len = input.size();
    static float *host_buffer = nullptr;
    static std::once_flag flag;
    
    std::call_once(flag, [len](){
        cudaMallocHost(&host_buffer, len*sizeof(float));
    });
        
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GetResult, 0, 0);

    int gridSize = (len + blockSize - 1) / blockSize;

    // cudaMemcpy(host_buffer, input.data(), len * sizeof(float), cudaMemcpyHostToDevice);

    GetResult<<<gridSize, blockSize>>>(len, host_buffer);
    
    cudaDeviceSynchronize();

    std::vector<float> output(len);
    memcpy(output.data(), host_buffer, len * sizeof(float));
    // cudaMemcpy(output.data(), host_buffer, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFreeHost(host_buffer);


    return output;
}