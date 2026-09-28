#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRows = 4;
constexpr int kTile = kBlockX;

__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * kTile + tx;
    const int row = blockIdx.y * kTile + ty * kRows;

    float acc[kRows] = {};

    for (int k0 = 0; k0 < n; k0 += kTile) {
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            as[ty * kRows + i][tx] = a[(row + i) * n + (k0 + tx)];
            bs[ty * kRows + i][tx] = b[(k0 + ty * kRows + i) * n + col];
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const float bv = bs[k][tx];
#pragma unroll
            for (int i = 0; i < kRows; ++i) {
                acc[i] += as[ty * kRows + i][k] * bv;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        c[(row + i) * n + col] = acc[i];
    }
}

__global__ void GemmSmall(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int cap = 0;
    if (n > cap) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cap = n;
    }

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n >= kTile) {
        dim3 block(kBlockX, kBlockY);
        dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    } else if (n > 0) {
        dim3 block(kBlockX, kBlockY);
        dim3 grid((n + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);
        GemmSmall<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(static_cast<size_t>(n) * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
