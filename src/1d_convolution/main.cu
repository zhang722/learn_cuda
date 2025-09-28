#include "cuda_buffer.hpp"

#include <vector>
#include <iostream>

__constant__ float kernel_constant[2048];

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = tid + bid * blockDim.x;

    int input_seg_size = blockDim.x + kernel_size - 1;
    for (int i = tid; i < input_seg_size; i += blockDim.x) {
        int global_idx = bid * blockDim.x + i;
        if (global_idx < input_size) {
            smem[i] = input[global_idx];
        }
    }  

    __syncthreads();
    if (id + kernel_size - 1 < input_size) {
        float sum = 0.f;
        for (int i = 0; i < kernel_size; i++) {
            sum += smem[tid + i] * kernel_constant[i];
        }
        output[id] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemcpyToSymbol(kernel_constant, kernel, kernel_size * sizeof(float));

    int inputSegSize = threadsPerBlock + kernel_size - 1;
    int sharedMemSize = (inputSegSize + kernel_size) * sizeof(float);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>
        (input, kernel, output, input_size, kernel_size);
    
    cudaDeviceSynchronize();
}

int main() {
    const int input_size = 4;
    const int kernel_size = 4;
    const int output_size = input_size - kernel_size + 1; // 注意：这里 output_size = 1
    
    std::vector<float> h_input = {1.0f, 2.0f, 3.0f, 4.0f}; // 示例输入
    std::vector<float> h_kernel = {1.0f, 0.5f, 0.25f, 0.125f}; // 示例核
    std::vector<float> h_output(output_size, 0.0f);

    try {
        // ✅ 修正：d_b 的大小应为 kernel_size
        CudaBuffer<float> d_input(h_input.data(), input_size);
        CudaBuffer<float> d_kernel(h_kernel.data(), kernel_size);
        CudaBuffer<float> d_output(output_size);

        solve(d_input.devicePtr(), d_kernel.devicePtr(), d_output.devicePtr(), input_size, kernel_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        d_output.deviceToHost(h_output.data());

        std::cout << "Result[0] = " << h_output[0] << std::endl;

    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}