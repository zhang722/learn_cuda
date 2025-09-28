#include "cuda_buffer.hpp"

#include <vector>
#include <iostream>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < N) {
        float temp = input[id];
        if (temp < 0.f) {
            temp *= 0.01f;
        }
        output[id] = temp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}


int main() {
    const int input_size = 4;
    
    std::vector<float> h_input = {1.0f, -2.0f, 3.0f, 4.0f}; // 示例输入
    std::vector<float> h_output(input_size); // 示例输入

    try {
        // ✅ 修正：d_b 的大小应为 kernel_size
        CudaBuffer<float> d_input(h_input.data(), input_size);
        CudaBuffer<float> d_output(input_size);

        solve(d_input.devicePtr(), d_output.devicePtr(), input_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        d_output.deviceToHost(h_output.data());

        for (float& a : h_output) {
            std::cout << a << ',';
        }
        std::cout << std::endl;
    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}