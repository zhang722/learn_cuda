#include "cuda_buffer.hpp"

#include <vector>
#include <iostream>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    int r = threadIdx.y + blockIdx.y * blockDim.y;

    if (c < N && r < N) {
        B[r * N + c] = A[r * N + c];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int blockSizeX = 16;
    int blockSizeY = 16;
    int gridSizeX = (N + blockSizeX - 1) / blockSizeX;
    int gridSizeY = (N + blockSizeY - 1) / blockSizeY;
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize(gridSizeX, gridSizeY);

    copy_matrix_kernel<<<gridSize, blockSize>>>(A, B, N);
    cudaDeviceSynchronize();
} 


int main() {
    const int input_size = 4;
    
    std::vector<float> h_input = {1, 2, 3, 4}; // 示例输入
    std::vector<float> h_output(input_size); // 示例输入

    try {
        // ✅ 修正：d_b 的大小应为 kernel_size
        CudaBuffer<float> d_input(h_input.data(), input_size);
        CudaBuffer<float> d_output(input_size);

        solve(d_input.devicePtr(), d_output.devicePtr(), input_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        d_output.deviceToHost(h_output.data());

        for (auto& a : h_output) {
            std::cout << a << ',';
        }
        std::cout << std::endl;
    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}