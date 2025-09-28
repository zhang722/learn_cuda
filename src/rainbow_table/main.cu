#include "cuda_buffer.hpp"

#include <vector>
#include <iostream>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < N) {
        unsigned int temp = input[id];
        for (int i = 0; i < R; i++) {
            temp = fnv1a_hash(temp);
        }
        output[id] = temp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}


int main() {
    const int input_size = 4;
    
    std::vector<int> h_input = {1, 2, 3, 4}; // 示例输入
    std::vector<unsigned int> h_output(input_size); // 示例输入

    try {
        // ✅ 修正：d_b 的大小应为 kernel_size
        CudaBuffer<int> d_input(h_input.data(), input_size);
        CudaBuffer<unsigned int> d_output(input_size);

        solve(d_input.devicePtr(), d_output.devicePtr(), input_size, 3);
        
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