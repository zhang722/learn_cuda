#include "cuda_buffer.hpp"

#include <vector>
#include <iostream>

__device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 归约 kernel
__global__ void reduce_kernel(const int* input, int* partial_sums, int n) {
    const int sharedMemSize = (blockDim.x + 31) / 32;
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;

    // 每个线程处理多个元素
    if (gid >= n)
        return;
    
    int sum = input[gid];

    // warp 内归约
    sum = warpReduceSum(sum);

    // 每个 warp 的 leader 写入结果
    if (tid % 32 == 0) {
        smem[tid / 32] = sum;
    }

    __syncthreads();

    if (tid == 0) {
        int temp = 0;
        for (int i = 0; i < sharedMemSize; i++) {
            temp += smem[i];
        }
        partial_sums[bid] = temp;
    }

}

// 递归归约函数
void recursive_reduce(const int* d_input, int* d_output, int n, int block_size = 256) {
    // 基础情况：如果数组很小，直接用一个 block 处理
    if (n <= block_size) {
        reduce_kernel<<<1, block_size>>>(d_input, d_output, n);
        return;
    }

    // 计算 grid size
    int grid_size = (n + block_size - 1) / block_size;

    // 每个 warp 产生一个部分和
    int partials_per_block = (block_size + 31) / 32;
    int num_partials = grid_size * partials_per_block;

    // 分配临时存储
    int *d_partials;
    CUDA_CHECK(cudaMalloc(&d_partials, num_partials * sizeof(int)));

    // 启动当前层归约
    reduce_kernel<<<grid_size, block_size>>>(d_input, d_partials, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 递归处理部分和数组
    recursive_reduce(d_partials, d_output, num_partials, block_size);

    // 释放临时内存
    cudaFree(d_partials);
}


__global__ void count_equal_kernel(const int* input, int* output, int *to_sum, int N, int K) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id >= N) {
        return; 
    }

    if (input[id] == K) {
        to_sum[id] = 1;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int* to_sum;
    int* partial_sum;
    cudaMalloc(&to_sum, N * sizeof(int));
    cudaMalloc(&partial_sum, N * sizeof(int));

    cudaMemset(to_sum, 0, N * sizeof(int));  // 先清零
    cudaMemset(partial_sum, 0, N * sizeof(int));  // 先清零
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, to_sum, N, K);
    recursive_reduce(to_sum, partial_sum, N);
    cudaMemcpy(output, partial_sum, sizeof(int), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(to_sum);
    cudaFree(partial_sum);
}


int main() {
    const int input_size = 4;
    
    std::vector<int> h_input = {1, 3, 3, 4}; // 示例输入
    std::vector<int> h_output(1); // 示例输入

    try {
        // ✅ 修正：d_b 的大小应为 kernel_size
        CudaBuffer<int> d_input(h_input.data(), input_size);
        CudaBuffer<int> d_output(1);

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