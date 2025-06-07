#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

constexpr int ThreadPerBlock = 2;
constexpr int ElementPerBlock = ThreadPerBlock * 2;

__global__ void scan_block(int* a, int* output, int *block_sums, int n) {
    // load
    __shared__ int temp[ElementPerBlock];
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //2
    int tid = threadIdx.x; //0
    int bid = blockIdx.x;
    if (2 * idx < n) {
        temp[2 * tid] = a[2 * idx]; 
    }
    if (2 * idx + 1 < n) {
        temp[2 * tid + 1] = a[2 * idx + 1];
    }

    int t = ElementPerBlock >> 1;
    for (int s = 1; s < ElementPerBlock; s *= 2) {
        __syncthreads();
        if (tid < t) {
            int k = tid * 2 * s;
            int i = k + s - 1; 
            int j = k + 2 * s - 1;

            temp[j] += temp[i];
        }
        t >>= 1;
    }

    if (tid == 0) {
        block_sums[bid] = temp[ElementPerBlock - 1];
        temp[ElementPerBlock - 1] = 0;
    }
    t = 1;
    for (int s = ElementPerBlock >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < t) {
            int k = tid * 2 * s;
            int i = k + s - 1; 
            int j = k + 2 * s - 1;

            int tt = temp[j];
            temp[j] += temp[i];
            temp[i] = tt; 
        }
        t *= 2;
    }

    __syncthreads();

    if (2 * idx < n) {
        output[2 * idx] =  temp[2 * tid]; 
    }
    if (2 * idx + 1 < n) {
        output[2 * idx + 1] = temp[2 * tid + 1];
    }
}



__global__ void add_kernel(int *output, int *sums, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int bid = blockIdx.x;

    if (2 * idx < n) {
        output[2 * idx] += sums[bid];
    }
    if (2 * idx + 1 < n) {
        output[2 * idx + 1] += sums[bid];
    }
}





void scan_large(int *input, int *output, int n) {
    int *d_input;
    int *d_output;
    int *d_sums;
    int *d_sums_sums;
    
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlock = (n + ElementPerBlock - 1) / ElementPerBlock;

    cudaMalloc(&d_sums, numBlock * sizeof(int));
    cudaMalloc(&d_sums_sums, numBlock * sizeof(int));

    {
        cudaEvent_t start, stop;
        float elapsedTime = 0.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        scan_block<<<numBlock, ThreadPerBlock>>>(d_input, d_output, d_sums, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("scan_large time: %f ms\n", elapsedTime);
    }
    if (numBlock != 1) {
        scan_large(d_sums, d_sums_sums, numBlock);
        cudaEvent_t start, stop;
        float elapsedTime = 0.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        add_kernel<<<numBlock, ThreadPerBlock>>>(d_output, d_sums_sums, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("add_kernel time: %f ms\n", elapsedTime);
    } 

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
}



int main() {
    int input[] = {1, 2, 3, 0, 1, 1, 1, 1, 1};
    int output[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    scan_large(input, output, 9);
    for (int i = 0; i < 9; i++) {
        std::cout << output[i] << ',';
    }
    std::cout << std::endl;
}