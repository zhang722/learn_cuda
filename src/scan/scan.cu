#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "options.h"

constexpr int ThreadPerBlock = 64;
constexpr int ElementPerBlock = ThreadPerBlock * 2;

// constexpr int NumBanks = 32;
constexpr int LogNumBanks = 5;
#define BANK_CONFLICT_FREE(n) ((n) >> LogNumBanks)
#define MAX_SHARE_SIZE (ElementPerBlock + BANK_CONFLICT_FREE(ElementPerBlock - 1))

constexpr int WarpSize = 32;

void print(int *d_a, int n) {
    int *a = new int[n];
    cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%d,", a[i]);
    }
    printf("\n");
    delete[] a;
}


__global__ void scan_block(int* a, int* output, int *block_sums, int n) {
    // load
    __shared__ int temp[MAX_SHARE_SIZE];
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //2
    int tid = threadIdx.x; //0
    int bid = blockIdx.x;
    int ai = 2 * tid;
    int bi = 2 * tid + 1;
    int bankOffsetA = BANK_CONFLICT_FREE(ai);
    int bankOffsetB = BANK_CONFLICT_FREE(bi);
    if (2 * idx < n) {
        temp[ai + bankOffsetA] = a[2 * idx]; 
    }
    if (2 * idx + 1 < n) {
        temp[bi + bankOffsetB] = a[2 * idx + 1];
    }

    int t = ElementPerBlock >> 1;
    for (int s = 1; s < ElementPerBlock; s *= 2) {
        __syncthreads();
        if (tid < t) {
            int k = tid * 2 * s;
            int i = k + s - 1; 
            int j = k + 2 * s - 1;

            i += BANK_CONFLICT_FREE(i);
            j += BANK_CONFLICT_FREE(j);

            temp[j] += temp[i];
        }
        t >>= 1;
    }

    if (tid == 0) {
        block_sums[bid] = temp[MAX_SHARE_SIZE - 1];
        temp[MAX_SHARE_SIZE - 1] = 0;
    }
    t = 1;
    for (int s = ElementPerBlock >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < t) {
            int k = tid * 2 * s;
            int i = k + s - 1; 
            int j = k + 2 * s - 1;
            i += BANK_CONFLICT_FREE(i);
            j += BANK_CONFLICT_FREE(j);

            int tt = temp[j];
            temp[j] += temp[i];
            temp[i] = tt; 
        }
        t *= 2;
    }

    __syncthreads();

    if (2 * idx < n) {
        output[2 * idx] =  temp[ai + bankOffsetA]; 
    }
    if (2 * idx + 1 < n) {
        output[2 * idx + 1] = temp[bi + bankOffsetB];
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





void scan_large(int *d_input, int *d_output, int n) {
    int *d_sums;
    int *d_sums_sums;

    int numBlock = (n + ElementPerBlock - 1) / ElementPerBlock;

    cudaMalloc(&d_sums, numBlock * sizeof(int));
    cudaMalloc(&d_sums_sums, numBlock * sizeof(int));

    scan_block<<<numBlock, ThreadPerBlock>>>(d_input, d_output, d_sums, n);

    if (numBlock != 1) {
        scan_large(d_sums, d_sums_sums, numBlock);
        add_kernel<<<numBlock, ThreadPerBlock>>>(d_output, d_sums_sums, n);
    } 
}

void scan_large_wrapper(int *input, int *output, int n) {
    int *d_input;
    int *d_output;
    
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    scan_large(d_input, d_output, n);
    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
}



__device__ int scan_warp(int val) {
    int x = val;
    for (int offset = 1; offset < WarpSize; offset <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, offset);
        if (threadIdx.x % WarpSize >= offset) {
            x += y;
        }
    }
    return x - val;
}

__global__ void scan_warp_block(int *a, int *output, int *block_sums, int n) {
    // constexpr int NumWarpsPerBlock = ThreadPerBlock / WarpSize;
    __shared__ int sums[WarpSize];
    __shared__ int temp[ThreadPerBlock];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockDim.x * bid + tid;
    int lane = tid % WarpSize;
    int wid = tid / WarpSize;
    if (idx < n) {
        temp[tid] = a[idx];
    }
    int val = temp[tid];

    // scan per warp
    int sum = scan_warp(temp[tid]);
    if (lane == WarpSize - 1) {
        sums[wid] = sum + temp[tid];
    }
    temp[tid] = sum;

    __syncthreads();

    if (wid == 0) {
        int all_sum = scan_warp(sums[lane]);
        sums[lane] = all_sum;
    }

    __syncthreads();
    temp[tid] += sums[wid];

    if (idx < n) {
        output[idx] = temp[tid];
    }

    if (tid == ThreadPerBlock - 1) {
        block_sums[bid] = temp[tid] + val;
    }
}



void scan_warp_large(int *d_input, int *d_output, int n) {
    int numBlock = (n + ThreadPerBlock - 1) / ThreadPerBlock;
    int *d_sums;
    int *d_sums_sums;
    cudaMalloc(&d_sums, numBlock * sizeof(int));
    cudaMalloc(&d_sums_sums, numBlock * sizeof(int));

    scan_warp_block<<<numBlock, ThreadPerBlock>>>(d_input, d_output, d_sums, n);

    if (numBlock != 1) {
        scan_warp_large(d_sums, d_sums_sums, numBlock);

        add_kernel<<<numBlock, ThreadPerBlock / 2>>>(d_output, d_sums_sums, n);
    } 
}

void scan_warp_large_wrapper(int *input, int *output, int n) {
    int *d_input;
    int *d_output;
    
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    scan_warp_large(d_input, d_output, n);
    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
}



__global__ void find_repeats_kernel(int *input, int *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n - 1) {
        if (input[idx] == input[idx + 1]) {
            output[idx] = 1;
        } else {
            output[idx] = 0;
        }
    }
}


__global__ void indexing_kernel(int *d_repeat, int *d_sums, int *d_output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        if (d_repeat[idx] == 1) {
            d_output[d_sums[idx]] = idx;
        }
    }
}


void find_repeats(int *input, int * &output, int n, int &n_output) {
    int *d_input;
    int *d_repeat;
    
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_repeat, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlock = (n + ThreadPerBlock - 1) / ThreadPerBlock;

    find_repeats_kernel<<<numBlock, ThreadPerBlock>>>(d_input, d_repeat, n);

    int *d_sums;
    cudaMalloc(&d_sums, n * sizeof(int));
    scan_warp_large(d_repeat, d_sums, n);

    int num_repeat;
    cudaMemcpy(&num_repeat, d_sums + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

    int *d_output;
    cudaMalloc(&d_output, num_repeat * sizeof(int));

    indexing_kernel<<<numBlock, ThreadPerBlock>>>(d_repeat, d_sums, d_output, n);

    n_output = num_repeat;
    int *out = new int[num_repeat];
    cudaMemcpy(out, d_output, num_repeat * sizeof(int), cudaMemcpyDeviceToHost);
    output = out;
}



int main(int argc, char* argv[]) {
    ProgramOptions opts = parse_arguments(argc, argv);

    std::cout << "test_type  = " << opts.test_type << "\n";
    std::cout << "input_type = " << opts.input_type << "\n";
    std::cout << "array_size = " << opts.array_size << "\n";
    std::cout << "use_thrust = " << (opts.use_thrust ? "true" : "false") << "\n";


    int N = opts.array_size;
    int *input = new int[N];
    int *output = new int[N];

    for (int i = 0; i < N; i++) {
        input[i] = 1;
    }

    if (opts.input_type == "scan") {
        scan_large_wrapper(input, output, N);
        // scan_warp_large_wrapper(input, output, N);
    } else if (opts.input_type == "find_repeats ") {
        int *out = nullptr;
        int num_repeat;
        find_repeats(input, out, N, num_repeat);
        delete[] out;
    }
}