#import "template.typ": *
#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm
#import "algos.typ": *
#import "figs.typ": *

#let title = "CUDA 并行scan实现"
#let authors = ("zs",)
#show: rest => project(title: title, authors: authors)[#rest]
@sweep 是upsweep的图示，假设一共有 $n=8$ 个元素，则一共会进行 $log_2(n) = 3$ 次处理，每次处理以步长 $2s$ 遍历数组。

#fig_upsweep<sweep>


#grid(
  columns: (1fr, 1fr),
  gutter: 12pt, // 两列之间的间距
  upsweep_box,
  upsweep_parallel_box
)

上面左边是upsweep的串行版本，右边是并行版本。并行版本中，需要同步的地方给出了注释。

+ "Sync here": 代表要进行线程块同步
+ "Sync warp here": 代表进行warp级别的同步，该同步要比线程块级别的快，并且有一些线程束函数可以使用，比如``` __shfl_up_sync ```、``` __shfl_down_sync ```等。



@downsweep 是downsweep的图示，处理的步长和upsweep一致，但是处理顺序相反，$s$ 应该从大到小。
#fig_downsweep<downsweep>


#grid(
  columns: (1fr, 1fr),
  gutter: 12pt, // 两列之间的间距
  downsweep_box,
  downsweep_parallel_box
)

#exclusive_scan_block_parallel_box


```cpp
constexpr int ThreadPerBlock = 64;
constexpr int ElementPerBlock = ThreadPerBlock * 2;

constexpr int LogNumBanks = 5;
#define BANK_CONFLICT_FREE(n) ((n) >> LogNumBanks)
#define MAX_SHARE_SIZE (ElementPerBlock + BANK_CONFLICT_FREE(ElementPerBlock - 1))

constexpr int WarpSize = 32;


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

```

#fig_scan_warp<scan_warp>


#grid(
  columns: (1fr, 1fr),
  gutter: 12pt, // 两列之间的间距
  scan_warp_box,
  scan_warp_parallel_box,
)

```cpp
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


```