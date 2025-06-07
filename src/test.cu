#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>


void get_cudaDeviceinfo(void)
{
 
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    printf("使用GPU device, %d, %s\n",dev, devProp.name);
    printf("SM的数量, %d\n",devProp.multiProcessorCount);
    printf("每个线程块的共享内存大小 ::%dKB\n" , devProp.sharedMemPerBlock / 1024);
    printf("每个线程块的最大线程数：::%d\n", devProp.maxThreadsPerBlock);
    printf("每个SM的最大线程数::::%d\n" , devProp.maxThreadsPerMultiProcessor);
    printf("每个SM的最大线程束数::%d\n" , devProp.maxThreadsPerMultiProcessor / 32);
}

int main() {
    get_cudaDeviceinfo();
    return 0;
}