#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring> // std::memcpy

// 自定义异常类，用于封装 CUDA 错误
class CudaException : public std::runtime_error {
public:
    explicit CudaException(const std::string& msg) : std::runtime_error(msg) {}
};

// 检查 CUDA 调用的宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw CudaException(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * @brief 一个模板类，用于管理成对的主机和设备内存，简化数据传输。
 * 
 * @tparam T 数据类型 (e.g., float, int, double)
 */
template<typename T>
class CudaBuffer {
private:
    T* h_ptr_; // 主机内存指针 (可选，如果不需要主机端拷贝则为 nullptr)
    T* d_ptr_; // 设备内存指针
    size_t size_; // 元素个数
    bool owns_host_; // 是否由本类管理主机内存

public:
    /**
     * @brief 构造函数：分配设备内存，并可选地分配主机内存。
     * 
     * @param size 元素个数
     * @param allocate_host 是否分配主机内存用于拷入/拷出
     */
    CudaBuffer(size_t size, bool allocate_host = true)
        : h_ptr_(nullptr), d_ptr_(nullptr), size_(size), owns_host_(allocate_host) 
    {
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_ptr_, size * sizeof(T)));
        
        // 如果需要，分配主机内存
        if (allocate_host) {
            CUDA_CHECK(cudaMallocHost(&h_ptr_, size * sizeof(T))); // 使用页锁定内存，传输更快
        }
    }

    /**
     * @brief 构造函数：从现有的主机数据初始化。
     *        分配设备内存，并将数据拷贝到设备。
     * 
     * @param host_data 指向主机数据的指针
     * @param size 元素个数
     */
    CudaBuffer(const T* host_data, size_t size)
        : h_ptr_(nullptr), d_ptr_(nullptr), size_(size), owns_host_(true) 
    {
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&d_ptr_, size * sizeof(T)));
        
        // 分配页锁定主机内存
        CUDA_CHECK(cudaMallocHost(&h_ptr_, size * sizeof(T)));
        
        // 拷贝数据到主机页锁定内存
        std::memcpy(h_ptr_, host_data, size * sizeof(T));
        
        // 将数据从主机拷贝到设备
        CUDA_CHECK(cudaMemcpy(d_ptr_, h_ptr_, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    // 禁用拷贝构造和拷贝赋值（深拷贝复杂且易错，通常不需要）
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // 启用移动构造和移动赋值
    CudaBuffer(CudaBuffer&& other) noexcept
        : h_ptr_(other.h_ptr_), d_ptr_(other.d_ptr_), size_(other.size_), owns_host_(other.owns_host_) 
    {
        other.h_ptr_ = nullptr;
        other.d_ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            // 释放当前资源
            reset();
            // 转移资源
            h_ptr_ = other.h_ptr_;
            d_ptr_ = other.d_ptr_;
            size_ = other.size_;
            owns_host_ = other.owns_host_;
            // 重置源对象
            other.h_ptr_ = nullptr;
            other.d_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief 析构函数：自动释放所有资源。
     */
    ~CudaBuffer() {
        reset();
    }

    /**
     * @brief 释放所有分配的内存。
     */
    void reset() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
        }
        if (h_ptr_ && owns_host_) {
            cudaFreeHost(h_ptr_);
            h_ptr_ = nullptr;
        }
    }

    /**
     * @brief 将数据从主机拷贝到设备。
     *        如果没有分配主机内存，需要传入数据指针。
     * 
     * @param host_data 指向主机数据的指针 (如果 h_ptr_ 不存在)
     */
    void hostToDevice(const T* host_data = nullptr) {
        T* src = (host_data != nullptr) ? const_cast<T*>(host_data) : h_ptr_;
        if (!src) {
            throw CudaException("No host data provided or allocated for hostToDevice transfer.");
        }
        CUDA_CHECK(cudaMemcpy(d_ptr_, src, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief 将数据从设备拷贝到主机。
     *        如果没有分配主机内存，需要传入目标指针。
     * 
     * @param host_dest 指向主机目标内存的指针 (如果 h_ptr_ 不存在)
     */
    void deviceToHost(T* host_dest = nullptr) {
        T* dest = (host_dest != nullptr) ? host_dest : h_ptr_;
        if (!dest) {
            throw CudaException("No host destination provided or allocated for deviceToHost transfer.");
        }
        CUDA_CHECK(cudaMemcpy(dest, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief 获取设备内存指针。
     * @return T* 设备指针
     */
    T* devicePtr() const { return d_ptr_; }

    /**
     * @brief 获取主机内存指针（如果已分配）。
     * @return T* 主机指针，如果未分配则返回 nullptr
     */
    T* hostPtr() const { return h_ptr_; }

    /**
     * @brief 获取缓冲区大小（元素个数）。
     * @return size_t 元素个数
     */
    size_t size() const { return size_; }

    /**
     * @brief 获取总字节数。
     * @return size_t 字节数
     */
    size_t sizeInBytes() const { return size_ * sizeof(T); }
};

// --- 使用示例 ---
/*
#include <iostream>
#include <vector>

__global__ void addKernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    
    // 在主机上准备数据
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);

    try {
        // 创建 CudaBuffer 对象，自动分配内存并拷贝数据
        CudaBuffer<float> d_a(h_a.data(), N); // 从 h_a.data() 拷贝
        CudaBuffer<float> d_b(h_b.data(), N); // 从 h_b.data() 拷贝
        CudaBuffer<float> d_c(N);             // 只分配设备内存

        // 执行内核
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        addKernel<<<gridSize, blockSize>>>(d_c.devicePtr(), d_a.devicePtr(), d_b.devicePtr(), N);
        
        // 检查内核启动错误
        CUDA_CHECK(cudaGetLastError());
        // 等待内核完成
        CUDA_CHECK(cudaDeviceSynchronize());

        // 将结果从设备拷贝回主机
        d_c.deviceToHost(h_c.data());

        // 此时 h_c 包含结果，CudaBuffer 对象在作用域结束时自动释放内存
        std::cout << "Result[0] = " << h_c[0] << std::endl; // 应为 3.0

    } catch (const CudaException& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
*/