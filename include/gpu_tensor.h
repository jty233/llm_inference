#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <stdexcept>

template<typename T>
class GpuTensor {
public:
    GpuTensor() : data_ptr(nullptr), size_(0), is_on_gpu(true) {}

    GpuTensor(const std::vector<T>& host_data, const std::vector<int>& shape) : shape_(shape), is_on_gpu(true) {
        size_ = 1;
        for (int s : shape_) {
            size_ *= s;
        }

        // 分配GPU内存
        cudaError_t err = cudaMalloc(&data_ptr, size_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }

        // 将数据从主机复制到GPU
        err = cudaMemcpy(data_ptr, host_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(data_ptr);
            throw std::runtime_error("Failed to copy data to GPU");
        }
    }

    GpuTensor(const std::vector<int>& shape) : shape_(shape), is_on_gpu(true) {
        size_ = 1;
        for (int s : shape_) {
            size_ *= s;
        }

        // 分配GPU内存并初始化为0
        cudaError_t err = cudaMalloc(&data_ptr, size_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }

        err = cudaMemset(data_ptr, 0, size_ * sizeof(T));
        if (err != cudaSuccess) {
            cudaFree(data_ptr);
            throw std::runtime_error("Failed to initialize GPU memory");
        }
    }

    // 拷贝构造函数
    GpuTensor(const GpuTensor& other) : shape_(other.shape_), size_(other.size_), is_on_gpu(true) {
        cudaError_t err = cudaMalloc(&data_ptr, size_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory in copy constructor");
        }

        err = cudaMemcpy(data_ptr, other.data_ptr, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            cudaFree(data_ptr);
            throw std::runtime_error("Failed to copy GPU data in copy constructor");
        }
    }

    // 移动构造函数
    GpuTensor(GpuTensor&& other) noexcept 
        : data_ptr(other.data_ptr), shape_(std::move(other.shape_)), size_(other.size_), is_on_gpu(other.is_on_gpu) {
        other.data_ptr = nullptr;
        other.size_ = 0;
    }

    // 赋值操作符
    GpuTensor& operator=(const GpuTensor& other) {
        if (this != &other) {
            if (data_ptr) {
                cudaFree(data_ptr);
            }
            
            shape_ = other.shape_;
            size_ = other.size_;
            is_on_gpu = true;
            
            cudaError_t err = cudaMalloc(&data_ptr, size_ * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory in assignment");
            }

            err = cudaMemcpy(data_ptr, other.data_ptr, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(data_ptr);
                throw std::runtime_error("Failed to copy GPU data in assignment");
            }
        }
        return *this;
    }

    // 移动赋值操作符
    GpuTensor& operator=(GpuTensor&& other) noexcept {
        if (this != &other) {
            if (data_ptr) {
                cudaFree(data_ptr);
            }
            
            data_ptr = other.data_ptr;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            is_on_gpu = other.is_on_gpu;
            
            other.data_ptr = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~GpuTensor() {
        if (data_ptr) {
            cudaFree(data_ptr);
        }
    }

    // 获取GPU指针
    T* get() const { return data_ptr; }

    // 获取形状
    const std::vector<int>& shape() const { return shape_; }

    // 获取大小
    size_t size() const { return size_; }

    // 将张量从GPU复制回CPU
    std::vector<T> toHost() const {
        std::vector<T> host_data(size_);
        cudaError_t err = cudaMemcpy(host_data.data(), data_ptr, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy data from GPU to host");
        }
        return host_data;
    }

    // 从主机更新GPU张量
    void updateFromHost(const std::vector<T>& host_data) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Host data size does not match tensor size");
        }
        
        cudaError_t err = cudaMemcpy(data_ptr, host_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to update GPU tensor from host");
        }
    }

    // 矩阵乘法
    GpuTensor matMul(const GpuTensor& other) const {
        // 确保是2D矩阵
        if (shape_.size() != 4 || other.shape_.size() != 4) {
            throw std::runtime_error("Matrix multiplication requires 2D tensors");
        }

        int m = shape_[2];  // 行数
        int k = shape_[3];  // 第一个矩阵的列数
        int n = other.shape_[3];  // 第二个矩阵的列数

        if (shape_[3] != other.shape_[2]) {
            throw std::runtime_error("Matrix dimensions incompatible for multiplication");
        }

        // 创建结果张量
        std::vector<int> result_shape = {1, 1, m, n};
        GpuTensor result(result_shape);

        // 使用cuBLAS进行矩阵乘法
        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    other.get(), n,
                    get(), k,
                    &beta,
                    result.get(), n);

        cublasDestroy(handle);
        return result;
    }

    // 元素级加法
    GpuTensor operator+(const GpuTensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor shapes must match for addition");
        }

        GpuTensor result(shape_);
        addKernel(data_ptr, other.data_ptr, result.data_ptr, size_);
        return result;
    }

    // 元素级乘法
    GpuTensor elementWiseMul(const GpuTensor& other) const {
        if (shape_ != other.shape_) {
            throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
        }

        GpuTensor result(shape_);
        mulKernel(data_ptr, other.data_ptr, result.data_ptr, size_);
        return result;
    }

    // 从Tensor<T>创建GpuTensor
    template<typename U>
    static GpuTensor fromTensor(const U& tensor) {
        std::vector<T> host_data = tensor.data;  // 假设tensor有data成员
        std::vector<int> shape = tensor.shape;   // 假设tensor有shape成员
        return GpuTensor(host_data, shape);
    }

    // 转换为Tensor<T>
    template<typename U>
    U toTensor() const {
        std::vector<T> host_data = toHost();
        return U(host_data, shape_);  // 假设Tensor构造函数接受数据和形状
    }

private:
    T* data_ptr;
    std::vector<int> shape_;
    size_t size_;
    bool is_on_gpu;

    // CUDA核函数声明
    static void addKernel(const T* a, const T* b, T* result, size_t size);
    static void mulKernel(const T* a, const T* b, T* result, size_t size);
};

// 定义CUDA核函数
template<typename T>
__global__ void addKernelImpl(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void mulKernelImpl(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

// 实现CUDA核函数
template<typename T>
void GpuTensor<T>::addKernel(const T* a, const T* b, T* result, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    addKernelImpl<<<gridSize, blockSize>>>(a, b, result, size);
    cudaDeviceSynchronize();
}

template<typename T>
void GpuTensor<T>::mulKernel(const T* a, const T* b, T* result, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    mulKernelImpl<<<gridSize, blockSize>>>(a, b, result, size);
    cudaDeviceSynchronize();
}