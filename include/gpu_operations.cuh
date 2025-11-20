#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>

// CUDA核函数实现
template<typename T>
__global__ void addKernel(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void mulKernel(const T* a, const T* b, T* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

template<typename T>
__global__ void geluKernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        const T sqrt_2_over_pi = sqrt(2.0f / M_PI);
        T x_cubed = x * x * x;
        T inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanh(inner));
    }
}

template<typename T>
__global__ void siluKernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        output[idx] = x / (1 + exp(-x));
    }
}

template<typename T>
__global__ void softmaxKernel(T* input, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        // 找到当前行的最大值
        T max_val = input[row * cols];
        for (int j = 1; j < cols; j++) {
            T val = input[row * cols + j];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 计算exp之和
        T exp_sum = 0;
        for (int j = 0; j < cols; j++) {
            input[row * cols + j] = exp(input[row * cols + j] - max_val);
            exp_sum += input[row * cols + j];
        }

        // 归一化
        for (int j = 0; j < cols; j++) {
            input[row * cols + j] /= exp_sum;
        }
    }
}

template<typename T>
__global__ void reluKernel(const T* input, T* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T x = input[idx];
        output[idx] = x > 0 ? x : 0;
    }
}

template<typename T>
__global__ void transposeKernel(const T* input, T* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // 转置：output[col][row] = input[row][col]
        output[col * rows + row] = input[row * cols + col];
    }
}

// CUDA操作类
class CudaOperations {
public:
    static cublasHandle_t getCublasHandle() {
        static cublasHandle_t handle = nullptr;
        if (!handle) {
            cublasCreate(&handle);
        }
        return handle;
    }

    // 矩阵乘法
    template<typename T>
    static void matMul(const T* A, const T* B, T* C, int m, int n, int k) {
        cublasHandle_t handle = getCublasHandle();
        
        const T alpha = 1.0f;
        const T beta = 0.0f;
        
        cublasStatus_t status = cublasSgemm(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           n, m, k,
                                           &alpha,
                                           B, n,
                                           A, k,
                                           &beta,
                                           C, n);
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS matrix multiplication failed");
        }
    }

    // 张量加法
    template<typename T>
    static void add(const T* a, const T* b, T* result, size_t size) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        addKernel<<<gridSize, blockSize>>>(a, b, result, size);
        cudaDeviceSynchronize();
    }

    // 张量乘法
    template<typename T>
    static void mul(const T* a, const T* b, T* result, size_t size) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        mulKernel<<<gridSize, blockSize>>>(a, b, result, size);
        cudaDeviceSynchronize();
    }

    // GELU激活函数
    template<typename T>
    static void gelu(const T* input, T* output, size_t size) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        geluKernel<<<gridSize, blockSize>>>(input, output, size);
        cudaDeviceSynchronize();
    }

    // SiLU激活函数
    template<typename T>
    static void silu(const T* input, T* output, size_t size) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        siluKernel<<<gridSize, blockSize>>>(input, output, size);
        cudaDeviceSynchronize();
    }

    // Softmax
    template<typename T>
    static void softmax(T* input, int rows, int cols) {
        dim3 blockSize(1, 256);
        dim3 gridSize(rows);
        softmaxKernel<<<gridSize, blockSize>>>(input, rows, cols);
        cudaDeviceSynchronize();
    }

    // ReLU激活函数
    template<typename T>
    static void relu(const T* input, T* output, size_t size) {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        reluKernel<<<gridSize, blockSize>>>(input, output, size);
        cudaDeviceSynchronize();
    }

    // 矩阵转置
    template<typename T>
    static void transpose(const T* input, T* output, int rows, int cols) {
        dim3 blockSize(16, 16);
        dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
        transposeKernel<<<gridSize, blockSize>>>(input, output, rows, cols);
        cudaDeviceSynchronize();
    }
};