#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

class GPUManager {
public:
    static GPUManager& getInstance() {
        static GPUManager instance;
        return instance;
    }

    bool isAvailable() const {
        return cuda_available;
    }

    int getDeviceCount() const {
        return device_count;
    }

    void* allocate(size_t size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }

    void copyHostToDevice(void* dst, const void* src, size_t size) {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy memory from host to device: " + std::string(cudaGetErrorString(err)));
        }
    }

    void copyDeviceToHost(void* dst, const void* src, size_t size) {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy memory from device to host: " + std::string(cudaGetErrorString(err)));
        }
    }

    void copyDeviceToDevice(void* dst, const void* src, size_t size) {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy memory from device to device: " + std::string(cudaGetErrorString(err)));
        }
    }

    void free(void* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    void synchronize() {
        cudaDeviceSynchronize();
    }

private:
    bool cuda_available = false;
    int device_count = 0;

    GPUManager() {
        cudaError_t err = cudaGetDeviceCount(&device_count);
        cuda_available = (err == cudaSuccess && device_count > 0);
    }

    ~GPUManager() = default;
    GPUManager(const GPUManager&) = delete;
    GPUManager& operator=(const GPUManager&) = delete;
};