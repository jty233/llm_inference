#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <vector>

// 包含CUDA头文件（如果可用）
#ifdef __CUDACC__
#include "gpu_operations.cuh"
#endif

template<typename T>
Tensor<T> gelu(const Tensor<T>& input) {
    Tensor<T> res;
    if (input.getDevice() == TensorDevice::GPU) {
#ifdef __CUDACC__
        // GPU实现
        res = input;  // 创建副本
        res.toCPU();  // 临时将结果转回CPU以保持兼容性
        // TODO: 实现完整的GPU GELU
        for (auto& x : res.data) {
            const float sqrt_2_over_pi = std::sqrt(2.0f / std::acos(-1));
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
            x = 0.5f * x * (1.0f + std::tanh(inner));
        }
#else
        // 如果没有CUDA支持，即使标记为GPU也使用CPU实现
        res = input;
        for (auto& x : res.data) {
            const float sqrt_2_over_pi = std::sqrt(2.0f / std::acos(-1));
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
            x = 0.5f * x * (1.0f + std::tanh(inner));
        }
#endif
    } else {
        // CPU实现
        res = input;
        for (auto& x : res.data) {
            const float sqrt_2_over_pi = std::sqrt(2.0f / std::acos(-1));
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
            x = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
    return res;
}


template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> res;
    if (input.getDevice() == TensorDevice::GPU) {
#ifdef __CUDACC__
        // GPU实现
        res = input;  // 创建副本
        res.toCPU();  // 临时将结果转回CPU以保持兼容性
        // TODO: 实现完整的GPU ReLU
        for (auto& x : res.data) {
            if (x < 0) {
                x = 0;
            }
        }
#else
        // 如果没有CUDA支持，即使标记为GPU也使用CPU实现
        res = input;
        for (auto& x : res.data) {
            if (x < 0) {
                x = 0;
            }
        }
#endif
    } else {
        // CPU实现
        res = input;
        for (auto& x : res.data) {
            if (x < 0) {
                x = 0;
            }
        }
    }
    return res;
}

template<typename T>
Tensor<T> silu(const Tensor<T>& input) {
    Tensor<T> res;
    if (input.getDevice() == TensorDevice::GPU) {
#ifdef __CUDACC__
        // GPU实现
        res = input;  // 创建副本
        res.toCPU();  // 临时将结果转回CPU以保持兼容性
        // TODO: 实现完整的GPU SiLU
        for (auto& x : res.data) {
            x /= 1 + std::exp(-x);
        }
#else
        // 如果没有CUDA支持，即使标记为GPU也使用CPU实现
        res = input;
        for (auto& x : res.data) {
            x /= 1 + std::exp(-x);
        }
#endif
    } else {
        // CPU实现
        res = input;
        for (auto& x : res.data) {
            x /= 1 + std::exp(-x);
        }
    }
    return res;
}



template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    Tensor<T> res = input;
    if (input.getDevice() == TensorDevice::GPU) {
#ifdef __CUDACC__
        // GPU实现
        if (input.getGpuPtr() != nullptr) {
            // 将数据复制到临时CPU张量进行计算（简化实现）
            res.toCPU();
            
            auto shape = res.shape;
            shape.pop_back();
            res.forEachDim(shape, [&] (std::vector<int> dim) {
                double max_val = -1e9;
                dim.push_back(0);
                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    max_val = std::max(max_val, (double)res.at(dim));
                }

                double exp_sum = 0;
                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    exp_sum += exp(res.at(dim) - max_val);
                }

                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    res.at(dim) = exp(res.at(dim) - max_val) / exp_sum;
                }
            });
            
            // 结果重新移到GPU
            res.toGPU();
        } else {
            // 如果GPU指针为空，使用CPU实现
            auto shape = res.shape;
            shape.pop_back();
            res.forEachDim(shape, [&] (std::vector<int> dim) {
                double max_val = -1e9;
                dim.push_back(0);
                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    max_val = std::max(max_val, (double)res.at(dim));
                }

                double exp_sum = 0;
                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    exp_sum += exp(res.at(dim) - max_val);
                }

                for (int i = 0; i < res.shape.back(); i++) {
                    dim.back() = i;
                    res.at(dim) = exp(res.at(dim) - max_val) / exp_sum;
                }
            });
        }
#else
        // 如果没有CUDA支持，即使标记为GPU也使用CPU实现
        auto shape = res.shape;
        shape.pop_back();
        res.forEachDim(shape, [&] (std::vector<int> dim) {
            double max_val = -1e9;
            dim.push_back(0);
            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                max_val = std::max(max_val, (double)res.at(dim));
            }

            double exp_sum = 0;
            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                exp_sum += exp(res.at(dim) - max_val);
            }

            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                res.at(dim) = exp(res.at(dim) - max_val) / exp_sum;
            }
        });
#endif
    } else {
        // CPU实现
        auto shape = res.shape;
        shape.pop_back();
        res.forEachDim(shape, [&] (std::vector<int> dim) {
            double max_val = -1e9;
            dim.push_back(0);
            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                max_val = std::max(max_val, (double)res.at(dim));
            }

            double exp_sum = 0;
            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                exp_sum += exp(res.at(dim) - max_val);
            }

            for (int i = 0; i < res.shape.back(); i++) {
                dim.back() = i;
                res.at(dim) = exp(res.at(dim) - max_val) / exp_sum;
            }
        });
    }
    return res;
}

template<typename T>
Tensor<T> apply_RoPE(const Tensor<T>& input, double rope_base, int f_token_num) {
    Tensor<T> res;
    auto shape = input.shape;
    res.asShape(shape);
    int d = shape.back();
    shape.pop_back();
    res.forEachDim(shape, [&] (std::vector<int> dim) {
        int m = dim.back() + f_token_num;
        dim.push_back(0);
        int off = input.idxs2Offset(dim);
        for (int i = 0; i < d / 2; i++) {
            double theta = std::pow(rope_base, -2. * i / d);
            res.data[off + i] = input.data[off + i] * cos(m * theta) - input.data[off + i + d / 2] * sin(m * theta);
        }
        for (int i = d / 2; i < d; i++) {
            double theta = std::pow(rope_base, -2. * (i - d / 2) / d); // NOLINT
            res.data[off + i] = input.data[off + i] * cos(m * theta) + input.data[off + i - d / 2] * sin(m * theta);
        }
    });
    return res;
}