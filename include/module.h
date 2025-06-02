#pragma once
#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <vector>

template<typename T>
Tensor<T> gelu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / std::acos(-1));
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        x = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return res;
}


template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        if (x < 0) {
            x = 0;
        }
    }
    return res;
}

template<typename T>
Tensor<T> silu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        x /= 1 + std::exp(-x);
    }
    return res;
}



template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    Tensor<T> res = input;
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