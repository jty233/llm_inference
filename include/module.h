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
}


template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    Tensor<T> res = input;
    for (auto& x : res.data) {
        if (x < 0) {
            x = 0;
        }
    }
}



template<typename T>
Tensor<T> softmax(const Tensor<T>& input) {
    Tensor<T> res = input;
    auto shape = res.shape;
    shape.pop_back();
    Tensor<T>::forEachDim(shape, [&] (const std::vector<int>& dim_) {
        auto dim = dim_;
        T max_val = -1e9;
        for (int i = 0; i < res.shape.back(); i++) {
            dim.push_back(i);
            max_val = std::max(max_val, res.at(dim));
            dim.pop_back();
        }

        double exp_sum = 0;
        for (int i = 0; i < res.shape.back(); i++) {
            dim.push_back(i);
            exp_sum += exp(res.at(dim));
            dim.pop_back();
        }

        for (int i = 0; i < res.shape.back(); i++) {
            dim.push_back(i);
            res.at(dim) = exp(res.at(dim)) / exp_sum;
            dim.pop_back();
        }
    });
    return res;
}